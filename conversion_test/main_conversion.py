#!/usr/bin/env python3
"""
Main conversion script to run inference on preprocessed TSV files.

This script:
1. Loads a TSV file from preprocessing.py
2. Runs inference for selected chat models
3. Compares model answers with correct answers
4. Outputs [domain]_converted.tsv with additional columns
"""

import json
import pandas as pd
import argparse
import asyncio
from pathlib import Path
from typing import Optional, Union, List
import re
import os
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import time

# Load API keys
def load_api_key(key_file: str) -> str:
    """Load API key from file."""
    try:
        with open(key_file, 'r') as f:
            api_key = f.read().strip()
        if not api_key:
            raise ValueError(f"API key file '{key_file}' is empty")
        return api_key
    except FileNotFoundError:
        raise FileNotFoundError(
            f"API key file '{key_file}' not found. "
            f"Please create the file with your API key."
        )

# Initialize API clients
openai_key = os.environ.get("OPENAI_API_KEY") or load_api_key("openai_key.txt")
together_key = os.environ.get("TOGETHER_API_KEY") or load_api_key("together_ai_key.txt")

openai_client = AsyncOpenAI(api_key=openai_key)
together_client = AsyncOpenAI(
    api_key=together_key,
    base_url="https://api.together.xyz/v1"
)

# Rate limiting for Together AI
class RateLimiter:
    """Rate limiter to prevent exceeding API limits."""
    def __init__(self, max_requests_per_minute: int = 200):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait if necessary to stay within rate limit."""
        async with self.lock:
            now = time.time()
            
            # Remove requests older than 60 seconds
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            # If we're at the limit, only wait for the oldest request to expire
            if len(self.requests) >= self.max_requests:
                oldest_request = self.requests[0]
                time_since_oldest = now - oldest_request
                sleep_time = 60 - time_since_oldest + 0.05
                
                if sleep_time > 0:
                    if sleep_time > 0.5:
                        print(f"Rate limit: waiting {sleep_time:.1f}s for slot to open...")
                    await asyncio.sleep(sleep_time)
                    
                    now = time.time()
                    self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            self.requests.append(now)

together_rate_limiter = RateLimiter(max_requests_per_minute=200)

# Model configurations
MODEL_CONFIGS = {
    "gpt-4o": {"provider": "openai", "model": "gpt-4o"},
    "qwen-coder": {"provider": "together", "model": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"},
    "llama-4": {"provider": "together", "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"}
}

async def ask_openai(prompt: str, model: str, semaphore: asyncio.Semaphore, is_timezone: bool = False) -> str:
    """Ask OpenAI model to perform conversion."""
    async with semaphore:
        try:
            if is_timezone:
                system_content = "You are a precise timezone conversion expert. Provide the time in the same format as the input (e.g., 1AM, 3:49PM), no explanations."
            else:
                system_content = "You are a precise unit conversion expert. Provide only the numerical answer with up to 4 decimal places, no units or explanations."
            
            response = await openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=500
            )
            content = response.choices[0].message.content.strip()
            finish_reason = response.choices[0].finish_reason
            
            if finish_reason == 'length':
                if not is_timezone:
                    numbers = re.findall(r'-?\d+\.?\d*[eE][+-]?\d*', content)
                    if not numbers:
                        numbers = re.findall(r'-?\d+\.?\d*', content)
                    if numbers:
                        return f"{numbers[-1]} [TRUNCATED]"
                return f"{content} [TRUNCATED]"
            
            return content
        except Exception as e:
            return f"ERROR: {str(e)}"

async def ask_together(prompt: str, model: str, semaphore: asyncio.Semaphore, retry_count: int = 0, is_timezone: bool = False) -> str:
    """Ask Together AI model to perform conversion."""
    await together_rate_limiter.acquire()
    
    async with semaphore:
        try:
            if is_timezone:
                system_content = "You are a precise timezone conversion expert. Provide the time in the same format as the input (e.g., 1AM, 3:49PM), no explanations."
            else:
                system_content = "You are a precise unit conversion expert. Provide only the numerical answer with up to 4 decimal places, no units or explanations."
            
            response = await together_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1000
            )
            content = response.choices[0].message.content.strip()
            finish_reason = response.choices[0].finish_reason
            
            if finish_reason == 'length':
                if not is_timezone:
                    numbers = re.findall(r'-?\d+\.?\d*[eE][+-]?\d*', content)
                    if not numbers:
                        numbers = re.findall(r'-?\d+\.?\d*', content)
                    if numbers:
                        return f"{numbers[-1]} [TRUNCATED]"
                return f"{content} [TRUNCATED]"
            
            return content
        except Exception as e:
            error_msg = str(e)
            if ("rate" in error_msg.lower() or "429" in error_msg) and retry_count < 3:
                wait_time = 5 * (2 ** retry_count)
                print(f"Rate limit error (attempt {retry_count + 1}/3), waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
                return await ask_together(prompt, model, semaphore, retry_count + 1, is_timezone)
            return f"ERROR: {error_msg}"

async def get_model_answer(
    prompt: str,
    model_name: str,
    semaphore: asyncio.Semaphore,
    is_timezone: bool = False
) -> str:
    """Route question to appropriate model provider."""
    config = MODEL_CONFIGS.get(model_name)
    if not config:
        return f"ERROR: Unknown model {model_name}"
    
    provider = config["provider"]
    model = config["model"]
    
    if provider == "openai":
        return await ask_openai(prompt, model, semaphore, is_timezone)
    elif provider == "together":
        return await ask_together(prompt, model, semaphore, is_timezone=is_timezone)
    else:
        return f"ERROR: Unknown provider {provider}"

def extract_number(answer: str) -> Optional[float]:
    """Extract numeric value from answer string, handling scientific notation."""
    if not answer or answer.startswith('ERROR:'):
        return None
    
    cleaned = answer.replace("[TRUNCATED]", "").replace(",", "").strip()
    
    # Try scientific notation first
    sci_pattern = re.search(r'-?\d+\.?\d*[eE][+-]?\d+', cleaned)
    if sci_pattern:
        try:
            return float(sci_pattern.group(0))
        except ValueError:
            pass
    
    # Fall back to regular number extraction
    matches = re.findall(r'-?\d+\.?\d*', cleaned)
    if matches:
        for match in reversed(matches):
            try:
                num = float(match)
                if abs(num) > 1e-10 or abs(num) == 0:
                    return num
            except ValueError:
                continue
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None

def extract_time_string(answer: str) -> Optional[str]:
    """Extract time string from answer (e.g., '1AM', '3:49PM')."""
    # Pattern 1: HH:MMAM/PM or HH:MM AM/PM
    time_pattern1 = re.search(r'(\d{1,2}):(\d{2})\s*(AM|PM)', answer, re.IGNORECASE)
    if time_pattern1:
        h = int(time_pattern1.group(1))
        m = int(time_pattern1.group(2))
        period = time_pattern1.group(3).upper()
        return f"{h}:{m:02d}{period}"
    
    # Pattern 2: HHAM/PM or HH AM/PM
    time_pattern2 = re.search(r'(\d{1,2})\s*(AM|PM)', answer, re.IGNORECASE)
    if time_pattern2:
        h = int(time_pattern2.group(1))
        period = time_pattern2.group(2).upper()
        return f"{h}{period}"
    
    return None

def extract_clothing_size(answer: str) -> Optional[str]:
    """Extract clothing size from answer (e.g., 'XS', 'M', 'L', '32', '34', '32B', '70A')."""
    # Try to match bra sizes first (format: number + letter, e.g., "32B", "70A")
    bra_pattern = re.search(r'\b(\d{2,3})([A-Z])\b', answer, re.IGNORECASE)
    if bra_pattern:
        return f"{bra_pattern.group(1)}{bra_pattern.group(2).upper()}"
    
    # Try to match standard alphanumeric sizes
    size_pattern = re.search(r'\b(XS|S|M|L|XL|XXL|XXXL)\b', answer, re.IGNORECASE)
    if size_pattern:
        return size_pattern.group(1).upper()
    
    # Try to match numeric sizes (EU sizes, pant sizes, etc.)
    numeric_pattern = re.search(r'\b(\d{1,3})\b', answer)
    if numeric_pattern:
        return numeric_pattern.group(1)
    
    return None

def calculate_loss(
    model_answer: Union[float, str, None],
    correct_answer: Union[float, str],
    answer_type: str = 'numeric'
) -> Optional[float]:
    """Calculate loss/difference between model answer and correct answer."""
    if model_answer is None:
        return None
    
    if answer_type == 'numeric':
        try:
            model_num = float(model_answer)
            correct_num = float(correct_answer)
            
            if correct_num == 0:
                return abs(model_num)
            else:
                # Relative error as percentage
                return abs((model_num - correct_num) / correct_num) * 100
        except (ValueError, TypeError):
            return None
    
    elif answer_type == 'timezone':
        # For timezone, calculate difference in minutes
        def parse_time_string(time_str: str) -> float:
            """Parse time string to hours (0-24)."""
            time_str = time_str.strip().upper()
            
            match = re.match(r'(\d{1,2}):(\d{2})(AM|PM)', time_str)
            if match:
                h = int(match.group(1))
                m = int(match.group(2))
                period = match.group(3)
                if period == 'PM' and h != 12:
                    h += 12
                elif period == 'AM' and h == 12:
                    h = 0
                return h + m / 60.0
            
            match = re.match(r'(\d{1,2})(AM|PM)', time_str)
            if match:
                h = int(match.group(1))
                period = match.group(2)
                if period == 'PM' and h != 12:
                    h += 12
                elif period == 'AM' and h == 12:
                    h = 0
                return float(h)
            
            raise ValueError(f"Could not parse time: {time_str}")
        
        try:
            model_hours = parse_time_string(str(model_answer))
            correct_hours = parse_time_string(str(correct_answer))
            
            diff_minutes = abs((model_hours - correct_hours) * 60)
            
            # Handle day rollover
            if diff_minutes > 12 * 60:
                diff_minutes = 24 * 60 - diff_minutes
            
            return diff_minutes
        except (ValueError, TypeError):
            return None
    
    elif answer_type == 'clothing':
        # For clothing sizes, return 0 if match, 1 if mismatch
        return 0.0 if str(model_answer).strip().upper() == str(correct_answer).strip().upper() else 1.0
    
    else:
        # For other types, return 0 if exact match, 1 otherwise
        return 0.0 if str(model_answer).strip() == str(correct_answer).strip() else 1.0

def determine_answer_type(domain: str, answer: Union[float, str]) -> str:
    """Determine the type of answer based on domain and answer value."""
    if domain == 'timezone' or 'timezone' in domain:
        return 'timezone'
    elif 'clothing_sizes' in domain or 'bra_size' in domain:
        return 'clothing'
    elif isinstance(answer, str) and not answer.replace('.', '').replace('-', '').isdigit():
        # Non-numeric string
        return 'string'
    else:
        return 'numeric'

async def process_row(
    row: pd.Series,
    model_name: str,
    semaphore: asyncio.Semaphore
) -> dict:
    """Process a single row: run inference and extract answer."""
    prompt = row['prompt']
    domain = row['domain']
    correct_answer = row['answer']
    
    # Determine if this is a timezone conversion
    is_timezone = (domain == 'timezone')
    
    # Get raw model response
    raw_response = await get_model_answer(prompt, model_name, semaphore, is_timezone)
    
    # Extract model answer based on domain
    answer_type = determine_answer_type(domain, correct_answer)
    
    if answer_type == 'timezone':
        model_answer = extract_time_string(raw_response)
    elif answer_type == 'clothing':
        model_answer = extract_clothing_size(raw_response)
    else:
        model_answer = extract_number(raw_response)
    
    # Calculate loss
    loss = calculate_loss(model_answer, correct_answer, answer_type)
    
    return {
        'raw_response': raw_response,
        'model_answer': model_answer,
        'loss': loss
    }

async def process_dataframe(
    df: pd.DataFrame,
    model_name: str,
    semaphore: asyncio.Semaphore
) -> pd.DataFrame:
    """Process entire dataframe with async inference."""
    results = []
    
    tasks = []
    for idx, row in df.iterrows():
        task = process_row(row, model_name, semaphore)
        tasks.append(task)
    
    # Run all tasks with progress bar
    results = await tqdm.gather(*tasks, desc=f"Processing {model_name}")
    
    # Add results to dataframe
    for idx, result in enumerate(results):
        df.at[idx, 'raw_response'] = result['raw_response']
        df.at[idx, 'model_answer'] = result['model_answer']
        df.at[idx, 'loss'] = result['loss']
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Run conversion inference on preprocessed TSV files')
    parser.add_argument('--domain', type=str, required=True,
                       help='Domain name (e.g., temperature, volume)')
    parser.add_argument('--input-file', type=str, required=True,
                       help='Input TSV file from preprocessing.py')
    parser.add_argument('--output-file', type=str, default=None,
                       help='Output TSV file (default: [domain]_converted.tsv)')
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['gpt-4o', 'qwen-coder', 'llama-4'],
                       help='Models to run inference on')
    parser.add_argument('--max-concurrent', type=int, default=10,
                       help='Maximum concurrent API requests')
    
    args = parser.parse_args()
    
    # Load input TSV
    input_file = Path(args.input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file, sep='\t')
    print(f"Loaded {len(df)} rows")
    
    # Determine output file
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = Path(f"{args.domain}_converted.tsv")
    
    # Initialize columns for results (single model or multiple models)
    if len(args.models) == 1:
        # Single model: use simple column names
        df['raw_response'] = None
        df['model_answer'] = None
        df['loss'] = None
    else:
        # Multiple models: use model suffix
        for model_name in args.models:
            df[f'raw_response_{model_name}'] = None
            df[f'model_answer_{model_name}'] = None
            df[f'loss_{model_name}'] = None
    
    # Process each model
    semaphore = asyncio.Semaphore(args.max_concurrent)
    
    for model_name in args.models:
        if model_name not in MODEL_CONFIGS:
            print(f"Warning: Unknown model {model_name}, skipping...")
            continue
        
        print(f"\nProcessing with {model_name}...")
        
        # Create a copy of the dataframe for this model
        df_model = df.copy()
        
        # Process dataframe
        df_model = asyncio.run(process_dataframe(df_model, model_name, semaphore))
        
        # Update main dataframe with results
        if len(args.models) == 1:
            df['raw_response'] = df_model['raw_response']
            df['model_answer'] = df_model['model_answer']
            df['loss'] = df_model['loss']
        else:
            df[f'raw_response_{model_name}'] = df_model['raw_response']
            df[f'model_answer_{model_name}'] = df_model['model_answer']
            df[f'loss_{model_name}'] = df_model['loss']
    
    # Save output
    print(f"\nSaving results to {output_file}...")
    df.to_csv(output_file, sep='\t', index=False)
    print(f"Saved {len(df)} rows to {output_file}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for model_name in args.models:
        if model_name not in MODEL_CONFIGS:
            continue
        
        if len(args.models) == 1:
            loss_col = 'loss'
        else:
            loss_col = f'loss_{model_name}'
        
        if loss_col in df.columns:
            valid_losses = df[loss_col].dropna()
            if len(valid_losses) > 0:
                print(f"\n{model_name}:")
                print(f"  Valid answers: {len(valid_losses)}/{len(df)}")
                print(f"  Mean loss: {valid_losses.mean():.4f}")
                print(f"  Median loss: {valid_losses.median():.4f}")
                print(f"  Zero loss (correct): {(valid_losses == 0).sum()}")

if __name__ == '__main__':
    main()
