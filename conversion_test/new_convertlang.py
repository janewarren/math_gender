import json
import os
import asyncio
import random
from openai import AsyncOpenAI
from typing import List, Dict, Tuple
from tqdm.asyncio import tqdm
import aiofiles
from datetime import datetime
import pandas as pd
import time
import signal
import sys

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
                # Calculate how long until the oldest request is 60 seconds old
                oldest_request = self.requests[0]
                time_since_oldest = now - oldest_request
                sleep_time = 60 - time_since_oldest + 0.05  # Small 50ms buffer
                
                if sleep_time > 0:
                    # Only print if sleep time is significant
                    if sleep_time > 0.5:
                        print(f"Rate limit: waiting {sleep_time:.1f}s for slot to open...")
                    await asyncio.sleep(sleep_time)
                    
                    # Clean up again after sleeping
                    now = time.time()
                    self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            # Record this request
            self.requests.append(now)

# Create rate limiters
together_rate_limiter = RateLimiter(max_requests_per_minute=200)  # Adjust based on your tier



# Graceful shutdown handler
class GracefulShutdown:
    """Handle graceful shutdown on Ctrl+C."""
    def __init__(self):
        self.shutdown_requested = False
        self.partial_results = []
        
    def request_shutdown(self, signum, frame):
        """Signal handler for graceful shutdown."""
        print("\n\n" + "="*60)
        print("SHUTDOWN REQUESTED - Saving progress...")
        print("="*60)
        self.shutdown_requested = True

shutdown_handler = GracefulShutdown()

# Model configurations
MODEL_CONFIGS = {
    "gpt-4o": {"provider": "openai", "model": "gpt-4o"},
    "qwen-coder": {"provider": "together", "model": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"},
    "llama-4": {"provider": "together", "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"}
    # ,
    # "deepseek-r1": {"provider": "together", "model": "deepseek-ai/DeepSeek-R1"}
}

# Conversion constants (to milliliters)
CONVERSION_TO_ML = {
    # US measurements
    'teaspoon': 4.92892,
    'tablespoon': 14.7868,
    'fluid_ounce': 29.5735,
    'cup': 236.588,
    'pint': 473.176,
    'quart': 946.353,
    # Metric measurements
    'milliliter': 1.0,
    'liter': 1000.0
}

# Display names for units
UNIT_DISPLAY_NAMES = {
    'teaspoon': 'teaspoon',
    'tablespoon': 'tablespoon',
    'fluid_ounce': 'fluid ounce',
    'cup': 'cup',
    'pint': 'pint',
    'quart': 'quart',
    'milliliter': 'milliliter',
    'liter': 'liter'
}

# US units and Metric units
US_UNITS = ['teaspoon', 'tablespoon', 'fluid_ounce', 'cup', 'pint', 'quart']
METRIC_UNITS = ['milliliter', 'liter']

# Standardized test values
TEST_VALUES = {
    'easy': [1, 5, 10, 20, 50],
    'hard': [4508.208, 1297.195, 18.333, 9.0241, 0.2994],
    'random': [5718, 1241.43, 3959.435, 12.505, 9717.519]
}

# Flatten into single list in order: easy, hard, random
ALL_TEST_VALUES = TEST_VALUES['easy'] + TEST_VALUES['hard'] + TEST_VALUES['random']

def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between any two volume units."""
    # Convert to milliliters first
    ml_value = value * CONVERSION_TO_ML[from_unit]
    # Then convert to target unit
    result = ml_value / CONVERSION_TO_ML[to_unit]
    return result



def load_dataset(file_path: str, sample_size: int = 50) -> pd.DataFrame:
    """Load dataset from CSV file and sample rows."""
    print(f"Loading dataset from {file_path}...")
    
    df = pd.read_csv(file_path)
    
    # Detect language columns automatically
    columns = df.columns.tolist()
    
    # Look for english_name column
    if 'english_name' not in columns:
        raise ValueError("CSV must contain 'english_name' column")
    
    # Find the other language column (should be second column with '_name' suffix)
    other_lang_col = None
    other_lang = None
    
    for col in columns:
        if col != 'english_name' and col.endswith('_name'):
            other_lang_col = col
            # Extract language name (e.g., 'hindi_name' -> 'hindi')
            other_lang = col.replace('_name', '')
            break
    
    if other_lang_col is None:
        raise ValueError(f"Could not find second language column. Columns found: {columns}")
    
    print(f"Detected languages: English and {other_lang.capitalize()}")
    
    # Rename columns to standardized names for processing
    df = df.rename(columns={
        'english_name': 'lang1_name',
        other_lang_col: 'lang2_name'
    })
    
    # Store language names as metadata
    df.attrs['lang1'] = 'english'
    df.attrs['lang2'] = other_lang
    
    # Sample rows
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    # Preserve metadata
    df_sample.attrs = df.attrs
    
    print(f"Sampled {len(df_sample)} rows")
    return df_sample

def create_conversion_tasks(df: pd.DataFrame, include_context_free: bool = True) -> List[Dict]:
    """Create conversion tasks from dataset with standardized test values."""
    tasks = []
    
    # Get language names from DataFrame metadata
    lang1 = df.attrs.get('lang1', 'english')
    lang2 = df.attrs.get('lang2', 'other')
    
    print(f"\nCreating conversion tasks for {lang1} and {lang2}...")
    print(f"Test values per conversion: {len(ALL_TEST_VALUES)}")
    
    # Generate all possible unit conversion pairs (US ↔ Metric only)
    conversion_pairs = []
    
    # US to Metric conversions
    for us_unit in US_UNITS:
        for metric_unit in METRIC_UNITS:
            conversion_pairs.append({
                'from_unit': us_unit,
                'to_unit': metric_unit,
                'conversion_type': 'us_to_metric'
            })
    
    # Metric to US conversions
    for metric_unit in METRIC_UNITS:
        for us_unit in US_UNITS:
            conversion_pairs.append({
                'from_unit': metric_unit,
                'to_unit': us_unit,
                'conversion_type': 'metric_to_us'
            })
    
    print(f"Total conversion pairs: {len(conversion_pairs)}")
    
    # For each ingredient
    for idx, row in df.iterrows():
        lang1_name = row['lang1_name']
        lang2_name = row['lang2_name']
        
        # For each conversion pair
        for pair in conversion_pairs:
            from_unit = pair['from_unit']
            to_unit = pair['to_unit']
            conversion_type = pair['conversion_type']
            
            # For each test value
            for test_value in ALL_TEST_VALUES:
                # Calculate correct answer
                correct_answer = convert_units(test_value, from_unit, to_unit)
                
                # Determine difficulty category
                if test_value in TEST_VALUES['easy']:
                    difficulty = 'easy'
                elif test_value in TEST_VALUES['hard']:
                    difficulty = 'hard'
                else:
                    difficulty = 'random'
                
                # Create task for first language (English)
                tasks.append({
                    'ingredient': lang1_name,
                    'language': lang1,
                    'context_type': 'ingredient',
                    'test_value': test_value,
                    'difficulty': difficulty,
                    'from_unit': from_unit,
                    'to_unit': to_unit,
                    'conversion_type': conversion_type,
                    'correct_answer': round(correct_answer, 4)
                })
                
                # Create task for second language
                tasks.append({
                    'ingredient': lang2_name,
                    'language': lang2,
                    'context_type': 'ingredient',
                    'test_value': test_value,
                    'difficulty': difficulty,
                    'from_unit': from_unit,
                    'to_unit': to_unit,
                    'conversion_type': conversion_type,
                    'correct_answer': round(correct_answer, 4)
                })
                
                # Add context-free version (no ingredient mentioned)
                if include_context_free:
                    tasks.append({
                        'ingredient': None,
                        'language': 'context_free',
                        'context_type': 'context_free',
                        'test_value': test_value,
                        'difficulty': difficulty,
                        'from_unit': from_unit,
                        'to_unit': to_unit,
                        'conversion_type': conversion_type,
                        'correct_answer': round(correct_answer, 4)
                    })
    
    context_free_count = sum(1 for t in tasks if t['context_type'] == 'context_free')
    print(f"\nCreated {len(tasks)} conversion tasks:")
    print(f"  - Ingredients: {len(df)}")
    print(f"  - Conversion pairs: {len(conversion_pairs)}")
    print(f"  - Test values per pair: {len(ALL_TEST_VALUES)}")
    print(f"  - Tasks per ingredient language: {len(conversion_pairs) * len(ALL_TEST_VALUES)}")
    print(f"  - With ingredient context: {len(tasks) - context_free_count}")
    if include_context_free:
        print(f"  - Context-free: {context_free_count}")
    
    return tasks

async def load_existing_results(output_dir: str) -> tuple[List[Dict], set]:
    """Load existing results from previous runs to enable resume."""
    existing_results = []
    completed_tasks = set()
    
    # Look for all result files in output directory
    result_files = []
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.startswith('language_conversion_') and filename.endswith('.json'):
                result_files.append(os.path.join(output_dir, filename))
            elif filename.startswith('checkpoint_') and filename.endswith('.json'):
                result_files.append(os.path.join(output_dir, filename))
    
    if not result_files:
        print("No existing results found - starting fresh")
        return existing_results, completed_tasks
    
    # Load most recent file
    result_files.sort(key=os.path.getmtime, reverse=True)
    most_recent = result_files[0]
    
    print(f"\nFound existing results: {os.path.basename(most_recent)}")
    
    try:
        async with aiofiles.open(most_recent, 'r') as f:
            content = await f.read()
            existing_results = json.loads(content)
        
        # Create set of completed task signatures
        for result in existing_results:
            # For context-free tasks, ignore ingredient in signature since prompt is identical
            ingredient = None if result.get('context_type') == 'context_free' else result.get('ingredient')
            task_signature = (
                result['model'],
                ingredient,
                result['language'],
                result['test_value'],
                result['from_unit'],
                result['to_unit']
            )
            completed_tasks.add(task_signature)
        
        print(f"Loaded {len(existing_results)} existing results")
        print(f"Unique completed tasks: {len(completed_tasks)}")
        
        return existing_results, completed_tasks
    
    except Exception as e:
        print(f"Error loading existing results: {e}")
        return [], set()

def filter_completed_tasks(tasks: List[Dict], completed_tasks: set, model_name: str) -> List[Dict]:
    """Filter out tasks that have already been completed for this model."""
    remaining_tasks = []
    
    for task in tasks:
        # For context-free tasks, ignore ingredient in signature since prompt is identical
        ingredient = None if task.get('context_type') == 'context_free' else task.get('ingredient')
        task_signature = (
            model_name,
            ingredient,
            task['language'],
            task['test_value'],
            task['from_unit'],
            task['to_unit']
        )
        
        if task_signature not in completed_tasks:
            remaining_tasks.append(task)
    
    return remaining_tasks

async def ask_openai(prompt: str, model: str, semaphore: asyncio.Semaphore) -> str:
    """Ask OpenAI model to perform conversion."""
    async with semaphore:
        try:
            response = await openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise unit conversion expert. Provide only the numerical answer with up to 4 decimal places, no units or explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"ERROR: {str(e)}"

async def ask_together(prompt: str, model: str, semaphore: asyncio.Semaphore, retry_count: int = 0) -> str:
    """Ask Together AI model to perform conversion."""
    # Apply rate limiting before making request
    await together_rate_limiter.acquire()
    
    async with semaphore:
        try:
            # Special handling for DeepSeek-R1
            if "deepseek" in model.lower():
                # For DeepSeek, we need VERY high limits for thinking
                # But we can't control thinking length, so maximize everything
                response = await together_client.chat.completions.create(
                    model=model,
                    messages=[
                        # Stronger prompt to discourage excessive thinking
                        {"role": "system", "content": "You are a unit conversion calculator. For simple unit conversions, think briefly and provide the numerical answer immediately."},
                        {"role": "user", "content": f"{prompt} Answer format: just the number."}
                    ],
                    temperature=0,
                    max_tokens=5000,  # Very high limit to capture full response
                )
            else:
                response = await together_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a precise unit conversion expert. Provide only the numerical answer with up to 4 decimal places, no units or explanations."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=1000
                )
            
            # Get full content and finish reason
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            
            # Store metadata about the response
            was_truncated = finish_reason == 'length'
            
            if was_truncated:
                # Response was cut off - still try to extract what we can
                content = content + "\n[RESPONSE TRUNCATED]"
            
            # For DeepSeek-R1, try to extract final answer
            if "deepseek" in model.lower():
                # Look for the answer after </think>
                if '</think>' in content:
                    parts = content.split('</think>')
                    final_answer = parts[-1].strip()
                    
                    # If we got truncated, the answer might be incomplete
                    if was_truncated and not final_answer:
                        # No answer after </think>, probably cut off during thinking
                        # Try to extract any number we can find
                        import re
                        numbers = re.findall(r'\d+\.?\d*', content)
                        if numbers:
                            # Return last number found as best guess
                            return f"{numbers[-1]} [EXTRACTED_FROM_TRUNCATED]"
                        else:
                            return "ERROR: Response truncated during thinking, no answer found"
                    
                    return final_answer if final_answer else "ERROR: Empty answer after </think>"
                
                # No </think> found
                if '<think>' in content and was_truncated:
                    # Started thinking but never finished
                    return "ERROR: Response truncated before completing thinking"
                
                # No think tags at all - return full content
                return content.strip()
            
            # Non-DeepSeek models
            return content.strip()
            
        except Exception as e:
            error_msg = str(e)
            # Check if it's a rate limit error
            if ("rate" in error_msg.lower() or "429" in error_msg) and retry_count < 3:
                # Exponential backoff: 5s, 10s, 20s
                wait_time = 5 * (2 ** retry_count)
                print(f"Rate limit error (attempt {retry_count + 1}/3), waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
                # Retry recursively
                return await ask_together(prompt, model, semaphore, retry_count + 1)
            return f"ERROR: {error_msg}"


async def get_model_answer(
    prompt: str,
    model_name: str,
    semaphore: asyncio.Semaphore
) -> str:
    """Route question to appropriate model provider."""
    config = MODEL_CONFIGS.get(model_name)
    if not config:
        return f"ERROR: Unknown model {model_name}"
    
    provider = config["provider"]
    model = config["model"]
    
    if provider == "openai":
        return await ask_openai(prompt, model, semaphore)
    elif provider == "together":
        return await ask_together(prompt, model, semaphore)
    else:
        return f"ERROR: Unknown provider {provider}"

def create_conversion_prompt(task: Dict) -> str:
    """Create prompt for unit conversion."""
    value = task['test_value']
    from_unit = UNIT_DISPLAY_NAMES[task['from_unit']]
    to_unit = UNIT_DISPLAY_NAMES[task['to_unit']]
    
    # Make plural if needed
    if value != 1:
        if from_unit != 'milliliter' and from_unit != 'liter':
            from_unit = from_unit + 's'
        if to_unit != 'milliliter' and to_unit != 'liter':
            to_unit = to_unit + 's'
    
    # Context-free conversion (no ingredient)
    if task.get('context_type') == 'context_free':
        prompt = f"Convert {value} {from_unit} to {to_unit}. Provide only the numerical value."
    else:
        # Ingredient-based conversion
        ingredient = task['ingredient']
        prompt = f"Convert {value} {from_unit} of {ingredient} to {to_unit}. Provide only the numerical value."
    
    return prompt

def extract_number(answer: str) -> float:
    """Extract numeric value from answer string, handling DeepSeek-R1 thinking tokens."""
    import re
    
    # Handle DeepSeek-R1 thinking tokens if present
    if '<think>' in answer or '</think>' in answer:
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
    
    # Remove common text, keep numbers
    cleaned = answer.replace(",", "").strip()
    
    # Extract the last number found
    matches = re.findall(r'-?\d+\.?\d*', cleaned)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None

def calculate_accuracy(model_answer: float, correct_answer: float, tolerance: float = 0.1) -> bool:
    """Check if model answer is within tolerance of correct answer."""
    if model_answer is None:
        return False
    
    if correct_answer == 0:
        return abs(model_answer) < tolerance
    
    percent_error = abs((model_answer - correct_answer) / correct_answer) * 100
    return percent_error <= tolerance

async def evaluate_task(
    task: Dict,
    model_name: str,
    semaphore: asyncio.Semaphore
) -> Dict:
    """Evaluate a single conversion task."""
    prompt = create_conversion_prompt(task)
    
    # Get model's answer
    model_answer_str = await get_model_answer(prompt, model_name, semaphore)
    model_answer_num = extract_number(model_answer_str)
    
    # Check accuracy
    is_correct = calculate_accuracy(model_answer_num, task['correct_answer'])
    
    # Check if there was an error
    is_error = model_answer_str.startswith('ERROR:')
    
    return {
        'model': model_name,
        'ingredient': task.get('ingredient'),
        'language': task['language'],
        'context_type': task.get('context_type', 'ingredient'),
        'test_value': task['test_value'],
        'difficulty': task['difficulty'],
        'from_unit': task['from_unit'],
        'to_unit': task['to_unit'],
        'conversion_type': task['conversion_type'],
        'correct_answer': task['correct_answer'],
        'model_answer_raw': model_answer_str,
        'model_answer': model_answer_num,
        'is_correct': is_correct,
        'is_error': is_error,
        'prompt': prompt  # Store prompt for retry
    }


async def evaluate_model_on_tasks(
    tasks: List[Dict],
    model_name: str,
    batch_size: int = 20,
    max_concurrent: int = 10
) -> List[Dict]:
    """Evaluate a model on all conversion tasks."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} on {len(tasks)} conversion tasks")
    print(f"{'='*60}\n")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Process in batches
    batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
    all_results = []
    
    for batch_idx, batch in enumerate(tqdm(batches, desc=f"Processing {model_name}")):
        # Check for shutdown request
        if shutdown_handler.shutdown_requested:
            print(f"\nShutdown requested - stopping at batch {batch_idx}/{len(batches)}")
            break
            
        batch_tasks = [
            evaluate_task(task, model_name, semaphore)
            for task in batch
        ]
        batch_results = await asyncio.gather(*batch_tasks)
        all_results.extend(batch_results)
    
    # Calculate accuracy
    correct_count = sum(1 for r in all_results if r['is_correct'])
    accuracy = (correct_count / len(all_results) * 100) if all_results else 0
    
    print(f"\n{model_name} Results:")
    print(f"  Total conversions: {len(all_results)}")
    print(f"  Correct: {correct_count}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    return all_results

async def retry_failed_queries(
    failed_results: List[Dict],
    batch_size: int = 20,
    max_concurrent: int = 10
) -> List[Dict]:
    """Retry failed API queries."""
    if not failed_results:
        print("\nNo failed queries to retry.")
        return []
    
    print(f"\n{'='*60}")
    print(f"RETRYING {len(failed_results)} FAILED QUERIES")
    print(f"{'='*60}\n")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    batches = [failed_results[i:i + batch_size] for i in range(0, len(failed_results), batch_size)]
    
    retry_results = []
    
    for batch in tqdm(batches, desc="Retrying failed queries"):
        batch_tasks = []
        for result in batch:
            # Reconstruct task from result
            task = {
                'ingredient': result.get('ingredient'),
                'language': result['language'],
                'context_type': result.get('context_type', 'ingredient'),
                'test_value': result['test_value'],
                'difficulty': result['difficulty'],
                'from_unit': result['from_unit'],
                'to_unit': result['to_unit'],
                'conversion_type': result['conversion_type'],
                'correct_answer': result['correct_answer']
            }
            batch_tasks.append(evaluate_task(task, result['model'], semaphore))
        
        batch_results = await asyncio.gather(*batch_tasks)
        retry_results.extend(batch_results)
    
    # Count successes
    success_count = sum(1 for r in retry_results if not r['is_error'])
    print(f"\nRetry Results:")
    print(f"  Successfully recovered: {success_count}/{len(retry_results)}")
    print(f"  Still failing: {len(retry_results) - success_count}")
    
    return retry_results

async def run_experiment(
    input_file: str,
    output_dir: str = "language_conversion_results",
    sample_size: int = 500,
    batch_size: int = 20,
    max_concurrent: int = 10,
    include_context_free: bool = True,
    retry_errors: bool = True,
    together_rpm: int = 200,
    resume: bool = True
):
    """Run the complete language-based conversion experiment."""
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, shutdown_handler.request_shutdown)
    
    print(f"\n{'='*60}")
    print(f"LANGUAGE-BASED CONVERSION EXPERIMENT")
    print(f"{'='*60}")
    print("Press Ctrl+C to save progress and exit gracefully")
    print(f"{'='*60}\n")
    
    # Update rate limiter with user-specified limit
    global together_rate_limiter
    together_rate_limiter = RateLimiter(max_requests_per_minute=together_rpm)
    print(f"Together AI rate limit set to: {together_rpm} requests/minute")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load existing results if resuming
    existing_results = []
    completed_tasks = set()
    
    if resume:
        existing_results, completed_tasks = await load_existing_results(output_dir)
    
    # Load dataset
    df = load_dataset(input_file, sample_size)
    
    # Create conversion tasks
    all_tasks = create_conversion_tasks(df, include_context_free=include_context_free)
    
    # Evaluate each model
    all_results = existing_results.copy()  # Start with existing results
    models = ["gpt-4o", "qwen-coder", "llama-4"]
            #   , "deepseek-r1"]
    
    try:
        for model in models:
            if shutdown_handler.shutdown_requested:
                print(f"\nSkipping remaining models due to shutdown request")
                break
            
            # Filter out completed tasks for this model
            remaining_tasks = filter_completed_tasks(all_tasks, completed_tasks, model)
            
            if not remaining_tasks:
                print(f"\n{'='*60}")
                print(f"{model} - All tasks already completed, skipping")
                print(f"{'='*60}\n")
                continue
            
            print(f"\n{'='*60}")
            print(f"{model} - {len(remaining_tasks)} tasks remaining (out of {len(all_tasks)} total)")
            print(f"{'='*60}\n")
            
            model_results = await evaluate_model_on_tasks(
                tasks=remaining_tasks,
                model_name=model,
                batch_size=batch_size,
                max_concurrent=max_concurrent
            )
            all_results.extend(model_results)
            
            # Update completed tasks
            for result in model_results:
                # For context-free tasks, ignore ingredient in signature since prompt is identical
                ingredient = None if result.get('context_type') == 'context_free' else result.get('ingredient')
                task_signature = (
                    result['model'],
                    ingredient,
                    result['language'],
                    result['test_value'],
                    result['from_unit'],
                    result['to_unit']
                )
                completed_tasks.add(task_signature)
            
            # Save checkpoint after each model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_file = os.path.join(output_dir, f"checkpoint_{model}_{timestamp}.json")
            async with aiofiles.open(checkpoint_file, 'w') as f:
                await f.write(json.dumps(all_results, indent=2))
            print(f"Checkpoint saved: {checkpoint_file}")
    
    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt detected during processing")
        shutdown_handler.shutdown_requested = True
    
    # Save results (partial or complete)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    status = "partial" if shutdown_handler.shutdown_requested else "complete"
    output_file = os.path.join(output_dir, f"language_conversion_{status}_{timestamp}.json")
    
    print(f"\nSaving {status} results to {output_file}...")
    async with aiofiles.open(output_file, 'w') as f:
        await f.write(json.dumps(all_results, indent=2))
    
    if shutdown_handler.shutdown_requested:
        print("\n" + "="*60)
        print("PARTIAL RESULTS SAVED")
        print("="*60)
        print(f"Completed: {len(all_results)} tasks")
        print(f"Output: {output_file}")
        print("\nTo resume: Simply re-run the same command")
        print("The script will automatically skip completed tasks")
        print("="*60 + "\n")
        return
    
    # Continue with normal processing if not interrupted...
    
    # Check for errors
    failed_results = [r for r in all_results if r.get('is_error', False)]
    
    if failed_results:
        print(f"\n{'='*60}")
        print(f"DETECTED {len(failed_results)} FAILED QUERIES")
        print(f"{'='*60}")
        
        # Save error log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_log_file = os.path.join(output_dir, f"error_log_{timestamp}.json")
        async with aiofiles.open(error_log_file, 'w') as f:
            await f.write(json.dumps(failed_results, indent=2))
        print(f"Error log saved to: {error_log_file}")
        
        # Retry if enabled
        if retry_errors:
            retry_results = await retry_failed_queries(
                failed_results=failed_results,
                batch_size=batch_size,
                max_concurrent=max_concurrent
            )
            
            # Update results with retry outcomes
            # Create a mapping of failed results for quick lookup
            failed_map = {
                (r['model'], r.get('ingredient'), r['language'], r['test_value'], r['from_unit'], r['to_unit']): i
                for i, r in enumerate(all_results) if r.get('is_error', False)
            }
            
            # Replace failed results with retry results
            for retry_result in retry_results:
                key = (
                    retry_result['model'],
                    retry_result.get('ingredient'),
                    retry_result['language'],
                    retry_result['test_value'],
                    retry_result['from_unit'],
                    retry_result['to_unit']
                )
                if key in failed_map:
                    idx = failed_map[key]
                    all_results[idx] = retry_result
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"language_conversion_results_{timestamp}.json")
    
    print(f"\nSaving results to {output_file}...")
    async with aiofiles.open(output_file, 'w') as f:
        await f.write(json.dumps(all_results, indent=2))
    
    # Create summary
    print("\n" + "="*60)
    print("SUMMARY BY MODEL AND LANGUAGE")
    print("="*60)
    
    results_df = pd.DataFrame(all_results)
    
    if not results_df.empty:
        # Filter out errors for accuracy calculation
        valid_results = results_df[~results_df['is_error']]
        
        # Summary by model and language
        summary = valid_results.groupby(['model', 'language'])['is_correct'].agg(['sum', 'count', 'mean']).reset_index()
        summary.columns = ['model', 'language', 'correct', 'total', 'accuracy']
        summary['accuracy'] = summary['accuracy'] * 100
        summary = summary.sort_values(['model', 'language'])
        
        summary_file = os.path.join(output_dir, f"summary_by_language_{timestamp}.csv")
        summary.to_csv(summary_file, index=False)
        
        print(summary.to_string(index=False))
        print(f"\nSummary saved to: {summary_file}")
        
        # Language comparison
        print("\n" + "="*60)
        print("LANGUAGE BIAS ANALYSIS")
        print("="*60)
        
        for model in models:
            model_data = valid_results[valid_results['model'] == model]
            
            # Get unique languages (excluding context_free)
            languages = model_data[model_data['language'] != 'context_free']['language'].unique()
            
            if len(languages) >= 2:
                lang1_acc = model_data[model_data['language'] == languages[0]]['is_correct'].mean() * 100
                lang2_acc = model_data[model_data['language'] == languages[1]]['is_correct'].mean() * 100
                difference = lang1_acc - lang2_acc
                
                print(f"\n{model.upper()}:")
                print(f"  {languages[0].capitalize()} accuracy: {lang1_acc:.2f}%")
                print(f"  {languages[1].capitalize()} accuracy: {lang2_acc:.2f}%")
                print(f"  Difference: {difference:+.2f}% {'(favors ' + languages[0] + ')' if difference > 0 else '(favors ' + languages[1] + ')' if difference < 0 else '(neutral)'}")
            
            # Context-free comparison if available
            if include_context_free:
                context_free_acc = model_data[model_data['language'] == 'context_free']['is_correct'].mean() * 100
                avg_context_acc = model_data[model_data['language'] != 'context_free']['is_correct'].mean() * 100
                print(f"  Context-free accuracy: {context_free_acc:.2f}%")
                print(f"  Average with context: {avg_context_acc:.2f}%")
                print(f"  Context effect: {context_free_acc - avg_context_acc:+.2f}%")
        
        # Final error summary
        remaining_errors = results_df['is_error'].sum()
        if remaining_errors > 0:
            print("\n" + "="*60)
            print(f"WARNING: {remaining_errors} queries still have errors")
            print("="*60)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Results: {output_file}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test LLM accuracy on language-based unit conversions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python language_conversion.py --input ingredients.csv --sample-size 500
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to CSV file with columns: english_name, <language>_name (e.g., hindi_name, japanese_name, spanish_name)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="language_conversion_results",
        help="Directory to save results (default: language_conversion_results)"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of ingredients to sample (default: 50)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Batch size for processing (default: 20)"
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent API calls (default: 10)"
    )
    
    parser.add_argument(
        "--no-context-free",
        action="store_true",
        help="Disable context-free conversions (only test with ingredients)"
    )
    
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="Disable automatic retry of failed queries"
    )
    
    parser.add_argument(
        "--together-rpm",
        type=int,
        default=200,
        help="Together AI rate limit in requests per minute (default: 200)"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from previous results"
    )
    
    args = parser.parse_args()
    
    asyncio.run(run_experiment(
        input_file=args.input,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
        include_context_free=not args.no_context_free,
        retry_errors=not args.no_retry,
        together_rpm=args.together_rpm,
        resume=not args.no_resume
    ))