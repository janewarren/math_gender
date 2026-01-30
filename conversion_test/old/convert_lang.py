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

# Model configurations
MODEL_CONFIGS = {
    "gpt-4o": {"provider": "openai", "model": "gpt-4o"},
    "qwen-coder": {"provider": "together", "model": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"},
    "llama-4": {"provider": "together", "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"},
    "deepseek-r1": {"provider": "together", "model": "deepseek-ai/DeepSeek-R1"}
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

def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between any two volume units."""
    # Convert to milliliters first
    ml_value = value * CONVERSION_TO_ML[from_unit]
    # Then convert to target unit
    result = ml_value / CONVERSION_TO_ML[to_unit]
    return result

def load_dataset(file_path: str, sample_size: int = 500) -> pd.DataFrame:
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
    """Create conversion tasks from dataset."""
    tasks = []
    
    # Get language names from DataFrame metadata
    lang1 = df.attrs.get('lang1', 'english')
    lang2 = df.attrs.get('lang2', 'other')
    
    print(f"\nCreating conversion tasks for {lang1} and {lang2}...")
    
    for idx, row in df.iterrows():
        lang1_name = row['lang1_name']
        lang2_name = row['lang2_name']
        
        # Generate random value between 1 and 10000
        random_value = random.uniform(1, 10000)
        
        # Randomly choose US to Metric or Metric to US
        conversion_type = random.choice(['us_to_metric', 'metric_to_us'])
        
        if conversion_type == 'us_to_metric':
            from_unit = random.choice(US_UNITS)
            to_unit = random.choice(METRIC_UNITS)
        else:
            from_unit = random.choice(METRIC_UNITS)
            to_unit = random.choice(US_UNITS)
        
        # Calculate correct answer
        correct_answer = convert_units(random_value, from_unit, to_unit)
        
        # Create task for first language (English)
        tasks.append({
            'ingredient': lang1_name,
            'language': lang1,
            'context_type': 'ingredient',
            'random_value': round(random_value, 2),
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
            'random_value': round(random_value, 2),
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
                'random_value': round(random_value, 2),
                'from_unit': from_unit,
                'to_unit': to_unit,
                'conversion_type': conversion_type,
                'correct_answer': round(correct_answer, 4)
            })
    
    context_free_count = sum(1 for t in tasks if t['context_type'] == 'context_free')
    print(f"Created {len(tasks)} conversion tasks:")
    print(f"  - {len(tasks) - context_free_count} with ingredient context ({len(df)} per language)")
    if include_context_free:
        print(f"  - {context_free_count} context-free")
    
    return tasks

async def ask_openai(prompt: str, model: str, semaphore: asyncio.Semaphore) -> str:
    """Ask OpenAI model to perform conversion."""
    async with semaphore:
        try:
            response = await openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise unit conversion expert. Provide only the numerical answer with up to 4 decimal places, no units or explanations. For example, if you were asked to convert 20 oz to mL, the answer would be 591.471."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"ERROR: {str(e)}"

async def ask_together(prompt: str, model: str, semaphore: asyncio.Semaphore) -> str:
    """Ask Together AI model to perform conversion."""
    async with semaphore:
        try:
            # Special handling for DeepSeek-R1
            if "deepseek" in model.lower():
                response = await together_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": f"You are a precise unit conversion expert. Provide ONLY the numerical answer with up to 4 decimal places, no units or explanations. For example, if you were asked to convert 20 oz to mL, the answer would be 591.471. Think only briefly to do this simple conversion task. Here is the question: {prompt}"}
                    ],
                    temperature=0.6,
                    max_completion_tokens=8000
                )
            else:
                response = await together_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a precise unit conversion expert. Provide ONLY the numerical answer with up to 4 decimal places, no units or explanations. For example, if you were asked to convert 20 oz to mL, the answer would be 591.471."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=100
                )
            
            # For DeepSeek-R1, extract the final answer after thinking
            content = response.choices[0].message.content.strip()
            
            finish_reason = response.choices[0].finish_reason
            if finish_reason == 'length':
                print(f"WARNING: DeepSeek response cut off (length limit). Content length: {len(content)}")
                # Still try to extract answer
            
            # Extract final answer after </think>
            if '</think>' in content:
                parts = content.split('</think>')
                if len(parts) > 1:
                    final_answer = parts[-1].strip()
                    return final_answer
                else:
                    # </think> found but incomplete
                    print(f"WARNING: Incomplete </think> tag in response")
                    return "ERROR: Response cut off before final answer"
            
            # If no </think> tag but response was cut off
            if finish_reason == 'length' and '<think>' in content:
                return "ERROR: Thinking tokens exceeded limit, no answer provided"
            
            # If no think tags, return full content
            return content
            
        except Exception as e:
            return f"ERROR: {str(e)}"

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
    ingredient = task['ingredient']
    value = task['random_value']
    from_unit = UNIT_DISPLAY_NAMES[task['from_unit']]
    to_unit = UNIT_DISPLAY_NAMES[task['to_unit']]
    
    # Make plural if needed
    if value != 1:
        if from_unit != 'milliliter' and from_unit != 'liter':
            from_unit = from_unit + 's'
        if to_unit != 'milliliter' and to_unit != 'liter':
            to_unit = to_unit + 's'
    
    if task.get('context_type') == 'context_free':
        prompt = f"Convert {value} {from_unit} to {to_unit}. Provide only the numerical value."
    else:
        # Ingredient-based conversion
        ingredient = task['ingredient']
        prompt = f"Convert {value} {from_unit} of {ingredient} to {to_unit}. Provide only the numerical value."
    
    return prompt

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
                'random_value': result['random_value'],
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
        'random_value': task['random_value'],
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
    
    for batch in tqdm(batches, desc=f"Processing {model_name}"):
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
async def run_experiment(
    input_file: str,
    output_dir: str = "language_conversion_results",
    sample_size: int = 500,
    batch_size: int = 20,
    max_concurrent: int = 10,
    include_context_free: bool = True,
    retry_errors: bool = True
):
    """Run the complete language-based conversion experiment."""
    print(f"\n{'='*60}")
    print(f"LANGUAGE-BASED CONVERSION EXPERIMENT")
    print(f"{'='*60}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    df = load_dataset(input_file, sample_size)
    
    # Create conversion tasks
    tasks = create_conversion_tasks(df, include_context_free=include_context_free)
    
    # Evaluate each model
    all_results = []
    models = ["gpt-4o", "qwen-coder", "llama-4", "deepseek-r1"]
    
    for model in models:
        model_results = await evaluate_model_on_tasks(
            tasks=tasks,
            model_name=model,
            batch_size=batch_size,
            max_concurrent=max_concurrent
        )
        all_results.extend(model_results)
    
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
                (r['model'], r.get('ingredient'), r['language'], r['random_value'], r['from_unit'], r['to_unit']): i
                for i, r in enumerate(all_results) if r.get('is_error', False)
            }
            
            # Replace failed results with retry results
            for retry_result in retry_results:
                key = (
                    retry_result['model'],
                    retry_result.get('ingredient'),
                    retry_result['language'],
                    retry_result['random_value'],
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
        default=500,
        help="Number of ingredients to sample (default: 500)"
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
    
    args = parser.parse_args()
    
    asyncio.run(run_experiment(
        input_file=args.input,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
        include_context_free=not args.no_context_free,
        retry_errors=not args.no_retry
    ))