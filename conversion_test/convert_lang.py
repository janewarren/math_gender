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
    
    # Verify required columns
    if 'english_name' not in df.columns or 'hindi_name' not in df.columns:
        raise ValueError("CSV must contain 'english_name' and 'hindi_name' columns")
    
    # Sample rows
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    print(f"Sampled {len(df_sample)} rows")
    return df_sample

def create_conversion_tasks(df: pd.DataFrame) -> List[Dict]:
    """Create conversion tasks from dataset."""
    tasks = []
    
    print("\nCreating conversion tasks...")
    
    for idx, row in df.iterrows():
        english_name = row['english_name']
        hindi_name = row['hindi_name']
        
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
        
        # Create task for English name
        tasks.append({
            'ingredient': english_name,
            'language': 'english',
            'random_value': round(random_value, 2),
            'from_unit': from_unit,
            'to_unit': to_unit,
            'conversion_type': conversion_type,
            'correct_answer': round(correct_answer, 4)
        })
        
        # Create task for Hindi name (same conversion parameters)
        tasks.append({
            'ingredient': hindi_name,
            'language': 'hindi',
            'random_value': round(random_value, 2),
            'from_unit': from_unit,
            'to_unit': to_unit,
            'conversion_type': conversion_type,
            'correct_answer': round(correct_answer, 4)
        })
    
    print(f"Created {len(tasks)} conversion tasks ({len(tasks)//2} per language)")
    return tasks

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

async def ask_together(prompt: str, model: str, semaphore: asyncio.Semaphore) -> str:
    """Ask Together AI model to perform conversion."""
    async with semaphore:
        try:
            # Special handling for DeepSeek-R1
            if "deepseek" in model.lower():
                response = await together_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a precise unit conversion expert. Provide only the numerical answer with up to 4 decimal places, no units or explanations."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.6,
                    max_tokens=2000,
                    max_completion_tokens=100
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
            
            # For DeepSeek-R1, extract the final answer after thinking
            content = response.choices[0].message.content.strip()
            
            if '</think>' in content:
                parts = content.split('</think>')
                if len(parts) > 1:
                    final_answer = parts[-1].strip()
                    return final_answer
            
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
    
    return {
        'model': model_name,
        'ingredient': task['ingredient'],
        'language': task['language'],
        'random_value': task['random_value'],
        'from_unit': task['from_unit'],
        'to_unit': task['to_unit'],
        'conversion_type': task['conversion_type'],
        'correct_answer': task['correct_answer'],
        'model_answer_raw': model_answer_str,
        'model_answer': model_answer_num,
        'is_correct': is_correct
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
    max_concurrent: int = 10
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
    tasks = create_conversion_tasks(df)
    
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
        # Summary by model and language
        summary = results_df.groupby(['model', 'language'])['is_correct'].agg(['sum', 'count', 'mean']).reset_index()
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
            model_data = results_df[results_df['model'] == model]
            english_acc = model_data[model_data['language'] == 'english']['is_correct'].mean() * 100
            hindi_acc = model_data[model_data['language'] == 'hindi']['is_correct'].mean() * 100
            difference = english_acc - hindi_acc
            
            print(f"\n{model.upper()}:")
            print(f"  English accuracy: {english_acc:.2f}%")
            print(f"  Hindi accuracy: {hindi_acc:.2f}%")
            print(f"  Difference: {difference:+.2f}% {'(favors English)' if difference > 0 else '(favors Hindi)' if difference < 0 else '(neutral)'}")
    
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
        help="Path to CSV file with columns: english_name, hindi_name"
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
    
    args = parser.parse_args()
    
    asyncio.run(run_experiment(
        input_file=args.input,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent
    ))