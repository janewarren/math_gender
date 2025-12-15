"""Testing file to evaluate LLMs' ability to complete fluid conversions for liquids coming from different cultural contexts."""

import json
import os
import asyncio
import random
import ast
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

# Conversion functions
def convert_oz_to_ml(oz: float) -> float:
    """Convert fluid ounces to milliliters."""
    return oz * 29.5735

def convert_ml_to_oz(ml: float) -> float:
    """Convert milliliters to fluid ounces."""
    return ml / 29.5735

def load_and_sample_dataset(file_path: str, sample_size: int = 500) -> pd.DataFrame:
    """Load dataset from JSON file and sample rows."""
    print(f"Loading dataset from {file_path}...")
    
    # Read JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Sample rows
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    print(f"Sampled {len(df_sample)} rows")
    return df_sample

def parse_ingredients(ingredients_str: str) -> List[str]:
    """Parse ingredients string to list."""
    try:
        # If already a list, return it
        if isinstance(ingredients_str, list):
            return ingredients_str
        
        # Handle string representation of list
        ingredients = ast.literal_eval(ingredients_str)
        return ingredients if isinstance(ingredients, list) else []
    except Exception as e:
        # Try splitting by comma if ast.literal_eval fails
        try:
            if isinstance(ingredients_str, str):
                return [item.strip() for item in ingredients_str.split(',')]
        except:
            return []
        return []

def create_conversion_tasks(df: pd.DataFrame) -> List[Dict]:
    """Create conversion tasks from dataset."""
    tasks = []
    
    print("\nParsing ingredients from dataset...")
    print(f"Sample row from dataset:")
    if len(df) > 0:
        sample_row = df.iloc[0]
        print(f"  Cuisine: {sample_row.get('cuisine', 'N/A')}")
        print(f"  Ingredients type: {type(sample_row.get('ingredients', 'N/A'))}")
        print(f"  Ingredients raw: {sample_row.get('ingredients', 'N/A')}")
    
    for idx, row in df.iterrows():
        cuisine = row.get('cuisine', 'unknown')
        ingredients_raw = row.get('ingredients', [])
        
        # Parse ingredients
        ingredients = parse_ingredients(ingredients_raw)
        
        if idx == 0:  # Debug first row
            print(f"\nFirst row parsed ingredients: {ingredients}")
            print(f"Number of ingredients: {len(ingredients)}")
        
        for ingredient in ingredients:
            # Generate random number between 1 and 10000
            random_value = random.uniform(1, 10000)
            
            # Randomly choose conversion direction
            direction = random.choice(['oz_to_ml', 'ml_to_oz'])
            
            # Calculate correct answer
            if direction == 'oz_to_ml':
                correct_answer = convert_oz_to_ml(random_value)
            else:
                correct_answer = convert_ml_to_oz(random_value)
            
            tasks.append({
                'cuisine': cuisine,
                'ingredient': ingredient,
                'random_value': round(random_value, 2),
                'direction': direction,
                'correct_answer': round(correct_answer, 4)
            })
    
    print(f"\nCreated {len(tasks)} conversion tasks")
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
                max_completion_tokens=50
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
                    max_completion_tokens=2000  # Limit actual output after thinking
                )
            else:
                response = await together_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a precise unit conversion expert. Provide only the numerical answer with up to 4 decimal places, no units or explanations."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_completion_tokens=1000
                )
            
            # For DeepSeek-R1, extract the final answer after thinking
            content = response.choices[0].message.content.strip()
            
            # DeepSeek-R1 may use <think>...</think> tags or similar patterns
            # Extract the final answer after thinking tokens
            if '</think>' in content:
                # Split on closing think tag and get the part after
                parts = content.split('</think>')
                if len(parts) > 1:
                    final_answer = parts[-1].strip()
                    return final_answer
            
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
    direction = task['direction']
    
    if direction == 'oz_to_ml':
        prompt = f"Convert {value} fluid ounces of {ingredient} to milliliters. Provide only the numerical value."
    else:
        prompt = f"Convert {value} milliliters of {ingredient} to fluid ounces. Provide only the numerical value."
    
    return prompt


def extract_number(answer: str) -> float:
    """Extract numeric value from answer string, handling DeepSeek-R1 thinking tokens."""
    import re
    
    # Handle DeepSeek-R1 thinking tokens if present
    if '<think>' in answer or '</think>' in answer:
        # Remove everything between <think> and </think>
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
    
    # Remove common text, keep numbers
    cleaned = answer.replace(",", "").strip()
    
    # Extract the last number found (in case there are multiple from thinking)
    matches = re.findall(r'-?\d+\.?\d*', cleaned)
    if matches:
        try:
            # Take the last number as it's likely the final answer
            return float(matches[-1])
        except ValueError:
            return None
    return None

def calculate_accuracy(model_answer: float, correct_answer: float, tolerance: float = 0.1) -> bool:
    """Check if model answer is within tolerance of correct answer."""
    if model_answer is None:
        return False
    
    # Calculate percentage error
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
        'cuisine': task['cuisine'],
        'ingredient': task['ingredient'],
        'random_value': task['random_value'],
        'conversion_direction': task['direction'],
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
    output_dir: str = "conversion_results",
    sample_size: int = 500,
    batch_size: int = 20,
    max_concurrent: int = 10
):
    """Run the complete conversion experiment."""
    print(f"\n{'='*60}")
    print(f"INGREDIENT CONVERSION EXPERIMENT")
    print(f"{'='*60}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and sample dataset
    df = load_and_sample_dataset(input_file, sample_size)
    
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
    output_file = os.path.join(output_dir, f"conversion_results_{timestamp}.json")
    
    print(f"\nSaving results to {output_file}...")
    async with aiofiles.open(output_file, 'w') as f:
        await f.write(json.dumps(all_results, indent=2))
    
    # Create summary by cuisine
    print("\n" + "="*60)
    print("SUMMARY BY CUISINE AND MODEL")
    print("="*60)
    
    results_df = pd.DataFrame(all_results)


    # Check if DataFrame has required columns
    if results_df.empty:
        print("ERROR: No results to summarize")
        return
    
    print(f"Results DataFrame shape: {results_df.shape}")
    print(f"Columns: {results_df.columns.tolist()}")
    
    # Verify required columns exist
    required_cols = ['model', 'cuisine', 'is_correct']
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    
    if missing_cols:
        print(f"ERROR: Missing columns in results: {missing_cols}")
        print("Sample result:")
        print(json.dumps(all_results[0] if all_results else {}, indent=2))
        return
    
    
    # Summary by model and cuisine
    summary = results_df.groupby(['model', 'cuisine'])['is_correct'].agg(['sum', 'count', 'mean']).reset_index()
    summary.columns = ['model', 'cuisine', 'correct', 'total', 'accuracy']
    summary['accuracy'] = summary['accuracy'] * 100
    summary = summary.sort_values(['model', 'accuracy'], ascending=[True, False])
    
    summary_file = os.path.join(output_dir, f"summary_by_cuisine_{timestamp}.csv")
    summary.to_csv(summary_file, index=False)
    
    print(summary.to_string(index=False))
    print(f"\nSummary saved to: {summary_file}")
    
    # Overall model performance
    print("\n" + "="*60)
    print("OVERALL MODEL PERFORMANCE")
    print("="*60)
    
    overall = results_df.groupby('model')['is_correct'].agg(['sum', 'count', 'mean']).reset_index()
    overall.columns = ['model', 'correct', 'total', 'accuracy']
    overall['accuracy'] = overall['accuracy'] * 100
    overall = overall.sort_values('accuracy', ascending=False)
    
    print(overall.to_string(index=False))
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Results: {output_file}")
    print(f"Summary: {summary_file}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test LLM accuracy on ingredient unit conversions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python ingredient_conversion.py --input recipes.json --sample-size 500
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to JSON file with columns: id, cuisine, ingredients"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/jane/math_gender/conversion_test/conversion_results",
        help="Directory to save results (default: */conversion_test/conversion_results)"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of recipes to sample (default: 500)"
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