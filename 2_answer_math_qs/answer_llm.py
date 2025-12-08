import json
import os
import asyncio
import argparse
from openai import AsyncOpenAI
from typing import List, Dict, Optional
from tqdm.asyncio import tqdm
import aiofiles
from datetime import datetime

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
    # OpenAI models
    "gpt-4o": {
        "provider": "openai",
        "model": "gpt-4o"
    },
    
    # Together AI models
    "qwen-coder": {
        "provider": "together",
        "model": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
    },
    "llama-4": {
        "provider": "together",
        "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    },
    "deepseek-r1": {
        "provider": "together",
        "model": "deepseek-ai/DeepSeek-R1"
    }
}

def load_transformed_dataset(file_path: str) -> List[Dict]:
    """Load transformed MathQA dataset."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract all unique context types from the first question
    if data and 'transformations' in data[0] and 'new_contexts' in data[0]['transformations']:
        contexts = list(data[0]['transformations']['new_contexts'].keys())
        print(f"Found {len(contexts)} contexts: {', '.join(contexts)}")
        return data, contexts
    else:
        raise ValueError("No contexts found in the transformed dataset")


async def ask_openai(question: str, model: str, semaphore: asyncio.Semaphore) -> str:
    """Ask OpenAI model to solve a math problem."""
    prompt = f"""You are an expert at math.
    
    Solve this math problem. Provide ONLY the numerical answer, nothing else.
    
    If the answer involves currency or units, provide just the number with no symbols.

Problem: {question}

Answer:"""

    async with semaphore:
        try:
            response = await openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a mathematics expert. Provide only numerical answers to math problems."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_completion_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"ERROR: {str(e)}"

async def ask_together(question: str, model: str, semaphore: asyncio.Semaphore) -> str:
    """Ask Together AI model to solve a math problem."""
    prompt = f"""You are an expert at math.
    
    Solve this math problem. Provide ONLY the numerical answer, nothing else.
    
    If the answer involves currency or units, provide just the number with no symbols.

Problem: {question}

Answer:"""

    async with semaphore:
        try:
            response = await together_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a mathematics expert. Provide only numerical answers to math problems."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"ERROR: {str(e)}"

async def get_model_answer(
    question: str,
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
        return await ask_openai(question, model, semaphore)
    elif provider == "together":
        return await ask_together(question, model, semaphore)
    else:
        return f"ERROR: Unknown provider {provider}"

def extract_number(answer: str) -> str:
    """Extract numeric value from answer string."""
    import re
    # Remove common text, keep numbers
    cleaned = answer.replace("$", "").replace(",", "").strip()
    # Extract first number found
    match = re.search(r'-?\d+\.?\d*', cleaned)
    if match:
        return match.group(0)
    return answer

def check_answer(model_answer: str, correct_answer: str) -> bool:
    """Check if model answer matches correct answer."""
    # Extract numeric values
    model_num = extract_number(model_answer)
    correct_num = extract_number(str(correct_answer))
    
    try:
        # Try exact match first
        if model_num == correct_num:
            return True
        # Try numerical comparison with tolerance
        model_val = float(model_num)
        correct_val = float(correct_num)
        return abs(model_val - correct_val) < 0.01
    except (ValueError, TypeError):
        # Fallback to string comparison
        return model_num.lower() == correct_num.lower()


async def evaluate_question(
    item: Dict,
    context_type: str,
    model_name: str,
    semaphore: asyncio.Semaphore
) -> Dict:
    """Evaluate a single question with a model."""
    # Get the question for the specified context from new_contexts
    question = item["transformations"]["new_contexts"].get(context_type)
    
    if not question:
        return {
            "question": None,
            "model_answer": "ERROR: Context not found",
            "correct_answer": item.get("correct"),
            "is_correct": False
        }
    
    # Get model's answer
    model_answer = await get_model_answer(question, model_name, semaphore)
    
    # Check correctness
    correct_answer = item.get("correct")
    is_correct = check_answer(model_answer, correct_answer)
    
    return {
        "question": question,
        "model_answer": model_answer,
        "correct_answer": correct_answer,
        "is_correct": is_correct,
        "original_problem": item.get("Problem"),
        "category": item.get("category"),
        "rationale": item.get("Rationale")
    }



async def evaluate_dataset(
    input_file: str,
    model_name: str,
    context_type: str,
    questions: List[Dict],
    output_dir: str = "evaluation_results",
    max_questions: Optional[int] = None,
    batch_size: int = 20,
    max_concurrent: int = 5
):
    """Evaluate entire dataset with specified model."""
    print(f"\n{'='*60}")
    print(f"EVALUATION CONFIGURATION")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Context: {context_type}")
    print(f"Input: {input_file}")
    print(f"{'='*60}\n")
    
    # Apply max_questions limit if specified
    eval_questions = questions[:max_questions] if max_questions else questions
    
    print(f"Evaluating {len(eval_questions)} questions...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Process in batches
    batches = [eval_questions[i:i + batch_size] for i in range(0, len(eval_questions), batch_size)]
    all_results = []
    
    for batch in tqdm(batches, desc=f"Evaluating {model_name} on {context_type}"):
        # Create tasks for batch
        tasks = [
            evaluate_question(item, context_type, model_name, semaphore)
            for item in batch
        ]
        
        # Wait for batch to complete
        batch_results = await asyncio.gather(*tasks)
        all_results.extend(batch_results)
    
    
    
    # Calculate accuracy
    correct_count = sum(1 for r in all_results if r["is_correct"])
    total_count = len(all_results)
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    
    # Prepare output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace(".", "_").replace("-", "_")
    output_file = os.path.join(
        output_dir,
        f"{safe_model_name}_{context_type}_{timestamp}.json"
    )
    
    output_data = {
        "metadata": {
            "model": model_name,
            "context_type": context_type,
            "input_file": input_file,
            "total_questions": total_count,
            "correct_answers": correct_count,
            "accuracy": accuracy,
            "timestamp": timestamp
        },
        "results": all_results
    }
    
    # Save results
    print(f"\nSaving results to {output_file}...")
    async with aiofiles.open(output_file, 'w') as f:
        await f.write(json.dumps(output_data, indent=2))
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Context: {context_type}")
    print(f"Total Questions: {total_count}")
    print(f"Correct Answers: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")
    
    return output_data


async def evaluate_multiple_models(
    input_file: str,
    models: List[str],
    output_dir: str = "evaluation_results",
    max_questions: Optional[int] = None,
    batch_size: int = 20,
    max_concurrent: int = 5
):
    """Evaluate dataset with multiple models on all contexts."""
    # Load dataset and extract contexts
    print(f"Loading dataset from {input_file}...")
    questions, contexts = load_transformed_dataset(input_file)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION SETUP")
    print(f"{'='*60}")
    print(f"Models: {', '.join(models)}")
    print(f"Contexts: {', '.join(contexts)}")
    print(f"Total questions: {len(questions)}")
    if max_questions:
        print(f"Limiting to: {max_questions} questions")
    print(f"{'='*60}\n")
    
    all_results = {}
    
    for model in models:
        for context in contexts:
            key = f"{model}_{context}"
            print(f"\n>>> Evaluating {model} on {context} context...\n")
            
            result = await evaluate_dataset(
                input_file=input_file,
                model_name=model,
                context_type=context,
                questions=questions,
                output_dir=output_dir,
                max_questions=max_questions,
                batch_size=batch_size,
                max_concurrent=max_concurrent
            )
            
            all_results[key] = result["metadata"]
    
    # Save summary
    summary_file = os.path.join(output_dir, "evaluation_summary.json")
    async with aiofiles.open(summary_file, 'w') as f:
        await f.write(json.dumps(all_results, indent=2))
    
    print(f"\n{'='*60}")
    print(f"ALL EVALUATIONS COMPLETE")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*60}\n")
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLMs on transformed math questions (automatically evaluates all contexts)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
  OpenAI: gpt-4o
  Together AI: qwen-coder, llama-4, deepseek-r1

Examples:
  # Evaluate single model on all contexts
  python evaluate_llm.py --input data.json --models gpt-4o
  
  # Evaluate all models on all contexts
  python evaluate_llm.py --input data.json --models gpt-4o qwen-coder llama-4 deepseek-r1
  
  # Test with first 10 questions
  python evaluate_llm.py --input data.json --models gpt-4o --max-questions 10

Note: All contexts in the dataset will be automatically evaluated. No need to specify contexts.
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to transformed MathQA JSON file"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="List of models to evaluate (e.g., gpt-4o qwen-coder llama-4 deepseek-r1)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results (default: evaluation_results)"
    )
    
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions to evaluate (default: all)"
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
        default=5,
        help="Maximum concurrent API calls (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Validate models
    invalid_models = [m for m in args.models if m not in MODEL_CONFIGS]
    if invalid_models:
        print(f"Error: Unknown models: {', '.join(invalid_models)}")
        print(f"Available models: {', '.join(MODEL_CONFIGS.keys())}")
        return
    
    # Run evaluation
    asyncio.run(evaluate_multiple_models(
        input_file=args.input,
        models=args.models,
        output_dir=args.output_dir,
        max_questions=args.max_questions,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent
    ))

if __name__ == "__main__":
    main()