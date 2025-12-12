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
    "gpt-4o": {"provider": "openai", "model": "gpt-4o"},
    "qwen-coder": {"provider": "together", "model": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"},
    "llama-4": {"provider": "together", "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"},
    "deepseek-r1": {"provider": "together", "model": "deepseek-ai/DeepSeek-R1"}
}

def load_transformed_dataset(file_path: str) -> List[Dict]:
    """Load transformed MathQA dataset."""
    with open(file_path, 'r') as f:
        return json.load(f)

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
                max_tokens=200
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
    cleaned = answer.replace("$", "").replace(",", "").strip()
    match = re.search(r'-?\d+\.?\d*', cleaned)
    if match:
        return match.group(0)
    return answer

def check_answer(model_answer: str, correct_answer: str) -> bool:
    """Check if model answer matches correct answer."""
    model_num = extract_number(model_answer)
    correct_num = extract_number(str(correct_answer))
    
    try:
        if model_num == correct_num:
            return True
        model_val = float(model_num)
        correct_val = float(correct_num)
        return abs(model_val - correct_val) < 0.01
    except (ValueError, TypeError):
        return model_num.lower() == correct_num.lower()

async def evaluate_question_no_context(
    item: Dict,
    model_name: str,
    semaphore: asyncio.Semaphore
) -> Dict:
    """Evaluate a single no-context question with a model."""
    # Get the context-free question
    question = item["transformations"].get("context_free")
    
    if not question:
        return {
            "question": None,
            "model_answer": "ERROR: No context-free version found",
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

async def evaluate_no_context(
    input_file: str,
    model_name: str,
    output_dir: str = "evaluation_results",
    max_questions: Optional[int] = None,
    batch_size: int = 20,
    max_concurrent: int = 5
):
    """Evaluate no-context questions for a single model."""
    print(f"\n{'='*60}")
    print(f"NO-CONTEXT EVALUATION")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Input: {input_file}")
    print(f"{'='*60}\n")
    
    # Load dataset
    print(f"Loading dataset from {input_file}...")
    questions = load_transformed_dataset(input_file)
    
    if max_questions:
        questions = questions[:max_questions]
    
    print(f"Evaluating {len(questions)} no-context questions...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Process in batches
    batches = [questions[i:i + batch_size] for i in range(0, len(questions), batch_size)]
    all_results = []
    
    for batch in tqdm(batches, desc=f"Evaluating {model_name} on no-context"):
        tasks = [
            evaluate_question_no_context(item, model_name, semaphore)
            for item in batch
        ]
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
        f"{safe_model_name}_no_context_{timestamp}.json"
    )
    
    output_data = {
        "metadata": {
            "model": model_name,
            "context_type": "no_context",
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
    print(f"NO-CONTEXT EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Total Questions: {total_count}")
    print(f"Correct Answers: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")
    
    return output_data

async def evaluate_all_models_no_context(
    input_file: str,
    models: List[str],
    output_dir: str = "evaluation_results",
    max_questions: Optional[int] = None,
    batch_size: int = 20,
    max_concurrent: int = 5
):
    """Evaluate no-context questions for all models."""
    all_results = {}
    
    for model in models:
        print(f"\n>>> Evaluating {model} on no-context questions...\n")
        
        result = await evaluate_no_context(
            input_file=input_file,
            model_name=model,
            output_dir=output_dir,
            max_questions=max_questions,
            batch_size=batch_size,
            max_concurrent=max_concurrent
        )
        
        all_results[f"{model}_no_context"] = result["metadata"]
    
    # Update summary file (append no-context results)
    summary_file = os.path.join(output_dir, "/data/jane/math_gender/2_answer_math_qs/evaluation_results/evaluation_summary.json")
    
    # Load existing summary if it exists
    if os.path.exists(summary_file):
        async with aiofiles.open(summary_file, 'r') as f:
            content = await f.read()
            existing_summary = json.loads(content)
    else:
        existing_summary = {}
    
    # Merge with new results
    existing_summary.update(all_results)
    
    # Save updated summary
    async with aiofiles.open(summary_file, 'w') as f:
        await f.write(json.dumps(existing_summary, indent=2))
    
    print(f"\n{'='*60}")
    print(f"NO-CONTEXT EVALUATIONS COMPLETE")
    print(f"Summary updated in: {summary_file}")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLMs on no-context (context-free) math questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
  OpenAI: gpt-4o
  Together AI: qwen-coder, llama-4, deepseek-r1

Examples:
  # Evaluate single model on no-context questions
  python evaluate_no_context.py --input mathqa_transformed.json --models gpt-4o
  
  # Evaluate all models on no-context questions
  python evaluate_no_context.py --input mathqa_transformed.json --models gpt-4o qwen-coder llama-4 deepseek-r1
  
  # Test with first 10 questions
  python evaluate_no_context.py --input mathqa_transformed.json --models gpt-4o --max-questions 10

Note: This script evaluates the context-free versions of questions and updates the existing evaluation_summary.json
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
    asyncio.run(evaluate_all_models_no_context(
        input_file=args.input,
        models=args.models,
        output_dir=args.output_dir,
        max_questions=args.max_questions,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent
    ))

if __name__ == "__main__":
    main()