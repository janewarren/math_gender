import json
import os
import asyncio
from openai import AsyncOpenAI
from typing import List, Dict
from tqdm.asyncio import tqdm
import aiofiles

def load_api_key(key_file: str = "openai_key.txt") -> str:
    """Load OpenAI API key from file."""
    try:
        with open(key_file, 'r') as f:
            api_key = f.read().strip()
        if not api_key:
            raise ValueError(f"API key file '{key_file}' is empty")
        return api_key
    except FileNotFoundError:
        raise FileNotFoundError(
            f"API key file '{key_file}' not found. "
            f"Please create the file with your OpenAI API key."
        )

# Load API key from file or environment variable
api_key = os.environ.get("OPENAI_API_KEY") or load_api_key("openai_key.txt")


client = AsyncOpenAI(api_key=api_key)

def load_mathqa_dataset(file_path: str) -> List[Dict]:
    """Load MathQA dataset from JSON or JSONL file."""
    with open(file_path, 'r') as f:
        # Try to load as JSON first
        try:
            data = json.load(f)
            # If it's a list, return it directly
            if isinstance(data, list):
                return data
            # If it's a dict with a key containing the questions, extract them
            elif isinstance(data, dict):
                # Common keys: 'data', 'questions', 'train', etc.
                for key in ['data', 'questions', 'train', 'test', 'examples']:
                    if key in data and isinstance(data[key], list):
                        return data[key]
                # If no common key found, wrap the dict in a list
                return [data]
        except json.JSONDecodeError:
            # If JSON fails, try JSONL format
            f.seek(0)
            questions = []
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))
            return questions

def extract_numeric_answer(options: str, correct: str) -> str:
    """Extract the numeric answer from multiple choice options so we can convert from MCQ to open ended."""
    # Parse options string like "a ) rs . 400 , b ) rs . 300 , c ) rs . 500"
    import re
    
    # Find the option that matches the correct letter
    pattern = rf'{correct}\s*\)\s*([^,]+)'
    match = re.search(pattern, options, re.IGNORECASE)
    
    if match:
        answer_text = match.group(1).strip()
        # Extract numeric value - handle formats like "rs . 400", "$ 50", "20%", etc.
        numeric_match = re.search(r'[\d,]+\.?\d*', answer_text.replace(',', ''))
        if numeric_match:
            return numeric_match.group(0)
    
    # Fallback: return the correct letter if extraction fails
    return correct

async def remove_context(question: str, semaphore: asyncio.Semaphore) -> str:    
    """Use GPT to transform word problem into context-free numerical question."""
    prompt = f"""You are a math problem simplifier. 
        Transform the given math word problem into a pure math question without any story or context. 
        Keep the same numbers and mathematical operations, but remove all narrative elements. 
        The result should be a direct mathematical question. 
    
        Original question: {question}

        Output ONLY the transformed question, nothing else."""

    async with semaphore:
        response = await client.chat.completions.create(
            model="gpt-5.1", 
            messages=[
                {"role": "system", "content": "You are a math problem transformer that removes context from word problems."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_completion_tokens=500
        )
        
    return response.choices[0].message.content

async def add_context(context_free_question: str, topic: str, numeric_answer: str, semaphore: asyncio.Semaphore) -> str:
    """Use GPT to transform context-free question into story problem with given context."""
   
    prompt = f"""You are a creative writer. 
    Transform the given pure math question into a word problem about {topic}. 
    Keep the same numbers and mathematical structure, but add a story/context around it. 
    Preserve all the original mathematical relationships.
    IMPORTANT: Do not change any numbers or operations - the answer must remain exactly {numeric_answer}.
    
    Pure math question: {context_free_question}

    Output ONLY the transformed question, nothing else."""

    async with semaphore:
        response = await client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": f"You are a math problem writer specializing in {topic}."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_completion_tokens=500
        )
    
    return response.choices[0].message.content


async def remove_all_contexts(
    questions: List[Dict],
    semaphore: asyncio.Semaphore,
    batch_size: int = 20
) -> List[Dict]:
    """Phase 1: Remove context from all questions in batches."""
    print("\n=== PHASE 1: Removing context from all questions ===")
    
    results = []
    
    # Process in batches for better progress tracking
    batches = [questions[i:i + batch_size] for i in range(0, len(questions), batch_size)]
    
    for batch_idx, batch in enumerate(tqdm(batches, desc="Removing contexts")):
        # Create tasks for all questions in batch
        tasks = []
        for item in batch:
            original_q = item.get('Problem', item.get('question', ''))
            task = remove_context(original_q, semaphore)
            
            # Extract numeric answer from multiple choice
            options = item.get('options', '')
            correct = item.get('correct', '')
            numeric_answer = extract_numeric_answer(options, correct)
            
            tasks.append((item, original_q, task, numeric_answer))
        
        # Wait for all tasks in batch to complete
        for item, original_q, task, numeric_answer in tasks:
            try:
                context_free = await task
                results.append({
                    **item,
                    "original_question": original_q,
                    "context_free": context_free,
                    "numeric_answer": numeric_answer
                })
            except Exception as e:
                print(f"\nError removing context: {e}")
                results.append({
                    **item,
                    "original_question": original_q,
                    "context_free": None,
                    "numeric_answer": item.get('correct', ''),
                    "error": str(e)
                })
    
    print(f"Completed Phase 1: {len(results)} questions processed")
    return results

async def add_all_contexts(
    context_free_questions: List[Dict],
    target_contexts: List[str],
    semaphore: asyncio.Semaphore,
    batch_size: int = 20
) -> List[Dict]:
    """Phase 2: Add new contexts to all context-free questions."""
    print(f"\n=== PHASE 2: Adding new contexts {target_contexts} ===")
    
    results = []
    
    # Process in batches
    batches = [context_free_questions[i:i + batch_size] for i in range(0, len(context_free_questions), batch_size)]
    
    for batch_idx, batch in enumerate(tqdm(batches, desc="Adding new contexts")):
        # Create tasks for all question-context combinations in batch
        tasks = []
        for item in batch:
            context_free = item.get('context_free')
            numeric_answer = item.get('numeric_answer', item.get('correct', ''))
            
            if context_free is None:
                # Skip questions that failed in phase 1 - preserve original format
                result_item = item.copy()
                result_item['transformations'] = {
                    'context_free': None,
                    'new_contexts': {}
                }
                result_item['correct'] = numeric_answer  # Convert to numeric
                result_item.pop('options', None)  # Remove multiple choice options
                result_item.pop('original_question', None)
                result_item.pop('numeric_answer', None)
                results.append(result_item)
                continue
            
            # Create tasks for all contexts for this question
            context_tasks = {
                context: add_context(context_free, context, numeric_answer, semaphore)
                for context in target_contexts
            }
            tasks.append((item, context_tasks, numeric_answer))
        
        # Wait for all tasks in batch to complete
        for item, context_tasks, numeric_answer in tasks:
            try:
                # Gather all context versions for this question
                new_contexts = {}
                for context, task in context_tasks.items():
                    new_contexts[context] = await task
                
                # Create result preserving ALL original fields except options
                result_item = item.copy()
                result_item['transformations'] = {
                    'context_free': item['context_free'],
                    'new_contexts': new_contexts
                }
                # Convert answer format from multiple choice to numeric
                result_item['correct'] = numeric_answer
                result_item.pop('options', None)  # Remove multiple choice options
                # Remove temporary fields used only during processing
                result_item.pop('original_question', None)
                result_item.pop('context_free', None)
                result_item.pop('numeric_answer', None)
                
                results.append(result_item)
            except Exception as e:
                print(f"\nError adding contexts: {e}")
                result_item = item.copy()
                result_item['transformations'] = {
                    'context_free': item.get('context_free'),
                    'new_contexts': {},
                    'error': str(e)
                }
                result_item['correct'] = numeric_answer
                result_item.pop('options', None)
                result_item.pop('original_question', None)
                result_item.pop('context_free', None)
                result_item.pop('numeric_answer', None)
                results.append(result_item)
    
    print(f"Completed Phase 2: {len(results)} questions with {len(target_contexts)} new contexts each")
    return results


async def process_dataset_async(
    input_file: str,
    output_file: str,
    target_contexts: List[str] = ["business", "sports", "knitting"],
    max_questions: int = None,
    batch_size: int = 20,
    max_concurrent: int = 10,
    save_intermediate: bool = True
):
    """Process entire MathQA dataset in two phases: remove context, then add new contexts."""
    # Load dataset
    print(f"Loading dataset from {input_file}...")
    questions = load_mathqa_dataset(input_file)
    
    if max_questions:
        questions = questions[:max_questions]
    
    print(f"\nTotal questions to process: {len(questions)}")
    print(f"Target contexts: {target_contexts}")
    print(f"Batch size: {batch_size}, Max concurrent requests: {max_concurrent}")
    
    # Create semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # PHASE 1: Remove all contexts
    context_free_results = await remove_all_contexts(questions, semaphore, batch_size)
    
    # Save intermediate results after phase 1
    if save_intermediate:
        phase1_file = output_file.replace('.json', '_phase1_context_free.json')
        print(f"\nSaving Phase 1 results to {phase1_file}...")
        async with aiofiles.open(phase1_file, 'w') as f:
            await f.write(json.dumps(context_free_results, indent=2))
    
    # PHASE 2: Add new contexts to all questions
    final_results = await add_all_contexts(context_free_results, target_contexts, semaphore, batch_size)
    
    # Save final results
    print(f"\nSaving final results to {output_file}...")
    async with aiofiles.open(output_file, 'w') as f:
        await f.write(json.dumps(final_results, indent=2))
    
    # Print summary
    successful = sum(1 for r in final_results if r.get('context_free') and r.get('new_contexts'))
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total questions processed: {len(final_results)}")
    print(f"Successful transformations: {successful}")
    print(f"Failed transformations: {len(final_results) - successful}")
    print(f"Contexts generated per question: {len(target_contexts)}")
    print(f"Total new questions created: {successful * len(target_contexts)}")
    print(f"{'='*60}")
    
    return final_results


def process_dataset(
    input_file: str,
    output_file: str,
    target_contexts: List[str] = ["business", "sports", "knitting"],
    max_questions: int = None,
    batch_size: int = 20,
    max_concurrent: int = 10,
    save_intermediate: bool = True
):
    """Synchronous wrapper for async processing."""
    return asyncio.run(process_dataset_async(
        input_file=input_file,
        output_file=output_file,
        target_contexts=target_contexts,
        max_questions=max_questions,
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        save_intermediate=save_intermediate
    ))


# Example usage
if __name__ == "__main__":
    # Example 1: Process with default settings
    # process_dataset(
    #     input_file="mathqa_train.jsonl",
    #     output_file="mathqa_transformed.json",
    #     target_contexts=["business", "sports", "knitting"],
    #     max_questions=50,   # Remove to process all
    #     batch_size=20,      # Questions per batch
    #     max_concurrent=10   # Max simultaneous API calls
    # )
    
    # Example 2: Custom contexts
    process_dataset(
        input_file="/data/jane/math_gender/mathQA/train.json",
        output_file="/data/jane/math_gender/1_process_data/mathqa_transformed.json",
        target_contexts=["cooking", "gardening", "fashion", "dance", "baking", "makeup", "shopping", "romance novels", "knitting", "construction", "sports", "video games", "cars", "weightlifting", "finance", "fishing", "woodworking", "hiking"],
        max_questions=50,
        batch_size=20,
        max_concurrent=15
    )