import json
import os
import asyncio
from openai import AsyncOpenAI
from typing import List, Dict
from tqdm.asyncio import tqdm
import aiofiles

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def load_mathqa_dataset(file_path: str) -> List[Dict]:
    """Load MathQA dataset from JSONL file."""
    questions = []
    with open(file_path, 'r') as f:
        for line in f:
            questions.append(json.loads(line))
    return questions

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
            model="gpt-5.1",  # Use "gpt-4" as GPT-5.1 isn't available yet
            messages=[
                {"role": "system", "content": "You are a math problem transformer that removes context from word problems."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        
    return response.output_text

async def add_context(context_free_question: str, topic: str, semaphore: asyncio.Semaphore) -> str:
    """Use GPT to transform context-free question into story problem with given context."""
   
    prompt = f"""You are a creative writer. 
    Transform the given pure math question into a word problem about {topic}. 
    Keep the same numbers and mathematical structure, but add a story/context around it. 
    Preserve all the original mathematical relationships.
    
    Pure math question: {context_free_question}

    Output ONLY the transformed question, nothing else."""

    async with semaphore:
        response = await client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": f"You are a math problem writer specializing in {topic}. You expertly transform "},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )
    
    return response.output_text


async def transform_question(original_question: str, target_contexts: List[str], semaphore: asyncio.Semaphore) -> Dict:
    """Complete transformation pipeline for a single question."""
    print(f"\nOriginal: {original_question[:100]}...")
    """Complete transformation pipeline for a single question."""
    try:
        # Step 1: Remove context
        context_free = await remove_context(original_question, semaphore)
        
        # Step 2: Add new contexts (in parallel)
        context_tasks = [
            add_context(context_free, context, semaphore)
            for context in target_contexts
        ]
        new_contexts_list = await asyncio.gather(*context_tasks)
        
        # Organize results
        transformed = {
            "original": original_question,
            "context_free": context_free,
            "new_contexts": dict(zip(target_contexts, new_contexts_list))
        }
        
        return transformed
        
    except Exception as e:
        print(f"Error transforming question: {e}")
        return {
            "original": original_question,
            "context_free": None,
            "new_contexts": {},
            "error": str(e)
        }

async def process_batch(
    batch: List[Dict],
    target_contexts: List[str],
    semaphore: asyncio.Semaphore
) -> List[Dict]:
    """Process a batch of questions asynchronously."""
    tasks = []
    for item in batch:
        original_q = item.get('Problem', item.get('question', ''))
        task = transform_question(original_q, target_contexts, semaphore)
        tasks.append((item, task))
    
    results = []
    for item, task in tasks:
        transformed = await task
        result = {
            **item,
            "transformations": transformed
        }
        results.append(result)
    
    return results


async def process_dataset_async(
    input_file: str,
    output_file: str,
    target_contexts: List[str] = ["business", "sports", "knitting"],
    max_questions: int = None,
    batch_size: int = 10,
    max_concurrent: int = 5
):
    """Process entire MathQA dataset asynchronously with batching."""
    # Load dataset
    print(f"Loading dataset from {input_file}...")
    questions = load_mathqa_dataset(input_file)
    
    if max_questions:
        questions = questions[:max_questions]
    
    print(f"Processing {len(questions)} questions...")
    print(f"Batch size: {batch_size}, Max concurrent requests: {max_concurrent}")
    
    # Create semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Split into batches
    batches = [questions[i:i + batch_size] for i in range(0, len(questions), batch_size)]
    
    all_results = []
    
    # Process batches with progress bar
    for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
        print(f"\nProcessing batch {batch_idx + 1}/{len(batches)}")
        batch_results = await process_batch(batch, target_contexts, semaphore)
        all_results.extend(batch_results)
        
        # Optional: Save intermediate results after each batch
        if (batch_idx + 1) % 5 == 0:
            intermediate_file = output_file.replace('.json', f'_checkpoint_{batch_idx + 1}.json')
            async with aiofiles.open(intermediate_file, 'w') as f:
                await f.write(json.dumps(all_results, indent=2))
            print(f"Checkpoint saved to {intermediate_file}")
    
    # Save final results
    print(f"\nSaving final results to {output_file}...")
    async with aiofiles.open(output_file, 'w') as f:
        await f.write(json.dumps(all_results, indent=2))
    
    print(f"Complete! Processed {len(all_results)} questions.")
    return all_results

def process_dataset(
    input_file: str,
    output_file: str,
    target_contexts: List[str] = ["business", "sports", "knitting"],
    max_questions: int = None,
    batch_size: int = 10,
    max_concurrent: int = 5
):
    """Synchronous wrapper for async processing."""
    return asyncio.run(process_dataset_async(
        input_file=input_file,
        output_file=output_file,
        target_contexts=target_contexts,
        max_questions=max_questions,
        batch_size=batch_size,
        max_concurrent=max_concurrent
    ))


# Example usage
if __name__ == "__main__":
    # Example 1: Process with default settings
    process_dataset(
        input_file="mathqa_train.jsonl",
        output_file="mathqa_transformed.json",
        target_contexts=["business", "sports", "knitting"],
        max_questions=100,  # Remove to process all
        batch_size=10,      # Questions per batch
        max_concurrent=5    # Max simultaneous API calls
    )
    
    # Example 2: Higher throughput for larger datasets
    # process_dataset(
    #     input_file="mathqa_train.jsonl",
    #     output_file="mathqa_transformed.json",
    #     target_contexts=["business", "sports", "knitting"],
    #     batch_size=50,
    #     max_concurrent=20
    # )
