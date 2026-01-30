#!/usr/bin/env python3
"""
Test script to run a small sample for each domain to check performance.

This script:
1. Creates test TSV files with limited test values (5 easy + 5 hard numbers)
2. Runs inference on each domain with a small sample
3. Collects and displays results
"""

import json
import pandas as pd
import argparse
import asyncio
from pathlib import Path
import subprocess
import sys
import os
from datetime import datetime

def create_test_numbers_file(output_file: Path):
    """Create a test numbers.json with just a few values."""
    test_numbers = {
        "easy": [1, 5, 10, 20, 50],
        "hard": [4508.208, 1297.195, 18.333, 9.0241, 0.2994]
    }
    
    with open(output_file, 'w') as f:
        json.dump(test_numbers, f, indent=2)
    
    print(f"Created test numbers file: {output_file}")
    return test_numbers

def create_test_times_file(output_file: Path):
    """Create a test times.json with just a few time values."""
    test_times = {
        "easy": ["1AM", "1PM", "12PM", "3PM", "9PM"],
        "hard": ["11:59AM", "3:49PM", "2:48AM", "6:58PM", "8:11PM"]
    }
    
    with open(output_file, 'w') as f:
        json.dump(test_times, f, indent=2)
    
    print(f"Created test times file: {output_file}")
    return test_times

def run_preprocessing_test(conversions_dir: Path, test_numbers_file: Path, 
                          test_times_file: Path, output_dir: Path):
    """Run preprocessing with test numbers."""
    print("\n" + "="*60)
    print("RUNNING PREPROCESSING WITH TEST VALUES")
    print("="*60)
    
    cmd = [
        sys.executable, "preprocessing.py",
        "--conversions-dir", str(conversions_dir),
        "--numbers-file", str(test_numbers_file),
        "--times-file", str(test_times_file),
        "--output-dir", str(output_dir)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: Preprocessing failed")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True

def run_inference_test(domain: str, input_file: Path, output_file: Path, 
                      models: list = None, max_rows: int = None):
    """Run inference on a single domain."""
    if models is None:
        models = ["gpt-4o"]  # Use just one model for testing
    
    print(f"\nRunning inference for {domain}...")
    print(f"  Input: {input_file}")
    print(f"  Output: {output_file}")
    
    # If max_rows is specified, create a limited version of the input file
    if max_rows is not None:
        try:
            df = pd.read_csv(input_file, sep='\t')
            if len(df) > max_rows:
                # Sample rows (stratified by difficulty if available)
                if 'difficulty' in df.columns:
                    # Sample proportionally from each difficulty
                    sampled = df.groupby('difficulty', group_keys=False).apply(
                        lambda x: x.sample(min(len(x), max_rows // 2))
                    ).head(max_rows)
                else:
                    sampled = df.head(max_rows)
                
                temp_input = input_file.parent / f"{input_file.stem}_limited.tsv"
                sampled.to_csv(temp_input, sep='\t', index=False)
                input_file = temp_input
                print(f"  Limited to {len(sampled)} rows for testing")
        except Exception as e:
            print(f"  WARNING: Could not limit rows: {e}")
    
    # Check if input file exists
    if not input_file.exists():
        print(f"  WARNING: Input file not found: {input_file}")
        return None
    
    # Count rows in input
    try:
        df = pd.read_csv(input_file, sep='\t')
        num_rows = len(df)
        print(f"  Rows in file: {num_rows}")
    except Exception as e:
        print(f"  ERROR: Could not read input file: {e}")
        return None
    
    # Run main_conversion.py
    cmd = [
        sys.executable, "main_conversion.py",
        "--domain", domain,
        "--input-file", str(input_file),
        "--output-file", str(output_file)
    ]
    
    # Add model arguments if specified
    if models:
        cmd.extend(["--models"] + models)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ERROR: Inference failed")
        print(result.stderr)
        return None
    
    # Load and analyze results
    try:
        result_df = pd.read_csv(output_file, sep='\t')
        
        # Calculate statistics
        total = len(result_df)
        if total == 0:
            return None
        
        stats = {
            'domain': domain,
            'total_prompts': total,
            'models_tested': models if models else ['default']
        }
        
        # Handle single model vs multiple models
        if len(models) == 1:
            model_name = models[0]
            loss_col = 'loss'
            answer_col = 'model_answer'
        else:
            # For multiple models, aggregate across all
            model_name = models[0]  # Use first model for stats
            loss_col = f'loss_{model_name}'
            answer_col = f'model_answer_{model_name}'
        
        # Check if columns exist
        has_loss = loss_col in result_df.columns
        has_answer = answer_col in result_df.columns
        
        if has_loss:
            # Filter out non-numeric losses
            numeric_losses = pd.to_numeric(result_df[loss_col], errors='coerce').dropna()
            if len(numeric_losses) > 0:
                stats['avg_loss'] = float(numeric_losses.mean())
                stats['median_loss'] = float(numeric_losses.median())
                # Count correct (loss == 0)
                correct = (numeric_losses == 0).sum()
                stats['correct'] = int(correct)
                stats['accuracy'] = (correct / len(numeric_losses) * 100) if len(numeric_losses) > 0 else 0
                stats['valid_answers'] = int(len(numeric_losses))
        
        if has_answer:
            # Count how many got answers
            answered = result_df[answer_col].notna().sum()
            stats['answered'] = int(answered)
            stats['answer_rate'] = (answered / total * 100) if total > 0 else 0
        
        return stats
    
    except Exception as e:
        print(f"  ERROR: Could not analyze results: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Test all domains with small samples')
    parser.add_argument('--models', type=str, nargs='+', default=['gpt-4o'],
                       help='Models to test (default: gpt-4o)')
    parser.add_argument('--test-dir', type=str, default='test_output',
                       help='Directory for test outputs')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip preprocessing if test files already exist')
    parser.add_argument('--max-rows', type=int, default=None,
                       help='Maximum number of rows to process per domain (for faster testing)')
    
    args = parser.parse_args()
    
    # Setup directories
    test_dir = Path(args.test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessed_dir = test_dir / 'preprocessed'
    results_dir = test_dir / 'results'
    preprocessed_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    conversions_dir = Path('conversions')
    
    print("="*60)
    print("DOMAIN TESTING SCRIPT")
    print("="*60)
    print(f"Test directory: {test_dir}")
    print(f"Models: {args.models}")
    print()
    
    # Create test numbers and times files
    test_numbers_file = test_dir / 'test_numbers.json'
    test_times_file = test_dir / 'test_times.json'
    
    if not args.skip_preprocessing or not test_numbers_file.exists():
        create_test_numbers_file(test_numbers_file)
        create_test_times_file(test_times_file)
    
    # Run preprocessing
    if not args.skip_preprocessing:
        success = run_preprocessing_test(
            conversions_dir, 
            test_numbers_file,
            test_times_file,
            preprocessed_dir
        )
        if not success:
            print("ERROR: Preprocessing failed. Exiting.")
            return
    else:
        print("Skipping preprocessing (using existing files)")
    
    # Define all domains
    domains = [
        "bits_bytes",
        "clothing_sizes_clothing_size",
        "clothing_sizes_pant_size",
        "clothing_sizes_shoe_size",
        "clothing_sizes_bra_size",
        "cooking",
        "currency",
        "density",
        "energy",
        "moles_to_particles",
        "speed",
        "temperature",
        "timezone",
        "volume"
    ]
    
    # Test each domain
    print("\n" + "="*60)
    print("RUNNING INFERENCE TESTS")
    print("="*60)
    
    all_stats = []
    
    for domain in domains:
        input_file = preprocessed_dir / f"{domain}.tsv"
        output_file = results_dir / f"{domain}_converted.tsv"
        
        stats = run_inference_test(domain, input_file, output_file, args.models, args.max_rows)
        if stats:
            all_stats.append(stats)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    print()
    
    if all_stats:
        # Create summary DataFrame
        summary_data = []
        for stats in all_stats:
            row = {
                'Domain': stats['domain'],
                'Total Prompts': stats['total_prompts']
            }
            
            if 'answered' in stats:
                row['Answered'] = stats['answered']
                row['Answer Rate %'] = f"{stats['answer_rate']:.1f}"
            
            if 'valid_answers' in stats:
                row['Valid Answers'] = stats['valid_answers']
            
            if 'correct' in stats:
                row['Correct'] = stats['correct']
                row['Accuracy %'] = f"{stats['accuracy']:.1f}"
            
            if 'avg_loss' in stats:
                row['Avg Loss'] = f"{stats['avg_loss']:.4f}"
                row['Median Loss'] = f"{stats['median_loss']:.4f}"
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Save summary
        summary_file = test_dir / f"test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary saved to: {summary_file}")
    else:
        print("No results to summarize")
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
    print(f"\nTest outputs in: {test_dir}")
    print(f"  - Preprocessed files: {preprocessed_dir}")
    print(f"  - Results: {results_dir}")

if __name__ == '__main__':
    main()
