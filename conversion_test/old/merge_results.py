#!/usr/bin/env python3
"""
Merge and normalize conversion results from different experiments.

This script:
1. Loads results from scientific_conversion_results and language_conversion_results
2. Normalizes the schema to a common format
3. Merges all results into a unified dataset
4. Provides comparison capabilities across contexts
"""

import json
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import argparse


def load_json_file(filepath: str) -> List[Dict[str, Any]]:
    """Load a JSON file containing results."""
    print(f"Loading: {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} entries")
    return data


def normalize_scientific_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a scientific conversion result to common format."""
    normalized = {
        # Common fields
        'model': result.get('model'),
        'test_value': result.get('test_value'),
        'difficulty': result.get('difficulty'),
        'from_unit': result.get('from_unit'),
        'to_unit': result.get('to_unit'),
        'correct_answer': result.get('correct_answer'),
        'model_answer_raw': result.get('model_answer_raw'),
        'model_answer': result.get('model_answer'),
        'is_correct': result.get('is_correct', False),
        'is_error': result.get('is_error', False),
        'prompt': result.get('prompt'),
        
        # Scientific-specific fields
        'experiment_type': 'scientific',
        'conversion_name': result.get('conversion_name'),
        'conversion_type': result.get('conversion_type'),  # linear, temperature, etc.
        'context_type': result.get('context_type', 'context_free'),
        'context': result.get('context'),  # substance name or None
        
        # Normalized fields for comparison
        'category': result.get('conversion_name'),  # e.g., "speed", "volume", "temperature"
        'subcategory': result.get('conversion_type'),  # e.g., "linear", "temperature"
        'has_context': result.get('context') is not None,
        'context_name': result.get('context'),
        'language': 'english',  # Scientific conversions are in English
        'ingredient': None,  # Not applicable for scientific
    }
    
    return normalized


def normalize_language_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a language conversion result to common format."""
    normalized = {
        # Common fields
        'model': result.get('model'),
        'test_value': result.get('test_value'),
        'difficulty': result.get('difficulty'),
        'from_unit': result.get('from_unit'),
        'to_unit': result.get('to_unit'),
        'correct_answer': result.get('correct_answer'),
        'model_answer_raw': result.get('model_answer_raw'),
        'model_answer': result.get('model_answer'),
        'is_correct': result.get('is_correct', False),
        'is_error': result.get('is_error', False),
        'prompt': result.get('prompt'),
        
        # Language-specific fields
        'experiment_type': 'language',
        'ingredient': result.get('ingredient'),
        'language': result.get('language', 'english'),
        'conversion_type': result.get('conversion_type'),  # us_to_metric, metric_to_us
        'context_type': result.get('context_type', 'ingredient'),
        
        # Normalized fields for comparison
        'category': 'cooking',  # All language conversions are cooking-related
        'subcategory': result.get('conversion_type'),  # us_to_metric, metric_to_us
        'has_context': True,  # Language conversions always have ingredients
        'context_name': result.get('ingredient'),
        'conversion_name': None,  # Not applicable for language
    }
    
    return normalized


def load_all_results(
    scientific_dir: str = "scientific_conversion_results",
    language_dir: str = "language_conversion_results",
    scientific_files: Optional[List[str]] = None,
    language_files: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Load and normalize all result files."""
    all_results = []
    
    # Load scientific conversion results
    if os.path.exists(scientific_dir):
        if scientific_files:
            files = [os.path.join(scientific_dir, f) for f in scientific_files]
        else:
            # Find all complete result files (not checkpoints)
            files = [
                os.path.join(scientific_dir, f)
                for f in os.listdir(scientific_dir)
                if f.startswith('conversion_results_') and f.endswith('.json')
                and 'checkpoint' not in f
            ]
        
        for filepath in files:
            if os.path.exists(filepath):
                try:
                    results = load_json_file(filepath)
                    normalized = [normalize_scientific_result(r) for r in results]
                    all_results.extend(normalized)
                except Exception as e:
                    print(f"  ERROR loading {filepath}: {e}")
    
    # Load language conversion results
    if os.path.exists(language_dir):
        if language_files:
            files = [os.path.join(language_dir, f) for f in language_files]
        else:
            # Find all language result files
            files = [
                os.path.join(language_dir, f)
                for f in os.listdir(language_dir)
                if f.startswith('language_conversion_results_') and f.endswith('.json')
            ]
        
        for filepath in files:
            if os.path.exists(filepath):
                try:
                    results = load_json_file(filepath)
                    normalized = [normalize_language_result(r) for r in results]
                    all_results.extend(normalized)
                except Exception as e:
                    print(f"  ERROR loading {filepath}: {e}")
    
    return all_results


def create_comparison_summary(df: pd.DataFrame, output_file: str):
    """Create summary statistics for model comparison."""
    print("\n" + "="*60)
    print("CREATING COMPARISON SUMMARY")
    print("="*60)
    
    # Filter out errors
    df_clean = df[~df['is_error']].copy()
    
    summaries = []
    
    # Overall accuracy by model
    print("\n1. Overall Accuracy by Model:")
    overall = df_clean.groupby('model')['is_correct'].agg(['sum', 'count', 'mean']).reset_index()
    overall.columns = ['model', 'correct', 'total', 'accuracy']
    overall['accuracy'] = overall['accuracy'] * 100
    overall = overall.sort_values('accuracy', ascending=False)
    print(overall.to_string(index=False))
    summaries.append(('overall_by_model', overall))
    
    # By experiment type
    print("\n2. Accuracy by Model and Experiment Type:")
    by_type = df_clean.groupby(['model', 'experiment_type'])['is_correct'].agg(['sum', 'count', 'mean']).reset_index()
    by_type.columns = ['model', 'experiment_type', 'correct', 'total', 'accuracy']
    by_type['accuracy'] = by_type['accuracy'] * 100
    by_type = by_type.sort_values(['model', 'experiment_type'])
    print(by_type.to_string(index=False))
    summaries.append(('by_experiment_type', by_type))
    
    # By category (scientific conversion name or "cooking")
    print("\n3. Accuracy by Model and Category:")
    by_category = df_clean.groupby(['model', 'category'])['is_correct'].agg(['sum', 'count', 'mean']).reset_index()
    by_category.columns = ['model', 'category', 'correct', 'total', 'accuracy']
    by_category['accuracy'] = by_category['accuracy'] * 100
    by_category = by_category.sort_values(['model', 'category'])
    print(by_category.to_string(index=False))
    summaries.append(('by_category', by_category))
    
    # By difficulty
    print("\n4. Accuracy by Model and Difficulty:")
    by_difficulty = df_clean.groupby(['model', 'difficulty'])['is_correct'].agg(['sum', 'count', 'mean']).reset_index()
    by_difficulty.columns = ['model', 'difficulty', 'correct', 'total', 'accuracy']
    by_difficulty['accuracy'] = by_difficulty['accuracy'] * 100
    by_difficulty = by_difficulty.sort_values(['model', 'difficulty'])
    print(by_difficulty.to_string(index=False))
    summaries.append(('by_difficulty', by_difficulty))
    
    # By context (with/without)
    print("\n5. Accuracy by Model and Context:")
    by_context = df_clean.groupby(['model', 'has_context'])['is_correct'].agg(['sum', 'count', 'mean']).reset_index()
    by_context.columns = ['model', 'has_context', 'correct', 'total', 'accuracy']
    by_context['accuracy'] = by_context['accuracy'] * 100
    by_context['has_context'] = by_context['has_context'].map({True: 'with_context', False: 'no_context'})
    by_context = by_context.sort_values(['model', 'has_context'])
    print(by_context.to_string(index=False))
    summaries.append(('by_context', by_context))
    
    # Language-specific: by language
    if 'language' in df_clean.columns:
        lang_data = df_clean[df_clean['experiment_type'] == 'language']
        if len(lang_data) > 0:
            print("\n6. Language Conversion Accuracy by Model and Language:")
            by_language = lang_data.groupby(['model', 'language'])['is_correct'].agg(['sum', 'count', 'mean']).reset_index()
            by_language.columns = ['model', 'language', 'correct', 'total', 'accuracy']
            by_language['accuracy'] = by_language['accuracy'] * 100
            by_language = by_language.sort_values(['model', 'language'])
            print(by_language.to_string(index=False))
            summaries.append(('by_language', by_language))
    
    # Save summaries
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else "."
    
    for name, summary_df in summaries:
        summary_file = os.path.join(summary_dir, f"comparison_{name}_{timestamp}.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSaved: {summary_file}")
    
    return summaries


def main():
    parser = argparse.ArgumentParser(
        description="Merge and normalize conversion results from different experiments"
    )
    parser.add_argument(
        '--scientific-dir',
        type=str,
        default='scientific_conversion_results',
        help='Directory containing scientific conversion results'
    )
    parser.add_argument(
        '--language-dir',
        type=str,
        default='language_conversion_results',
        help='Directory containing language conversion results'
    )
    parser.add_argument(
        '--scientific-files',
        type=str,
        nargs='+',
        help='Specific scientific result files to load (relative to scientific-dir)'
    )
    parser.add_argument(
        '--language-files',
        type=str,
        nargs='+',
        help='Specific language result files to load (relative to language-dir)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='merged_results.json',
        help='Output file for merged results'
    )
    parser.add_argument(
        '--output-csv',
        type=str,
        help='Also save as CSV (optional)'
    )
    parser.add_argument(
        '--create-summary',
        action='store_true',
        help='Create comparison summary statistics'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("MERGING CONVERSION RESULTS")
    print("="*60)
    print()
    
    # Load all results
    all_results = load_all_results(
        scientific_dir=args.scientific_dir,
        language_dir=args.language_dir,
        scientific_files=args.scientific_files,
        language_files=args.language_files
    )
    
    print(f"\nTotal merged results: {len(all_results)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Print basic statistics
    print(f"\nModels: {df['model'].unique().tolist()}")
    print(f"Experiment types: {df['experiment_type'].unique().tolist()}")
    print(f"Categories: {df['category'].unique().tolist()}")
    if 'language' in df.columns:
        print(f"Languages: {df[df['experiment_type'] == 'language']['language'].unique().tolist()}")
    
    # Save merged results
    print(f"\nSaving merged results to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save as CSV if requested
    if args.output_csv:
        print(f"Saving as CSV to: {args.output_csv}")
        df.to_csv(args.output_csv, index=False)
    elif args.create_summary:
        # Auto-generate CSV filename
        csv_file = args.output.replace('.json', '.csv')
        print(f"Saving as CSV to: {csv_file}")
        df.to_csv(csv_file, index=False)
    
    # Create summary if requested
    if args.create_summary:
        create_comparison_summary(df, args.output)
    
    print("\n" + "="*60)
    print("MERGE COMPLETE")
    print("="*60)
    print(f"\nMerged {len(all_results)} results from all experiments")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
