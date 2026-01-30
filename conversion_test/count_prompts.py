#!/usr/bin/env python3
"""
Count the number of prompts that will be created for each domain.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

def count_prompts_for_domain(
    json_file: Path,
    numbers_file: Path,
    times_file: Optional[Path] = None,
    substances_file: Optional[Path] = None
) -> Dict[str, Any]:
    """Count prompts for a single conversion domain."""
    
    # Load conversion config
    with open(json_file, 'r') as f:
        config = json.load(f)
    
    # Load numbers
    with open(numbers_file, 'r') as f:
        numbers_data = json.load(f)
    
    easy_numbers = numbers_data.get('easy', [])
    hard_numbers = numbers_data.get('hard', [])
    
    # Load times if provided
    easy_times = []
    hard_times = []
    if times_file and times_file.exists():
        with open(times_file, 'r') as f:
            times_data = json.load(f)
        easy_times = times_data.get('easy', [])
        hard_times = times_data.get('hard', [])
    
    # Load substances (contexts) if provided
    substances_contexts = []
    if substances_file and substances_file.exists():
        with open(substances_file, 'r') as f:
            substances_data = json.load(f)
        substances_contexts = substances_data.get('contexts', [])
    
    conversion_name = config.get('name', json_file.stem)
    conversion_type = config.get('conversion_type', 'linear')
    
    # Determine if this conversion should use substances as contexts
    use_substances_contexts = conversion_name in ['density', 'volume', 'moles_to_particles']
    
    # Add contexts to config if needed
    if use_substances_contexts and substances_contexts:
        if 'contexts' not in config:
            config['contexts'] = []
        # Merge with existing contexts, avoiding duplicates
        existing_contexts = set(config.get('contexts', []))
        for ctx in substances_contexts:
            if ctx not in existing_contexts:
                config['contexts'].append(ctx)
    
    # Check if this is a multi-section config (like clothing_sizes)
    sub_sections = ['clothing_size', 'pant_size', 'shoe_size', 'bra_size']
    has_sub_sections = any(section in config for section in sub_sections)
    
    results = {}
    
    if has_sub_sections:
        # Process each sub-section separately
        for section_name in sub_sections:
            if section_name not in config:
                continue
            
            section_config = config[section_name]
            # Add contexts to section_config if this conversion uses substances
            if use_substances_contexts and substances_contexts:
                if 'contexts' not in section_config:
                    section_config['contexts'] = []
                # Merge with existing contexts, avoiding duplicates
                existing_contexts = set(section_config.get('contexts', []))
                for ctx in substances_contexts:
                    if ctx not in existing_contexts:
                        section_config['contexts'].append(ctx)
            
            count_info = count_section_prompts(
                section_config,
                conversion_name,
                section_name,
                conversion_type,
                easy_numbers,
                hard_numbers,
                easy_times,
                hard_times,
                full_config=config
            )
            
            domain_name = f"{conversion_name}_{section_name}"
            results[domain_name] = count_info
    else:
        # Standard single-section processing
        # Add contexts to config if this conversion uses substances
        if use_substances_contexts and substances_contexts:
            if 'contexts' not in config:
                config['contexts'] = []
            # Merge with existing contexts, avoiding duplicates
            existing_contexts = set(config.get('contexts', []))
            for ctx in substances_contexts:
                if ctx not in existing_contexts:
                    config['contexts'].append(ctx)
        
        count_info = count_section_prompts(
            config,
            conversion_name,
            None,
            conversion_type,
            easy_numbers,
            hard_numbers,
            easy_times,
            hard_times,
            full_config=config
        )
        
        results[conversion_name] = count_info
    
    return results

def count_section_prompts(
    section_config: Dict,
    base_conversion_name: str,
    section_name: Optional[str],
    conversion_type: str,
    easy_numbers: List,
    hard_numbers: List,
    easy_times: List,
    hard_times: List,
    full_config: Optional[Dict] = None
) -> Dict[str, Any]:
    """Count prompts for a single section."""
    
    unit_pairs = section_config.get('unit_pairs', [])
    contexts = section_config.get('contexts', [])
    
    # Check if this is a currency conversion with easy/hard currencies
    is_currency = (base_conversion_name == 'currency' or conversion_type == 'currency')
    easy_currencies = []
    hard_currencies = []
    if is_currency and full_config:
        easy_currencies = full_config.get('easy_currencies', [])
        hard_currencies = full_config.get('hard_currencies', [])
    
    total_prompts = 0
    context_free_prompts = 0
    context_prompts = 0
    unit_pair_counts = []
    
    # Process each unit pair
    for pair in unit_pairs:
        from_unit = pair['from']
        to_unit = pair['to']
        
        # Determine test values
        # For currency conversions, use all 200 numbers (easy + hard) for all unit pairs
        if is_currency:
            # Use all numbers (easy + hard) for all currency pairs
            test_values = easy_numbers + hard_numbers
        elif section_config.get('size_mappings'):
            size_mappings = section_config.get('size_mappings', {})
            mapping_key = None
            for key in size_mappings.keys():
                if key.startswith(f"{from_unit}_to_"):
                    mapping_key = key
                    break
            
            if mapping_key:
                mapping = size_mappings[mapping_key]
                test_values = list(mapping.keys())
            else:
                test_values = easy_numbers[:10] + hard_numbers[:10]
        elif conversion_type == 'timezone':
            if easy_times or hard_times:
                test_values = easy_times + hard_times
            else:
                test_values = []
                for num in easy_numbers + hard_numbers:
                    if isinstance(num, (int, float)):
                        h = int(num) % 24
                        if h == 0:
                            test_values.append("12AM")
                        elif h < 12:
                            test_values.append(f"{h}AM")
                        elif h == 12:
                            test_values.append("12PM")
                        else:
                            test_values.append(f"{h-12}PM")
        else:
            test_values = easy_numbers + hard_numbers
        
        # Count prompts for this unit pair
        num_test_values = len(test_values)
        num_contexts = len(contexts)
        
        # Context-free: 1 per test value
        pair_context_free = num_test_values
        # With context: num_test_values × num_contexts
        pair_context = num_test_values * num_contexts
        
        pair_total = pair_context_free + pair_context
        
        unit_pair_counts.append({
            'from': from_unit,
            'to': to_unit,
            'test_values': num_test_values,
            'context_free': pair_context_free,
            'with_context': pair_context,
            'total': pair_total
        })
        
        context_free_prompts += pair_context_free
        context_prompts += pair_context
        total_prompts += pair_total
    
    return {
        'unit_pairs': len(unit_pairs),
        'test_values_easy': len(easy_numbers) if conversion_type != 'timezone' else len(easy_times),
        'test_values_hard': len(hard_numbers) if conversion_type != 'timezone' else len(hard_times),
        'test_values_total': len(easy_numbers) + len(hard_numbers) if conversion_type != 'timezone' else len(easy_times) + len(hard_times),
        'contexts': len(contexts),
        'context_free_prompts': context_free_prompts,
        'context_prompts': context_prompts,
        'total_prompts': total_prompts,
        'unit_pair_details': unit_pair_counts
    }

def main():
    conversions_dir = Path('conversions')
    numbers_file = Path('conversions/numbers.json')
    times_file = Path('conversions/times.json')
    
    # Get all JSON files (except numbers.json, times.json, and substances.json)
    json_files = [
        f for f in conversions_dir.glob('*.json')
        if f.name not in ['numbers.json', 'times.json', 'substances.json']
    ]
    
    # Load substances file
    substances_file = conversions_dir / 'substances.json'
    
    print("="*80)
    print("PROMPT COUNT BY DOMAIN")
    print("="*80)
    print()
    
    all_results = {}
    grand_total = 0
    
    for json_file in sorted(json_files):
        try:
            results = count_prompts_for_domain(json_file, numbers_file, times_file, substances_file)
            all_results.update(results)
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue
    
    # Print summary
    print(f"{'Domain':<40} {'Unit Pairs':<12} {'Test Values':<15} {'Contexts':<10} {'Context-Free':<15} {'With Context':<15} {'Total':<10}")
    print("-" * 120)
    
    for domain, info in sorted(all_results.items()):
        unit_pairs = info['unit_pairs']
        test_vals = info['test_values_total']
        contexts = info['contexts']
        cf_prompts = info['context_free_prompts']
        ctx_prompts = info['context_prompts']
        total = info['total_prompts']
        grand_total += total
        
        print(f"{domain:<40} {unit_pairs:<12} {test_vals:<15} {contexts:<10} {cf_prompts:<15} {ctx_prompts:<15} {total:<10}")
    
    print("-" * 120)
    print(f"{'GRAND TOTAL':<40} {'':<12} {'':<15} {'':<10} {'':<15} {'':<15} {grand_total:<10}")
    print()
    
    # Print detailed breakdown for each domain
    print("="*80)
    print("DETAILED BREAKDOWN BY DOMAIN")
    print("="*80)
    print()
    
    for domain, info in sorted(all_results.items()):
        print(f"\n{domain}:")
        print(f"  Unit pairs: {info['unit_pairs']}")
        print(f"  Test values: {info['test_values_total']} (Easy: {info['test_values_easy']}, Hard: {info['test_values_hard']})")
        print(f"  Contexts: {info['contexts']}")
        print(f"  Context-free prompts: {info['context_free_prompts']}")
        print(f"  Prompts with context: {info['context_prompts']}")
        print(f"  Total prompts: {info['total_prompts']}")
        
        if info['unit_pair_details']:
            print(f"  Breakdown by unit pair:")
            for pair_info in info['unit_pair_details']:
                print(f"    {pair_info['from']} → {pair_info['to']}: "
                      f"{pair_info['test_values']} test values × "
                      f"(1 context-free + {info['contexts']} contexts) = "
                      f"{pair_info['total']} prompts")

if __name__ == '__main__':
    main()
