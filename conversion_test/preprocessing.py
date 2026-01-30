#!/usr/bin/env python3
"""
Preprocessing script to convert JSON configuration files to TSV format.

This script:
1. Loads conversion JSON files and numbers.json
2. Generates prompts for each conversion
3. Calculates correct answers
4. Outputs TSV files with columns: domain, distractor, prompt, number, answer, difficulty
"""

import json
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import re
from decimal import Decimal, getcontext

# Set precision for decimal calculations
getcontext().prec = 28

# Conversion functions (simplified versions from convert.py)
def convert_linear(value: float, from_unit: str, to_unit: str, conversion_factors: Dict[str, float], base_unit: str) -> float:
    """Convert between linear units using conversion factors."""
    if from_unit == to_unit:
        return value
    
    # Convert to base unit
    if from_unit in conversion_factors:
        base_value = value / conversion_factors[from_unit]
    else:
        raise ValueError(f"Unknown unit: {from_unit}")
    
    # Convert from base unit to target
    if to_unit in conversion_factors:
        result = base_value * conversion_factors[to_unit]
    else:
        raise ValueError(f"Unknown unit: {to_unit}")
    
    return float(result)

def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between temperature units."""
    if from_unit == to_unit:
        return value
    
    # Convert to Celsius first
    if from_unit.lower() in ['celsius', 'c']:
        celsius = value
    elif from_unit.lower() in ['fahrenheit', 'f']:
        celsius = (value - 32) * 5 / 9
    elif from_unit.lower() in ['kelvin', 'k']:
        celsius = value - 273.15
    else:
        raise ValueError(f"Unknown temperature unit: {from_unit}")
    
    # Convert from Celsius to target
    if to_unit.lower() in ['celsius', 'c']:
        return celsius
    elif to_unit.lower() in ['fahrenheit', 'f']:
        return celsius * 9 / 5 + 32
    elif to_unit.lower() in ['kelvin', 'k']:
        return celsius + 273.15
    else:
        raise ValueError(f"Unknown temperature unit: {to_unit}")

def convert_timezone(value: str, from_unit: str, to_unit: str, timezone_offsets: Dict[str, float], city_to_tz: Dict[str, str]) -> str:
    """Convert time between timezones."""
    # Parse time string to hours (0-24)
    def parse_time_string(time_str: str) -> float:
        """Parse time string like '1AM' or '3:49PM' to hours (0-24)."""
        time_str = time_str.strip().upper()
        
        # Pattern 1: HH:MMAM/PM
        match = re.match(r'(\d{1,2}):(\d{2})(AM|PM)', time_str)
        if match:
            h = int(match.group(1))
            m = int(match.group(2))
            period = match.group(3)
            
            if period == 'PM' and h != 12:
                h += 12
            elif period == 'AM' and h == 12:
                h = 0
            
            return h + m / 60.0
        
        # Pattern 2: HHAM/PM
        match = re.match(r'(\d{1,2})(AM|PM)', time_str)
        if match:
            h = int(match.group(1))
            period = match.group(2)
            
            if period == 'PM' and h != 12:
                h += 12
            elif period == 'AM' and h == 12:
                h = 0
            
            return float(h)
        
        raise ValueError(f"Could not parse time string: {time_str}")
    
    def format_time_string(hours: float) -> str:
        """Format hours (0-24) to time string like '1AM' or '3:49PM'."""
        h = int(hours) % 24
        m = int((hours % 1) * 60)
        
        if m == 0:
            if h == 0:
                return "12AM"
            elif h < 12:
                return f"{h}AM"
            elif h == 12:
                return "12PM"
            else:
                return f"{h-12}PM"
        else:
            if h == 0:
                return f"12:{m:02d}AM"
            elif h < 12:
                return f"{h}:{m:02d}AM"
            elif h == 12:
                return f"12:{m:02d}PM"
            else:
                return f"{h-12}:{m:02d}PM"
    
    # Get timezone offsets
    from_tz = city_to_tz.get(from_unit, from_unit)
    to_tz = city_to_tz.get(to_unit, to_unit)
    
    from_offset = timezone_offsets.get(from_tz, 0.0)
    to_offset = timezone_offsets.get(to_tz, 0.0)
    
    # Parse time to hours
    hours = parse_time_string(value)
    
    # Convert to UTC first
    utc_hours = (hours - from_offset) % 24
    
    # Convert from UTC to target timezone
    target_hours = (utc_hours + to_offset) % 24
    
    return format_time_string(target_hours)

def convert_custom(value: Union[float, str], from_unit: str, to_unit: str, config: Dict) -> Union[float, str]:
    """Convert using custom conversion logic."""
    # Check for currency first
    exchange_rates = config.get('exchange_rates', {})
    if exchange_rates and from_unit in exchange_rates and to_unit in exchange_rates:
        # Convert to USD first, then to target
        usd_value = float(value) / exchange_rates[from_unit]
        result = usd_value * exchange_rates[to_unit]
        return float(result)
    
    # Check for size mappings (clothing sizes, etc.)
    size_mappings = config.get('size_mappings', {})
    if size_mappings:
        # Try direct mapping first
        mapping_key = f"{from_unit}_to_{to_unit}"
        if mapping_key in size_mappings:
            mapping = size_mappings[mapping_key]
            value_str = str(value)
            if value_str in mapping:
                return mapping[value_str]
            # Try case-insensitive match
            for k, v in mapping.items():
                if k.upper() == value_str.upper():
                    return v
        
        # Try reverse lookup (find mapping where value exists and source matches)
        for key, mapping in size_mappings.items():
            if key.endswith(f"_to_{to_unit}"):
                source_unit = key.replace(f"_to_{to_unit}", "")
                if source_unit == from_unit:
                    value_str = str(value)
                    if value_str in mapping:
                        return mapping[value_str]
                    # Try case-insensitive
                    for k, v in mapping.items():
                        if k.upper() == value_str.upper():
                            return v
        
        # Try multi-step conversion (e.g., inches -> cm)
        # Check if we can go through an intermediate unit
        # For now, raise error if no direct mapping found
        raise ValueError(f"No mapping found for {from_unit} to {to_unit}")
    
    # Fall back to linear conversion
    conversion_factors = config.get('conversion_factors', {})
    base_unit = config.get('base_unit')
    if conversion_factors and base_unit:
        return convert_linear(float(value), from_unit, to_unit, conversion_factors, base_unit)
    else:
        raise ValueError(f"Custom conversion not implemented for {from_unit} to {to_unit}")

def calculate_conversion(
    value: Union[float, str],
    from_unit: str,
    to_unit: str,
    conversion_type: str,
    config: Dict
) -> Union[float, str]:
    """Calculate conversion result based on conversion type."""
    if conversion_type == 'linear':
        conversion_factors = config.get('conversion_factors', {})
        base_unit = config.get('base_unit')
        if not conversion_factors or not base_unit:
            raise ValueError("Linear conversion requires conversion_factors and base_unit")
        return convert_linear(float(value), from_unit, to_unit, conversion_factors, base_unit)
    
    elif conversion_type == 'temperature':
        return convert_temperature(float(value), from_unit, to_unit)
    
    elif conversion_type == 'timezone':
        timezone_offsets = config.get('timezone_offsets', {})
        city_to_tz = config.get('city_to_timezone', {})
        if not timezone_offsets:
            raise ValueError("Timezone conversion requires timezone_offsets")
        return convert_timezone(str(value), from_unit, to_unit, timezone_offsets, city_to_tz)
    
    elif conversion_type == 'custom':
        return convert_custom(value, from_unit, to_unit, config)
    
    else:
        raise ValueError(f"Unknown conversion type: {conversion_type}")

def create_prompt(
    value: Union[float, str],
    from_unit: str,
    to_unit: str,
    conversion_type: str,
    config: Dict,
    context: Optional[str] = None
) -> str:
    """Create prompt for conversion."""
    display_names = config.get('display_names', {})
    from_display = display_names.get(from_unit, from_unit)
    to_display = display_names.get(to_unit, to_unit)
    
    # Special handling for clothing size conversions
    # Check if this is a clothing size conversion by looking for size_mappings
    is_clothing_size = bool(config.get('size_mappings'))
    
    if is_clothing_size:
        # Get sizing type from config (e.g., "clothing", "shoe", "bra", "pant")
        sizing_type = config.get('sizing_type', 'clothing')
        
        # Map section names to readable sizing types
        sizing_type_map = {
            'clothing_size': 'clothing',
            'pant_size': 'pant',
            'shoe_size': 'shoe',
            'bra_size': 'bra'
        }
        
        # If sizing_type is a section name, map it
        if sizing_type in sizing_type_map:
            sizing_type = sizing_type_map[sizing_type]
        
        # For clothing sizes, use a more natural syntax with sizing type
        # e.g., "Convert clothing size XS in US to UK" or "Convert shoe size 8 in US to EU"
        return f"Convert {sizing_type} size {value} in {from_display} to {to_display} sizing. Provide only the size value."
    
    # Handle pluralization for some units
    if conversion_type not in ['temperature', 'timezone', 'custom']:
        if isinstance(value, (int, float)) and value != 1:
            if not from_display.endswith('s') and not from_display.endswith('es'):
                from_display = from_display + 's'
            if not to_display.endswith('s') and not to_display.endswith('es'):
                to_display = to_display + 's'
    
    # Special handling for timezone conversions
    if conversion_type == 'timezone':
        return f"Convert {value} in {from_display} time to {to_display} time. Assume you are thinking about standard time, not daylight savings. Provide the time in the same format (e.g., 1AM, 3:49PM)."
    
    # Handle context (distractor)
    if context:
        return f"Convert {value} {from_display} of {context} to {to_display}. Provide only the numerical value."
    else:
        return f"Convert {value} {from_display} to {to_display}. Provide only the numerical value."

def process_conversion_file(
    json_file: Path,
    numbers_file: Path,
    times_file: Optional[Path],
    output_dir: Path,
    substances_file: Optional[Path] = None
) -> None:
    """Process a single conversion JSON file and create TSV output."""
    print(f"\nProcessing {json_file.name}...")
    
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
    # density, volume, and moles_to_particles should use contexts
    use_substances_contexts = conversion_name in ['density', 'volume', 'moles_to_particles']
    
    # Check if this is a multi-section config (like clothing_sizes)
    # Look for known sub-sections
    sub_sections = ['clothing_size', 'pant_size', 'shoe_size', 'bra_size']
    has_sub_sections = any(section in config for section in sub_sections)
    
    rows = []
    
    if has_sub_sections:
        # Process each sub-section separately and create separate output files
        for section_name in sub_sections:
            if section_name not in config:
                continue
            
            print(f"  Processing {section_name}...")
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
            
            # Process this sub-section (pass full config for currency difficulty checking)
            section_rows = process_section(
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
            
            # Create DataFrame for this section
            if section_rows:
                df = pd.DataFrame(section_rows)
                df = df[['domain', 'distractor', 'prompt', 'number', 'answer', 'difficulty']]
                
                # Output TSV file for this section
                domain_name = f"{conversion_name}_{section_name}"
                output_file = output_dir / f"{domain_name}.tsv"
                df.to_csv(output_file, sep='\t', index=False)
                print(f"    Created {len(df)} rows")
                print(f"    Saved to {output_file}")
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
        
        section_rows = process_section(
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
        
        # Create DataFrame
        if section_rows:
            df = pd.DataFrame(section_rows)
            df = df[['domain', 'distractor', 'prompt', 'number', 'answer', 'difficulty']]
            
            # Output TSV file
            output_file = output_dir / f"{conversion_name}.tsv"
            df.to_csv(output_file, sep='\t', index=False)
            
            print(f"  Created {len(df)} rows")
            print(f"  Saved to {output_file}")

def process_section(
    section_config: Dict,
    base_conversion_name: str,
    section_name: Optional[str],
    conversion_type: str,
    easy_numbers: List,
    hard_numbers: List,
    easy_times: List,
    hard_times: List,
    full_config: Optional[Dict] = None
) -> List[Dict]:
    """Process a single section (or main config) and return rows."""
    unit_pairs = section_config.get('unit_pairs', [])
    contexts = section_config.get('contexts', [])
    display_names = section_config.get('display_names', {})
    
    # Create domain name
    if section_name:
        domain_name = f"{base_conversion_name}_{section_name}"
    else:
        domain_name = base_conversion_name
    
    rows = []
    
    # Check if this is a currency conversion with easy/hard currencies
    is_currency = (base_conversion_name == 'currency' or conversion_type == 'currency')
    easy_currencies = []
    hard_currencies = []
    if is_currency and full_config:
        easy_currencies = full_config.get('easy_currencies', [])
        hard_currencies = full_config.get('hard_currencies', [])
    
    # Process each unit pair
    for pair in unit_pairs:
        from_unit = pair['from']
        to_unit = pair['to']
        
        # For currency conversions, use all 200 numbers (easy + hard) for all unit pairs
        if is_currency:
            # Use all numbers (easy + hard) for all currency pairs
            test_values = easy_numbers + hard_numbers
        # For clothing sizes, we need to get valid test values from the mappings
        elif section_config.get('size_mappings'):
            size_mappings = section_config.get('size_mappings', {})
            # Get valid input values from size mappings
            # Find a mapping that has from_unit as source
            mapping_key = None
            for key in size_mappings.keys():
                if key.startswith(f"{from_unit}_to_"):
                    mapping_key = key
                    break
            
            if mapping_key:
                mapping = size_mappings[mapping_key]
                # Get all keys from the mapping as test values
                test_values = list(mapping.keys())
            else:
                # Try reverse lookup or use numbers
                test_values = easy_numbers[:10] + hard_numbers[:10]  # Limit for clothing sizes
        elif conversion_type == 'timezone':
            # For timezone, use time strings from times.json
            if easy_times or hard_times:
                test_values = easy_times + hard_times
            else:
                # Fallback: generate from numbers if times.json not provided
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
                        test_values.append(str(num))
        else:
            test_values = easy_numbers + hard_numbers
        
        # Process each test value
        for test_value in test_values:
            # Determine difficulty
            if conversion_type == 'timezone':
                # Check if test_value is in easy_times or hard_times
                if easy_times and test_value in easy_times:
                    difficulty = 'Easy'
                elif hard_times and test_value in hard_times:
                    difficulty = 'Hard'
                else:
                    # Fallback: determine from format
                    if isinstance(test_value, str) and ('AM' in test_value or 'PM' in test_value):
                        # If it has minutes (contains ':'), it's likely hard
                        if ':' in test_value:
                            difficulty = 'Hard'
                        else:
                            difficulty = 'Easy'
                    else:
                        difficulty = 'Easy'
            elif section_config.get('size_mappings'):
                # For clothing sizes, use Easy by default (can be adjusted)
                difficulty = 'Easy'
            elif is_currency:
                # For currency, difficulty is determined by which numbers were used
                if test_value in easy_numbers:
                    difficulty = 'Easy'
                elif test_value in hard_numbers:
                    difficulty = 'Hard'
                else:
                    difficulty = 'Easy'  # Default
            else:
                if test_value in easy_numbers:
                    difficulty = 'Easy'
                elif test_value in hard_numbers:
                    difficulty = 'Hard'
                else:
                    difficulty = 'Easy'
            
            # Calculate correct answer
            # For clothing sizes, we need to pass the section_config with size_mappings
            try:
                if section_config.get('size_mappings'):
                    # Create a config dict with size_mappings for the conversion function
                    temp_config = section_config.copy()
                    temp_config['conversion_type'] = conversion_type
                    answer = calculate_conversion(test_value, from_unit, to_unit, conversion_type, temp_config)
                else:
                    # Use the section config (or full config for currency exchange rates)
                    if is_currency and full_config:
                        temp_config = full_config.copy()
                    else:
                        temp_config = section_config.copy()
                    temp_config['conversion_type'] = conversion_type
                    answer = calculate_conversion(test_value, from_unit, to_unit, conversion_type, temp_config)
            except Exception as e:
                print(f"  Warning: Could not calculate {from_unit} -> {to_unit} for {test_value}: {e}")
                continue
            
            # Create context-free prompt
            temp_config_for_prompt = section_config.copy()
            temp_config_for_prompt['display_names'] = display_names
            # Pass size_mappings and sizing_type to prompt creation for clothing sizes
            if section_config.get('size_mappings'):
                temp_config_for_prompt['size_mappings'] = section_config.get('size_mappings')
                # Pass section_name as sizing_type for clothing size prompts
                if section_name:
                    temp_config_for_prompt['sizing_type'] = section_name
            prompt = create_prompt(test_value, from_unit, to_unit, conversion_type, temp_config_for_prompt, context=None)
            
            rows.append({
                'domain': domain_name,
                'distractor': None,
                'prompt': prompt,
                'number': test_value,
                'answer': answer,
                'difficulty': difficulty
            })
            
            # Create prompts with contexts if available
            for context in contexts:
                # For clothing sizes, contexts typically don't apply, but handle if needed
                temp_config_for_context = temp_config_for_prompt.copy()
                if section_config.get('size_mappings'):
                    temp_config_for_context['size_mappings'] = section_config.get('size_mappings')
                    # Pass section_name as sizing_type for clothing size prompts
                    if section_name:
                        temp_config_for_context['sizing_type'] = section_name
                prompt_with_context = create_prompt(test_value, from_unit, to_unit, conversion_type, temp_config_for_context, context=context)
                
                rows.append({
                    'domain': domain_name,
                    'distractor': context,
                    'prompt': prompt_with_context,
                    'number': test_value,
                    'answer': answer,
                    'difficulty': difficulty
                })
    
    return rows

def main():
    parser = argparse.ArgumentParser(description='Preprocess conversion JSON files to TSV format')
    parser.add_argument('--conversions-dir', type=str, default='conversions',
                       help='Directory containing conversion JSON files')
    parser.add_argument('--numbers-file', type=str, default='conversions/numbers.json',
                       help='JSON file containing easy and hard numbers')
    parser.add_argument('--times-file', type=str, default='conversions/times.json',
                       help='JSON file containing easy and hard times for timezone conversions')
    parser.add_argument('--output-dir', type=str, default='preprocessed',
                       help='Output directory for TSV files')
    parser.add_argument('--files', type=str, nargs='+', default=None,
                       help='Specific JSON files to process (default: all in conversions directory)')
    
    args = parser.parse_args()
    
    conversions_dir = Path(args.conversions_dir)
    numbers_file = Path(args.numbers_file)
    times_file = Path(args.times_file) if args.times_file else None
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify numbers file exists
    if not numbers_file.exists():
        raise FileNotFoundError(f"Numbers file not found: {numbers_file}")
    
    # Times file is optional (only needed for timezone conversions)
    if times_file and not times_file.exists():
        print(f"Warning: Times file not found: {times_file}. Timezone conversions will use fallback method.")
        times_file = None
    
    # Get list of JSON files to process
    if args.files:
        json_files = [Path(f) for f in args.files]
    else:
        # Process all JSON files in conversions directory (except numbers.json)
        json_files = [f for f in conversions_dir.glob('*.json') if f.name != 'numbers.json']
    
    print(f"Processing {len(json_files)} conversion file(s)...")
    
    # Process each file
    for json_file in json_files:
        if not json_file.exists():
            print(f"Warning: File not found: {json_file}, skipping...")
            continue
        
        try:
            process_conversion_file(json_file, numbers_file, times_file, output_dir)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nPreprocessing complete! Output files in {output_dir}")

if __name__ == '__main__':
    main()
