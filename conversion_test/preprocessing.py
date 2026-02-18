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
        total_minutes = round(hours * 60)       # avoid float truncation
        h = (total_minutes // 60) % 24
        m = total_minutes % 60
        
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
        
        # Try multi-step conversion (e.g., US_alphanumeric -> US_numeric -> EU_numeric)
        # Find an intermediate unit that connects from_unit to to_unit
        intermediate_found = False
        for intermediate_key, intermediate_mapping in size_mappings.items():
            # Check if this mapping starts with from_unit
            if intermediate_key.startswith(f"{from_unit}_to_"):
                intermediate_unit = intermediate_key.replace(f"{from_unit}_to_", "")
                # Check if there's a mapping from intermediate_unit to to_unit
                final_key = f"{intermediate_unit}_to_{to_unit}"
                if final_key in size_mappings:
                    # Two-step conversion: from_unit -> intermediate_unit -> to_unit
                    intermediate_mapping_dict = intermediate_mapping
                    final_mapping_dict = size_mappings[final_key]
                    
                    value_str = str(value)
                    # First step: from_unit -> intermediate_unit
                    if value_str in intermediate_mapping_dict:
                        intermediate_value = intermediate_mapping_dict[value_str]
                    else:
                        # Try case-insensitive
                        for k, v in intermediate_mapping_dict.items():
                            if k.upper() == value_str.upper():
                                intermediate_value = v
                                break
                        else:
                            continue  # Try next intermediate
                    
                    # Second step: intermediate_unit -> to_unit
                    intermediate_str = str(intermediate_value)
                    if intermediate_str in final_mapping_dict:
                        return final_mapping_dict[intermediate_str]
                    # Try case-insensitive
                    for k, v in final_mapping_dict.items():
                        if k.upper() == intermediate_str.upper():
                            return v
                    intermediate_found = True
        
        if not intermediate_found:
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

def get_conversion_factor(
    from_unit: str,
    to_unit: str,
    conversion_type: str,
    config: Dict,
    full_config: Optional[Dict] = None
) -> Optional[float]:
    """Calculate the direct conversion factor for a unit pair.
    
    Returns the factor to multiply the value by to get the result.
    Returns None if this conversion type doesn't support direct factors.
    
    Examples:
    - mL to tbsp: returns 0.067628045 (5 * 0.067628045 = 0.338...)
    - USD to EUR: returns 0.85 (5 * 0.85 = 4.25)
    """
    if from_unit == to_unit:
        return 1.0
    
    if conversion_type == 'linear':
        conversion_factors = config.get('conversion_factors', {})
        if from_unit in conversion_factors and to_unit in conversion_factors:
            # Factor = to_factor / from_factor
            # This gives us: value * (to_factor / from_factor) = result
            return conversion_factors[to_unit] / conversion_factors[from_unit]
    
    elif conversion_type == 'custom':
        # Check for currency (exchange rates)
        exchange_rates = None
        if full_config and full_config.get('exchange_rates'):
            exchange_rates = full_config.get('exchange_rates', {})
        elif config.get('exchange_rates'):
            exchange_rates = config.get('exchange_rates', {})
        
        if exchange_rates and from_unit in exchange_rates and to_unit in exchange_rates:
            # Factor = to_rate / from_rate
            # This gives us: value * (to_rate / from_rate) = result
            return exchange_rates[to_unit] / exchange_rates[from_unit]
    
    # Temperature, timezone, and clothing sizes don't support direct factors
    return None

def get_math_expression(
    value: Union[float, str],
    from_unit: str,
    to_unit: str,
    conversion_type: str,
    config: Dict,
    full_config: Optional[Dict] = None
) -> Optional[str]:
    """Generate a mathematical expression for the conversion.
    
    Returns a string like "(5*9/5)+32" for temperature or "(14.5+3)%24" for timezone.
    Returns None if this conversion type doesn't support math expressions.
    """
    if from_unit == to_unit:
        return str(value)
    
    if conversion_type == 'temperature':
        from_unit_lower = from_unit.lower()
        to_unit_lower = to_unit.lower()
        
        # Convert to Celsius first, then to target
        if from_unit_lower in ['celsius', 'c']:
            celsius_expr = str(value)
        elif from_unit_lower in ['fahrenheit', 'f']:
            celsius_expr = f"({value}-32)*5/9"
        elif from_unit_lower in ['kelvin', 'k']:
            celsius_expr = f"{value}-273.15"
        else:
            return None
        
        # Convert from Celsius to target
        if to_unit_lower in ['celsius', 'c']:
            return celsius_expr
        elif to_unit_lower in ['fahrenheit', 'f']:
            return f"({celsius_expr}*9/5)+32"
        elif to_unit_lower in ['kelvin', 'k']:
            return f"{celsius_expr}+273.15"
        else:
            return None
    
    elif conversion_type == 'timezone':
        # Parse time string to hours
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
        
        # Get timezone offsets
        timezone_offsets = config.get('timezone_offsets', {})
        city_to_tz = config.get('city_to_timezone', {})
        
        from_tz = city_to_tz.get(from_unit, from_unit)
        to_tz = city_to_tz.get(to_unit, to_unit)
        
        from_offset = timezone_offsets.get(from_tz, 0.0)
        to_offset = timezone_offsets.get(to_tz, 0.0)
        
        # Parse time to hours
        if isinstance(value, str):
            hours = parse_time_string(value)
        else:
            hours = float(value)
        
        # Calculate offset difference
        offset_diff = to_offset - from_offset
        
        # Create expression: (hours + offset_diff) % 24
        # Format hours to avoid unnecessary decimals
        if hours == int(hours):
            hours_str = str(int(hours))
        else:
            hours_str = str(hours)
        
        if offset_diff == 0:
            return f"{hours_str}%24"
        elif offset_diff > 0:
            return f"({hours_str}+{offset_diff})%24"
        else:
            # offset_diff is already negative, so this produces (hours-offset) correctly
            return f"({hours_str}{offset_diff})%24"
    
    return None

def create_conversion_guide(
    from_unit: str,
    to_unit: str,
    conversion_type: str,
    config: Dict,
    full_config: Optional[Dict] = None
) -> str:
    """Create a conversion guide to include in the prompt."""
    guide_parts = []
    
    if conversion_type == 'currency' or (conversion_type == 'custom' and config.get('exchange_rates')):
        # Include exchange rates (check full_config first for currency)
        exchange_rates = None
        if full_config and full_config.get('exchange_rates'):
            exchange_rates = full_config.get('exchange_rates', {})
        elif config.get('exchange_rates'):
            exchange_rates = config.get('exchange_rates', {})
        
        if exchange_rates:
            guide_parts.append("Exchange rates (relative to USD):")
            # Sort for readability
            sorted_rates = sorted(exchange_rates.items(), key=lambda x: x[1])
            for currency, rate in sorted_rates:
                guide_parts.append(f"  {currency}: {rate}")
    
    elif conversion_type == 'linear':
        # Include conversion factors
        conversion_factors = config.get('conversion_factors', {})
        base_unit = config.get('base_unit', '')
        if conversion_factors:
            guide_parts.append(f"Conversion factors (relative to {base_unit}):")
            for unit, factor in sorted(conversion_factors.items()):
                guide_parts.append(f"  {unit}: {factor}")
    
    elif conversion_type == 'temperature':
        # Include temperature conversion formulas
        guide_parts.append("Temperature conversion formulas:")
        guide_parts.append("  Celsius to Fahrenheit: F = (C × 9/5) + 32")
        guide_parts.append("  Fahrenheit to Celsius: C = (F - 32) × 5/9")
        guide_parts.append("  Celsius to Kelvin: K = C + 273.15")
        guide_parts.append("  Kelvin to Celsius: C = K - 273.15")
        guide_parts.append("  Fahrenheit to Kelvin: K = (F - 32) × 5/9 + 273.15")
        guide_parts.append("  Kelvin to Fahrenheit: F = (K - 273.15) × 9/5 + 32")
    
    elif conversion_type == 'timezone':
        # Include city timezone offsets relative to GMT
        city_to_timezone = config.get('city_to_timezone', {})
        timezone_offsets = config.get('timezone_offsets', {})
        
        if city_to_timezone and timezone_offsets:
            guide_parts.append("City timezones (relative to GMT):")
            # Create list of (city, offset) tuples
            city_offsets = []
            for city, tz in city_to_timezone.items():
                if tz in timezone_offsets:
                    offset = timezone_offsets[tz]
                    city_offsets.append((city, offset))
            
            # Sort by offset for readability
            city_offsets.sort(key=lambda x: x[1])
            
            # Format as "City: GMT+offset" or "City: GMT-offset"
            for city, offset in city_offsets:
                if offset >= 0:
                    guide_parts.append(f"  {city}: GMT+{offset:.1f}")
                else:
                    guide_parts.append(f"  {city}: GMT{offset:.1f}")
    
    elif config.get('size_mappings'):
        # Include relevant size mappings for clothing sizes
        size_mappings = config.get('size_mappings', {})
        display_names = config.get('display_names', {})
        
        # Find the exact mapping for this conversion (from_unit -> to_unit)
        exact_mapping_key = f"{from_unit}_to_{to_unit}"
        mapping_key = None
        is_multi_step = False
        
        # First try exact match
        if exact_mapping_key in size_mappings:
            mapping_key = exact_mapping_key
        else:
            # Try to find a multi-step path (e.g., US_alphanumeric -> US_numeric -> EU_numeric)
            # Check if we can go through an intermediate unit
            for intermediate_key, intermediate_mapping in size_mappings.items():
                if intermediate_key.startswith(f"{from_unit}_to_"):
                    intermediate_unit = intermediate_key.replace(f"{from_unit}_to_", "")
                    final_key = f"{intermediate_unit}_to_{to_unit}"
                    if final_key in size_mappings:
                        # Multi-step conversion found
                        # We'll show both mappings
                        is_multi_step = True
                        mapping_key = intermediate_key  # Use first step for now
                        break
            
            if not is_multi_step:
                # Fallback: find any mapping that starts with from_unit
                # (for cases where we might need multi-step, but show what we have)
                for key in size_mappings.keys():
                    if key.startswith(f"{from_unit}_to_"):
                        mapping_key = key
                        break
        
        if mapping_key and mapping_key in size_mappings:
            mapping = size_mappings[mapping_key]
            from_display = display_names.get(from_unit, from_unit)
            to_display = display_names.get(to_unit, to_unit)
            
            if is_multi_step:
                # For multi-step, show both mappings
                intermediate_unit = mapping_key.replace(f"{from_unit}_to_", "")
                final_key = f"{intermediate_unit}_to_{to_unit}"
                final_mapping = size_mappings[final_key]
                intermediate_display = display_names.get(intermediate_unit, intermediate_unit)
                
                guide_parts.append(f"Size conversion table ({from_display} to {to_display}):")
                guide_parts.append(f"  (Two-step conversion: {from_display} → {intermediate_display} → {to_display})")
                
                # Show combined mapping by going through intermediate
                combined_mapping = {}
                for key, intermediate_val in mapping.items():
                    if str(intermediate_val) in final_mapping:
                        combined_mapping[key] = final_mapping[str(intermediate_val)]
                    # Try case-insensitive match
                    else:
                        for k, v in final_mapping.items():
                            if str(k).upper() == str(intermediate_val).upper():
                                combined_mapping[key] = v
                                break
                
                if combined_mapping:
                    # Show all mappings, sorted if numeric
                    try:
                        sorted_items = sorted(combined_mapping.items(), key=lambda x: float(x[0]) if str(x[0]).replace('.', '').isdigit() else x[0])
                    except:
                        sorted_items = sorted(combined_mapping.items())
                    
                    for key, val in sorted_items:
                        guide_parts.append(f"  {key} → {val}")
                else:
                    # Fallback: show first step mapping
                    guide_parts.append(f"  Step 1: {from_display} → {intermediate_display}")
                    try:
                        sorted_items = sorted(mapping.items(), key=lambda x: float(x[0]) if str(x[0]).replace('.', '').isdigit() else x[0])
                    except:
                        sorted_items = sorted(mapping.items())
                    for key, val in sorted_items[:5]:  # Show first 5
                        guide_parts.append(f"    {key} → {val}")
                    guide_parts.append(f"  Step 2: {intermediate_display} → {to_display}")
                    try:
                        sorted_items = sorted(final_mapping.items(), key=lambda x: float(x[0]) if str(x[0]).replace('.', '').isdigit() else x[0])
                    except:
                        sorted_items = sorted(final_mapping.items())
                    for key, val in sorted_items[:5]:  # Show first 5
                        guide_parts.append(f"    {key} → {val}")
            else:
                guide_parts.append(f"Size conversion table ({from_display} to {to_display}):")
                # Show all mappings, sorted if numeric
                try:
                    # Try to sort numerically if possible
                    sorted_items = sorted(mapping.items(), key=lambda x: float(x[0]) if str(x[0]).replace('.', '').isdigit() else x[0])
                except:
                    sorted_items = sorted(mapping.items())
                
                for key, val in sorted_items:
                    guide_parts.append(f"  {key} → {val}")
    
    if guide_parts:
        return "\n".join(guide_parts)
    return ""

def create_prompt(
    value: Union[float, str],
    from_unit: str,
    to_unit: str,
    conversion_type: str,
    config: Dict,
    context: Optional[str] = None,
    full_config: Optional[Dict] = None,
    include_guide: bool = True,
    math_only: bool = False
) -> str:
    """Create prompt for conversion.
    
    If math_only=True, creates a pure mathematical prompt like "what is 5*0.067628"
    for linear/currency conversions, or "what is (5*9/5)+32" for temperature,
    or "what is (14.5+3)%24" for timezone.
    """
    # Handle math-only mode
    if math_only:
        # Try direct factor first (for linear and currency)
        factor = get_conversion_factor(from_unit, to_unit, conversion_type, config, full_config)
        if factor is not None:
            # Format the factor to a reasonable precision
            # Use enough precision to be accurate but not excessive
            if abs(factor) < 0.001 or abs(factor) > 1000:
                # Use scientific notation for very small/large numbers
                factor_str = f"{factor:.6e}"
            else:
                # Use regular decimal notation
                factor_str = f"{factor:.10f}".rstrip('0').rstrip('.')
            
            return f"what is {value}*{factor_str}"
        
        # Try math expression (for temperature and timezone)
        math_expr = get_math_expression(value, from_unit, to_unit, conversion_type, config, full_config)
        if math_expr is not None:
            return f"what is {math_expr}"
        
        # This conversion type doesn't support math-only mode
        # This should have been caught in process_section, but handle gracefully
        raise ValueError(f"Math-only mode not supported for conversion type '{conversion_type}' from {from_unit} to {to_unit}")
    
    # Regular prompt creation (existing logic)
    display_names = config.get('display_names', {})
    from_display = display_names.get(from_unit, from_unit)
    to_display = display_names.get(to_unit, to_unit)
    
    # Create conversion guide (only if include_guide is True)
    conversion_guide = None
    if include_guide:
        conversion_guide = create_conversion_guide(from_unit, to_unit, conversion_type, config, full_config)
    
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
        
        # Get gender information from config name or explicit gender field
        # Check if base conversion name indicates gender
        base_name = config.get('base_conversion_name', '')
        if 'women' in base_name.lower() or 'men' in base_name.lower():
            if 'women' in base_name.lower():
                gender_text = "women's"
            else:
                gender_text = "men's"
        else:
            # Check explicit gender field
            gender = config.get('gender', None)
            if gender:
                gender_text = "men's" if gender == "men" else "women's"
            else:
                gender_text = None
        
        # For clothing sizes, use a more natural syntax with sizing type and gender
        # e.g., "Convert women's clothing size XS in US to UK" or "Convert men's shoe size 8 in US to EU"
        base_prompt = ""
        if gender_text:
            base_prompt = f"Convert {gender_text} {sizing_type} size {value} in {from_display} to {to_display} sizing."
        else:
            base_prompt = f"Convert {sizing_type} size {value} in {from_display} to {to_display} sizing."
        
        if conversion_guide:
            return f"{base_prompt}\n\nConversion guide:\n{conversion_guide}\n\nProvide only the size and nothing else."
        else:
            return f"{base_prompt} Provide only the size and nothing else."
    
    # Handle pluralization for some units
    if conversion_type not in ['temperature', 'timezone', 'custom']:
        if isinstance(value, (int, float)) and value != 1:
            if not from_display.endswith('s') and not from_display.endswith('es'):
                from_display = from_display + 's'
            if not to_display.endswith('s') and not to_display.endswith('es'):
                to_display = to_display + 's'
    
    # Special handling for timezone conversions
    if conversion_type == 'timezone':
        base_prompt = f"Convert {value} in {from_display} time to {to_display} time. Assume you are thinking about standard time, not daylight savings."
        if conversion_guide:
            return f"{base_prompt}\n\nConversion guide:\n{conversion_guide}\n\nProvide the time in the same format (e.g., 1AM, 3:49PM)."
        else:
            return f"{base_prompt} Provide the time in the same format (e.g., 1AM, 3:49PM)."
    
    # Handle context (distractor)
    base_prompt = ""
    if context:
        base_prompt = f"Convert {value} {from_display} of {context} to {to_display}."
    else:
        base_prompt = f"Convert {value} {from_display} to {to_display}."
    
    if conversion_guide:
        return f"{base_prompt}\n\nConversion guide:\n{conversion_guide}\n\nProvide only the numerical value."
    else:
        return f"{base_prompt} Provide only the numerical value."

def process_conversion_file(
    json_file: Path,
    numbers_file: Path,
    times_file: Optional[Path],
    output_dir: Path,
    substances_file: Optional[Path] = None,
    include_guide: bool = True,
    math_only: bool = False
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
    
    # Use the JSON "name" field if it exists, otherwise use the filename stem
    # This handles cases where the filename differs from the expected domain name
    # (e.g., moles.json has "name": "moles_to_particles")
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
                full_config=config,
                include_guide=include_guide,
                math_only=math_only
            )
            
            # Group rows by domain (to handle gender-specific domains)
            if section_rows:
                df = pd.DataFrame(section_rows)
                df = df[['domain', 'distractor', 'prompt', 'number', 'answer', 'difficulty']]
                
                # Output separate files for each domain (e.g., clothing_size_men, clothing_size_women)
                for domain in df['domain'].unique():
                    domain_df = df[df['domain'] == domain]
                    # Replace None/NaN with "null" for distractor column
                    domain_df['distractor'] = domain_df['distractor'].fillna('null')
                    # Add suffix based on flags
                    suffix_parts = []
                    if math_only:
                        suffix_parts.append("_math_only")
                    if not include_guide:
                        suffix_parts.append("_no_guide")
                    suffix = "".join(suffix_parts)
                    output_file = output_dir / f"{domain}{suffix}.tsv"
                    domain_df.to_csv(output_file, sep='\t', index=False, na_rep='null')
                    print(f"    Created {len(domain_df)} rows for {domain}{suffix}")
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
            full_config=config,
            include_guide=include_guide,
            math_only=math_only
        )
        
        # Create DataFrame
        if section_rows:
            df = pd.DataFrame(section_rows)
            df = df[['domain', 'distractor', 'prompt', 'number', 'answer', 'difficulty']]
            
            # Group rows by domain (to handle gender-specific domains)
            for domain in df['domain'].unique():
                domain_df = df[df['domain'] == domain]
                # Replace None/NaN with "null" for distractor column
                domain_df['distractor'] = domain_df['distractor'].fillna('null')
                # Add suffix based on flags
                suffix_parts = []
                if math_only:
                    suffix_parts.append("_math_only")
                if not include_guide:
                    suffix_parts.append("_no_guide")
                suffix = "".join(suffix_parts)
                output_file = output_dir / f"{domain}{suffix}.tsv"
                domain_df.to_csv(output_file, sep='\t', index=False, na_rep='null')
                print(f"  Created {len(domain_df)} rows for {domain}{suffix}")
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
    full_config: Optional[Dict] = None,
    include_guide: bool = True,
    math_only: bool = False
) -> List[Dict]:
    """Process a single section (or main config) and return rows."""
    unit_pairs = section_config.get('unit_pairs', [])
    contexts = section_config.get('contexts', [])
    display_names = section_config.get('display_names', {})
    
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
        
        # Create domain name
        # Gender is now embedded in base_conversion_name (e.g., clothing_sizes_women, clothing_sizes_men)
        if section_name:
            domain_name = f"{base_conversion_name}_{section_name}"
        else:
            domain_name = base_conversion_name
        
        # For currency conversions, use all 200 numbers (easy + hard) for all unit pairs
        if is_currency:
            # Use all numbers (easy + hard) for all currency pairs
            test_values = easy_numbers + hard_numbers
        # For clothing sizes, we need to get valid test values from the mappings
        elif section_config.get('size_mappings'):
            size_mappings = section_config.get('size_mappings', {})
            # Get valid input values from size mappings
            # Find a mapping that has from_unit as source and to_unit as target
            mapping_key = None
            # First, try exact match
            exact_key = f"{from_unit}_to_{to_unit}"
            if exact_key in size_mappings:
                mapping_key = exact_key
            else:
                # Otherwise, find any mapping that has from_unit as source
                for key in size_mappings.keys():
                    if key.startswith(f"{from_unit}_to_"):
                        mapping_key = key
                        break
            
            if mapping_key:
                mapping = size_mappings[mapping_key]
                # Get all keys from the mapping as test values
                test_values = list(mapping.keys())
            else:
                # Try to find a multi-step path or reverse lookup
                # For multi-step: check if we can go through an intermediate unit
                found_multi_step = False
                for intermediate_key, intermediate_mapping in size_mappings.items():
                    if intermediate_key.startswith(f"{from_unit}_to_"):
                        intermediate_unit = intermediate_key.replace(f"{from_unit}_to_", "")
                        final_key = f"{intermediate_unit}_to_{to_unit}"
                        if final_key in size_mappings:
                            # Use values from the first step mapping
                            test_values = list(intermediate_mapping.keys())
                            found_multi_step = True
                            break
                
                if not found_multi_step:
                    # Try reverse lookup: find any mapping that ends with to_unit
                    # This handles cases where we have UK_to_EUR but need US_to_EUR
                    # We need to find a way to convert from from_unit to the source of that mapping
                    for key, mapping in size_mappings.items():
                        if key.endswith(f"_to_{to_unit}"):
                            source_unit = key.replace(f"_to_{to_unit}", "")
                            # Check if we can convert from_unit -> source_unit
                            # If from_unit == source_unit, we can use these values directly
                            if from_unit == source_unit:
                                test_values = list(mapping.keys())
                                found_multi_step = True
                                break
                            # Otherwise, check if there's a mapping from_unit -> source_unit
                            conversion_key = f"{from_unit}_to_{source_unit}"
                            if conversion_key in size_mappings:
                                # We can convert from_unit -> source_unit -> to_unit
                                # Use values from the from_unit -> source_unit mapping
                                conversion_mapping = size_mappings[conversion_key]
                                test_values = list(conversion_mapping.keys())
                                found_multi_step = True
                                break
                
                if not found_multi_step:
                    # For clothing sizes, we should NEVER use numeric fallback
                    # Instead, collect all available size values from any mapping
                    # that could potentially work (any mapping that has from_unit as source)
                    all_size_values = set()
                    for key, mapping in size_mappings.items():
                        if key.startswith(f"{from_unit}_to_"):
                            all_size_values.update(mapping.keys())
                    
                    if all_size_values:
                        # Use all available size values from mappings starting with from_unit
                        test_values = list(all_size_values)
                        print(f"  Warning: No direct or multi-step mapping found for {from_unit} -> {to_unit}, using size values from {from_unit} mappings: {len(test_values)} values")
                    else:
                        # Last resort: use any size values from any mapping
                        for key, mapping in size_mappings.items():
                            all_size_values.update(mapping.keys())
                        if all_size_values:
                            test_values = list(all_size_values)
                            print(f"  Warning: No mapping path found for {from_unit} -> {to_unit}, using all available size values: {len(test_values)} values")
                        else:
                            # This should never happen for clothing sizes
                            raise ValueError(f"No size mappings found for clothing size conversion {from_unit} -> {to_unit}")
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
            
            # Check if this conversion type supports math-only mode
            supports_math_only = False
            if math_only:
                # Math-only works for:
                # - Linear conversions (simple multiplication)
                # - Currency (with exchange rates, simple multiplication)
                # - Temperature (formulas like (C*9/5)+32)
                # - Timezone (addition/subtraction with modulo)
                if conversion_type == 'linear':
                    supports_math_only = True
                elif conversion_type == 'custom' and (section_config.get('exchange_rates') or (full_config and full_config.get('exchange_rates'))):
                    supports_math_only = True
                elif conversion_type == 'temperature':
                    supports_math_only = True
                elif conversion_type == 'timezone':
                    supports_math_only = True
                
                # Skip if this conversion type doesn't support math-only
                if not supports_math_only:
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
                # Pass base_conversion_name to help determine gender from filename
                temp_config_for_prompt['base_conversion_name'] = base_conversion_name
            # For currency, also include exchange rates from full_config
            if is_currency and full_config:
                temp_config_for_prompt['exchange_rates'] = full_config.get('exchange_rates', {})
            prompt = create_prompt(test_value, from_unit, to_unit, conversion_type, temp_config_for_prompt, context=None, full_config=full_config, include_guide=include_guide, math_only=math_only)
            
            rows.append({
                'domain': domain_name,
                'distractor': None,
                'prompt': prompt,
                'number': test_value,
                'answer': answer,
                'difficulty': difficulty
            })
            
            # Create prompts with contexts if available (skip for math-only mode)
            if not math_only:
                for context in contexts:
                    # For clothing sizes, contexts typically don't apply, but handle if needed
                    temp_config_for_context = temp_config_for_prompt.copy()
                    if section_config.get('size_mappings'):
                        temp_config_for_context['size_mappings'] = section_config.get('size_mappings')
                        # Pass section_name as sizing_type for clothing size prompts
                        if section_name:
                            temp_config_for_context['sizing_type'] = section_name
                        # Pass base_conversion_name to help determine gender from filename
                        temp_config_for_context['base_conversion_name'] = base_conversion_name
                    # For currency, also include exchange rates from full_config
                    if is_currency and full_config:
                        temp_config_for_context['exchange_rates'] = full_config.get('exchange_rates', {})
                    prompt_with_context = create_prompt(test_value, from_unit, to_unit, conversion_type, temp_config_for_context, context=context, full_config=full_config, include_guide=include_guide, math_only=math_only)
                    
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
    parser.add_argument('--no-guide', action='store_true',
                       help='Exclude conversion guides from prompts (creates files with _no_guide suffix)')
    parser.add_argument('--math-only', action='store_true',
                       help='Create pure mathematical prompts (e.g., "what is 5*0.067628") instead of conversion prompts. Only works for linear conversions and currency.')
    
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
        # Process all JSON files in conversions directory (except numbers.json and substances.json)
        json_files = [f for f in conversions_dir.glob('*.json') 
                     if f.name not in ['numbers.json', 'substances.json', 'times.json']]
    
    # Find substances file (needed for density, volume, moles_to_particles)
    substances_file = conversions_dir / 'substances.json'
    if not substances_file.exists():
        print(f"Warning: Substances file not found: {substances_file}. Density/volume/moles conversions may not have contexts.")
        substances_file = None
    
    print(f"Processing {len(json_files)} conversion file(s)...")
    
    # Process each file
    for json_file in json_files:
        if not json_file.exists():
            print(f"Warning: File not found: {json_file}, skipping...")
            continue
        
        try:
            process_conversion_file(json_file, numbers_file, times_file, output_dir, substances_file, include_guide=not args.no_guide, math_only=args.math_only)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nPreprocessing complete! Output files in {output_dir}")

if __name__ == '__main__':
    main()
