import json
import os
import asyncio
from openai import AsyncOpenAI
from typing import List, Dict, Optional, Callable, Any
from tqdm.asyncio import tqdm
import aiofiles
from datetime import datetime
import pandas as pd
import time
import signal
import sys
import re
from decimal import Decimal, getcontext

# Set precision for decimal calculations
getcontext().prec = 28

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

# Rate limiting for Together AI
class RateLimiter:
    """Rate limiter to prevent exceeding API limits."""
    def __init__(self, max_requests_per_minute: int = 200):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait if necessary to stay within rate limit."""
        async with self.lock:
            now = time.time()
            
            # Remove requests older than 60 seconds
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            # If we're at the limit, only wait for the oldest request to expire
            if len(self.requests) >= self.max_requests:
                oldest_request = self.requests[0]
                time_since_oldest = now - oldest_request
                sleep_time = 60 - time_since_oldest + 0.05
                
                if sleep_time > 0:
                    if sleep_time > 0.5:
                        print(f"Rate limit: waiting {sleep_time:.1f}s for slot to open...")
                    await asyncio.sleep(sleep_time)
                    
                    now = time.time()
                    self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            self.requests.append(now)

together_rate_limiter = RateLimiter(max_requests_per_minute=200)

# Graceful shutdown handler
class GracefulShutdown:
    """Handle graceful shutdown on Ctrl+C."""
    def __init__(self):
        self.shutdown_requested = False
        
    def request_shutdown(self, signum, frame):
        """Signal handler for graceful shutdown."""
        print("\n\n" + "="*60)
        print("SHUTDOWN REQUESTED - Saving progress...")
        print("="*60)
        self.shutdown_requested = True

shutdown_handler = GracefulShutdown()

# Model configurations
MODEL_CONFIGS = {
    "gpt-4o": {"provider": "openai", "model": "gpt-4o"},
    "qwen-coder": {"provider": "together", "model": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"},
    "llama-4": {"provider": "together", "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"}
}

# Standardized test values (same as new_convertlang.py)
TEST_VALUES = {
    'easy': [1, 5, 10, 20, 50],
    'hard': [4508.208, 1297.195, 18.333, 9.0241, 0.2994],
    'random': [5718, 1241.43, 3959.435, 12.505, 9717.519]
}

TIME_VALUES = {
    'easy': ['1AM', '1PM', '12PM', '3PM', '9PM'],
    'hard': ['11:59AM', '3:49PM', '2:48AM', '6:58PM', '8:11PM']
}

ALL_TEST_VALUES = TEST_VALUES['easy'] + TEST_VALUES['hard'] + TEST_VALUES['random']
ALL_TIME_VALUES = TIME_VALUES['easy'] + TIME_VALUES['hard']

# Conversion types that should have context variations (only density and moles)
CONVERSIONS_WITH_CONTEXT = {'density', 'moles_to_particles', 'volume'}

def parse_time_string(time_str: str) -> float:
    """Parse time string (e.g., '1AM', '3:49PM') into numeric hours (0-24)."""
    time_str = time_str.strip().upper()
    
    # Remove AM/PM
    is_pm = 'PM' in time_str
    time_str = time_str.replace('AM', '').replace('PM', '').strip()
    
    # Parse hours and minutes
    if ':' in time_str:
        parts = time_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
    else:
        hours = int(time_str)
        minutes = 0
    
    # Convert to 24-hour format
    if is_pm and hours != 12:
        hours += 12
    elif not is_pm and hours == 12:
        hours = 0
    
    # Convert to decimal hours
    return hours + minutes / 60.0

def format_time_string(hours: float) -> str:
    """Format numeric hours (0-24) into time string (e.g., '1AM', '3:49PM')."""
    # Handle day rollover
    while hours < 0:
        hours += 24
    while hours >= 24:
        hours -= 24
    
    h = int(hours)
    m = int((hours - h) * 60)
    
    # Convert to 12-hour format
    is_pm = h >= 12
    if h == 0:
        h = 12
    elif h > 12:
        h -= 12
    
    if m == 0:
        return f"{h}{'PM' if is_pm else 'AM'}"
    else:
        return f"{h}:{m:02d}{'PM' if is_pm else 'AM'}"

# Conversion calculation functions
def convert_linear(value: float, from_unit: str, to_unit: str, conversion_factors: Dict[str, float], base_unit: str) -> float:
    """Convert between units using linear conversion factors via a base unit.
    
    conversion_factors represent: how many of this unit equals 1 base unit.
    Example: if base_unit is 'moles':
    - moles: 1.0 means 1 mole = 1.0 moles (base)
    - particles: 6.022e23 means 6.022e23 particles = 1 mole (base)
    """
    # Convert to base unit first
    if from_unit == base_unit:
        base_value = value  # Already in base units
    else:
        # Divide by factor: if 6.022e23 particles = 1 mole, then 1 particle = 1/6.022e23 moles
        base_value = value / conversion_factors.get(from_unit, 1.0)
    
    # Convert from base unit to target unit
    if to_unit == base_unit:
        result = base_value  # Already in base units
    else:
        # Multiply by factor: if 6.022e23 particles = 1 mole, then 1 mole = 6.022e23 particles
        result = base_value * conversion_factors.get(to_unit, 1.0)
    
    return result

def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between temperature units (Celsius, Fahrenheit, Kelvin)."""
    from_lower = from_unit.lower()
    to_lower = to_unit.lower()
    
    # First convert to Kelvin
    if from_lower in ['c', 'celsius', 'degrees c', '°c', 'deg c']:
        kelvin = value + 273.15
    elif from_lower in ['f', 'fahrenheit', 'degrees f', '°f', 'deg f']:
        kelvin = (value - 32) * 5/9 + 273.15
    elif from_lower in ['k', 'kelvin']:
        kelvin = value
    else:
        raise ValueError(f"Unknown temperature unit: {from_unit}")
    
    # Then convert from Kelvin to target
    if to_lower in ['c', 'celsius', 'degrees c', '°c', 'deg c']:
        return kelvin - 273.15
    elif to_lower in ['f', 'fahrenheit', 'degrees f', '°f', 'deg f']:
        return (kelvin - 273.15) * 9/5 + 32
    elif to_lower in ['k', 'kelvin']:
        return kelvin
    else:
        raise ValueError(f"Unknown temperature unit: {to_unit}")

def convert_timezone(value: float, from_tz: str, to_tz: str, timezone_offsets: Dict[str, float], city_to_tz: Optional[Dict[str, str]] = None) -> float:
    """Convert time between timezones (value is hours, e.g., 14.5 = 2:30 PM).
    
    Supports both timezone abbreviations (EST, PST) and city names (Los Angeles, New York).
    If city_to_tz mapping is provided, city names will be looked up to find their timezone.
    """
    # Look up city names to timezone abbreviations if mapping provided
    if city_to_tz:
        from_tz_actual = city_to_tz.get(from_tz, from_tz)
        to_tz_actual = city_to_tz.get(to_tz, to_tz)
    else:
        from_tz_actual = from_tz
        to_tz_actual = to_tz
    
    from_offset = timezone_offsets.get(from_tz_actual, 0.0)
    to_offset = timezone_offsets.get(to_tz_actual, 0.0)
    
    # Convert to UTC first
    utc_time = value - from_offset
    # Then convert to target timezone
    result = utc_time + to_offset
    
    # Handle day rollover
    if result < 0:
        result += 24
    elif result >= 24:
        result -= 24
    
    return result

def convert_number_base(value: float, from_base: int, to_base: int) -> float:
    """Convert number from one base to another (e.g., binary to decimal)."""
    # For integer conversions
    if value == int(value):
        # Convert integer to decimal first
        if from_base != 10:
            # Parse as string in source base
            decimal_val = int(str(int(value)), from_base)
        else:
            decimal_val = int(value)
        
        # Convert to target base
        if to_base == 10:
            return float(decimal_val)
        else:
            # Convert to target base and return as decimal representation
            # For display purposes, we'll return the decimal value
            # The actual base conversion string would be different
            return float(decimal_val)
    else:
        # For non-integer, treat as decimal and convert
        return value

def convert_custom(value: float, from_unit: str, to_unit: str, conversion_func: Optional[Callable] = None, **kwargs) -> float:
    """Convert using a custom function if provided, otherwise use linear conversion."""
    if conversion_func:
        return conversion_func(value, from_unit, to_unit, **kwargs)
    else:
        # Default to linear conversion
        conversion_factors = kwargs.get('conversion_factors', {})
        base_unit = kwargs.get('base_unit', list(conversion_factors.keys())[0] if conversion_factors else from_unit)
        return convert_linear(value, from_unit, to_unit, conversion_factors, base_unit)

def load_conversion_config(config_file: str) -> Dict:
    """Load conversion configuration from JSON file."""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def calculate_conversion(
    value: float,
    from_unit: str,
    to_unit: str,
    conversion_type: str,
    config: Dict
) -> float:
    """Calculate the correct answer for a conversion based on conversion type."""
    if conversion_type == 'linear':
        conversion_factors = config.get('conversion_factors', {})
        base_unit = config.get('base_unit', list(conversion_factors.keys())[0] if conversion_factors else from_unit)
        return convert_linear(value, from_unit, to_unit, conversion_factors, base_unit)
    
    elif conversion_type == 'temperature':
        return convert_temperature(value, from_unit, to_unit)
    
    elif conversion_type == 'timezone':
        timezone_offsets = config.get('timezone_offsets', {})
        city_to_tz = config.get('city_to_timezone', {})
        return convert_timezone(value, from_unit, to_unit, timezone_offsets, city_to_tz)
    
    elif conversion_type == 'number_base':
        from_base = config.get('from_base', 10)
        to_base = config.get('to_base', 10)
        return convert_number_base(value, from_base, to_base)
    
    elif conversion_type == 'custom':
        # For custom types, the config should specify the calculation method
        # This could be extended to support Python code or formulas
        conversion_factors = config.get('conversion_factors', {})
        base_unit = config.get('base_unit')
        return convert_custom(value, from_unit, to_unit, conversion_factors=conversion_factors, base_unit=base_unit)
    
    else:
        raise ValueError(f"Unknown conversion type: {conversion_type}")

def create_conversion_tasks(
    conversion_configs: List[Dict],
    include_context_free: bool = True,
    contexts: Optional[List[str]] = None,
    max_unit_pairs: Optional[int] = None,
    max_test_values: Optional[int] = None
) -> List[Dict]:
    """Create conversion tasks from conversion configurations."""
    tasks = []
    
    contexts = contexts or []
    
    for config in conversion_configs:
        conversion_type = config.get('conversion_type', 'linear')
        conversion_name = config.get('name', 'unknown')
        unit_pairs = config.get('unit_pairs', [])
        
        print(f"\nProcessing {conversion_name} ({conversion_type})...")
        print(f"  Unit pairs: {len(unit_pairs)}")
        
        # Limit unit pairs if max_unit_pairs is specified
        if max_unit_pairs and max_unit_pairs > 0:
            unit_pairs = unit_pairs[:max_unit_pairs]
            print(f"  Limited to {len(unit_pairs)} unit pairs for testing")
        
        for pair in unit_pairs:
            from_unit = pair['from']
            to_unit = pair['to']
            
            # Use TIME_VALUES for timezone conversions, otherwise use ALL_TEST_VALUES
            if conversion_type == 'timezone':
                test_values = ALL_TIME_VALUES
            else:
                test_values = ALL_TEST_VALUES
            
            # Limit test values if max_test_values is specified
            if max_test_values and max_test_values > 0:
                test_values = test_values[:max_test_values]
            
            for test_value in test_values:
                # For timezone conversions, parse time string to numeric hours
                if conversion_type == 'timezone':
                    numeric_value = parse_time_string(test_value)
                else:
                    numeric_value = test_value
                
                # Calculate correct answer
                try:
                    correct_answer = calculate_conversion(
                        numeric_value,
                        from_unit,
                        to_unit,
                        conversion_type,
                        config
                    )
                except Exception as e:
                    print(f"  Warning: Could not calculate conversion {from_unit} -> {to_unit}: {e}")
                    continue
                
                # Determine difficulty
                if conversion_type == 'timezone':
                    if test_value in TIME_VALUES['easy']:
                        difficulty = 'easy'
                    elif test_value in TIME_VALUES['hard']:
                        difficulty = 'hard'
                    else:
                        difficulty = 'random'
                else:
                    if test_value in TEST_VALUES['easy']:
                        difficulty = 'easy'
                    elif test_value in TEST_VALUES['hard']:
                        difficulty = 'hard'
                    else:
                        difficulty = 'random'
                
                # Format answer appropriately
                if conversion_type == 'timezone':
                    # Convert numeric hours back to time string
                    correct_answer = format_time_string(correct_answer)
                elif conversion_type == 'number_base':
                    correct_answer = round(correct_answer, 0)  # Integer for base conversions
                # For all other conversions, keep full precision (no rounding)
                # JSON will preserve float precision automatically
                
                # Get city popularity metadata for timezone conversions
                city_popularity = None
                if conversion_type == 'timezone':
                    city_popularity_map = config.get('city_popularity', {})
                    from_popularity = None
                    to_popularity = None
                    
                    for level, cities in city_popularity_map.items():
                        if from_unit in cities:
                            from_popularity = level
                        if to_unit in cities:
                            to_popularity = level
                    
                    # Use the "lower" popularity level (more obscure) for the pair
                    # Handle both standard popularity levels and custom keys (like "easy", "random", "hard")
                    if from_popularity and to_popularity:
                        popularity_order = ['very_well_known', 'well_known', 'moderately_known', 'less_known', 'obscure']
                        # Check if both are in the standard popularity order
                        if from_popularity in popularity_order and to_popularity in popularity_order:
                            from_idx = popularity_order.index(from_popularity)
                            to_idx = popularity_order.index(to_popularity)
                            city_popularity = popularity_order[max(from_idx, to_idx)]
                        else:
                            # If using custom keys (e.g., "easy", "random", "hard"), just use the first one found
                            # or create a combined key
                            city_popularity = from_popularity if from_popularity else to_popularity
                    elif from_popularity:
                        city_popularity = from_popularity
                    elif to_popularity:
                        city_popularity = to_popularity
                
                # Create context-free task
                if include_context_free:
                    task_data = {
                        'conversion_name': conversion_name,
                        'conversion_type': conversion_type,
                        'context_type': 'context_free',
                        'context': None,
                        'test_value': test_value,
                        'difficulty': difficulty,
                        'from_unit': from_unit,
                        'to_unit': to_unit,
                        'correct_answer': correct_answer
                    }
                    if city_popularity:
                        task_data['city_popularity'] = city_popularity
                    tasks.append(task_data)
                
                # Create tasks with contexts (only for density, volume, moles conversions)
                if conversion_name in CONVERSIONS_WITH_CONTEXT:
                    for context in contexts:
                        task_data = {
                            'conversion_name': conversion_name,
                            'conversion_type': conversion_type,
                            'context_type': 'context',
                            'context': context,
                            'test_value': test_value,
                            'difficulty': difficulty,
                            'from_unit': from_unit,
                            'to_unit': to_unit,
                            'correct_answer': correct_answer
                        }
                        if city_popularity:
                            task_data['city_popularity'] = city_popularity
                        tasks.append(task_data)
    
    context_free_count = sum(1 for t in tasks if t['context_type'] == 'context_free')
    print(f"\nCreated {len(tasks)} conversion tasks:")
    print(f"  - Context-free: {context_free_count}")
    print(f"  - With context: {len(tasks) - context_free_count}")
    
    return tasks

async def load_existing_results(output_dir: str) -> tuple[List[Dict], set]:
    """Load existing results from previous runs to enable resume."""
    existing_results = []
    completed_tasks = set()
    
    result_files = []
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.startswith('conversion_results_') and filename.endswith('.json'):
                result_files.append(os.path.join(output_dir, filename))
            elif filename.startswith('checkpoint_') and filename.endswith('.json'):
                result_files.append(os.path.join(output_dir, filename))
    
    if not result_files:
        print("No existing results found - starting fresh")
        return existing_results, completed_tasks
    
    result_files.sort(key=os.path.getmtime, reverse=True)
    most_recent = result_files[0]
    
    print(f"\nFound existing results: {os.path.basename(most_recent)}")
    
    try:
        async with aiofiles.open(most_recent, 'r') as f:
            content = await f.read()
            existing_results = json.loads(content)
        
        for result in existing_results:
            task_signature = (
                result['model'],
                result['conversion_name'],
                result.get('context'),
                result['test_value'],
                result['from_unit'],
                result['to_unit']
            )
            completed_tasks.add(task_signature)
        
        print(f"Loaded {len(existing_results)} existing results")
        print(f"Unique completed tasks: {len(completed_tasks)}")
        
        return existing_results, completed_tasks
    
    except Exception as e:
        print(f"Error loading existing results: {e}")
        return [], set()

def filter_completed_tasks(tasks: List[Dict], completed_tasks: set, model_name: str) -> List[Dict]:
    """Filter out tasks that have already been completed for this model."""
    remaining_tasks = []
    
    for task in tasks:
        task_signature = (
            model_name,
            task['conversion_name'],
            task.get('context'),
            task['test_value'],
            task['from_unit'],
            task['to_unit']
        )
        
        if task_signature not in completed_tasks:
            remaining_tasks.append(task)
    
    return remaining_tasks

async def ask_openai(prompt: str, model: str, semaphore: asyncio.Semaphore, is_timezone: bool = False) -> str:
    """Ask OpenAI model to perform conversion."""
    async with semaphore:
        try:
            if is_timezone:
                system_content = "You are a precise timezone conversion expert. Provide the time in the same format as the input (e.g., 1AM, 3:49PM), no explanations."
            else:
                system_content = "You are a precise unit conversion expert. Provide only the numerical answer with up to 4 decimal places, no units or explanations."
            
            response = await openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=500  # Increased from 100 to handle scientific notation and longer responses
            )
            content = response.choices[0].message.content.strip()
            finish_reason = response.choices[0].finish_reason
            
            # Check if response was truncated
            if finish_reason == 'length':
                # Response was cut off - try to extract what we can
                # For scientific notation, check if we have a partial number
                if not is_timezone:
                    # Try to extract any number from the truncated response
                    numbers = re.findall(r'-?\d+\.?\d*[eE][+-]?\d*', content)
                    if not numbers:
                        numbers = re.findall(r'-?\d+\.?\d*', content)
                    if numbers:
                        # Return the last number found (might be incomplete)
                        return f"{numbers[-1]} [TRUNCATED]"
                return f"{content} [TRUNCATED]"
            
            return content
        except Exception as e:
            return f"ERROR: {str(e)}"

async def ask_together(prompt: str, model: str, semaphore: asyncio.Semaphore, retry_count: int = 0, is_timezone: bool = False) -> str:
    """Ask Together AI model to perform conversion."""
    await together_rate_limiter.acquire()
    
    async with semaphore:
        try:
            if is_timezone:
                system_content = "You are a precise timezone conversion expert. Provide the time in the same format as the input (e.g., 1AM, 3:49PM), no explanations."
            else:
                system_content = "You are a precise unit conversion expert. Provide only the numerical answer with up to 4 decimal places, no units or explanations."
            
            response = await together_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1000
            )
            content = response.choices[0].message.content.strip()
            finish_reason = response.choices[0].finish_reason
            
            # Check if response was truncated
            if finish_reason == 'length':
                # Response was cut off - try to extract what we can
                if not is_timezone:
                    # Try to extract any number from the truncated response
                    numbers = re.findall(r'-?\d+\.?\d*[eE][+-]?\d*', content)
                    if not numbers:
                        numbers = re.findall(r'-?\d+\.?\d*', content)
                    if numbers:
                        # Return the last number found (might be incomplete)
                        return f"{numbers[-1]} [TRUNCATED]"
                return f"{content} [TRUNCATED]"
            
            return content
        except Exception as e:
            error_msg = str(e)
            if ("rate" in error_msg.lower() or "429" in error_msg) and retry_count < 3:
                wait_time = 5 * (2 ** retry_count)
                print(f"Rate limit error (attempt {retry_count + 1}/3), waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
                return await ask_together(prompt, model, semaphore, retry_count + 1, is_timezone)
            return f"ERROR: {error_msg}"

async def get_model_answer(
    prompt: str,
    model_name: str,
    semaphore: asyncio.Semaphore,
    is_timezone: bool = False
) -> str:
    """Route question to appropriate model provider."""
    config = MODEL_CONFIGS.get(model_name)
    if not config:
        return f"ERROR: Unknown model {model_name}"
    
    provider = config["provider"]
    model = config["model"]
    
    if provider == "openai":
        return await ask_openai(prompt, model, semaphore, is_timezone)
    elif provider == "together":
        return await ask_together(prompt, model, semaphore, is_timezone=is_timezone)
    else:
        return f"ERROR: Unknown provider {provider}"

def create_conversion_prompt(task: Dict, config: Dict) -> str:
    """Create prompt for unit conversion."""
    value = task['test_value']
    from_unit = task['from_unit']
    to_unit = task['to_unit']
    context = task.get('context')
    
    # Get display names if available
    display_names = config.get('display_names', {})
    from_display = display_names.get(from_unit, from_unit)
    to_display = display_names.get(to_unit, to_unit)
    
    # Handle pluralization for some units
    if value != 1 and task['conversion_type'] not in ['temperature', 'timezone', 'number_base']:
        # Simple pluralization (can be improved)
        if not from_display.endswith('s') and not from_display.endswith('es'):
            from_display = from_display + 's'
        if not to_display.endswith('s') and not to_display.endswith('es'):
            to_display = to_display + 's'
    
    # Special handling for timezone conversions with cities
    if task['conversion_type'] == 'timezone':
        # For timezone, format as "Convert [time] in [city] to [city]"
        # value is already a time string like '1AM' or '3:49PM'
        # Timezone conversions never have context
        prompt = f"Convert {value} in {from_display} time to {to_display} time. Assume you are thinking about standard time, not daylight savings. Provide the time in the same format (e.g., 1AM, 3:49PM)."
    elif context and task['conversion_name'] in CONVERSIONS_WITH_CONTEXT:
        # Only add context for conversions that support it (density and moles)
        prompt = f"Convert {value} {from_display} of {context} to {to_display}. Provide only the numerical value."
    else:
        # Context-free prompt for all other conversions
        prompt = f"Convert {value} {from_display} to {to_display}. Provide only the numerical value."
    
    return prompt

def extract_number(answer: str) -> Optional[float]:
    """Extract numeric value from answer string, handling scientific notation."""
    if not answer or answer.startswith('ERROR:'):
        return None
    
    # Remove [TRUNCATED] marker if present
    cleaned = answer.replace("[TRUNCATED]", "").replace(",", "").strip()
    
    # First try to match scientific notation (e.g., 6.0221e+23, 3.5E-10, 1e23)
    # Look for complete scientific notation first
    sci_pattern = re.search(r'-?\d+\.?\d*[eE][+-]?\d+', cleaned)
    if sci_pattern:
        try:
            return float(sci_pattern.group(0))
        except ValueError:
            pass
    
    # Try to match partial scientific notation (e.g., "6.0221e+" from truncated response)
    sci_partial = re.search(r'-?\d+\.?\d*[eE][+-]?', cleaned)
    if sci_partial:
        # This is likely a truncated scientific notation - try to extract what we can
        # For truncated sci notation, we can't parse it, so return None
        pass
    
    # Fall back to regular number extraction (for non-scientific notation)
    # Try to find the most likely answer number (usually the first or largest standalone number)
    matches = re.findall(r'-?\d+\.?\d*', cleaned)
    if matches:
        # Prefer numbers that look like answers (not part of explanations)
        # If there are multiple numbers, try to find the one that's most likely the answer
        # For now, try all matches and pick the one that's most reasonable
        for match in reversed(matches):  # Start from the end (often the answer comes last)
            try:
                num = float(match)
                # Skip very small numbers that might be part of explanations (like "0.1%")
                if abs(num) > 1e-10 or abs(num) == 0:
                    return num
            except ValueError:
                continue
        # If none worked, try the last one anyway
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None

def extract_time_string(answer: str) -> Optional[str]:
    """Extract time string from answer (e.g., '1AM', '3:49PM')."""
    # Try to find time patterns
    # Pattern 1: HH:MMAM/PM or HH:MM AM/PM
    time_pattern1 = re.search(r'(\d{1,2}):(\d{2})\s*(AM|PM)', answer, re.IGNORECASE)
    if time_pattern1:
        h = int(time_pattern1.group(1))
        m = int(time_pattern1.group(2))
        period = time_pattern1.group(3).upper()
        return f"{h}:{m:02d}{period}"
    
    # Pattern 2: HHAM/PM or HH AM/PM
    time_pattern2 = re.search(r'(\d{1,2})\s*(AM|PM)', answer, re.IGNORECASE)
    if time_pattern2:
        h = int(time_pattern2.group(1))
        period = time_pattern2.group(2).upper()
        return f"{h}{period}"
    
    return None

def calculate_accuracy(model_answer: Optional[float], correct_answer: float, tolerance: float = 0.1) -> bool:
    """Check if model answer is within tolerance of correct answer."""
    if model_answer is None:
        return False
    
    if correct_answer == 0:
        return abs(model_answer) < tolerance
    
    percent_error = abs((model_answer - correct_answer) / correct_answer) * 100
    return percent_error <= tolerance

def calculate_time_accuracy(model_answer_str: Optional[str], correct_answer_str: str, tolerance_minutes: int = 5) -> bool:
    """Check if model time answer matches correct time within tolerance."""
    if model_answer_str is None:
        return False
    
    try:
        model_hours = parse_time_string(model_answer_str)
        correct_hours = parse_time_string(correct_answer_str)
        
        # Calculate difference in minutes
        diff_minutes = abs((model_hours - correct_hours) * 60)
        
        # Handle day rollover (e.g., 11:59PM vs 12:01AM)
        if diff_minutes > 12 * 60:  # More than 12 hours difference
            diff_minutes = 24 * 60 - diff_minutes
        
        return diff_minutes <= tolerance_minutes
    except (ValueError, AttributeError):
        return False

async def evaluate_task(
    task: Dict,
    config: Dict,
    model_name: str,
    semaphore: asyncio.Semaphore
) -> Dict:
    """Evaluate a single conversion task."""
    prompt = create_conversion_prompt(task, config)
    is_timezone = task['conversion_type'] == 'timezone'
    
    model_answer_str = await get_model_answer(prompt, model_name, semaphore, is_timezone)
    is_error = model_answer_str.startswith('ERROR:')
    
    # Handle timezone conversions differently (time strings vs numeric)
    if task['conversion_type'] == 'timezone':
        model_answer_time = extract_time_string(model_answer_str)
        is_correct = calculate_time_accuracy(model_answer_time, task['correct_answer'])
        model_answer_num = parse_time_string(model_answer_time) if model_answer_time else None
    else:
        model_answer_num = extract_number(model_answer_str)
        is_correct = calculate_accuracy(model_answer_num, task['correct_answer'])
    
    result = {
        'model': model_name,
        'conversion_name': task['conversion_name'],
        'conversion_type': task['conversion_type'],
        'context_type': task['context_type'],
        'context': task.get('context'),
        'test_value': task['test_value'],
        'difficulty': task['difficulty'],
        'from_unit': task['from_unit'],
        'to_unit': task['to_unit'],
        'correct_answer': task['correct_answer'],
        'model_answer_raw': model_answer_str,
        'model_answer': model_answer_num,
        'is_correct': is_correct,
        'is_error': is_error,
        'prompt': prompt
    }
    
    # Include city popularity metadata if available
    if 'city_popularity' in task:
        result['city_popularity'] = task['city_popularity']
    
    # For timezone, also include parsed time string
    if task['conversion_type'] == 'timezone' and model_answer_time:
        result['model_answer_time'] = model_answer_time
    
    return result

async def evaluate_model_on_tasks(
    tasks: List[Dict],
    configs: Dict[str, Dict],
    model_name: str,
    batch_size: int = 20,
    max_concurrent: int = 10
) -> List[Dict]:
    """Evaluate a model on all conversion tasks."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} on {len(tasks)} conversion tasks")
    print(f"{'='*60}\n")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
    all_results = []
    
    for batch_idx, batch in enumerate(tqdm(batches, desc=f"Processing {model_name}")):
        if shutdown_handler.shutdown_requested:
            print(f"\nShutdown requested - stopping at batch {batch_idx}/{len(batches)}")
            break
        
        batch_tasks = []
        for task in batch:
            config = configs.get(task['conversion_name'], {})
            batch_tasks.append(evaluate_task(task, config, model_name, semaphore))
        
        batch_results = await asyncio.gather(*batch_tasks)
        all_results.extend(batch_results)
    
    correct_count = sum(1 for r in all_results if r['is_correct'])
    accuracy = (correct_count / len(all_results) * 100) if all_results else 0
    
    print(f"\n{model_name} Results:")
    print(f"  Total conversions: {len(all_results)}")
    print(f"  Correct: {correct_count}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    return all_results

async def run_experiment(
    config_files: List[str],
    output_dir: str = "conversion_results",
    batch_size: int = 20,
    max_concurrent: int = 10,
    include_context_free: bool = True,
    contexts_file: Optional[str] = None,
    together_rpm: int = 200,
    resume: bool = True,
    max_unit_pairs: Optional[int] = None,
    max_test_values: Optional[int] = None,
    test_models: Optional[List[str]] = None
):
    """Run the complete conversion experiment."""
    signal.signal(signal.SIGINT, shutdown_handler.request_shutdown)
    
    print(f"\n{'='*60}")
    print(f"SCIENTIFIC CONVERSION EXPERIMENT")
    print(f"{'='*60}")
    print("Press Ctrl+C to save progress and exit gracefully")
    print(f"{'='*60}\n")
    
    global together_rate_limiter
    together_rate_limiter = RateLimiter(max_requests_per_minute=together_rpm)
    print(f"Together AI rate limit set to: {together_rpm} requests/minute")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load conversion configurations
    conversion_configs = []
    configs_dict = {}
    
    for config_file in config_files:
        print(f"Loading conversion config: {config_file}")
        config = load_conversion_config(config_file)
        conversion_configs.append(config)
        configs_dict[config.get('name', 'unknown')] = config
    
    # Load contexts if provided
    contexts = []
    if contexts_file and os.path.exists(contexts_file):
        with open(contexts_file, 'r') as f:
            contexts_data = json.load(f)
            contexts = contexts_data.get('contexts', [])
        print(f"Loaded {len(contexts)} contexts from {contexts_file}")
    
    # Load existing results if resuming
    existing_results = []
    completed_tasks = set()
    
    if resume:
        existing_results, completed_tasks = await load_existing_results(output_dir)
    
    # Create conversion tasks
    all_tasks = create_conversion_tasks(
        conversion_configs,
        include_context_free=include_context_free,
        contexts=contexts,
        max_unit_pairs=max_unit_pairs,
        max_test_values=max_test_values
    )
    
    # Evaluate each model
    all_results = existing_results.copy()
    models = test_models if test_models else ["gpt-4o", "qwen-coder", "llama-4"]
    
    try:
        for model in models:
            if shutdown_handler.shutdown_requested:
                print(f"\nSkipping remaining models due to shutdown request")
                break
            
            remaining_tasks = filter_completed_tasks(all_tasks, completed_tasks, model)
            
            if not remaining_tasks:
                print(f"\n{'='*60}")
                print(f"{model} - All tasks already completed, skipping")
                print(f"{'='*60}\n")
                continue
            
            print(f"\n{'='*60}")
            print(f"{model} - {len(remaining_tasks)} tasks remaining (out of {len(all_tasks)} total)")
            print(f"{'='*60}\n")
            
            model_results = await evaluate_model_on_tasks(
                tasks=remaining_tasks,
                configs=configs_dict,
                model_name=model,
                batch_size=batch_size,
                max_concurrent=max_concurrent
            )
            all_results.extend(model_results)
            
            # Update completed tasks
            for result in model_results:
                task_signature = (
                    result['model'],
                    result['conversion_name'],
                    result.get('context'),
                    result['test_value'],
                    result['from_unit'],
                    result['to_unit']
                )
                completed_tasks.add(task_signature)
            
            # Save checkpoint
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_file = os.path.join(output_dir, f"checkpoint_{model}_{timestamp}.json")
            async with aiofiles.open(checkpoint_file, 'w') as f:
                await f.write(json.dumps(all_results, indent=2))
            print(f"Checkpoint saved: {checkpoint_file}")
    
    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt detected during processing")
        shutdown_handler.shutdown_requested = True
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    status = "partial" if shutdown_handler.shutdown_requested else "complete"
    output_file = os.path.join(output_dir, f"conversion_results_{status}_{timestamp}.json")
    
    print(f"\nSaving {status} results to {output_file}...")
    async with aiofiles.open(output_file, 'w') as f:
        await f.write(json.dumps(all_results, indent=2))
    
    if shutdown_handler.shutdown_requested:
        print("\n" + "="*60)
        print("PARTIAL RESULTS SAVED")
        print("="*60)
        print(f"Completed: {len(all_results)} tasks")
        print(f"Output: {output_file}")
        print("\nTo resume: Simply re-run the same command")
        print("="*60 + "\n")
        return
    
    # Create summary
    print("\n" + "="*60)
    print("SUMMARY BY MODEL AND CONVERSION TYPE")
    print("="*60)
    
    results_df = pd.DataFrame(all_results)
    
    if not results_df.empty:
        valid_results = results_df[~results_df['is_error']]
        
        summary = valid_results.groupby(['model', 'conversion_name', 'difficulty'])['is_correct'].agg(['sum', 'count', 'mean']).reset_index()
        summary.columns = ['model', 'conversion_name', 'difficulty', 'correct', 'total', 'accuracy']
        summary['accuracy'] = summary['accuracy'] * 100
        summary = summary.sort_values(['model', 'conversion_name', 'difficulty'])
        
        summary_file = os.path.join(output_dir, f"summary_{timestamp}.csv")
        summary.to_csv(summary_file, index=False)
        
        print(summary.to_string(index=False))
        print(f"\nSummary saved to: {summary_file}")
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Results: {output_file}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test LLM accuracy on scientific unit conversions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python convert.py --config conversions/moles.json conversions/temperature.json
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        nargs='+',
        required=True,
        help="Path(s) to JSON configuration file(s) defining conversions"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scientific_conversion_results",
        help="Directory to save results (default: conversion_results)"
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
    
    parser.add_argument(
        "--no-context-free",
        action="store_true",
        help="Disable context-free conversions"
    )
    
    parser.add_argument(
        "--contexts",
        type=str,
        help="JSON file containing list of contexts to use (optional)"
    )
    
    parser.add_argument(
        "--together-rpm",
        type=int,
        default=200,
        help="Together AI rate limit in requests per minute (default: 200)"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from previous results"
    )
    
    parser.add_argument(
        "--max-unit-pairs",
        type=int,
        help="Limit number of unit pairs per conversion type (for testing)"
    )
    
    parser.add_argument(
        "--max-test-values",
        type=int,
        help="Limit number of test values per unit pair (for testing)"
    )
    
    parser.add_argument(
        "--test-model",
        type=str,
        help="Test with only one model (e.g., 'gpt-4o')"
    )
    
    args = parser.parse_args()
    
    test_models = [args.test_model] if args.test_model else None
    
    asyncio.run(run_experiment(
        config_files=args.config,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
        include_context_free=not args.no_context_free,
        contexts_file=args.contexts,
        together_rpm=args.together_rpm,
        resume=not args.no_resume,
        max_unit_pairs=args.max_unit_pairs,
        max_test_values=args.max_test_values,
        test_models=test_models
    ))
