# Conversion Configuration Files

This directory contains JSON configuration files that define different types of scientific conversions to test.

## Configuration File Format

Each JSON file should have the following structure:

```json
{
  "name": "conversion_name",
  "conversion_type": "linear|temperature|timezone|number_base|custom",
  "description": "Description of the conversion type",
  "display_names": {
    "unit_key": "Display Name"
  },
  "unit_pairs": [
    {"from": "unit1", "to": "unit2"}
  ]
}
```

### Conversion Types

#### 1. Linear Conversions (`linear`)
For conversions that use a simple multiplier (e.g., moles to particles, bits to bytes).

Required fields:
- `conversion_factors`: Dictionary mapping unit names to their conversion factor relative to base unit
- `base_unit`: The base unit for conversions

Example:
```json
{
  "name": "moles_to_particles",
  "conversion_type": "linear",
  "base_unit": "moles",
  "conversion_factors": {
    "moles": 1.0,
    "particles": 6.02214076e23
  },
  "unit_pairs": [
    {"from": "moles", "to": "particles"},
    {"from": "particles", "to": "moles"}
  ]
}
```

#### 2. Temperature Conversions (`temperature`)
For temperature conversions (Celsius, Fahrenheit, Kelvin).

Example:
```json
{
  "name": "temperature",
  "conversion_type": "temperature",
  "unit_pairs": [
    {"from": "celsius", "to": "fahrenheit"},
    {"from": "fahrenheit", "to": "kelvin"}
  ]
}
```

#### 3. Timezone Conversions (`timezone`)
For converting time between timezones.

Required fields:
- `timezone_offsets`: Dictionary mapping timezone names to their UTC offset in hours

Example:
```json
{
  "name": "timezone",
  "conversion_type": "timezone",
  "timezone_offsets": {
    "EST": -5.0,
    "PST": -8.0,
    "GMT": 0.0
  },
  "unit_pairs": [
    {"from": "EST", "to": "PST"}
  ]
}
```

## Usage

Run the conversion script with one or more configuration files:

```bash
python convert.py --config conversions/moles.json conversions/temperature.json
```

With contexts (e.g., chemical compounds):

```bash
python convert.py --config conversions/moles.json --contexts conversions/contexts_example.json
```

## Available Configuration Files

- `moles.json` - Moles to particles/atoms/molecules conversions
- `temperature.json` - Temperature unit conversions
- `bits_bytes.json` - Digital storage unit conversions
- `timezone.json` - Timezone conversions
- `energy.json` - Energy unit conversions
- `density.json` - Density unit conversions
- `contexts_example.json` - Example contexts for chemistry conversions

## Adding New Conversion Types

To add a new conversion type:

1. Create a new JSON file in this directory
2. Define the conversion type and unit pairs
3. If using `linear` type, specify conversion factors and base unit
4. Run the script with your new config file
