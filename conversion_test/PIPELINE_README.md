# Conversion Test Pipeline

This pipeline processes conversion tasks through two main stages: preprocessing and inference.

## Overview

1. **Preprocessing** (`preprocessing.py`): Converts JSON configuration files to TSV format with prompts
2. **Main Conversion** (`main_conversion.py`): Runs inference on preprocessed TSV files and compares results

## Conversion Types

### Scientific Conversions
- `volume.json` - Volume unit conversions
- `speed.json` - Speed unit conversions
- `density.json` - Density unit conversions
- `bits_bytes.json` - Digital storage conversions
- `temperature.json` - Temperature conversions
- `moles.json` - Moles to particles conversions

### General Conversions
- `energy.json` - Energy/calorie conversions
- `clothing_sizes.json` - Clothing size conversions (US, UK, EU)
- `currency.json` - Currency conversions
- `cooking.json` - Cooking measurement conversions
- `timezone.json` - Timezone conversions

## Usage

### Step 1: Preprocessing

Convert JSON files to TSV format:

```bash
python preprocessing.py --conversions-dir conversions --numbers-file conversions/numbers.json --output-dir preprocessed
```

Or process specific files:

```bash
python preprocessing.py --files conversions/temperature.json conversions/volume.json --output-dir preprocessed
```

**Output**: TSV files with columns: `domain`, `distractor`, `prompt`, `number`, `answer`, `difficulty`

### Step 2: Main Conversion

Run inference on preprocessed TSV files:

```bash
python main_conversion.py --domain temperature --input-file preprocessed/temperature.tsv --models gpt-4o
```

Or run multiple models:

```bash
python main_conversion.py --domain temperature --input-file preprocessed/temperature.tsv --models gpt-4o qwen-coder llama-4
```

**Output**: `[domain]_converted.tsv` with additional columns: `raw_response`, `model_answer`, `loss`

## File Structure

```
conversion_test/
├── conversions/
│   ├── *.json          # Conversion configuration files
│   └── numbers.json    # Easy/hard numbers for testing
├── preprocessed/       # Output from preprocessing.py
│   └── *.tsv          # Preprocessed TSV files
├── preprocessing.py    # Preprocessing script
└── main_conversion.py # Inference script
```

## Notes

- JSON files are left with space for human adjustment
- The `numbers.json` file contains easy and hard numbers that determine difficulty
- Distractor column contains context items (e.g., "peanut butter" in cooking conversions)
- Loss column shows relative error (%) for numeric answers, difference in minutes for timezone, and 0/1 for exact match for other types
