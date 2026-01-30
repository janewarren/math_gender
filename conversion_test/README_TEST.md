# Domain Testing Script

The `test_domains.py` script allows you to quickly test all conversion domains with a small sample to verify everything is working correctly.

## Usage

### Basic test (single model, small sample):
```bash
python3 test_domains.py
```

### Test with specific model(s):
```bash
python3 test_domains.py --models gpt-4o qwen-coder
```

### Test with limited rows per domain (faster):
```bash
python3 test_domains.py --max-rows 20
```

### Skip preprocessing if test files already exist:
```bash
python3 test_domains.py --skip-preprocessing
```

## What it does:

1. **Creates test data files:**
   - `test_output/test_numbers.json` - 5 easy + 5 hard numbers
   - `test_output/test_times.json` - 5 easy + 5 hard times

2. **Runs preprocessing:**
   - Processes all conversion JSON files
   - Creates TSV files in `test_output/preprocessed/`
   - Uses limited test values for fast execution

3. **Runs inference:**
   - Tests each domain with the specified model(s)
   - Saves results to `test_output/results/`

4. **Generates summary:**
   - Prints a summary table with statistics
   - Saves CSV summary to `test_output/test_summary_[timestamp].csv`

## Output

The script generates:
- **Preprocessed TSV files**: `test_output/preprocessed/[domain].tsv`
- **Result TSV files**: `test_output/results/[domain]_converted.tsv`
- **Summary CSV**: `test_output/test_summary_[timestamp].csv`

## Example Output

```
Domain                          Total Prompts  Answered  Answer Rate %  Valid Answers  Correct  Accuracy %
bits_bytes                      60             60        100.0           60              58       96.7
currency                        90             90        100.0           90              87       96.7
density                         60             60        100.0           60              55       91.7
...
```

## Notes

- The test uses only 5 easy + 5 hard numbers (10 total) instead of the full 200
- This makes testing much faster while still validating the pipeline
- For full runs, use `preprocessing.py` and `main_conversion.py` directly
