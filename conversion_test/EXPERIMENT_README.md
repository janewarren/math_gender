# Full Experiment Runner

This document describes how to run the full conversion experiment using SLURM batch jobs.

## Overview

The `run_full_experiment.sh` script:
1. Runs preprocessing for all three prompt types (regular, no-guide, math-only)
2. Submits separate SLURM batch jobs for each domain
3. Each job processes all models (gpt-4o, qwen-coder, llama-4) for all three types

## Usage

### Basic Usage

```bash
cd /data/jane/convert/math_gender/conversion_test
./run_full_experiment.sh
```

### What It Does

1. **Preprocessing Phase** (runs sequentially):
   - Generates `test_output/preprocessed/*.tsv` files for regular prompts
   - Generates `test_output/preprocessed/*_no_guide.tsv` files
   - Generates `test_output/preprocessed/*_math_only.tsv` files

2. **Job Submission Phase**:
   - Creates one SLURM job per domain (15 domains total)
   - Each job processes:
     - Regular prompts → `test_output/results/<model>/<domain>_converted.tsv`
     - No-guide prompts → `test_output/results_no_guide/<model>/<domain>_no_guide_converted.tsv`
     - Math-only prompts → `test_output/results_math_only/<model>/<domain>_math_only_converted.tsv`
   - All jobs run in parallel (subject to SLURM queue limits)

## Domains Processed

The script processes the following 15 domains:
- bits_bytes
- clothing_sizes_men_shoe_size
- clothing_sizes_men_pant_size
- clothing_sizes_women_shoe_size
- clothing_sizes_women_bra_size
- clothing_sizes_women_pant_size
- cooking
- currency
- density
- energy
- moles_to_particles
- speed
- temperature
- timezone
- volume

## Models

Each domain is tested with three models:
- gpt-4o
- qwen-coder
- llama-4

## Output Structure

```
test_output/
├── preprocessed/
│   ├── bits_bytes.tsv
│   ├── bits_bytes_no_guide.tsv
│   ├── bits_bytes_math_only.tsv
│   └── ... (for each domain)
├── results/
│   ├── gpt-4o/
│   │   ├── bits_bytes_converted.tsv
│   │   └── ...
│   ├── qwen-coder/
│   │   └── ...
│   └── llama-4/
│       └── ...
├── results_no_guide/
│   ├── gpt-4o/
│   │   ├── bits_bytes_no_guide_converted.tsv
│   │   └── ...
│   └── ... (same structure)
└── results_math_only/
    ├── gpt-4o/
    │   ├── bits_bytes_math_only_converted.tsv
    │   └── ...
    └── ... (same structure)
```

## Job Management

### Monitor Jobs

```bash
# View all submitted jobs
squeue -u $USER

# View specific job details
scontrol show job <job_id>
```

### Check Job Outputs

Job outputs are saved in `sbatch_jobs/`:
- `sbatch_jobs/conv_<domain>.out` - Standard output
- `sbatch_jobs/conv_<domain>.err` - Standard error

```bash
# View output for a specific domain
tail -f sbatch_jobs/conv_temperature.out

# Check for errors
grep -i error sbatch_jobs/*.err
```

### Cancel Jobs

```bash
# Cancel all jobs for this experiment
scancel -n conv_

# Cancel specific job
scancel <job_id>
```

## Customization

### Modify Resource Requirements

Edit the SBATCH directives in the script (around line 60-65):

```bash
#SBATCH --time=24:00:00    # Maximum runtime
#SBATCH --mem=16G          # Memory per job
#SBATCH --cpus-per-task=4  # CPUs per job
```

### Modify Models

Edit the `MODELS` array in the script:

```bash
MODELS=("gpt-4o" "qwen-coder" "llama-4")
```

### Modify Domains

Edit the `DOMAINS` array in the script to process only specific domains:

```bash
DOMAINS=(
    "temperature"
    "timezone"
    "currency"
)
```

### Add Environment Activation

If you need to activate a virtual environment, uncomment and modify this section in the job script template:

```bash
# Activate environment if needed
source /path/to/venv/bin/activate
```

## Troubleshooting

### Preprocessing Fails

- Check that `conversions/numbers.json` and `conversions/times.json` exist
- Verify all JSON configuration files are valid
- Check Python dependencies are installed

### Jobs Fail to Start

- Check SLURM queue limits: `squeue`
- Verify account/partition settings if required
- Check job script permissions: `ls -l sbatch_jobs/*.sh`

### Jobs Complete but No Results

- Check job output files for errors: `cat sbatch_jobs/conv_<domain>.err`
- Verify input files exist: `ls test_output/preprocessed/<domain>*.tsv`
- Check API keys/credentials for model access

## Running Individual Domains

If you want to test a single domain first:

1. Comment out other domains in the `DOMAINS` array
2. Or create a separate script for a single domain
3. Or run manually:

```bash
# Preprocess
python3 preprocessing.py --files conversions/temperature.json \
    --numbers-file conversions/numbers.json \
    --times-file conversions/times.json \
    --output-dir test_output/preprocessed

# Run inference
python3 main_conversion.py \
    --domain temperature \
    --input-file test_output/preprocessed/temperature.tsv \
    --output-file test_output/results/gpt-4o/temperature_converted.tsv \
    --models gpt-4o qwen-coder llama-4
```

## Notes

- Preprocessing runs sequentially (takes ~5-10 minutes)
- Each domain job processes all models sequentially within the job
- Total runtime depends on:
  - Number of prompts per domain
  - API response times
  - SLURM queue wait times
- Jobs are independent and can be rerun individually if needed
