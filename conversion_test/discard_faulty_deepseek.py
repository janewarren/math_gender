#!/usr/bin/env python3
"""discard_faulty_deepseek.py — Back up originals, then null out rows missing <think> traces."""
import pandas as pd
import glob
import shutil
from pathlib import Path

RESULT_COLS = ["raw_response", "model_answer", "loss", "reasoning_tokens", "call_seconds"]
BACKUP_DIR = Path("full_results_deepseek_backup")

files = sorted(glob.glob("full_results/*/deepseek-v3.1/*_converted.tsv"))

# Step 1: Copy all original files to backup
print("=== Backing up original files ===")
for f in files:
    src = Path(f)
    dst = BACKUP_DIR / src.relative_to("full_results")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
print(f"  Backed up {len(files)} files to {BACKUP_DIR}/\n")

# Step 2: Null out rows without <think> traces
print("=== Clearing faulty rows ===")
total_kept, total_cleared = 0, 0
for f in files:
    df = pd.read_csv(f, sep="\t")
    has_response = df["raw_response"].notna() & (df["raw_response"].astype(str) != "null")
    has_trace = df["raw_response"].astype(str).str.contains(r"</think>", na=False)

    kept = (has_response & has_trace).sum()
    cleared = (has_response & ~has_trace).sum()

    if cleared > 0:
        mask = has_response & ~has_trace
        for col in RESULT_COLS:
            if col in df.columns:
                df.loc[mask, col] = None
        df.to_csv(f, sep="\t", index=False)

    total_kept += kept
    total_cleared += cleared
    short = f.split("deepseek-v3.1/")[-1]
    print(f"  {short:55s}  kept={kept:5d}  cleared={cleared:5d}")

print(f"\nDone. Kept {total_kept} rows with traces, cleared {total_cleared} for redo.")
print(f"Originals backed up to: {BACKUP_DIR.resolve()}")
