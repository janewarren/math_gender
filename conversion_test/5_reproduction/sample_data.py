#!/usr/bin/env python3
"""
Step 1: Stratified sampling from the full preprocessed dataset.

For each domain × condition TSV, sample N_PER_GROUP rows (stratified by
difficulty when available) to create a small but representative test set.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

PREPROC_DIR = Path(__file__).resolve().parent.parent / "full_results" / "preprocessed"
OUTPUT_DIR = Path(__file__).resolve().parent / "preprocessed"

N_PER_GROUP = 5   # rows per domain × condition
SEED = 42

CONDITIONS_SUFFIXES = {
    "regular": "",
    "no_guide": "_no_guide",
    "math_only": "_math_only",
}


def discover_domains() -> list[str]:
    """Find unique base domain names from the preprocessed directory."""
    names = set()
    for f in PREPROC_DIR.glob("*.tsv"):
        name = f.stem
        for suf in ("_no_guide", "_math_only"):
            name = name.replace(suf, "")
        names.add(name)
    return sorted(names)


def sample_tsv(input_path: Path, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample n rows from a TSV, stratified by difficulty if the column exists."""
    df = pd.read_csv(input_path, sep="\t")

    if "difficulty" in df.columns and df["difficulty"].nunique() > 1:
        groups = df.groupby("difficulty")
        per_stratum = max(1, n // df["difficulty"].nunique())
        samples = []
        for _, grp in groups:
            k = min(per_stratum, len(grp))
            samples.append(grp.sample(n=k, random_state=int(rng.integers(1e9))))
        result = pd.concat(samples, ignore_index=True)
    else:
        k = min(n, len(df))
        result = df.sample(n=k, random_state=int(rng.integers(1e9)))

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    domains = discover_domains()

    total = 0
    for domain in domains:
        for cond_name, suffix in CONDITIONS_SUFFIXES.items():
            src = PREPROC_DIR / f"{domain}{suffix}.tsv"
            if not src.exists():
                continue
            sample = sample_tsv(src, N_PER_GROUP, rng)
            out = OUTPUT_DIR / f"{domain}{suffix}.tsv"
            sample.to_csv(out, sep="\t", index=False)
            total += len(sample)
            print(f"  {domain:40s} {cond_name:12s}  {len(sample):3d} rows  (from {src.name})")

    print(f"\nTotal sampled: {total} rows across {len(list(OUTPUT_DIR.glob('*.tsv')))} files")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
