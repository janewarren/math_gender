#!/usr/bin/env python3
"""
Step 3: Aggregate inference results and compare with original gpt-4o results.

Mirrors the validation.ipynb logic: merge all *_converted.tsv files into a
single results.csv, compute accuracy, then compare against the original
project's gpt-4o numbers.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

REPRO_DIR = Path(__file__).resolve().parent
RESULTS_DIR = REPRO_DIR / "results" / "gpt-5.4-mini"
ORIGINAL_RESULTS = REPRO_DIR.parent / "2_analysis" / "results.csv"

CONDITION_MAP = {
    "results": "in_domain_with_guide",
}

SUFFIX_TO_CONDITION = {
    "": "in_domain_with_guide",
    "_no_guide": "in_domain_no_guide",
    "_math_only": "math_only",
}


def infer_condition(filename: str) -> str:
    """Derive the condition from the converted TSV filename."""
    stem = filename.replace("_converted", "")
    if "_math_only" in stem:
        return "math_only"
    elif "_no_guide" in stem:
        return "in_domain_no_guide"
    else:
        return "in_domain_with_guide"


def infer_domain(filename: str) -> str:
    """Derive the base domain from the converted TSV filename."""
    stem = filename.replace("_converted.tsv", "")
    for suf in ("_math_only", "_no_guide"):
        stem = stem.replace(suf, "")
    # Normalize clothing sub-domains to a single domain label
    if stem.startswith("clothing_sizes"):
        return "clothing_size"
    return stem


def load_repro_results() -> pd.DataFrame:
    """Load and merge all converted TSV files from the reproduction run."""
    frames = []
    for tsv in sorted(RESULTS_DIR.glob("*_converted.tsv")):
        df = pd.read_csv(tsv, sep="\t")
        df["model"] = "gpt-5.4-mini"
        df["condition"] = infer_condition(tsv.name)
        df["source_file"] = tsv.name
        frames.append(df)

    if not frames:
        print("ERROR: No converted TSV files found. Run run_inference.py first.")
        sys.exit(1)

    merged = pd.concat(frames, ignore_index=True)

    # Normalize domain names to match original analysis
    merged["domain"] = merged["source_file"].apply(infer_domain)
    merged["is_correct"] = merged["loss"] == 0.0

    return merged


def load_original_results() -> pd.DataFrame:
    """Load original results for gpt-5.2 (closest GPT reasoning model) and gpt-4o."""
    if not ORIGINAL_RESULTS.exists():
        print(f"WARNING: Original results not found at {ORIGINAL_RESULTS}")
        return pd.DataFrame()
    df = pd.read_csv(ORIGINAL_RESULTS, sep="\t",
                     usecols=["domain", "condition", "model", "loss", "is_correct"])
    return df[df["model"].isin(["gpt-5.2", "gpt-4o"])].copy()


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def main():
    # ── Load reproduction results ─────────────────────────────────
    repro = load_repro_results()
    print_section("Reproduction Run Summary")
    print(f"Total rows:  {len(repro)}")
    print(f"Domains:     {sorted(repro['domain'].unique())}")
    print(f"Conditions:  {sorted(repro['condition'].unique())}")

    valid = repro[repro["loss"].notna()]
    errors = repro[repro["raw_response"].astype(str).str.startswith("ERROR:")]
    correct = valid[valid["is_correct"]]
    print(f"Valid rows:  {len(valid)}  (errors: {len(errors)})")
    print(f"Correct:     {len(correct)} / {len(valid)}  ({len(correct)/len(valid)*100:.1f}%)")

    # ── Accuracy by condition ─────────────────────────────────────
    print_section("Accuracy by Condition")
    cond_acc = (
        valid.groupby("condition")["is_correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n"})
    )
    cond_acc["accuracy_%"] = (cond_acc["accuracy"] * 100).round(1)
    print(cond_acc[["n", "accuracy_%"]].to_string())

    # ── Accuracy by domain ────────────────────────────────────────
    print_section("Accuracy by Domain")
    dom_acc = (
        valid.groupby("domain")["is_correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n"})
        .sort_values("accuracy")
    )
    dom_acc["accuracy_%"] = (dom_acc["accuracy"] * 100).round(1)
    print(dom_acc[["n", "accuracy_%"]].to_string())

    # ── Accuracy by domain × condition ────────────────────────────
    print_section("Accuracy by Domain × Condition")
    cross = (
        valid.groupby(["domain", "condition"])["is_correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n"})
    )
    cross["accuracy_%"] = (cross["accuracy"] * 100).round(1)
    pivot = cross["accuracy_%"].unstack("condition").fillna("-")
    print(pivot.to_string())

    # ── Comparison with original ──────────────────────────────────
    original = load_original_results()
    if not original.empty:
        for ref_model in ["gpt-5.2", "gpt-4o"]:
            ref = original[original["model"] == ref_model]
            if ref.empty:
                continue
            print_section(f"Comparison: gpt-5.4-mini vs Original {ref_model}")

            orig_by_cond = (
                ref.groupby("condition")["is_correct"]
                .mean().mul(100).round(1)
                .rename(f"{ref_model}_%")
            )
            repro_by_cond = (
                valid.groupby("condition")["is_correct"]
                .mean().mul(100).round(1)
                .rename("gpt-5.4-mini_%")
            )
            compare_cond = pd.concat([orig_by_cond, repro_by_cond], axis=1).fillna("-")
            compare_cond["delta"] = compare_cond.apply(
                lambda r: f"{r['gpt-5.4-mini_%'] - r[f'{ref_model}_%']:+.1f}"
                if r["gpt-5.4-mini_%"] != "-" else "-", axis=1)
            print("By Condition:")
            print(compare_cond.to_string())

            print()

            orig_by_dom = (
                ref.groupby("domain")["is_correct"]
                .mean().mul(100).round(1)
                .rename(f"{ref_model}_%")
            )
            repro_by_dom = (
                valid.groupby("domain")["is_correct"]
                .mean().mul(100).round(1)
                .rename("gpt-5.4-mini_%")
            )
            compare_dom = pd.concat([orig_by_dom, repro_by_dom], axis=1).fillna("-")
            compare_dom["delta"] = compare_dom.apply(
                lambda r: f"{float(r['gpt-5.4-mini_%']) - float(r[f'{ref_model}_%']):+.1f}"
                if r["gpt-5.4-mini_%"] != "-" and r[f"{ref_model}_%"] != "-" else "-", axis=1)
            print("By Domain:")
            print(compare_dom.to_string())
            print()
    else:
        print("\nSkipping comparison — original results.csv not found.")

    # ── Save merged results ───────────────────────────────────────
    out_path = REPRO_DIR / "results.csv"
    repro.to_csv(out_path, sep="\t", index=False)
    print(f"\nSaved merged results to {out_path}")


if __name__ == "__main__":
    main()
