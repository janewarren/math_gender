#!/usr/bin/env python3
"""
Measure inference time for models to tune timeout and max_workers.

Uses the same call_model() as the main pipeline so timings match production.
Run before adding new models to the full experiment to set config timeouts.

Usage:
  python benchmark_inference_time.py --models deepseek-r1 llama-3.3-70b
  python benchmark_inference_time.py --models deepseek-r1 --n-calls 10 --domain currency
  python benchmark_inference_time.py --input full_results/preprocessed/currency.tsv --n-calls 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root so config and api can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODEL_CONFIGS, DEFAULT_BASE_DIR, PREPROCESSED_SUBDIR, setup_api_keys
from api import call_model, FatalAPIError


def main():
    p = argparse.ArgumentParser(
        description="Benchmark inference time for conversion models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["deepseek-r1", "glm4.7-fp8", "llama-3.3-70b", "mistral-small-24b"],
        help="Model keys from config (default: the four prospective Together models)",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Preprocessed TSV with prompt/domain/answer (default: full_results/preprocessed/currency.tsv)",
    )
    p.add_argument(
        "--n-calls",
        type=int,
        default=5,
        help="Number of API calls per model (default: 5)",
    )
    p.add_argument(
        "--domain",
        type=str,
        default="currency",
        help="Domain name for prompt selection if using default input path",
    )
    args = p.parse_args()

    if args.input is None:
        args.input = DEFAULT_BASE_DIR / PREPROCESSED_SUBDIR / f"{args.domain}.tsv"
    if not args.input.exists():
        print(f"Input file not found: {args.input}")
        sys.exit(1)

    setup_api_keys()
    df = pd.read_csv(args.input, sep="\t", nrows=max(args.n_calls, 50))
    if "prompt" not in df.columns or "domain" not in df.columns:
        df["domain"] = args.domain
    prompts = df["prompt"].fillna("").tolist()
    domains = df["domain"].fillna(args.domain).tolist() if "domain" in df.columns else [args.domain] * len(prompts)
    if len(prompts) < args.n_calls:
        prompts = prompts * (args.n_calls // len(prompts) + 1)
        domains = domains * (args.n_calls // len(domains) + 1)
    prompts = prompts[: args.n_calls]
    domains = domains[: args.n_calls]

    results = []
    for model_name in args.models:
        if model_name not in MODEL_CONFIGS:
            print(f"Unknown model '{model_name}' (skip). Known: {list(MODEL_CONFIGS.keys())}")
            continue
        times = []
        tokens_list = []
        print(f"\n{model_name}: running {args.n_calls} calls...")
        for i in range(args.n_calls):
            try:
                out = call_model(model_name, prompts[i], domains[i])
                t = out.get("call_seconds")
                if t is not None:
                    times.append(t)
                if out.get("reasoning_tokens") is not None:
                    tokens_list.append(out["reasoning_tokens"])
            except FatalAPIError as e:
                print(f"  Fatal error: {e}")
                break
            except Exception as e:
                print(f"  Call {i+1} failed: {e}")
        if not times:
            results.append({
                "model": model_name,
                "n": 0,
                "mean_s": np.nan,
                "p50_s": np.nan,
                "p95_s": np.nan,
                "max_s": np.nan,
                "suggested_timeout": "—",
                "suggested_max_workers": "—",
            })
            continue
        t_arr = np.array(times)
        p95 = float(np.percentile(t_arr, 95))
        suggested_timeout = int(np.ceil(p95 * 2))  # 2× p95 for safety
        suggested_timeout = max(suggested_timeout, 60)
        # Rough max_workers: assume ~2–4 RPS for Together; want latency * RPS ≈ 10–20
        mean_s = float(t_arr.mean())
        suggested_workers = max(10, min(80, int(20 / mean_s))) if mean_s > 0 else 30
        results.append({
            "model": model_name,
            "n": len(times),
            "mean_s": round(t_arr.mean(), 2),
            "p50_s": round(float(np.median(t_arr)), 2),
            "p95_s": round(p95, 2),
            "max_s": round(float(t_arr.max()), 2),
            "suggested_timeout": suggested_timeout,
            "suggested_max_workers": suggested_workers,
        })
        if tokens_list:
            results[-1]["reasoning_tokens_avg"] = round(np.mean(tokens_list), 0)

    if not results:
        print("No results.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Inference time summary (seconds)")
    print("=" * 80)
    out_df = pd.DataFrame(results)
    cols = ["model", "n", "mean_s", "p50_s", "p95_s", "max_s", "suggested_timeout", "suggested_max_workers"]
    if "reasoning_tokens_avg" in out_df.columns:
        cols.insert(cols.index("suggested_timeout"), "reasoning_tokens_avg")
    print(out_df[cols].to_string(index=False))
    print()
    print("Suggested config: set timeout to suggested_timeout (or higher for safety).")
    print("Set max_workers from suggested_max_workers; reduce if you hit rate limits.")
    print("Re-run with --n-calls 10 or more for more stable estimates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
