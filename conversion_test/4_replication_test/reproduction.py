#!/usr/bin/env python3
"""
Replication / consistency test.

For each prompt in replication_sample.tsv, queries every model N_RUNS times
and records each answer separately. The output lets us measure answer
variability across repeated runs on identical prompts.

Usage:
  python reproduction.py                               # all models, 5 runs
  python reproduction.py --models gpt-4o gpt-5.2       # specific models
  python reproduction.py --n-runs 3                     # fewer repetitions
  python reproduction.py --list-tasks                   # show progress
"""

import sys
import time
import signal
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_INFERENCE_DIR = _SCRIPT_DIR.parent / "1_run_inference"
sys.path.insert(0, str(_INFERENCE_DIR))

from config import MODEL_CONFIGS, RESULT_COLS, setup_api_keys
from extractors import extract_answer, determine_answer_type, calculate_loss
from api import call_model, FatalAPIError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("replication")

_shutdown = False

def _signal_handler(signum, _frame):
    global _shutdown
    _shutdown = True
    log.warning("Signal %s received — saving checkpoint and stopping.", signum)

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


SAMPLE_FILE = _SCRIPT_DIR / "replication_sample.tsv"
OUTPUT_DIR = _SCRIPT_DIR / "replication_results"
DEFAULT_N_RUNS = 5
TOLERANCE_PERCENT = 0.1
TOLERANCE_MINUTES = 1.0
CHECKPOINT_EVERY = 100


def process_single(prompt: str, domain: str, correct_answer, model_name: str) -> dict:
    """Call a model once and return extracted results."""
    config = MODEL_CONFIGS.get(model_name, {})
    try:
        result = call_model(model_name, prompt, domain)
    except FatalAPIError:
        raise
    except Exception as exc:
        return {c: (f"ERROR: {exc}" if c == "raw_response" else None) for c in RESULT_COLS}

    raw_response = result["response"]
    model_answer = extract_answer(
        raw_response, domain, correct_answer, config.get("reasoning", False)
    )
    answer_type = determine_answer_type(domain, correct_answer)
    loss = calculate_loss(
        model_answer, correct_answer, answer_type,
        TOLERANCE_PERCENT, TOLERANCE_MINUTES,
    )
    return {
        "raw_response": raw_response,
        "model_answer": model_answer,
        "loss": loss,
        "reasoning_tokens": result["reasoning_tokens"],
        "call_seconds": result.get("call_seconds"),
    }


def run_model(model_name: str, sample: pd.DataFrame, n_runs: int):
    """Run all prompts × n_runs for one model, checkpointing as we go."""
    out_file = OUTPUT_DIR / f"{model_name}.tsv"
    max_workers = MODEL_CONFIGS.get(model_name, {}).get("max_workers", 10)

    # Build the full result frame: one row per (prompt_idx, run_id)
    rows = []
    for idx in range(len(sample)):
        for run_id in range(1, n_runs + 1):
            r = sample.iloc[idx].to_dict()
            r["prompt_idx"] = idx
            r["run_id"] = run_id
            for col in RESULT_COLS:
                r[col] = None
            rows.append(r)
    df = pd.DataFrame(rows)

    # Resume from checkpoint
    pending = list(range(len(df)))
    if out_file.exists():
        try:
            ckpt = pd.read_csv(out_file, sep="\t")
            if "raw_response" in ckpt.columns and len(ckpt) == len(df):
                for col in RESULT_COLS:
                    if col in ckpt.columns:
                        df[col] = ckpt[col]
                done_mask = (
                    ckpt["raw_response"].notna()
                    & ~ckpt["raw_response"].astype(str).isin(["null", "nan", ""])
                    & ~ckpt["raw_response"].astype(str).str.startswith("ERROR:")
                )
                pending = [i for i in range(len(df)) if not done_mask.iloc[i]]
                n_done = len(df) - len(pending)
                log.info("  Resuming %s: %d / %d done, %d pending.",
                         model_name, n_done, len(df), len(pending))
        except Exception as exc:
            log.warning("  Checkpoint unreadable (%s). Starting fresh.", exc)

    if not pending:
        log.info("  %s already complete (%d rows). Skipping.", model_name, len(df))
        return

    all_times = []
    task_start = time.time()

    for batch_off in range(0, len(pending), CHECKPOINT_EVERY):
        if _shutdown:
            log.warning("  Shutdown — saving checkpoint.")
            _save(df, out_file)
            return

        batch = pending[batch_off : batch_off + CHECKPOINT_EVERY]

        model_rpm = MODEL_CONFIGS.get(model_name, {}).get("max_rpm")
        if model_rpm and max_workers <= 2:
            secs_per_req = 60.0 / model_rpm + 45
            batch_timeout = max(int(len(batch) * secs_per_req * 1.5), 600)
        else:
            batch_timeout = max(len(batch) * 10, 600)

        results = {}
        fatal = False

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    process_single,
                    str(df.iloc[i]["prompt"]),
                    str(df.iloc[i]["domain"]),
                    df.iloc[i]["answer"],
                    model_name,
                ): i
                for i in batch
            }
            done_so_far = len(df) - len(pending) + batch_off + len(batch)
            desc = f"  {model_name} [{done_so_far}/{len(df)}]"
            try:
                for fut in tqdm(as_completed(futures, timeout=batch_timeout),
                                total=len(futures), desc=desc, leave=False):
                    try:
                        res = fut.result()
                        results[futures[fut]] = res
                    except FatalAPIError as exc:
                        log.error("FATAL: %s", exc)
                        fatal = True
                        break
                    except Exception as exc:
                        results[futures[fut]] = {
                            c: (f"ERROR: {exc}" if c == "raw_response" else None)
                            for c in RESULT_COLS
                        }
            except TimeoutError:
                log.warning("  Batch timed out after %ds.", batch_timeout)

            if fatal:
                for fut in futures:
                    if futures[fut] not in results:
                        fut.cancel()

        for idx, res in results.items():
            for col in RESULT_COLS:
                df.at[idx, col] = res[col]

        batch_times = [r["call_seconds"] for r in results.values() if r.get("call_seconds")]
        all_times.extend(batch_times)
        _save(df, out_file)

        if fatal:
            log.error("Stopping %s due to fatal API error.", model_name)
            raise SystemExit(1)

    elapsed = time.time() - task_start
    valid = pd.to_numeric(df["loss"], errors="coerce").dropna()
    if len(valid):
        correct = (valid == 0).sum()
        log.info("  %s done — %d/%d correct (%.1f%%) in %s",
                 model_name, correct, len(valid), correct / len(valid) * 100,
                 _fmt_dur(elapsed))


def _save(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = df.copy()
    if "distractor" in tmp.columns:
        tmp["distractor"] = tmp["distractor"].fillna("null")
    tmp.to_csv(path, sep="\t", index=False, na_rep="null")


def _fmt_dur(s):
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{sec:02d}s" if h else (f"{m}m{sec:02d}s" if m else f"{sec}s")


def main():
    p = argparse.ArgumentParser(description="Replication consistency test.")
    p.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS.keys()))
    p.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS,
                   help=f"Times to repeat each prompt (default {DEFAULT_N_RUNS})")
    p.add_argument("--list-tasks", action="store_true",
                   help="Show progress without running anything")
    args = p.parse_args()

    setup_api_keys()

    if not SAMPLE_FILE.exists():
        log.error("Sample file not found: %s\nRun sample.ipynb first.", SAMPLE_FILE)
        raise SystemExit(1)

    sample = pd.read_csv(SAMPLE_FILE, sep="\t")
    log.info("Loaded %d prompts from %s", len(sample), SAMPLE_FILE.name)
    total_calls = len(sample) * args.n_runs
    log.info("Each model will make %d calls (%d prompts × %d runs).",
             total_calls, len(sample), args.n_runs)

    models = [m for m in args.models if m in MODEL_CONFIGS]
    if not models:
        log.error("No valid models specified.")
        raise SystemExit(1)

    if args.list_tasks:
        for model in models:
            out_file = OUTPUT_DIR / f"{model}.tsv"
            if out_file.exists():
                try:
                    ckpt = pd.read_csv(out_file, sep="\t")
                    done = ckpt["raw_response"].notna() & ~ckpt["raw_response"].astype(str).isin(["null","nan",""])
                    print(f"  {model:25s}  {done.sum():>6d} / {len(ckpt):>6d} done")
                except Exception:
                    print(f"  {model:25s}  checkpoint unreadable")
            else:
                print(f"  {model:25s}  not started")
        return

    for model in models:
        if _shutdown:
            break
        log.info("\n=== %s ===", model)
        run_model(model, sample, args.n_runs)

    log.info("\nAll done.")


if __name__ == "__main__":
    main()
