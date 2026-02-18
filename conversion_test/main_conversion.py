#!/usr/bin/env python3
"""
Parallelized conversion inference script.

Usage:
  python main_conversion.py                             # run all
  python main_conversion.py --models gpt-4o gpt-5.2     # filter models
  python main_conversion.py --conditions regular         # filter conditions
  python main_conversion.py --list-tasks                 # show status
"""

import time
import signal
import argparse
import logging
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from tqdm import tqdm

from config import (
    MODEL_CONFIGS, CONDITIONS, DEFAULT_BASE_DIR,
    PREPROCESSED_SUBDIR, RESULT_COLS, setup_api_keys,
)
from extractors import extract_answer, determine_answer_type, calculate_loss
from api import call_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("conversion")

# Graceful shutdown on SIGTERM / SIGINT
_shutdown = False

def _signal_handler(signum, _frame):
    global _shutdown
    _shutdown = True
    log.warning("Signal %s received — saving checkpoint and stopping.", signum)

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


def _error_result(msg: str) -> dict:
    """Build a result dict for a failed row."""
    return {c: (f"ERROR: {msg}" if c == "raw_response" else None) for c in RESULT_COLS}


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def _log_timing(label: str, times: list[float], elapsed: float):
    """Log percentile stats for a list of call durations."""
    if not times:
        return
    t = np.array(times)
    log.info("  %s: mean=%.1fs  p50=%.1fs  p95=%.1fs  min=%.1fs  max=%.1fs  | %s",
             label, t.mean(), np.median(t), np.percentile(t, 95), t.min(), t.max(),
             _fmt_duration(elapsed))


# ── Row Processing (runs in worker threads) ──────────────────────

def process_row(prompt, domain, correct_answer, model_name, tolerance_percent, tolerance_minutes) -> dict:
    """Process one row: call model → extract answer → compute loss."""
    config = MODEL_CONFIGS.get(model_name, {})
    try:
        result = call_model(model_name, prompt, domain)
    except Exception as exc:
        return _error_result(str(exc))

    raw_response = result["response"]
    model_answer = extract_answer(raw_response, domain, correct_answer, config.get("reasoning", False))
    answer_type = determine_answer_type(domain, correct_answer)
    loss = calculate_loss(model_answer, correct_answer, answer_type, tolerance_percent, tolerance_minutes)

    return {
        "raw_response": raw_response,
        "model_answer": model_answer,
        "loss": loss,
        "reasoning_tokens": result["reasoning_tokens"],
        "call_seconds": result.get("call_seconds"),
    }


# ── Checkpointing ────────────────────────────────────────────────

def save_checkpoint(df: pd.DataFrame, output_file: Path):
    tmp = df.copy()
    if "distractor" in tmp.columns:
        tmp["distractor"] = tmp["distractor"].fillna("null")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp.to_csv(output_file, sep="\t", index=False, na_rep="null")


def load_checkpoint(output_file: Path, df: pd.DataFrame) -> list[int]:
    """Restore results from existing output file.

    Returns list of row indices that still need processing (supports sparse checkpoints).
    """
    if not output_file.exists():
        return list(range(len(df)))
    try:
        ckpt = pd.read_csv(output_file, sep="\t")
        if "raw_response" not in ckpt.columns:
            return list(range(len(df)))
        for col in RESULT_COLS:
            if col in ckpt.columns:
                df[col] = ckpt[col]
        done_mask = ckpt["raw_response"].notna()
        done_mask &= ~ckpt["raw_response"].astype(str).isin(["null", "nan", ""])
        done_mask &= ~ckpt["raw_response"].astype(str).str.startswith("ERROR:")
        n_done = int(done_mask.sum())
        pending = [i for i in range(len(df)) if i >= len(ckpt) or not done_mask.iloc[i]]
        log.info("  Resuming: %d / %d rows done, %d pending.", n_done, len(df), len(pending))
        return pending
    except Exception as exc:
        log.warning("  Checkpoint unreadable (%s). Starting fresh.", exc)
        return list(range(len(df)))


# ── Task Processing ──────────────────────────────────────────────

def process_task(*, model_name, input_file, output_file,
                 max_workers=10, checkpoint_every=200,
                 tolerance_percent=0.1, tolerance_minutes=1.0) -> bool:
    """Run inference for one model × domain × condition task."""
    df = pd.read_csv(input_file, sep="\t")
    for col in RESULT_COLS:
        df[col] = None

    pending = load_checkpoint(output_file, df)
    if not pending:
        log.info("  Already complete (%d rows). Skipping.", len(df))
        return True

    all_times: list[float] = []
    task_start = time.time()

    for batch_off in range(0, len(pending), checkpoint_every):
        if _shutdown:
            log.warning("  Shutdown — saving checkpoint.")
            save_checkpoint(df, output_file)
            return True

        batch = pending[batch_off : batch_off + checkpoint_every]
        results: dict[int, dict] = {}

        # Wall-clock timeout for entire batch (generous upper bound)
        batch_timeout = max(len(batch) * 10, 600)

        # ── Stall detection: abort batch if no row completes for STALL_LIMIT ──
        STALL_LIMIT = 600  # 10 minutes with zero progress → API is likely down
        last_progress = time.time()
        consecutive_errors = 0
        CIRCUIT_BREAKER_THRESHOLD = 10  # consecutive errors before cooldown

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(process_row, str(df.iloc[i]["prompt"]), str(df.iloc[i]["domain"]),
                            df.iloc[i]["answer"], model_name, tolerance_percent, tolerance_minutes): i
                for i in batch
            }
            n_done = len(df) - len(pending) + batch_off + len(batch)
            desc = f"  [{n_done}/{len(df)}]"
            done_iter = as_completed(futures, timeout=batch_timeout)
            stalled = False
            try:
                for fut in tqdm(done_iter, total=len(futures), desc=desc, leave=False):
                    try:
                        res = fut.result()
                        results[futures[fut]] = res
                        is_error = str(res.get("raw_response", "")).startswith("ERROR:")
                        if is_error:
                            consecutive_errors += 1
                        else:
                            consecutive_errors = 0
                            last_progress = time.time()
                    except Exception as exc:
                        results[futures[fut]] = _error_result(str(exc))
                        consecutive_errors += 1

                    # Circuit breaker: too many consecutive errors → API is down
                    if consecutive_errors >= CIRCUIT_BREAKER_THRESHOLD:
                        log.warning("  Circuit breaker: %d consecutive errors. Pausing 5min.",
                                    consecutive_errors)
                        time.sleep(300)
                        consecutive_errors = 0
                        last_progress = time.time()  # reset stall timer after cooldown

                    # Stall detection: no successful row for too long
                    if time.time() - last_progress > STALL_LIMIT:
                        log.warning("  Stall detected: no progress for %ds. Aborting batch.",
                                    STALL_LIMIT)
                        stalled = True
                        break

            except TimeoutError:
                log.warning("  Batch timed out after %ds — %d/%d rows completed.",
                            batch_timeout, len(results), len(batch))

            # Mark any unfinished rows as errors (don't save them → they stay pending)
            if stalled:
                for fut, idx in futures.items():
                    if idx not in results:
                        fut.cancel()

        for idx, res in results.items():
            for col in RESULT_COLS:
                df.at[idx, col] = res[col]

        batch_times = [r["call_seconds"] for r in results.values() if r.get("call_seconds")]
        all_times.extend(batch_times)
        _log_timing(f"[{n_done}/{len(df)}]", batch_times, time.time() - task_start)
        save_checkpoint(df, output_file)

        # If stalled, return False so the task-retry logic can handle it
        if stalled:
            log.warning("  Task paused due to stall. Will retry later.")
            return False

    # Final summary
    valid = pd.to_numeric(df["loss"], errors="coerce").dropna()
    if len(valid):
        correct = (valid == 0).sum()
        log.info("  Done — %d/%d correct (%.1f%%), mean loss %.4f",
                 correct, len(valid), correct / len(valid) * 100, valid.mean())
    _log_timing("Total", all_times, time.time() - task_start)
    return True


def run_task_with_retry(max_retries=3, **kwargs) -> bool:
    for attempt in range(1, max_retries + 1):
        try:
            ok = process_task(**kwargs)
            if ok:
                return True
            # process_task returned False (stall) — wait longer before retry
            if attempt < max_retries:
                wait = min(120 * 2 ** (attempt - 1), 600)
                log.warning("  Task stalled (attempt %d/%d) — retry in %ds", attempt, max_retries, wait)
                time.sleep(wait)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if attempt == max_retries:
                log.error("  Failed after %d attempts: %s", max_retries, exc)
                return False
            wait = min(30 * 2 ** (attempt - 1), 300)
            log.warning("  Attempt %d/%d failed: %s — retry in %ds", attempt, max_retries, exc, wait)
            time.sleep(wait)
    return False


# ── Task Discovery ───────────────────────────────────────────────

def discover_tasks(base_dir: Path, models: list[str], conditions: list[str],
                   domains: Optional[list[str]] = None) -> list[dict]:
    """Build the full task list from preprocessed directory."""
    preproc = base_dir / PREPROCESSED_SUBDIR

    if domains is None:
        names = set()
        for f in preproc.glob("*.tsv"):
            name = f.stem
            for suf in ("_no_guide", "_math_only"):
                name = name.replace(suf, "")
            names.add(name)
        domains = sorted(names)

    tasks = []
    for model in models:
        if model not in MODEL_CONFIGS:
            log.warning("Unknown model '%s' — skipping.", model)
            continue
        for domain in domains:
            for cond in conditions:
                cfg = CONDITIONS[cond]
                inp = preproc / f"{domain}{cfg['suffix']}.tsv"
                if not inp.exists():
                    continue
                out = base_dir / cfg["output_dir"] / model / f"{domain}{cfg['suffix']}_converted.tsv"

                status, processed = "pending", 0
                if out.exists():
                    try:
                        n_in = len(pd.read_csv(inp, sep="\t", usecols=["domain"]))
                        done = pd.read_csv(out, sep="\t", usecols=["raw_response"])["raw_response"]
                        processed = int((done.notna() & (done.astype(str) != "null")).sum())
                        status = "complete" if processed >= n_in else ("partial" if processed else "pending")
                    except Exception:
                        pass

                tasks.append(dict(model=model, domain=domain, condition=cond,
                                  input_file=inp, output_file=out,
                                  status=status, processed=processed))
    return tasks


# ── CLI ──────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Run conversion inference.", formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    p.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS.keys()))
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS.keys()), choices=list(CONDITIONS.keys()))
    p.add_argument("--domains", nargs="+", default=None)
    p.add_argument("--max-workers", type=int, default=None, help="Threads per task (default: auto per model)")
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--max-task-retries", type=int, default=3)
    p.add_argument("--tolerance-percent", type=float, default=0.1)
    p.add_argument("--tolerance-minutes", type=float, default=1.0)
    p.add_argument("--list-tasks", action="store_true")
    args = p.parse_args()

    setup_api_keys()
    tasks = discover_tasks(args.base_dir, args.models, args.conditions, args.domains)

    by_status = {s: sum(1 for t in tasks if t["status"] == s) for s in ("complete", "partial", "pending")}
    log.info("Discovered %d tasks: %d complete, %d partial, %d pending.",
             len(tasks), by_status["complete"], by_status["partial"], by_status["pending"])

    if args.list_tasks:
        for t in tasks:
            print(f"  {t['status']:8s}  {t['model']:25s}  {t['condition']:10s}  {t['domain']:40s}  ({t['processed']} rows)")
        return

    work = [t for t in tasks if t["status"] != "complete"]
    if not work:
        log.info("All tasks complete!")
        return

    log.info("Processing %d tasks.", len(work))
    succeeded = failed = 0

    for i, task in enumerate(work, 1):
        if _shutdown:
            log.warning("Shutdown after %d/%d tasks.", i - 1, len(work))
            break

        workers = args.max_workers or MODEL_CONFIGS.get(task["model"], {}).get("max_workers", 10)
        log.info("\n[%d/%d] %s / %s / %s  (status: %s, workers: %d)",
                 i, len(work), task["model"], task["domain"], task["condition"], task["status"], workers)

        ok = run_task_with_retry(
            max_retries=args.max_task_retries, model_name=task["model"],
            input_file=task["input_file"], output_file=task["output_file"],
            max_workers=workers, checkpoint_every=args.checkpoint_every,
            tolerance_percent=args.tolerance_percent, tolerance_minutes=args.tolerance_minutes,
        )
        succeeded += ok
        failed += not ok

    log.info("\nDone: %d succeeded, %d failed, %d already complete.", succeeded, failed, by_status["complete"])


if __name__ == "__main__":
    main()
