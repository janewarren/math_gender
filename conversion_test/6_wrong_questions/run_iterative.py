#!/usr/bin/env python3
"""
Iterative wrong-question finder.

Round 1: run gpt-5 on ALL math_only + no_guide prompts.
Round 2+: re-run only the questions that were wrong in the previous round.
Stop after 100 rounds or when every question has been answered correctly.

State is checkpointed after each round so runs can be resumed.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

_INFERENCE_DIR = str(Path(__file__).resolve().parent.parent / "1_run_inference")
if _INFERENCE_DIR not in sys.path:
    sys.path.insert(0, _INFERENCE_DIR)

from extractors import extract_answer, determine_answer_type, calculate_loss
from config import get_system_prompt, RESULT_COLS

import litellm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("iterative")
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ── Config ────────────────────────────────────────────────────────
REPRO_DIR = Path(__file__).resolve().parent
PREPROC_DIR = REPRO_DIR.parent / "full_results" / "preprocessed"
ROUNDS_DIR = REPRO_DIR / "rounds"
STATE_FILE = REPRO_DIR / "state.json"
MASTER_FILE = REPRO_DIR / "master.tsv"

MODEL = "gpt-5"
MAX_ROUNDS = 100
MAX_WORKERS = 50
TOLERANCE_PERCENT = 0.1
TOLERANCE_MINUTES = 1.0


def setup_keys():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)
    if not os.environ.get("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY not found in %s", env_path)
        sys.exit(1)


def call_gpt5(prompt: str, domain: str) -> dict:
    is_timezone = "timezone" in domain
    messages = [
        {"role": "system", "content": get_system_prompt(is_timezone, is_reasoning=True)},
        {"role": "user", "content": prompt},
    ]
    t0 = time.time()
    resp = litellm.completion(
        model=MODEL,
        messages=messages,
        max_tokens=16000,
        reasoning_effort="medium",
        timeout=600,
    )
    call_seconds = time.time() - t0

    msg = resp.choices[0].message
    content = msg.content or ""

    reasoning_tokens = None
    if resp.usage and hasattr(resp.usage, "completion_tokens_details"):
        details = resp.usage.completion_tokens_details
        if details:
            reasoning_tokens = getattr(details, "reasoning_tokens", None)

    if resp.choices[0].finish_reason == "length":
        content += " [TRUNCATED]"

    return {"response": content, "reasoning_tokens": reasoning_tokens, "call_seconds": call_seconds}


def process_row(idx: int, prompt: str, domain: str, correct_answer, row_id: int) -> dict:
    """Process one row: call model -> extract answer -> compute loss."""
    try:
        result = call_gpt5(prompt, domain)
    except Exception as exc:
        return {"row_id": row_id, "raw_response": f"ERROR: {exc}",
                "model_answer": None, "loss": None,
                "reasoning_tokens": None, "call_seconds": None, "is_correct": False}

    raw_response = result["response"]
    model_answer = extract_answer(raw_response, domain, correct_answer, is_reasoning=True)
    answer_type = determine_answer_type(domain, correct_answer)
    loss = calculate_loss(model_answer, correct_answer, answer_type,
                          TOLERANCE_PERCENT, TOLERANCE_MINUTES)
    is_correct = (loss == 0.0) if loss is not None else False

    return {
        "row_id": row_id,
        "raw_response": raw_response,
        "model_answer": model_answer,
        "loss": loss,
        "reasoning_tokens": result["reasoning_tokens"],
        "call_seconds": result["call_seconds"],
        "is_correct": is_correct,
    }


def load_master() -> pd.DataFrame:
    """Load or build the master question list from preprocessed files."""
    if MASTER_FILE.exists():
        log.info("Loading existing master file (%s)", MASTER_FILE)
        return pd.read_csv(MASTER_FILE, sep="\t")

    log.info("Building master question list from preprocessed files...")
    frames = []
    for f in sorted(PREPROC_DIR.glob("*_no_guide.tsv")):
        df = pd.read_csv(f, sep="\t")
        df["condition"] = "no_guide"
        df["source_file"] = f.name
        frames.append(df)

    master = pd.concat(frames, ignore_index=True)
    n_raw = len(master)

    # Keep only rows with no distractor
    master = master[
        master["distractor"].isna() | master["distractor"].astype(str).isin(["null", "nan"])
    ].copy()
    log.info("Filtered to no-distractor: %d → %d rows", n_raw, len(master))

    n_before = len(master)
    master = master.drop_duplicates(subset=["prompt", "answer"]).reset_index(drop=True)
    log.info("Deduplicated: %d rows → %d unique (prompt, answer) pairs", n_before, len(master))

    master["row_id"] = range(len(master))
    master["first_correct_round"] = -1  # -1 = never correct yet

    MASTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(MASTER_FILE, sep="\t", index=False)
    log.info("Master file: %d rows from %d files", len(master), len(frames))
    return master


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"completed_rounds": 0, "remaining_ids": None}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def run_round(master: pd.DataFrame, pending_ids: list[int], round_num: int) -> list[int]:
    """Run one round of inference on pending questions. Returns IDs still wrong."""
    pending = master[master["row_id"].isin(set(pending_ids))].copy()
    log.info("Round %d: %d questions to process", round_num, len(pending))

    results = {}
    t0 = time.time()
    checkpoint_interval = 5000

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        for _, row in pending.iterrows():
            fut = pool.submit(
                process_row,
                row.name, str(row["prompt"]), str(row["domain"]),
                row["answer"], int(row["row_id"]),
            )
            futures[fut] = int(row["row_id"])

        done_count = 0
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f"  Round {round_num}", leave=False):
            rid = futures[fut]
            try:
                res = fut.result()
                results[rid] = res
            except Exception as exc:
                results[rid] = {
                    "row_id": rid, "raw_response": f"ERROR: {exc}",
                    "model_answer": None, "loss": None,
                    "reasoning_tokens": None, "call_seconds": None, "is_correct": False,
                }
            done_count += 1

    elapsed = time.time() - t0

    round_df = pd.DataFrame(results.values())
    round_file = ROUNDS_DIR / f"round_{round_num:03d}.tsv"
    round_df.to_csv(round_file, sep="\t", index=False)

    n_correct = round_df["is_correct"].sum()
    n_errors = round_df["raw_response"].astype(str).str.startswith("ERROR:").sum()
    still_wrong_ids = round_df[~round_df["is_correct"]]["row_id"].tolist()

    # Update master: mark newly correct questions
    newly_correct = round_df[round_df["is_correct"]]["row_id"].values
    master.loc[master["row_id"].isin(newly_correct), "first_correct_round"] = round_num
    master.to_csv(MASTER_FILE, sep="\t", index=False)

    call_times = round_df["call_seconds"].dropna()
    log.info("  Round %d done in %.0fs: %d correct, %d still wrong, %d errors  "
             "(mean call: %.1fs)",
             round_num, elapsed, n_correct, len(still_wrong_ids), n_errors,
             call_times.mean() if len(call_times) else 0)

    return still_wrong_ids


def main():
    setup_keys()
    ROUNDS_DIR.mkdir(parents=True, exist_ok=True)

    master = load_master()
    state = load_state()

    start_round = state["completed_rounds"] + 1
    if state["remaining_ids"] is not None:
        pending_ids = state["remaining_ids"]
    else:
        pending_ids = master["row_id"].tolist()

    log.info("Starting from round %d with %d pending questions (of %d total)",
             start_round, len(pending_ids), len(master))

    for round_num in range(start_round, MAX_ROUNDS + 1):
        if not pending_ids:
            log.info("All questions answered correctly! Stopping at round %d.", round_num - 1)
            break

        still_wrong = run_round(master, pending_ids, round_num)

        state["completed_rounds"] = round_num
        state["remaining_ids"] = still_wrong
        save_state(state)

        pending_ids = still_wrong
        log.info("  → %d questions remain after round %d\n", len(pending_ids), round_num)

    # Final summary
    never_correct = master[master["first_correct_round"] == -1]
    log.info("=" * 60)
    log.info("FINAL: %d / %d questions were NEVER answered correctly across %d rounds",
             len(never_correct), len(master), state["completed_rounds"])
    log.info("=" * 60)

    never_correct.to_csv(REPRO_DIR / "always_wrong.tsv", sep="\t", index=False)
    log.info("Saved always_wrong.tsv (%d rows)", len(never_correct))


if __name__ == "__main__":
    main()
