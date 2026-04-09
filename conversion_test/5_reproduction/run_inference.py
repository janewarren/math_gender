#!/usr/bin/env python3
"""
Step 2: Run gpt-5.4-mini inference on the sampled dataset.

Imports the *exact* extraction/scoring logic from the original pipeline
(1_run_inference/) so results are directly comparable.
"""

import os
import sys
import time
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Wire up the original inference code so we reuse extractors + system prompts
_INFERENCE_DIR = str(Path(__file__).resolve().parent.parent / "1_run_inference")
if _INFERENCE_DIR not in sys.path:
    sys.path.insert(0, _INFERENCE_DIR)

from extractors import extract_answer, determine_answer_type, calculate_loss
from config import get_system_prompt, RESULT_COLS

import litellm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("reproduction")

litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ── Paths ─────────────────────────────────────────────────────────
REPRO_DIR = Path(__file__).resolve().parent
PREPROC_DIR = REPRO_DIR / "preprocessed"
RESULTS_DIR = REPRO_DIR / "results" / "gpt-5.4-mini"

MODEL_NAME = "gpt-5.4-mini"
TOLERANCE_PERCENT = 0.1
TOLERANCE_MINUTES = 1.0


def setup_keys():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        log.error("OPENAI_API_KEY not found in %s", env_path)
        sys.exit(1)
    log.info("Loaded API key from %s", env_path)


def call_model(prompt: str, domain: str) -> dict:
    """Single synchronous call to gpt-5.4-mini via LiteLLM."""
    is_timezone = "timezone" in domain
    messages = [
        {"role": "system", "content": get_system_prompt(is_timezone, is_reasoning=True)},
        {"role": "user", "content": prompt},
    ]
    t0 = time.time()
    resp = litellm.completion(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=16000,
        timeout=120,
    )
    call_seconds = time.time() - t0

    msg = resp.choices[0].message
    content = msg.content or ""

    # Extract reasoning tokens if available
    reasoning_tokens = None
    if resp.usage and hasattr(resp.usage, "completion_tokens_details"):
        details = resp.usage.completion_tokens_details
        if details:
            reasoning_tokens = getattr(details, "reasoning_tokens", None)

    if resp.choices[0].finish_reason == "length":
        content += " [TRUNCATED]"

    return {"response": content, "reasoning_tokens": reasoning_tokens, "call_seconds": call_seconds}


def process_file(input_file: Path, output_file: Path):
    """Run inference on every row of one preprocessed TSV."""
    df = pd.read_csv(input_file, sep="\t")
    for col in RESULT_COLS:
        df[col] = None

    # Resume support: skip rows that already have results
    if output_file.exists():
        try:
            ckpt = pd.read_csv(output_file, sep="\t")
            if "raw_response" in ckpt.columns and len(ckpt) == len(df):
                done = ckpt["raw_response"].notna() & ~ckpt["raw_response"].astype(str).isin(["null", "nan", ""])
                if done.all():
                    log.info("  Already complete — skipping %s", input_file.name)
                    return
                for col in RESULT_COLS:
                    if col in ckpt.columns:
                        df[col] = ckpt[col]
                pending = [i for i in range(len(df)) if not done.iloc[i]]
                log.info("  Resuming: %d/%d done, %d pending", done.sum(), len(df), len(pending))
            else:
                pending = list(range(len(df)))
        except Exception:
            pending = list(range(len(df)))
    else:
        pending = list(range(len(df)))

    for idx in pending:
        row = df.iloc[idx]
        prompt = str(row["prompt"])
        domain = str(row["domain"])
        correct_answer = row["answer"]

        try:
            result = call_model(prompt, domain)
        except Exception as exc:
            log.warning("  Row %d failed: %s", idx, exc)
            df.at[idx, "raw_response"] = f"ERROR: {exc}"
            continue

        raw_response = result["response"]
        model_answer = extract_answer(raw_response, domain, correct_answer, is_reasoning=False)
        answer_type = determine_answer_type(domain, correct_answer)
        loss = calculate_loss(model_answer, correct_answer, answer_type,
                              TOLERANCE_PERCENT, TOLERANCE_MINUTES)

        df.at[idx, "raw_response"] = raw_response
        df.at[idx, "model_answer"] = model_answer
        df.at[idx, "loss"] = loss
        df.at[idx, "reasoning_tokens"] = result["reasoning_tokens"]
        df.at[idx, "call_seconds"] = result["call_seconds"]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, sep="\t", index=False, na_rep="null")
    valid = pd.to_numeric(df["loss"], errors="coerce").dropna()
    if len(valid):
        correct = (valid == 0).sum()
        log.info("  → %d/%d correct (%.1f%%)", correct, len(valid), correct / len(valid) * 100)


def main():
    setup_keys()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tsv_files = sorted(PREPROC_DIR.glob("*.tsv"))
    if not tsv_files:
        log.error("No preprocessed TSV files found in %s. Run sample_data.py first.", PREPROC_DIR)
        sys.exit(1)

    log.info("Found %d TSV files to process.", len(tsv_files))
    total_rows = 0

    for i, tsv in enumerate(tsv_files, 1):
        out_name = tsv.stem + "_converted.tsv"
        output_file = RESULTS_DIR / out_name
        n_rows = len(pd.read_csv(tsv, sep="\t"))
        total_rows += n_rows
        log.info("[%d/%d] %s  (%d rows)", i, len(tsv_files), tsv.name, n_rows)
        process_file(tsv, output_file)

    log.info("Done. Processed %d total rows. Results in %s", total_rows, RESULTS_DIR)


if __name__ == "__main__":
    main()
