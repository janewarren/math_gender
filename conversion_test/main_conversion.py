#!/usr/bin/env python3
"""
Parallelized conversion inference script.

Uses LiteLLM for unified API calls (OpenAI + Together AI) and
ThreadPoolExecutor for parallel row processing within each task.

Features:
  - Discovers all model × domain × condition tasks automatically
  - Resumes from per-file checkpoints (compatible with old output files)
  - Exponential-backoff retry via tenacity for transient API errors
  - Thread-safe rate limiter per provider
  - Auto-restart failed tasks up to N times
  - Graceful shutdown on SIGTERM / SIGINT (saves checkpoint)

Usage:
  # Run everything (all models, all domains, all conditions)
  python main_conversion.py

  # Filter to specific models / conditions / domains
  python main_conversion.py --models gpt-4o gpt-5.2
  python main_conversion.py --conditions regular no_guide
  python main_conversion.py --domains cooking temperature

  # Tune concurrency
  python main_conversion.py --max-workers 20 --checkpoint-every 100
"""

import os
import sys
import re
import time
import signal
import argparse
import logging
from pathlib import Path
from typing import Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd
import numpy as np
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import litellm

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("conversion")

# Suppress noisy litellm debug output
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ── Graceful Shutdown ─────────────────────────────────────────────
_shutdown = False


def _signal_handler(signum, _frame):
    global _shutdown
    _shutdown = True
    log.warning("Signal %s received — will save checkpoint and stop after current batch.", signum)


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════

DEFAULT_BASE_DIR = Path("full_results")
PREPROCESSED_SUBDIR = "preprocessed"

# Model registry — each entry carries everything LiteLLM needs.
# "litellm_model" uses the provider prefix that LiteLLM expects.
MODEL_CONFIGS = {
    # ── Standard models ──────────────────────────────────────────
    "gpt-4o": {
        "litellm_model": "gpt-4o",
        "reasoning": False,
    },
    "qwen-coder": {
        "litellm_model": "together_ai/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        "reasoning": False,
    },
    "llama-4": {
        "litellm_model": "together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "reasoning": False,
    },
    # ── Reasoning / CoT models ───────────────────────────────────
    "gpt-5.2": {
        "litellm_model": "gpt-5.2",
        "reasoning": True,
        "extra_params": {"reasoning_effort": "medium"},
    },
    "deepseek-v3.1": {
        "litellm_model": "together_ai/deepseek-ai/DeepSeek-V3.1",
        "reasoning": True,
        "extra_body": {"thinking": {"type": "enabled"}},
    },
    "qwen3-235b-thinking": {
        "litellm_model": "together_ai/Qwen/Qwen3-235B-A22B-Thinking-2507",
        "reasoning": True,
        "extra_body": {"enable_thinking": True},
    },
    "qwen3-next-thinking": {
        "litellm_model": "together_ai/Qwen/Qwen3-Next-80B-A3B-Thinking",
        "reasoning": True,
        "extra_body": {"enable_thinking": True},
    },
}

# Condition → (input-file suffix, output subdirectory)
CONDITIONS = {
    "regular":   {"suffix": "",            "output_dir": "results"},
    "no_guide":  {"suffix": "_no_guide",   "output_dir": "results_no_guide"},
    "math_only": {"suffix": "_math_only",  "output_dir": "results_math_only"},
}

# ══════════════════════════════════════════════════════════════════
#  API KEY SETUP
# ══════════════════════════════════════════════════════════════════

def _load_key_file(path: str) -> str:
    with open(path) as f:
        return f.read().strip()


def setup_api_keys():
    """Ensure LiteLLM can find API keys (env vars take priority over files).

    LiteLLM expects OPENAI_API_KEY and TOGETHER_AI_API_KEY.
    """
    # OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        try:
            os.environ["OPENAI_API_KEY"] = _load_key_file("openai_key.txt")
        except FileNotFoundError:
            log.warning("No OPENAI_API_KEY env var or openai_key.txt — OpenAI models will fail.")

    # Together AI — litellm checks TOGETHER_AI_API_KEY
    together_key = (
        os.environ.get("TOGETHER_AI_API_KEY")
        or os.environ.get("TOGETHERAI_API_KEY")
        or os.environ.get("TOGETHER_API_KEY")
    )
    if not together_key:
        try:
            together_key = _load_key_file("together_ai_key.txt")
        except FileNotFoundError:
            log.warning("No Together AI API key found — Together models will fail.")
    if together_key:
        os.environ["TOGETHER_AI_API_KEY"] = together_key

# ══════════════════════════════════════════════════════════════════
#  THREAD-SAFE RATE LIMITER
# ══════════════════════════════════════════════════════════════════

class RateLimiter:
    """Sliding-window rate limiter (thread-safe)."""

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self._timestamps: list[float] = []
        self._lock = Lock()

    def acquire(self):
        while True:
            with self._lock:
                now = time.time()
                self._timestamps = [t for t in self._timestamps if now - t < 60]
                if len(self._timestamps) < self.max_per_minute:
                    self._timestamps.append(now)
                    return  # slot acquired
                sleep_for = 60.0 - (now - self._timestamps[0]) + 0.1
            # sleep *outside* the lock so other threads aren't blocked
            time.sleep(max(sleep_for, 0.05))


# One limiter per provider
_rate_limiters = {
    "openai":   RateLimiter(max_per_minute=500),
    "together": RateLimiter(max_per_minute=200),
}

# ══════════════════════════════════════════════════════════════════
#  SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════════

def get_system_prompt(is_timezone: bool, is_reasoning: bool) -> str:
    """Return the system prompt appropriate for the domain and model type."""
    if is_reasoning:
        if is_timezone:
            return (
                "You are a precise timezone conversion expert. "
                "Provide your final answer within <answer> and </answer> tags in the same "
                "format as the input (e.g., <answer>1AM</answer>, <answer>3:49PM</answer>)."
            )
        return (
            "You are a precise conversion expert. "
            "Provide your final answer within <answer> and </answer> tags. "
            "For numerical answers, provide only the number (e.g., <answer>42.5</answer>). "
            "For size answers, provide only the size (e.g., <answer>M</answer> or <answer>32</answer>)."
        )
    else:
        if is_timezone:
            return (
                "You are a precise timezone conversion expert. Provide the time in the same "
                "format as the input (e.g., 1AM, 3:49PM), within <answer> and </answer> tags "
                "(e.g., <answer>1AM</answer>, <answer>3:49PM</answer>)."
            )
        return (
            "You are a precise unit conversion expert. Provide only the numerical answer "
            "with up to 4 decimal places, within <answer> and </answer> tags "
            "(e.g., <answer>42.5</answer>). "
        )

# ══════════════════════════════════════════════════════════════════
#  RESPONSE HELPERS
# ══════════════════════════════════════════════════════════════════

def _build_full_response(response) -> str:
    """Build the full raw response string, capturing any reasoning/thinking content."""
    msg = response.choices[0].message
    content = (msg.content or "").strip()

    reasoning = ""

    # 1. LiteLLM stores provider-specific data (e.g. Together reasoning) here
    psf = getattr(msg, "provider_specific_fields", None) or {}
    for key in ("reasoning_content", "reasoning", "thinking_content"):
        val = psf.get(key)
        if val and isinstance(val, str) and val.strip():
            reasoning = val.strip()
            break

    # 2. Direct message attributes (OpenAI reasoning models, etc.)
    if not reasoning:
        for attr in ("reasoning_content", "thinking_content", "reasoning"):
            val = getattr(msg, attr, None)
            if val and isinstance(val, str) and val.strip():
                reasoning = val.strip()
                break

    # 3. model_extra dict (fallback for various providers)
    if not reasoning:
        extra = getattr(msg, "model_extra", None) or {}
        val = extra.get("reasoning", "")
        reasoning = val.strip() if isinstance(val, str) else ""

    if reasoning:
        return f"[REASONING]\n{reasoning}\n[/REASONING]\n{content}"
    return content


def _get_reasoning_tokens(response) -> Optional[int]:
    """Extract reasoning-token count from response usage stats."""
    try:
        usage = response.usage
        if usage and hasattr(usage, "completion_tokens_details"):
            details = usage.completion_tokens_details
            if details and hasattr(details, "reasoning_tokens"):
                return details.reasoning_tokens
    except Exception:
        pass
    return None

# ══════════════════════════════════════════════════════════════════
#  MODEL API CALL  (with retry + rate limiting)
# ══════════════════════════════════════════════════════════════════

class RetryableAPIError(Exception):
    """Raised for transient API errors that should be retried."""
    pass


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(p in msg for p in (
        "rate", "429", "500", "502", "503", "504",
        "timeout", "connection", "overloaded", "capacity",
        "try again", "temporarily", "server_error",
    ))


@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=4, max=120),
    retry=retry_if_exception_type(RetryableAPIError),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)
def call_model(model_name: str, prompt: str, domain: str) -> dict:
    """Call a model via LiteLLM.  Returns {'response': str, 'reasoning_tokens': int|None}."""
    config = MODEL_CONFIGS[model_name]
    is_reasoning = config.get("reasoning", False)
    is_timezone = ("timezone" in domain)

    # Rate-limit before sending
    provider = "together" if "together_ai" in config["litellm_model"] else "openai"
    _rate_limiters[provider].acquire()

    messages = [
        {"role": "system", "content": get_system_prompt(is_timezone, is_reasoning)},
        {"role": "user", "content": prompt},
    ]

    params: dict = {
        "model": config["litellm_model"],
        "messages": messages,
    }

    if is_reasoning:
        params["max_tokens"] = 16000
    else:
        params["temperature"] = 0
        params["max_tokens"] = 1000 if provider == "together" else 500

    # Model-specific extra parameters
    if "extra_body" in config:
        params["extra_body"] = config["extra_body"]
    if "extra_params" in config:
        params.update(config["extra_params"])

    try:
        response = litellm.completion(**params)
    except Exception as exc:
        if _is_retryable(exc):
            raise RetryableAPIError(str(exc)) from exc
        # Non-retryable → return as ERROR row
        return {"response": f"ERROR: {exc}", "reasoning_tokens": None}

    full_response = _build_full_response(response)
    reasoning_tokens = _get_reasoning_tokens(response)

    # Flag truncated responses
    if response.choices[0].finish_reason == "length":
        full_response += " [TRUNCATED]"

    return {"response": full_response, "reasoning_tokens": reasoning_tokens}

# ══════════════════════════════════════════════════════════════════
#  ANSWER EXTRACTION
# ══════════════════════════════════════════════════════════════════

def extract_answer_from_tags(response: str) -> Optional[str]:
    """Extract the last <answer>…</answer> value.  Returns None if no tag found."""
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    return matches[-1].strip() if matches else None


def extract_number(answer: str) -> Optional[float]:
    """Extract a numeric value from *answer*, handling scientific notation."""
    if not answer or answer.startswith("ERROR:"):
        return None
    cleaned = answer.replace("[TRUNCATED]", "").replace(",", "").strip()

    # Scientific notation first
    sci = re.search(r"-?\d+\.?\d*[eE][+-]?\d+", cleaned)
    if sci:
        try:
            return float(sci.group(0))
        except ValueError:
            pass

    # Plain numbers (take the last reasonable one)
    matches = re.findall(r"-?\d+\.?\d*", cleaned)
    for m in reversed(matches):
        try:
            num = float(m)
            if abs(num) > 1e-10 or num == 0:
                return num
        except ValueError:
            continue
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            pass
    return None


def extract_time_string(answer: str) -> Optional[str]:
    """Extract a time string like '1AM' or '3:49PM'."""
    m = re.search(r"(\d{1,2}):(\d{2})\s*(AM|PM)", answer, re.IGNORECASE)
    if m:
        return f"{int(m.group(1))}:{int(m.group(2)):02d}{m.group(3).upper()}"
    m = re.search(r"(\d{1,2})\s*(AM|PM)", answer, re.IGNORECASE)
    if m:
        return f"{int(m.group(1))}{m.group(2).upper()}"
    return None


def extract_clothing_size(answer: str) -> Optional[str]:
    """Extract a clothing size (e.g. 'M', '32B', '46.5')."""
    if not answer:
        return None
    # Bra size: number + letter  (32B, 70A)
    m = re.search(r"\b(\d{2,3})([A-Z])\b", answer, re.IGNORECASE)
    if m:
        return f"{m.group(1)}{m.group(2).upper()}"
    # Alpha sizes
    m = re.search(r"\b(XS|S|M|L|XL|XXL|XXXL)\b", answer, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Numeric sizes (shoe sizes, pants)
    m = re.search(r"\b(\d{1,3}\.?\d*)\b", answer)
    if m:
        try:
            num = float(m.group(1))
            return str(int(num)) if num == int(num) else f"{num:.1f}".rstrip("0").rstrip(".")
        except ValueError:
            return m.group(1)
    return None

# ══════════════════════════════════════════════════════════════════
#  ANSWER TYPE + LOSS CALCULATION
# ══════════════════════════════════════════════════════════════════

def determine_answer_type(domain: str, answer) -> str:
    if "timezone" in domain:
        return "timezone"
    if "clothing_sizes" in domain or "bra_size" in domain:
        return "clothing"
    if isinstance(answer, str) and not answer.replace(".", "").replace("-", "").isdigit():
        return "string"
    return "numeric"


def calculate_loss(
    model_answer,
    correct_answer,
    answer_type: str = "numeric",
    tolerance_percent: float = 0.1,
    tolerance_minutes: float = 1.0,
) -> Optional[float]:
    """Return 0.0 if within tolerance, the error magnitude otherwise, or None on parse failure."""
    if model_answer is None:
        return None

    # ── Numeric ──────────────────────────────────────────────────
    if answer_type == "numeric":
        try:
            m, c = float(model_answer), float(correct_answer)
        except (ValueError, TypeError):
            return None
        if c == 0:
            return 0.0 if abs(m) < 0.001 else abs(m)
        rel = abs((m - c) / c) * 100
        return 0.0 if rel <= tolerance_percent else rel

    # ── Timezone ─────────────────────────────────────────────────
    if answer_type == "timezone":
        def _to_hours(s: str) -> float:
            s = s.strip().upper()
            m2 = re.match(r"(\d{1,2}):(\d{2})(AM|PM)", s)
            if m2:
                h, mi, p = int(m2.group(1)), int(m2.group(2)), m2.group(3)
                if p == "PM" and h != 12:
                    h += 12
                elif p == "AM" and h == 12:
                    h = 0
                return h + mi / 60.0
            m2 = re.match(r"(\d{1,2})(AM|PM)", s)
            if m2:
                h, p = int(m2.group(1)), m2.group(2)
                if p == "PM" and h != 12:
                    h += 12
                elif p == "AM" and h == 12:
                    h = 0
                return float(h)
            raise ValueError(f"Cannot parse time: {s}")
        try:
            diff = abs((_to_hours(str(model_answer)) - _to_hours(str(correct_answer))) * 60)
            if diff > 720:
                diff = 1440 - diff
            return 0.0 if diff <= tolerance_minutes else diff
        except (ValueError, TypeError):
            return None

    # ── Clothing ─────────────────────────────────────────────────
    if answer_type == "clothing":
        try:
            return 0.0 if abs(float(str(model_answer).strip()) - float(str(correct_answer).strip())) < 0.001 else 1.0
        except (ValueError, TypeError):
            return 0.0 if str(model_answer).strip().upper() == str(correct_answer).strip().upper() else 1.0

    # ── Fallback (exact string match) ────────────────────────────
    return 0.0 if str(model_answer).strip() == str(correct_answer).strip() else 1.0

# ══════════════════════════════════════════════════════════════════
#  SINGLE-ROW PROCESSING  (runs inside a worker thread)
# ══════════════════════════════════════════════════════════════════

def process_row(
    prompt: str,
    domain: str,
    correct_answer,
    model_name: str,
    tolerance_percent: float,
    tolerance_minutes: float,
) -> dict:
    """Process one row end-to-end.  Thread-safe (no shared mutable state)."""
    config = MODEL_CONFIGS.get(model_name, {})
    is_reasoning = config.get("reasoning", False)

    # 1) Call model
    result = call_model(model_name, prompt, domain)
    raw_response = result["response"]
    reasoning_tokens = result["reasoning_tokens"]

    # 2) Extract answer
    answer_type = determine_answer_type(domain, correct_answer)

    if is_reasoning:
        tagged = extract_answer_from_tags(raw_response)
        source = tagged if tagged is not None else raw_response
    else:
        source = raw_response

    if answer_type == "timezone":
        model_answer = extract_time_string(source)
    elif answer_type == "clothing":
        model_answer = extract_clothing_size(source)
    else:
        model_answer = extract_number(source)

    # 3) Compute loss
    loss = calculate_loss(model_answer, correct_answer, answer_type,
                          tolerance_percent, tolerance_minutes)

    return {
        "raw_response": raw_response,
        "model_answer": model_answer,
        "loss": loss,
        "reasoning_tokens": reasoning_tokens,
    }

# ══════════════════════════════════════════════════════════════════
#  CHECKPOINT  (per output file)
# ══════════════════════════════════════════════════════════════════

_ckpt_lock = Lock()

RESULT_COLS = ("raw_response", "model_answer", "loss", "reasoning_tokens")


def save_checkpoint(df: pd.DataFrame, output_file: Path):
    """Write current DataFrame state to the output TSV (thread-safe)."""
    with _ckpt_lock:
        tmp = df.copy()
        if "distractor" in tmp.columns:
            tmp["distractor"] = tmp["distractor"].fillna("null")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        tmp.to_csv(output_file, sep="\t", index=False, na_rep="null")


def load_checkpoint(output_file: Path, df: pd.DataFrame) -> int:
    """If *output_file* already exists, restore results into *df* in-place.

    Returns the number of rows already processed (= index to resume from).
    """
    if not output_file.exists():
        return 0
    try:
        ckpt = pd.read_csv(output_file, sep="\t")
        if "raw_response" not in ckpt.columns:
            return 0
        mask = ckpt["raw_response"].notna() & (ckpt["raw_response"].astype(str) != "null")
        start = int(mask.sum())
        if start > 0 and start <= len(df):
            for col in RESULT_COLS:
                if col in ckpt.columns:
                    df[col] = ckpt[col]
            log.info("  Resuming from checkpoint: %d / %d rows already done.", start, len(df))
        return start
    except Exception as exc:
        log.warning("  Could not load checkpoint (%s). Starting from scratch.", exc)
        return 0

# ══════════════════════════════════════════════════════════════════
#  TASK PROCESSING  (one model × domain × condition)
# ══════════════════════════════════════════════════════════════════

def process_task(
    *,
    model_name: str,
    input_file: Path,
    output_file: Path,
    max_workers: int = 10,
    checkpoint_every: int = 50,
    tolerance_percent: float = 0.1,
    tolerance_minutes: float = 1.0,
) -> bool:
    """Run inference for one task.  Returns True on success."""

    # Load input
    df = pd.read_csv(input_file, sep="\t")
    for col in RESULT_COLS:
        df[col] = None

    # Resume from checkpoint
    start_idx = load_checkpoint(output_file, df)
    if start_idx >= len(df):
        log.info("  Already complete (%d rows). Skipping.", len(df))
        return True

    pending = list(range(start_idx, len(df)))
    log.info("  %d rows to process (%d already done).", len(pending), start_idx)

    # Process in checkpoint-sized batches
    for batch_off in range(0, len(pending), checkpoint_every):
        if _shutdown:
            log.warning("  Shutdown requested — saving checkpoint.")
            save_checkpoint(df, output_file)
            return True

        batch_idx = pending[batch_off : batch_off + checkpoint_every]
        results: dict[int, dict] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for idx in batch_idx:
                row = df.iloc[idx]
                fut = pool.submit(
                    process_row,
                    prompt=str(row["prompt"]),
                    domain=str(row["domain"]),
                    correct_answer=row["answer"],
                    model_name=model_name,
                    tolerance_percent=tolerance_percent,
                    tolerance_minutes=tolerance_minutes,
                )
                futures[fut] = idx

            done_count = start_idx + batch_off
            desc = f"  [{done_count + len(batch_idx)}/{len(df)}]"
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=desc, leave=False):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:
                    results[idx] = {
                        "raw_response": f"ERROR: {exc}",
                        "model_answer": None,
                        "loss": None,
                        "reasoning_tokens": None,
                    }

        # Write results into DataFrame (main thread only — safe)
        for idx, res in results.items():
            for col in RESULT_COLS:
                df.at[idx, col] = res[col]

        # Checkpoint
        save_checkpoint(df, output_file)

    # Final summary
    valid = pd.to_numeric(df["loss"], errors="coerce").dropna()
    if len(valid):
        correct = (valid == 0).sum()
        log.info("  Done — %d/%d correct (%.1f%%), mean loss %.4f",
                 correct, len(valid), correct / len(valid) * 100, valid.mean())
    else:
        log.info("  Done — no valid losses computed.")
    return True

# ══════════════════════════════════════════════════════════════════
#  TASK-LEVEL AUTO-RESTART
# ══════════════════════════════════════════════════════════════════

def run_task_with_retry(max_retries: int = 3, **task_kwargs) -> bool:
    """Run process_task with auto-restart on failure."""
    for attempt in range(1, max_retries + 1):
        try:
            return process_task(**task_kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if attempt == max_retries:
                log.error("  Task failed after %d attempts: %s", max_retries, exc)
                return False
            wait = min(30 * (2 ** (attempt - 1)), 300)
            log.warning("  Task failed (attempt %d/%d): %s  — retrying in %ds",
                        attempt, max_retries, exc, wait)
            time.sleep(wait)
    return False

# ══════════════════════════════════════════════════════════════════
#  TASK DISCOVERY
# ══════════════════════════════════════════════════════════════════

def discover_tasks(
    base_dir: Path,
    models: list[str],
    conditions: list[str],
    domains: Optional[list[str]] = None,
) -> list[dict]:
    """Scan the preprocessed directory and build the full task list.

    Each task is a dict with keys:
        model, domain, condition, input_file, output_file, status, processed
    """
    preproc_dir = base_dir / PREPROCESSED_SUBDIR

    # Auto-discover domains from preprocessed filenames
    if domains is None:
        raw_names = set()
        for f in preproc_dir.glob("*.tsv"):
            name = f.stem
            for suf in ("_no_guide", "_math_only"):
                name = name.replace(suf, "")
            raw_names.add(name)
        domains = sorted(raw_names)

    tasks = []
    for model in models:
        if model not in MODEL_CONFIGS:
            log.warning("Unknown model '%s' — skipping.", model)
            continue
        for domain in domains:
            for cond in conditions:
                cfg = CONDITIONS[cond]
                input_file = preproc_dir / f"{domain}{cfg['suffix']}.tsv"
                if not input_file.exists():
                    continue

                output_dir = base_dir / cfg["output_dir"] / model
                output_file = output_dir / f"{domain}{cfg['suffix']}_converted.tsv"

                # Quick status check (use pd.read_csv for correct multi-line TSV handling)
                status, processed = "pending", 0
                if output_file.exists():
                    try:
                        n_input = len(pd.read_csv(input_file, sep="\t", usecols=["domain"]))
                        ckpt = pd.read_csv(output_file, sep="\t", usecols=["raw_response"])
                        mask = ckpt["raw_response"].notna() & (ckpt["raw_response"].astype(str) != "null")
                        processed = int(mask.sum())
                        status = "complete" if processed >= n_input else ("partial" if processed > 0 else "pending")
                    except Exception:
                        pass

                tasks.append({
                    "model": model,
                    "domain": domain,
                    "condition": cond,
                    "input_file": input_file,
                    "output_file": output_file,
                    "status": status,
                    "processed": processed,
                })

    return tasks

# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run conversion inference across all model × domain × condition tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-dir", type=Path, default=DEFAULT_BASE_DIR,
        help="Root results directory (default: %(default)s)",
    )
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_CONFIGS.keys()),
        help="Models to run (default: all). Choices: " + ", ".join(MODEL_CONFIGS.keys()),
    )
    parser.add_argument(
        "--conditions", nargs="+", default=list(CONDITIONS.keys()),
        choices=list(CONDITIONS.keys()),
        help="Conditions to run (default: all).",
    )
    parser.add_argument(
        "--domains", nargs="+", default=None,
        help="Domains to run (default: auto-discover from preprocessed/).",
    )
    parser.add_argument(
        "--max-workers", type=int, default=10,
        help="Concurrent threads per task for API calls (default: 10).",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=50,
        help="Save checkpoint every N rows (default: 50).",
    )
    parser.add_argument(
        "--max-task-retries", type=int, default=3,
        help="Auto-restart failed tasks up to N times (default: 3).",
    )
    parser.add_argument(
        "--tolerance-percent", type=float, default=0.1,
        help="Numeric answer tolerance in %% (default: 0.1).",
    )
    parser.add_argument(
        "--tolerance-minutes", type=float, default=1.0,
        help="Timezone answer tolerance in minutes (default: 1.0).",
    )
    parser.add_argument(
        "--list-tasks", action="store_true",
        help="List all tasks and their status, then exit.",
    )
    args = parser.parse_args()

    # Setup
    setup_api_keys()

    # Discover tasks
    tasks = discover_tasks(args.base_dir, args.models, args.conditions, args.domains)

    n_complete = sum(1 for t in tasks if t["status"] == "complete")
    n_partial = sum(1 for t in tasks if t["status"] == "partial")
    n_pending = sum(1 for t in tasks if t["status"] == "pending")

    log.info("Discovered %d tasks: %d complete, %d partial, %d pending.",
             len(tasks), n_complete, n_partial, n_pending)

    if args.list_tasks:
        fmt = "  {status:8s}  {model:25s}  {condition:10s}  {domain:40s}  ({processed} rows)"
        for t in tasks:
            print(fmt.format(**t))
        return

    # Process tasks (skip completed ones)
    work = [t for t in tasks if t["status"] != "complete"]
    if not work:
        log.info("All tasks already complete. Nothing to do!")
        return

    log.info("Processing %d tasks (%d workers, checkpoint every %d rows).",
             len(work), args.max_workers, args.checkpoint_every)

    succeeded, failed, skipped = 0, 0, 0
    for i, task in enumerate(work, 1):
        if _shutdown:
            log.warning("Shutdown — stopping after %d/%d tasks.", i - 1, len(work))
            break

        label = f"{task['model']} / {task['domain']} / {task['condition']}"
        log.info("\n[%d/%d] %s  (status: %s)", i, len(work), label, task["status"])

        ok = run_task_with_retry(
            max_retries=args.max_task_retries,
            model_name=task["model"],
            input_file=task["input_file"],
            output_file=task["output_file"],
            max_workers=args.max_workers,
            checkpoint_every=args.checkpoint_every,
            tolerance_percent=args.tolerance_percent,
            tolerance_minutes=args.tolerance_minutes,
        )
        if ok:
            succeeded += 1
        else:
            failed += 1

    log.info("\nFinished: %d succeeded, %d failed, %d skipped (already complete).",
             succeeded, failed, n_complete)


if __name__ == "__main__":
    main()
