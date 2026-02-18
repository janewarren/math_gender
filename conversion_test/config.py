"""
Shared configuration for the conversion inference pipeline.

Contains model registry, experimental conditions, constants, and API key setup.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

log = logging.getLogger("conversion")

# ── Paths & Constants ────────────────────────────────────────────

DEFAULT_BASE_DIR = Path("full_results")
PREPROCESSED_SUBDIR = "preprocessed"
RESULT_COLS = ("raw_response", "model_answer", "loss", "reasoning_tokens", "call_seconds")

# ── Model Registry ───────────────────────────────────────────────
# Each entry carries everything LiteLLM needs.
# "litellm_model" uses the provider prefix that LiteLLM expects.

MODEL_CONFIGS = {
    # ── Standard models ──────────────────────────────────────────
    "gpt-4o": {
        "litellm_model": "gpt-4o",
        "reasoning": False,
        "max_workers": 20,      # ~1-2s latency × 7.5 RPS
    },
    "qwen-coder": {
        "litellm_model": "together_ai/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        "reasoning": False,
        "max_workers": 40,      # ~1-3s latency × 14 RPS
    },
    "llama-4": {
        "litellm_model": "together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "reasoning": False,
        "max_workers": 40,      # ~1-3s latency × 14 RPS
    },
    # ── Reasoning / CoT models ───────────────────────────────────
    "gpt-5.2": {
        "litellm_model": "gpt-5.2",
        "reasoning": True,
        "extra_params": {"reasoning_effort": "medium"},
        "max_workers": 60,      # ~3-10s latency × 7.5 RPS
    },
    "deepseek-v3.1": {
        "litellm_model": "together_ai/deepseek-ai/DeepSeek-V3.1",
        "reasoning": True,
        "extra_body": {"thinking": {"type": "enabled"}},
        "extra_params": {"frequency_penalty": 0.2},
        "timeout": 300,         # 5 min — DeepSeek can be slow
        "max_workers": 100,     # ~5-20s latency × 14 RPS
    },
    "qwen3-235b-thinking": {
        "litellm_model": "together_ai/Qwen/Qwen3-235B-A22B-Thinking-2507",
        "reasoning": True,
        "extra_body": {"enable_thinking": True},
        "stream": False,        # non-streaming — streaming endpoint is unstable
        "timeout": 180,         # 3 min — Qwen thinking is fast
        "max_workers": 15,      # low concurrency — model prone to hangs
    },
    "qwen3-next-thinking": {
        "litellm_model": "together_ai/Qwen/Qwen3-Next-80B-A3B-Thinking",
        "reasoning": True,
        "extra_body": {"enable_thinking": True},
        "stream": False,        # non-streaming — streaming endpoint is unstable
        "timeout": 180,         # 3 min — Qwen thinking is fast
        "max_workers": 15,      # low concurrency — model prone to hangs
    },
}

# ── Experimental Conditions ──────────────────────────────────────

CONDITIONS = {
    "regular":   {"suffix": "",            "output_dir": "results"},
    "no_guide":  {"suffix": "_no_guide",   "output_dir": "results_no_guide"},
    "math_only": {"suffix": "_math_only",  "output_dir": "results_math_only"},
}

# ── System Prompts ───────────────────────────────────────────────

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

# ── API Key Setup ────────────────────────────────────────────────

def setup_api_keys():
    """Load API keys from .env file.  Existing env vars take priority."""
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        log.warning("OPENAI_API_KEY not set — OpenAI models will fail.")
    if not os.environ.get("TOGETHER_AI_API_KEY"):
        log.warning("TOGETHER_AI_API_KEY not set — Together models will fail.")
