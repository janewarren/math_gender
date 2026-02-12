"""
LLM API layer — rate limiting, model calls, and response parsing.

Handles LiteLLM calls with tenacity retry and thread-safe rate limiting.
"""

import time
import logging
from typing import Optional
from threading import Lock

import litellm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from config import MODEL_CONFIGS, get_system_prompt

log = logging.getLogger("conversion")

# Suppress noisy litellm debug output
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ── Thread-Safe Rate Limiter ─────────────────────────────────────

class RateLimiter:
    """Sliding-window rate limiter (thread-safe) with acquire timeout."""

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self._timestamps: list[float] = []
        self._lock = Lock()

    def acquire(self, timeout: float = 300):
        """Acquire a rate-limit slot. Raises TimeoutError after *timeout* seconds."""
        deadline = time.time() + timeout
        while True:
            with self._lock:
                now = time.time()
                if now >= deadline:
                    raise TimeoutError(
                        f"Rate limiter could not acquire a slot within {timeout}s"
                    )
                self._timestamps = [t for t in self._timestamps if now - t < 60]
                if len(self._timestamps) < self.max_per_minute:
                    self._timestamps.append(now)
                    return  # slot acquired
                sleep_for = 60.0 - (now - self._timestamps[0]) + 0.1
            # sleep *outside* the lock so other threads aren't blocked
            time.sleep(max(min(sleep_for, deadline - time.time()), 0.05))


# One limiter per provider — stay ~10-15% under hard caps to avoid 429 storms
rate_limiters = {
    "openai":   RateLimiter(max_per_minute=450),    # hard cap 500
    "together": RateLimiter(max_per_minute=850),    # hard cap 1000
}

# ── Response Parsing ─────────────────────────────────────────────

_REASONING_KEYS = ("reasoning_content", "reasoning", "thinking_content")


def _build_full_response(response) -> str:
    """Build the full raw response string, capturing any reasoning/thinking content."""
    msg = response.choices[0].message
    content = (msg.content or "").strip()

    # Search three dict-like sources for reasoning text (first non-empty wins)
    sources = [
        getattr(msg, "provider_specific_fields", None) or {},   # LiteLLM / Together
        {k: getattr(msg, k, None) for k in _REASONING_KEYS},   # direct attributes
        getattr(msg, "model_extra", None) or {},                # fallback
    ]
    for src in sources:
        for key in _REASONING_KEYS:
            val = src.get(key)
            if isinstance(val, str) and val.strip():
                return f"[REASONING]\n{val.strip()}\n[/REASONING]\n{content}"

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

# ── Model API Call (with retry + rate limiting) ──────────────────

RETRYABLE_EXCEPTIONS = (
    litellm.RateLimitError,
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
    litellm.Timeout,
    litellm.APIConnectionError,
    TimeoutError,          # from our RateLimiter
    ConnectionError,
)


@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=4, max=120),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)
def call_model(model_name: str, prompt: str, domain: str) -> dict:
    """Call a model via LiteLLM.

    Returns {'response': str, 'reasoning_tokens': int|None, 'call_seconds': float|None}.
    """
    config = MODEL_CONFIGS[model_name]
    is_reasoning = config.get("reasoning", False)
    is_timezone = ("timezone" in domain)

    # Rate-limit before sending
    provider = "together" if "together_ai" in config["litellm_model"] else "openai"
    rate_limiters[provider].acquire()

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

    t0 = time.time()
    response = litellm.completion(**params)
    call_duration = time.time() - t0

    full_response = _build_full_response(response)
    reasoning_tokens = _get_reasoning_tokens(response)

    # Flag truncated responses
    if response.choices[0].finish_reason == "length":
        full_response += " [TRUNCATED]"

    return {"response": full_response, "reasoning_tokens": reasoning_tokens, "call_seconds": call_duration}
