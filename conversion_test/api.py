"""
LLM API layer — rate limiting, model calls, and response parsing.

Handles LiteLLM calls with tenacity retry and thread-safe rate limiting.
"""

import time
import logging
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

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
litellm.request_timeout = 120  # global fallback
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
        deadline = time.time() + timeout
        while True:
            with self._lock:
                now = time.time()
                if now >= deadline:
                    raise TimeoutError(f"Rate limiter: no slot within {timeout}s")
                self._timestamps = [t for t in self._timestamps if now - t < 60]
                if len(self._timestamps) < self.max_per_minute:
                    self._timestamps.append(now)
                    return
                sleep_for = 60.0 - (now - self._timestamps[0]) + 0.1
            time.sleep(max(min(sleep_for, deadline - time.time()), 0.05))


rate_limiters = {
    "openai":   RateLimiter(max_per_minute=450),
    "together": RateLimiter(max_per_minute=425),
}

# ── Response Parsing ─────────────────────────────────────────────

_REASONING_KEYS = ("reasoning_content", "reasoning", "thinking_content")


def _do_non_streaming_call(params: dict, model_name: str) -> dict:
    """Execute a non-streaming LiteLLM call.

    More reliable than streaming for models with unstable streaming endpoints.
    """
    # Remove stream-specific params
    ns_params = {k: v for k, v in params.items()
                 if k not in ("stream", "stream_options")}
    ns_params["stream"] = False

    resp = litellm.completion(**ns_params)

    msg = resp.choices[0].message
    content = (msg.content or "").strip()

    # Extract reasoning from message
    reasoning_parts = []
    for key in _REASONING_KEYS:
        val = getattr(msg, key, None)
        if val:
            reasoning_parts.append(val)
            break
    psf = getattr(msg, "provider_specific_fields", None) or {}
    if not reasoning_parts:
        for key in _REASONING_KEYS:
            val = psf.get(key)
            if val:
                reasoning_parts.append(val)
                break
    reasoning = "".join(reasoning_parts).strip()

    full_response = f"[REASONING]\n{reasoning}\n[/REASONING]\n{content}" if reasoning else content

    reasoning_tokens = None
    if resp.usage and hasattr(resp.usage, "completion_tokens_details"):
        details = resp.usage.completion_tokens_details
        if details and hasattr(details, "reasoning_tokens"):
            reasoning_tokens = details.reasoning_tokens

    finish_reason = resp.choices[0].finish_reason
    if finish_reason == "length":
        full_response += " [TRUNCATED]"

    return {"response": full_response, "reasoning_tokens": reasoning_tokens}


def _do_streaming_call(params: dict, timeout_val: int, model_name: str, provider: str) -> dict:
    """Execute a streaming LiteLLM call and consume all chunks.

    This runs inside a dedicated thread so we can enforce a hard wall-clock
    timeout from the caller via future.result(timeout=...).
    """
    stream = litellm.completion(**params)

    collected_content = []
    collected_reasoning = []
    finish_reason = None
    usage = None
    t0 = time.time()

    for chunk in stream:
        if time.time() - t0 > timeout_val:
            raise litellm.Timeout(
                message=f"Streaming exceeded {timeout_val}s wall clock",
                model=model_name, llm_provider=provider,
            )
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta:
            if delta.content:
                collected_content.append(delta.content)
            for key in _REASONING_KEYS:
                val = getattr(delta, key, None)
                if val:
                    collected_reasoning.append(val)
                    break
            psf = getattr(delta, "provider_specific_fields", None) or {}
            for key in _REASONING_KEYS:
                val = psf.get(key)
                if val:
                    collected_reasoning.append(val)
                    break
        if chunk.choices and chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason
        if hasattr(chunk, "usage") and chunk.usage:
            usage = chunk.usage

    content = "".join(collected_content).strip()
    reasoning = "".join(collected_reasoning).strip()
    full_response = f"[REASONING]\n{reasoning}\n[/REASONING]\n{content}" if reasoning else content

    reasoning_tokens = None
    if usage and hasattr(usage, "completion_tokens_details"):
        details = usage.completion_tokens_details
        if details and hasattr(details, "reasoning_tokens"):
            reasoning_tokens = details.reasoning_tokens

    if finish_reason == "length":
        full_response += " [TRUNCATED]"

    return {"response": full_response, "reasoning_tokens": reasoning_tokens}


# Single-thread pool used solely for hard timeout enforcement.
# Daemon threads so they don't block process exit.
_timeout_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="api-timeout")


# ── Model API Call (with retry + rate limiting) ──────────────────

RETRYABLE_EXCEPTIONS = (
    litellm.RateLimitError,
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
    litellm.Timeout,
    litellm.APIConnectionError,
    TimeoutError,
    FuturesTimeout,
    ConnectionError,
)


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=4, max=60),
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
    t_rl = time.time()
    rate_limiters[provider].acquire()
    rl_wait = time.time() - t_rl
    if rl_wait > 2:
        log.info("Rate-limiter wait: %.1fs (%s)", rl_wait, provider)

    messages = [
        {"role": "system", "content": get_system_prompt(is_timezone, is_reasoning)},
        {"role": "user", "content": prompt},
    ]

    params: dict = {
        "model": config["litellm_model"],
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    if is_reasoning:
        params["max_tokens"] = 16000
    else:
        params["temperature"] = 0
        params["max_tokens"] = 1000 if provider == "together" else 500

    if "extra_body" in config:
        params["extra_body"] = config["extra_body"]
    if "extra_params" in config:
        params.update(config["extra_params"])

    # Per-model timeout: config override > 600s (reasoning) > 120s (standard)
    timeout_val = config.get("timeout", 600 if is_reasoning else 120)
    params["timeout"] = timeout_val

    use_streaming = config.get("stream", True)

    t0 = time.time()

    if use_streaming:
        # Submit to a separate thread so we can enforce a HARD wall-clock timeout.
        # Even if the stream iterator blocks forever in __next__(), the
        # future.result(timeout=...) will raise FuturesTimeout.
        future = _timeout_pool.submit(_do_streaming_call, params, timeout_val, model_name, provider)

        try:
            result = future.result(timeout=timeout_val + 30)  # +30s grace for setup/teardown
        except FuturesTimeout:
            future.cancel()
            elapsed = time.time() - t0
            log.warning("HARD TIMEOUT: %s stuck for %.0fs — will retry", model_name, elapsed)
            raise TimeoutError(f"Hard timeout after {elapsed:.0f}s for {model_name}")
    else:
        result = _do_non_streaming_call(params, model_name)

    call_duration = time.time() - t0
    if call_duration > timeout_val * 0.8:
        log.warning("Slow call: %s took %.1fs (limit=%ds)", model_name, call_duration, timeout_val)

    result["call_seconds"] = call_duration
    return result
