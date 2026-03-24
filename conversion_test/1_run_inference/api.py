"""
LLM API layer — rate limiting, model calls, and response parsing.

Handles LiteLLM calls with tenacity retry and thread-safe rate limiting.
"""

import sys
import time
import logging
from pathlib import Path
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

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from config import MODEL_CONFIGS, get_system_prompt

log = logging.getLogger("conversion")


# ── Fatal (non-recoverable) Error Detection ──────────────────────

class FatalAPIError(Exception):
    """Non-recoverable API error (billing, auth) — should stop execution."""
    pass


FATAL_ERROR_PATTERNS = [
    "credit limit exceeded",
    "insufficient_quota",
    "billing hard limit",
    "invalid api key",
    "invalid_api_key",
    "account deactivated",
    "account has been disabled",
]


def _is_fatal(error_msg: str) -> bool:
    """Check whether an API error is non-recoverable (billing, auth, etc.)."""
    msg_lower = error_msg.lower()
    return any(pat in msg_lower for pat in FATAL_ERROR_PATTERNS)

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
    "openai":     RateLimiter(max_per_minute=350),
    "anthropic":  RateLimiter(max_per_minute=400),
    "together":   RateLimiter(max_per_minute=1000),
}
_model_rate_limiters: dict[str, RateLimiter] = {}


def _get_rate_limiter(model_name: str, provider: str) -> RateLimiter:
    """Return a per-model rate limiter if configured, else the provider-level one."""
    if model_name not in _model_rate_limiters:
        rpm = MODEL_CONFIGS.get(model_name, {}).get("max_rpm")
        if rpm:
            _model_rate_limiters[model_name] = RateLimiter(max_per_minute=rpm)
    if model_name in _model_rate_limiters:
        return _model_rate_limiters[model_name]
    return rate_limiters[provider]

# ── Response Parsing ─────────────────────────────────────────────


def _extract_reasoning_and_content(msg) -> tuple[str, str]:
    """Pull every piece of reasoning and content from a LiteLLM message.

    Handles:
      - Anthropic content-block lists  [{"type":"thinking","thinking":"…"}, {"type":"text","text":"…"}]
      - Top-level attrs: reasoning_content, reasoning, thinking_content, thinking
      - provider_specific_fields with the same keys
      - Plain string msg.content
    Returns (reasoning, content) as strings.
    """
    reasoning_parts: list[str] = []
    content_parts: list[str] = []

    # 1. Anthropic content-block list (msg.content is a list of dicts)
    raw_content = msg.content
    if isinstance(raw_content, list):
        for block in raw_content:
            if isinstance(block, dict):
                if block.get("type") == "thinking":
                    text = block.get("thinking") or block.get("text") or ""
                    if text:
                        reasoning_parts.append(text)
                elif block.get("type") == "text":
                    text = block.get("text") or ""
                    if text:
                        content_parts.append(text)
                else:
                    text = block.get("text") or block.get("content") or ""
                    if text:
                        content_parts.append(text)
            elif isinstance(block, str):
                content_parts.append(block)
    elif isinstance(raw_content, str) and raw_content:
        content_parts.append(raw_content)

    # 2. Top-level reasoning attributes (OpenAI, DeepSeek, Qwen via LiteLLM)
    _ALL_REASONING_KEYS = ("reasoning_content", "reasoning", "thinking_content", "thinking")
    for key in _ALL_REASONING_KEYS:
        val = getattr(msg, key, None)
        if val and isinstance(val, str) and val.strip():
            if val.strip() not in [r.strip() for r in reasoning_parts]:
                reasoning_parts.append(val.strip())

    # 3. provider_specific_fields
    psf = getattr(msg, "provider_specific_fields", None) or {}
    for key in _ALL_REASONING_KEYS:
        val = psf.get(key)
        if val and isinstance(val, str) and val.strip():
            if val.strip() not in [r.strip() for r in reasoning_parts]:
                reasoning_parts.append(val.strip())

    return "\n\n".join(reasoning_parts).strip(), "\n".join(content_parts).strip()


def _extract_reasoning_tokens(resp) -> int | None:
    """Extract reasoning/thinking token count from any provider's usage object."""
    if not resp.usage:
        return None

    # OpenAI: usage.completion_tokens_details.reasoning_tokens
    if hasattr(resp.usage, "completion_tokens_details"):
        details = resp.usage.completion_tokens_details
        if details:
            for attr in ("reasoning_tokens", "thinking_tokens"):
                val = getattr(details, attr, None)
                if val is not None:
                    return int(val)

    # Anthropic / generic: usage dict or attributes
    usage_dict = resp.usage if isinstance(resp.usage, dict) else getattr(resp.usage, "__dict__", {})
    for key in ("reasoning_tokens", "thinking_tokens", "cache_read_input_tokens"):
        val = usage_dict.get(key)
        if val is not None:
            return int(val)

    return None


def _do_non_streaming_call(params: dict, model_name: str) -> dict:
    """Execute a non-streaming LiteLLM call.

    More reliable than streaming for models with unstable streaming endpoints.
    """
    ns_params = {k: v for k, v in params.items()
                 if k not in ("stream", "stream_options")}
    ns_params["stream"] = False

    resp = litellm.completion(**ns_params)

    msg = resp.choices[0].message
    reasoning, content = _extract_reasoning_and_content(msg)

    full_response = f"[REASONING]\n{reasoning}\n[/REASONING]\n{content}" if reasoning else content
    reasoning_tokens = _extract_reasoning_tokens(resp)

    finish_reason = resp.choices[0].finish_reason
    if finish_reason == "length":
        full_response += " [TRUNCATED]"

    return {"response": full_response, "reasoning_tokens": reasoning_tokens}


_ALL_STREAMING_REASONING_KEYS = ("reasoning_content", "reasoning", "thinking_content", "thinking")


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
            for key in _ALL_STREAMING_REASONING_KEYS:
                val = getattr(delta, key, None)
                if val:
                    collected_reasoning.append(val)
                    break
            psf = getattr(delta, "provider_specific_fields", None) or {}
            for key in _ALL_STREAMING_REASONING_KEYS:
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
    if usage:
        # Reuse the same helper logic by building a mock
        class _UsageWrap:
            pass
        uw = _UsageWrap()
        uw.usage = usage
        uw.completion_tokens_details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = _extract_reasoning_tokens(uw) if hasattr(usage, "completion_tokens_details") else None
        if reasoning_tokens is None:
            # Fallback: check usage directly
            for attr in ("reasoning_tokens", "thinking_tokens"):
                val = getattr(usage, attr, None) if not isinstance(usage, dict) else usage.get(attr)
                if val is not None:
                    reasoning_tokens = int(val)
                    break

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
    lm = config["litellm_model"]
    if "together_ai" in lm:
        provider = "together"
    elif "anthropic/" in lm:
        provider = "anthropic"
    else:
        provider = "openai"
    t_rl = time.time()
    _get_rate_limiter(model_name, provider).acquire()
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

    try:
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
    except FatalAPIError:
        raise
    except Exception as exc:
        if _is_fatal(str(exc)):
            raise FatalAPIError(str(exc)) from exc
        raise

    call_duration = time.time() - t0
    if call_duration > timeout_val * 0.8:
        log.warning("Slow call: %s took %.1fs (limit=%ds)", model_name, call_duration, timeout_val)

    result["call_seconds"] = call_duration
    return result
