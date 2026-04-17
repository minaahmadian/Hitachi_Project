from __future__ import annotations

import os
import re
import threading
import time
from typing import Any, Final

from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq

_lock = threading.Lock()
_cooldown_lock = threading.Lock()
_cached_llm: ChatGroq | None = None
_last_llm_call_monotonic: float | None = None

_DEFAULT_MODEL: Final[str] = "llama-3.1-8b-instant"


def _parse_groq_retry_after_seconds(exc: BaseException) -> float | None:
    """Groq 429 bodies often include 'try again in 12.34s'."""
    s = str(exc)
    for pattern in (
        r"try again in ([0-9.]+)\s*s",
        r"Please try again in ([0-9.]+)\s*s",
    ):
        m = re.search(pattern, s, re.I)
        if m:
            return float(m.group(1)) + 2.0
    return None


def get_chat_groq(
    *,
    model: str | None = None,
    temperature: float = 0,
) -> ChatGroq:
    """
    Lazily construct a shared ChatGroq client.

    This avoids requiring GROQ_API_KEY at import time (only when the model is first used).
    """
    global _cached_llm
    if _cached_llm is not None:
        return _cached_llm

    resolved_model = model or os.getenv("GROQ_CHAT_MODEL", _DEFAULT_MODEL)
    max_retries = int(os.getenv("GROQ_MAX_RETRIES", "0"))
    # Cap completion size to stay under low TPM; JSON agents rarely need more.
    max_tokens = int(os.getenv("GROQ_MAX_OUTPUT_TOKENS", "900"))

    with _lock:
        if _cached_llm is None:
            _cached_llm = ChatGroq(
                model=resolved_model,
                temperature=temperature,
                max_retries=max_retries,
                max_tokens=max_tokens,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
    return _cached_llm


def invoke_chat_groq(messages: list[BaseMessage]) -> Any:
    """
    Invoke Groq chat with:
    - Cooldown between calls (default 65s) so ~6000 TPM free tiers can fit 3 agents in separate windows
    - Retries on 429 (parse server wait hint; up to GROQ_INVOKE_RATE_LIMIT_RETRIES)

    GROQ_LLM_CALL_DELAY_SEC=0 disables spacing (may hit 429 on free tier).
    """
    global _last_llm_call_monotonic
    delay = float(os.getenv("GROQ_LLM_CALL_DELAY_SEC", "65"))
    delay = max(0.0, delay)
    rate_retries = max(0, int(os.getenv("GROQ_INVOKE_RATE_LIMIT_RETRIES", "3")))

    with _cooldown_lock:
        if _last_llm_call_monotonic is not None and delay > 0:
            print(
                f"Groq cooldown: waiting {delay:.0f}s before next LLM call "
                f"(spreads ~6000 TPM; GROQ_LLM_CALL_DELAY_SEC)...",
                flush=True,
            )
            time.sleep(delay)

        attempt = 0
        while True:
            try:
                response = get_chat_groq().invoke(messages)
                _last_llm_call_monotonic = time.monotonic()
                return response
            except Exception as exc:
                wait = _parse_groq_retry_after_seconds(exc)
                err_s = str(exc).lower()
                is_429 = (
                    wait is not None
                    or "429" in str(exc)
                    or "rate_limit" in err_s
                    or "rate limit" in err_s
                )
                if is_429 and attempt < rate_retries:
                    sleep_s = wait if wait is not None else 70.0
                    sleep_s = min(max(sleep_s, 8.0), 150.0)
                    print(
                        f"Groq 429 rate limit; sleeping {sleep_s:.1f}s then retry "
                        f"({attempt + 1}/{rate_retries})...",
                        flush=True,
                    )
                    time.sleep(sleep_s)
                    attempt += 1
                    continue
                raise
