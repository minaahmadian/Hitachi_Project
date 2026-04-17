from __future__ import annotations

import os
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

    with _lock:
        if _cached_llm is None:
            # Default 0: Groq's SDK otherwise retries 429s with long sleeps; we prefer
            # immediate failure so agent nodes can use deterministic fallbacks. Set
            # GROQ_MAX_RETRIES=2 if you want SDK-level retries instead.
            max_retries = int(os.getenv("GROQ_MAX_RETRIES", "0"))
            _cached_llm = ChatGroq(
                model=resolved_model,
                temperature=temperature,
                max_retries=max_retries,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
    return _cached_llm


def invoke_chat_groq(messages: list[BaseMessage]) -> Any:
    """
    Invoke the shared ChatGroq client with an optional cooldown between calls.

    Spreads LLM usage across time so low TPM tiers (e.g. 6000/min) are less likely
    to return 429 when multiple agents run in one graph. First call does not sleep.

    Set GROQ_LLM_CALL_DELAY_SEC=0 to disable (default from env: 20).
    """
    global _last_llm_call_monotonic
    delay = float(os.getenv("GROQ_LLM_CALL_DELAY_SEC", "20"))
    delay = max(0.0, delay)

    with _cooldown_lock:
        if _last_llm_call_monotonic is not None and delay > 0:
            time.sleep(delay)
        response = get_chat_groq().invoke(messages)
        _last_llm_call_monotonic = time.monotonic()
        return response
