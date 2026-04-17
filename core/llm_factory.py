from __future__ import annotations

import os
import threading
from typing import Final

from langchain_groq import ChatGroq

_lock = threading.Lock()
_cached_llm: ChatGroq | None = None

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
