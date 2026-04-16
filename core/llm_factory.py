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
            _cached_llm = ChatGroq(
                model=resolved_model,
                temperature=temperature,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
    return _cached_llm
