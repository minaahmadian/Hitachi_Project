from __future__ import annotations

from abc import ABC, abstractmethod
import importlib
from typing import Callable, Final, Protocol, cast
import time
from typing_extensions import override

from vectordb import Vector


class _EmbeddingsApi(Protocol):
    def create(self, *, model: str, input: list[str]) -> object: ...


class _OpenAIClientProtocol(Protocol):
    embeddings: _EmbeddingsApi


class _SentenceTransformerProtocol(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class EmbeddingProvider(ABC):
    def __init__(self, batch_size: int = 100) -> None:
        self.batch_size: Final[int] = max(1, min(batch_size, 100))

    @abstractmethod
    def embed(self, texts: list[str]) -> list[Vector]:
        raise NotImplementedError

    def embed_query(self, text: str) -> Vector:
        """Embed a single query string and return one vector."""
        return self.embed([text])[0]


class GroqEmbedder(EmbeddingProvider):
    def __init__(
        self,
        *,
        model: str = "nomic-embed-text",
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
        api_key: str | None = None,
        client: object | None = None,
    ) -> None:
        super().__init__(batch_size=batch_size)
        self.model: Final[str] = model
        self.max_retries: Final[int] = max_retries
        self.retry_delay_seconds: Final[float] = retry_delay_seconds
        self._api_key: Final[str | None] = api_key
        self._client: _OpenAIClientProtocol | None = cast(_OpenAIClientProtocol | None, client)

    @override
    def embed(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []

        embeddings: list[Vector] = []
        for offset in range(0, len(texts), self.batch_size):
            batch = texts[offset : offset + self.batch_size]
            embeddings.extend(self._embed_batch_with_retry(batch))
        return embeddings

    def _embed_batch_with_retry(self, texts: list[str]) -> list[Vector]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                client = self._get_client()
                response = client.embeddings.create(model=self.model, input=texts)
                data = cast(list[object], getattr(response, "data"))
                return [list(cast(list[float], getattr(item, "embedding"))) for item in data]
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_delay_seconds * (attempt + 1))

        raise RuntimeError(f"Embedding batch failed after retries: {last_error}")

    def _get_client(self) -> _OpenAIClientProtocol:
        if self._client is not None:
            return self._client

        import os

        api_key = self._api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set. Set it in .env or pass api_key parameter.")

        try:
            openai_module = importlib.import_module("openai")
            openai_client_ctor = cast(Callable[[], object], getattr(openai_module, "OpenAI"))
        except Exception as exc:
            raise RuntimeError("OpenAI package not installed. Install 'openai' to use GroqEmbedder.") from exc

        self._client = cast(
            _OpenAIClientProtocol,
            openai_client_ctor(api_key=api_key, base_url="https://api.groq.com/openai/v1"),
        )
        return self._client


class LocalEmbedder(EmbeddingProvider):
    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 100,
        model: object | None = None,
    ) -> None:
        super().__init__(batch_size=batch_size)
        self.model_name: Final[str] = model_name
        self._model: _SentenceTransformerProtocol | None = cast(_SentenceTransformerProtocol | None, model)

    @override
    def embed(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []

        model = self._get_model()
        vectors: list[Vector] = []
        for offset in range(0, len(texts), self.batch_size):
            batch = texts[offset : offset + self.batch_size]
            encoded = model.encode(batch)
            for vector in encoded:
                vectors.append(list(vector))
        return vectors

    def _get_model(self) -> _SentenceTransformerProtocol:
        if self._model is not None:
            return self._model

        try:
            st_module = importlib.import_module("sentence_transformers")
            sentence_transformer_ctor = cast(Callable[[str], object], getattr(st_module, "SentenceTransformer"))
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers not installed. Install it to use LocalEmbedder."
            ) from exc

        self._model = cast(_SentenceTransformerProtocol, sentence_transformer_ctor(self.model_name))
        return self._model