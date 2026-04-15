from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from vectordb.types import SearchResult


class ContextStrategy(str, Enum):
    RECENT_FIRST = "RECENT_FIRST"
    HIGHEST_SCORE = "HIGHEST_SCORE"
    DIVERSITY = "DIVERSITY"


@dataclass(slots=True)
class ContextBuilder:
    strategy: ContextStrategy = ContextStrategy.HIGHEST_SCORE

    def build_context(self, search_results: list[SearchResult], max_tokens: int = 2000) -> str:
        if not search_results:
            return ""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        ordered = self._order_results(search_results)
        parts: list[str] = []
        used_tokens = 0

        for result in ordered:
            block = self.format_source(result)
            block_tokens = self._estimate_tokens(block)
            if used_tokens + block_tokens > max_tokens:
                remaining = max_tokens - used_tokens
                if remaining <= 0:
                    break

                truncated_block = self._truncate_to_tokens(block, remaining)
                if truncated_block:
                    parts.append(truncated_block.rstrip())
                break

            parts.append(block.rstrip())
            used_tokens += block_tokens

        return "\n".join(parts)

    def format_source(self, result: SearchResult) -> str:
        metadata = result.chunk.metadata
        filename = str(metadata.get("filename") or metadata.get("source") or "unknown")
        chunk_index = self._as_int(metadata.get("chunk_index"), default=0)
        total_chunks = self._as_int(metadata.get("total_chunks"), default=1)

        if total_chunks <= 0:
            total_chunks = 1
        if chunk_index < 0:
            chunk_index = 0

        human_chunk = chunk_index + 1
        if human_chunk > total_chunks:
            total_chunks = human_chunk

        score_text = f"{result.score:.4f}"
        header = f"[Source: {filename}, chunk {human_chunk}/{total_chunks}, score {score_text}]"
        body = result.chunk.text.strip()
        return f"{header}\n{body}\n---"

    def _order_results(self, results: list[SearchResult]) -> list[SearchResult]:
        if self.strategy == ContextStrategy.HIGHEST_SCORE:
            return sorted(results, key=lambda item: item.score, reverse=True)
        if self.strategy == ContextStrategy.RECENT_FIRST:
            return sorted(
                results,
                key=lambda item: self._extract_timestamp(item.chunk.metadata),
                reverse=True,
            )
        return self._diversify(results)

    def _diversify(self, results: list[SearchResult]) -> list[SearchResult]:
        ordered = sorted(results, key=lambda item: item.score, reverse=True)
        grouped: dict[str, list[SearchResult]] = {}
        for result in ordered:
            metadata = result.chunk.metadata or {}
            source = str(metadata.get("filename") or metadata.get("source") or "unknown")
            grouped.setdefault(source, []).append(result)

        diversified: list[SearchResult] = []
        source_names = list(grouped.keys())
        while True:
            added = False
            for source in source_names:
                if grouped[source]:
                    diversified.append(grouped[source].pop(0))
                    added = True
            if not added:
                break
        return diversified

    def _extract_timestamp(self, metadata: Mapping[str, object] | None) -> float:
        if not metadata:
            return float("-inf")
        for key in ("timestamp", "date", "created_at", "datetime"):
            value = metadata.get(key)
            as_float = self._to_timestamp(value)
            if as_float is not None:
                return as_float
        return float("-inf")

    @staticmethod
    def _to_timestamp(value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            normalized = value.strip().replace("Z", "+00:00")
            try:
                return datetime.fromisoformat(normalized).timestamp()
            except ValueError:
                return None
        return None

    @staticmethod
    def _as_int(value: object, default: int) -> int:
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return default
        return default

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, (len(text) + 3) // 4)

    def _truncate_to_tokens(self, text: str, token_limit: int) -> str:
        if token_limit <= 0:
            return ""
        char_limit = token_limit * 4
        if len(text) <= char_limit:
            return text
        if char_limit <= 3:
            return text[:char_limit]
        suffix = "\n[TRUNCATED: max_tokens reached]"
        if char_limit <= len(suffix) + 3:
            return text[:char_limit]
        return text[: char_limit - len(suffix) - 3].rstrip() + "..." + suffix