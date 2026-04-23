"""
Pure-Python BM25+ sparse retrieval index.

Why BM25 in a safety-critical RAG stack
---------------------------------------
Dense embeddings alone miss queries dominated by **specific technical terms**
(requirement IDs, abbreviations like PVIS / IAMS / RTM, subsystem acronyms).
BM25 is the opposite: it excels exactly where dense embeddings are weakest.
Running both in parallel and fusing their rankings (RRF) is the most reliable
single-machine retrieval strategy, consistently outperforming either path alone
in IR literature for domain-specific corpora.

This module depends on no external packages so it can run wherever the rest
of the pipeline runs. Correctness of the BM25+ formula is tested by
scripts/benchmark_rssom_retrieval.py.

BM25+ reference:  Lv & Zhai, "Lower-bounding term frequency normalization"
(CIKM 2011) — adds a ``delta`` term to the TF component so rarely seen
documents are not under-scored.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass


_TOKEN_RX = re.compile(r"[A-Za-z][A-Za-z0-9_\-]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer. Keeps hyphens so 'C6-APCS-1' stays one token."""
    if not text:
        return []
    return [tok.lower() for tok in _TOKEN_RX.findall(text)]


@dataclass(slots=True)
class BM25Hit:
    doc_id: str
    score: float
    rank: int
    text: str
    metadata: dict[str, object]


class BM25Index:
    """
    Okapi BM25+ sparse retrieval over a collection of text documents.

    Parameters
    ----------
    k1, b : standard BM25 tunables.  (k1=1.5, b=0.75 is the Lucene default and
        consistently strong on short technical text.)
    delta : BM25+ lower bound correction term (typical 1.0).
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75, delta: float = 1.0) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.delta = float(delta)
        self._doc_ids: list[str] = []
        self._doc_texts: list[str] = []
        self._doc_metas: list[dict[str, object]] = []
        self._doc_lengths: list[int] = []
        self._term_frequencies: list[dict[str, int]] = []
        self._doc_frequencies: dict[str, int] = {}
        self._avg_length: float = 0.0
        self._n_docs: int = 0

    def add(self, *, doc_id: str, text: str, metadata: dict[str, object] | None = None) -> None:
        tokens = _tokenize(text)
        self._doc_ids.append(doc_id)
        self._doc_texts.append(text)
        self._doc_metas.append(dict(metadata or {}))
        self._doc_lengths.append(len(tokens))
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1
        self._term_frequencies.append(tf)
        for tok in tf:
            self._doc_frequencies[tok] = self._doc_frequencies.get(tok, 0) + 1

    def finalize(self) -> None:
        self._n_docs = len(self._doc_ids)
        self._avg_length = (sum(self._doc_lengths) / self._n_docs) if self._n_docs else 0.0

    @property
    def ready(self) -> bool:
        return self._n_docs > 0 and self._avg_length > 0

    def search(self, query: str, *, top_k: int = 10) -> list[BM25Hit]:
        if not self.ready or top_k <= 0:
            return []
        query_terms = _tokenize(query)
        if not query_terms:
            return []

        k1, b, delta = self.k1, self.b, self.delta
        avg_len = self._avg_length
        n = self._n_docs

        scores: list[float] = [0.0] * n
        for term in set(query_terms):
            df = self._doc_frequencies.get(term)
            if not df:
                continue
            # Robertson-Sparck-Jones IDF, floored at 0 like Lucene does.
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            for idx in range(n):
                tf = self._term_frequencies[idx].get(term)
                if not tf:
                    continue
                doc_len = self._doc_lengths[idx] or 1
                norm = 1.0 - b + b * (doc_len / avg_len)
                tf_component = (tf * (k1 + 1.0)) / (tf + k1 * norm) + delta
                scores[idx] += idf * tf_component

        ranked = sorted(
            ((idx, score) for idx, score in enumerate(scores) if score > 0),
            key=lambda it: it[1],
            reverse=True,
        )[:top_k]

        return [
            BM25Hit(
                doc_id=self._doc_ids[idx],
                score=round(score, 6),
                rank=rank,
                text=self._doc_texts[idx],
                metadata=dict(self._doc_metas[idx]),
            )
            for rank, (idx, score) in enumerate(ranked, start=1)
        ]
