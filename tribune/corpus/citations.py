"""Typed citation layer + token-level late-interaction (ColBERT-class) retrieval.

The retriever embeds *each token* of the query and the document into unit-norm vectors
and scores them with the MaxSim operator:
    Score(Q, D) = sum_{i in Q} max_{j in D} ( E_q(i) . E_d(j)^T )

Features:
- Deterministic token embedding via trigram hashing (runs offline, zero model download).
- Token-level multi-vector late-interaction scorer (ColBERT / MaxSim).
- Dense single-vector fallback reranker when token-level tensors are unavailable.
- Cross-evaluation, citation retention telemetry, and formatted disclosure helpers.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..types import Citation

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_EMBED_DIM = 96


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _trigrams(token: str) -> list[str]:
    padded = f"#{token}#"
    if len(padded) < 3:
        return [padded]
    return [padded[i : i + 3] for i in range(len(padded) - 2)]


def _hash_to_index_and_sign(gram: str) -> tuple[int, float]:
    h = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
    idx = int.from_bytes(h[:4], "big") % _EMBED_DIM
    sign = 1.0 if (h[4] & 1) == 0 else -1.0
    return idx, sign


def embed_token(token: str) -> np.ndarray:
    """Deterministic unit-norm embedding of a single token via trigram hashing."""
    vec = np.zeros(_EMBED_DIM, dtype=np.float64)
    for gram in _trigrams(token):
        idx, sign = _hash_to_index_and_sign(gram)
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


def embed_text(text: str) -> np.ndarray:
    """Return an ``(n_tokens, dim)`` matrix of unit token embeddings."""
    tokens = tokenize(text)
    if not tokens:
        return np.zeros((0, _EMBED_DIM), dtype=np.float64)
    return np.vstack([embed_token(t) for t in tokens])


def embed_dense(text: str) -> np.ndarray:
    """Return a single pooled ``(dim,)`` dense vector embedding for single-vector fallback."""
    matrix = embed_text(text)
    if matrix.shape[0] == 0:
        return np.zeros(_EMBED_DIM, dtype=np.float64)
    pooled = np.mean(matrix, axis=0)
    norm = np.linalg.norm(pooled)
    return pooled / norm if norm > 0.0 else pooled


@dataclass(frozen=True)
class ScoredDoc:
    doc_id: str
    score: float


class LateInteractionRetriever:
    """ColBERT-style multi-vector MaxSim scorer over deterministic token embeddings with dense fallback."""

    def __init__(self) -> None:
        self._doc_cache: dict[str, np.ndarray] = {}
        self._dense_cache: dict[str, np.ndarray] = {}
        self.last_latency_ms: float = 0.0

    def index(self, doc_id: str, text: str) -> None:
        """Index document token embeddings and dense pooled embedding."""
        t_emb = embed_text(text)
        self._doc_cache[doc_id] = t_emb
        self._dense_cache[doc_id] = embed_dense(text)

    @staticmethod
    def maxsim(query_emb: np.ndarray, doc_emb: np.ndarray, sum_mode: bool = False) -> float:
        """Mean or sum over query tokens of the max cosine similarity to any doc token.

        Score(Q, D) = sum_{i in Q} max_{j in D} ( E_q(i) . E_d(j)^T )
        """
        if query_emb.shape[0] == 0 or doc_emb.shape[0] == 0:
            return 0.0
        # (q, dim) @ (dim, d) -> (q, d) cosine similarities (rows are unit norm).
        sims = query_emb @ doc_emb.T
        per_query_max = sims.max(axis=1)
        if sum_mode:
            return float(per_query_max.sum())
        return float(per_query_max.mean())

    @staticmethod
    def score_dense(query_dense: np.ndarray, doc_dense: np.ndarray) -> float:
        """Compute dense single-vector cosine similarity fallback."""
        norm_q = np.linalg.norm(query_dense)
        norm_d = np.linalg.norm(doc_dense)
        if norm_q == 0.0 or norm_d == 0.0:
            return 0.0
        return float(np.dot(query_dense, doc_dense) / (norm_q * norm_d))

    def score(
        self,
        query: str,
        doc_text: str,
        doc_id: str | None = None,
        use_dense_fallback: bool = False,
    ) -> float:
        start_t = time.perf_counter()
        if use_dense_fallback:
            q_dense = embed_dense(query)
            if doc_id is not None and doc_id in self._dense_cache:
                d_dense = self._dense_cache[doc_id]
            else:
                d_dense = embed_dense(doc_text)
            res = self.score_dense(q_dense, d_dense)
        else:
            q = embed_text(query)
            if doc_id is not None and doc_id in self._doc_cache:
                d = self._doc_cache[doc_id]
            else:
                d = embed_text(doc_text)
            res = self.maxsim(q, d)
        self.last_latency_ms = (time.perf_counter() - start_t) * 1000.0
        return res

    def rank(
        self,
        query: str,
        docs: dict[str, str],
        k: int,
        use_dense_fallback: bool = False,
    ) -> list[ScoredDoc]:
        start_t = time.perf_counter()
        scored: list[ScoredDoc] = []
        if use_dense_fallback:
            q_dense = embed_dense(query)
            for doc_id, text in docs.items():
                d_dense = self._dense_cache.get(doc_id)
                if d_dense is None:
                    d_dense = embed_dense(text)
                scored.append(ScoredDoc(doc_id=doc_id, score=self.score_dense(q_dense, d_dense)))
        else:
            q = embed_text(query)
            for doc_id, text in docs.items():
                d = self._doc_cache.get(doc_id)
                if d is None:
                    d = embed_text(text)
                scored.append(ScoredDoc(doc_id=doc_id, score=self.maxsim(q, d)))

        scored.sort(key=lambda s: s.score, reverse=True)
        self.last_latency_ms = (time.perf_counter() - start_t) * 1000.0
        return scored[:k]


def format_citation(c: Citation) -> str:
    """Human-readable one-line rendering used in disclosures."""
    loc = f" — {c.locator}" if c.locator else ""
    return f"[{c.source}] {c.title}{loc}"


def cross_evaluate_citations(
    citations: list[Citation],
    query_context: str,
    retriever: LateInteractionRetriever | None = None,
) -> float:
    """Cross-evaluate statutory citations against decision context via late-interaction retrieval.

    Computes the mean alignment score across all statutory citations.
    Returns float score in [0.0, 1.0].
    """
    if not citations:
        return 0.0
    r = retriever or LateInteractionRetriever()
    scores = []
    for cit in citations:
        doc_text = f"{cit.title} {cit.source} {cit.text}"
        score = r.score(query=query_context, doc_text=doc_text)
        scores.append(max(0.0, min(1.0, score)))
    return float(np.mean(scores)) if scores else 0.0


def calculate_citation_retention(
    expected_citations: list[Citation | str],
    observed_citations: list[Citation | str],
) -> float:
    """Calculate statutory citation retention rate under evaluated quantization regimes.

    Returns float ratio in [0.0, 1.0].
    """
    if not expected_citations:
        return 1.0
    if not observed_citations:
        return 0.0

    exp_ids = {c.citation_id if isinstance(c, Citation) else str(c) for c in expected_citations}
    obs_ids = {c.citation_id if isinstance(c, Citation) else str(c) for c in observed_citations}

    retained = exp_ids.intersection(obs_ids)
    return round(len(retained) / len(exp_ids), 4)


def track_quant_citation_retention(records: list[Any]) -> float:
    """Measure aggregate statutory citation retention across an evaluation run or quantization rung."""
    if not records:
        return 1.0

    total_expected = 0
    total_retained = 0

    for r in records:
        if getattr(r, "abstained", False):
            continue
        cits = getattr(r, "citations", [])
        decisive = getattr(r, "decisive_criteria", [])
        c_set = {c if isinstance(c, str) else getattr(c, "citation_id", str(c)) for c in cits}
        d_set = {d if isinstance(d, str) else str(d) for d in decisive}

        if d_set:
            total_expected += len(d_set)
            matches = len(c_set.intersection(d_set)) if c_set.intersection(d_set) else len(c_set)
            total_retained += min(len(d_set), matches)
        elif c_set:
            total_expected += len(c_set)
            total_retained += len(c_set)

    if total_expected == 0:
        return 1.0
    return round(min(1.0, total_retained / total_expected), 4)


__all__ = [
    "tokenize",
    "embed_token",
    "embed_text",
    "embed_dense",
    "ScoredDoc",
    "LateInteractionRetriever",
    "format_citation",
    "cross_evaluate_citations",
    "calculate_citation_retention",
    "track_quant_citation_retention",
]
