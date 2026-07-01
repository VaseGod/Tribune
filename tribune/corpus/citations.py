"""Typed citation layer + late-interaction (ColBERT-class) retrieval.

The retriever embeds *each token* of the query and the document and scores them
with MaxSim — for every query token, take its maximum similarity to any document
token, then average. This is the defining mechanic of late-interaction retrieval
(ColBERT). The token embeddings here are produced deterministically with the
hashing trick over character trigrams, so the whole retrieval stack runs offline,
reproducibly, with no model download — while implementing the *real* algorithm,
not a bag-of-words shortcut.

A hosted backend can swap in learned ColBERT embeddings without changing the
scoring contract (see :mod:`tribune.corpus.rule_store`).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

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


@dataclass(frozen=True)
class ScoredDoc:
    doc_id: str
    score: float


class LateInteractionRetriever:
    """ColBERT-style MaxSim scorer over deterministic token embeddings."""

    def __init__(self) -> None:
        self._doc_cache: dict[str, np.ndarray] = {}

    def index(self, doc_id: str, text: str) -> None:
        self._doc_cache[doc_id] = embed_text(text)

    @staticmethod
    def maxsim(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
        """Mean over query tokens of the max cosine similarity to any doc token."""
        if query_emb.shape[0] == 0 or doc_emb.shape[0] == 0:
            return 0.0
        # (q, dim) @ (dim, d) -> (q, d) cosine similarities (rows are unit norm).
        sims = query_emb @ doc_emb.T
        per_query_max = sims.max(axis=1)
        return float(per_query_max.mean())

    def score(self, query: str, doc_text: str, doc_id: str | None = None) -> float:
        q = embed_text(query)
        if doc_id is not None and doc_id in self._doc_cache:
            d = self._doc_cache[doc_id]
        else:
            d = embed_text(doc_text)
        return self.maxsim(q, d)

    def rank(self, query: str, docs: dict[str, str], k: int) -> list[ScoredDoc]:
        q = embed_text(query)
        scored: list[ScoredDoc] = []
        for doc_id, text in docs.items():
            d = self._doc_cache.get(doc_id)
            if d is None:
                d = embed_text(text)
            scored.append(ScoredDoc(doc_id=doc_id, score=self.maxsim(q, d)))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]


def format_citation(c: Citation) -> str:
    """Human-readable one-line rendering used in disclosures."""
    loc = f" — {c.locator}" if c.locator else ""
    return f"[{c.source}] {c.title}{loc}"
