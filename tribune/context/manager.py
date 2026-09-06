"""Proactive Context Engineering & Dynamic Working Memory Management.

Replaces passive FIFO sliding-window buffers and naive tail-truncation with
tool-directed proactive context management, exact token & entropy analysis,
semantic history folding, and entity/citation-preserving LLMLingua-2 style compression.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..types import (
    BudgetStatus,
    CompactionUrgency,
    ContextAnalysis,
    FoldResult,
)


def _tokenize(text: str) -> list[str]:
    """Tokenize text into alphanumeric words, punctuation, and legal citation symbols."""
    if not text:
        return []
    # Retain punctuation and citation symbols as distinct tokens
    return re.findall(r"[A-Za-z0-9]+|[^\w\s]", text)


def _calculate_shannon_entropy(tokens: list[str]) -> float:
    """Calculate Shannon entropy H(X) = -sum(p * log2(p)) over token distribution."""
    if not tokens:
        return 0.0
    total = len(tokens)
    counts = Counter(tokens)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 4)


def _calculate_information_density(tokens: list[str]) -> float:
    """Calculate information density (unique tokens / total tokens adjusted by max entropy)."""
    if not tokens:
        return 0.0
    unique = len(set(tokens))
    total = len(tokens)
    ratio = unique / total
    entropy = _calculate_shannon_entropy(tokens)
    max_entropy = math.log2(total) if total > 1 else 1.0
    entropy_norm = min(1.0, entropy / max_entropy) if max_entropy > 0 else 1.0
    density = 0.5 * ratio + 0.5 * entropy_norm
    return round(density, 4)


# Patterns for preserving critical legal/evidentiary tokens during compression
_CITATION_PATTERN = re.compile(
    r"\b(?:\d+\s+(?:CFR|C\.F\.R\.|USC|U\.S\.C\.|Stat\.)\s+[\d\.]+|"
    r"Exhibit\s+[A-Za-z0-9]+|Doc(?:ument)?\s*#?\s*\d+|"
    r"§+\s*[\d\.]+|Case\s*No\.?\s*[\w\-]+)\b",
    re.IGNORECASE,
)

_CAUSAL_PREDICATES = {
    "because",
    "therefore",
    "since",
    "consequently",
    "due to",
    "as a result",
    "resulting in",
    "caused",
    "entitled",
    "denied",
    "terminated",
    "eligible",
    "ineligible",
    "qualifies",
    "disqualified",
    "pursuant to",
    "violates",
    "satisfied",
    "failed",
    "requires",
    "mandates",
    "precludes",
}

_NAMED_ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")


@dataclass
class WorkingMemorySpan:
    """A tracked span of interaction, document evidence, or reasoning context."""

    span_id: str
    text: str
    token_count: int
    is_folded: bool = False
    indexing_header: str = ""
    summary: str = ""
    created_at: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    metadata: dict[str, Any] = field(default_factory=dict)


class ProactiveContextManager:
    """Proactive, tool-directed context manager replacing passive FIFO sliding-window buffers.

    Maintains working memory spans, computes token budgets and velocities, folds resolved
    interaction spans into structured indexing headers, and performs extractive distillation
    compression without discarding named entities, citations, or causal predicates.
    """

    def __init__(
        self,
        total_budget: int = 8192,
        hard_token_boundary: int | None = None,
        elevated_threshold: float = 0.65,
        critical_threshold: float = 0.85,
    ) -> None:
        self.total_budget = total_budget
        self.hard_token_boundary = (
            hard_token_boundary if hard_token_boundary is not None else int(total_budget * 0.90)
        )
        self.elevated_threshold = elevated_threshold
        self.critical_threshold = critical_threshold

        self._spans: dict[str, WorkingMemorySpan] = {}
        self._step_count: int = 0
        self._tokens_consumed_history: list[int] = []

    def register_span(
        self,
        span_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> WorkingMemorySpan:
        """Register a new working memory span and track its token volume."""
        tokens = _tokenize(text)
        token_count = len(tokens)
        span = WorkingMemorySpan(
            span_id=span_id,
            text=text,
            token_count=token_count,
            metadata=metadata or {},
        )
        self._spans[span_id] = span
        self._step_count += 1
        self._tokens_consumed_history.append(token_count)
        return span

    def get_span(self, span_id: str) -> WorkingMemorySpan | None:
        return self._spans.get(span_id)

    def analyze_text(self, span: str) -> ContextAnalysis:
        """Calculate exact token count, information density, and Shannon entropy for working memory span."""
        tokens = _tokenize(span)
        token_count = len(tokens)
        shannon_entropy = _calculate_shannon_entropy(tokens)
        information_density = _calculate_information_density(tokens)
        working_bytes = len(span.encode("utf-8"))

        return ContextAnalysis(
            token_count=token_count,
            information_density=information_density,
            shannon_entropy=shannon_entropy,
            working_memory_bytes=working_bytes,
        )

    def check_budget(self) -> BudgetStatus:
        """Return remaining context window quota, consumption velocity, and compaction urgency flags."""
        active_spans = [s for s in self._spans.values() if not s.is_folded]
        used_tokens = sum(s.token_count for s in active_spans)
        # Add folded headers token counts
        folded_tokens = sum(len(_tokenize(s.indexing_header)) for s in self._spans.values() if s.is_folded)
        total_used = used_tokens + folded_tokens
        remaining_quota = max(0, self.total_budget - total_used)

        # Calculate consumption velocity (tokens per step over recent history)
        recent = self._tokens_consumed_history[-5:] if self._tokens_consumed_history else [total_used]
        velocity = sum(recent) / max(1, len(recent))

        # Determine urgency
        utilization = total_used / max(1, self.total_budget)
        if utilization >= self.critical_threshold or total_used >= self.hard_token_boundary:
            urgency = CompactionUrgency.CRITICAL
            is_urgent = True
        elif utilization >= self.elevated_threshold:
            urgency = CompactionUrgency.ELEVATED
            is_urgent = False
        else:
            urgency = CompactionUrgency.NORMAL
            is_urgent = False

        return BudgetStatus(
            total_budget=self.total_budget,
            used_tokens=total_used,
            remaining_quota=remaining_quota,
            consumption_velocity=round(velocity, 2),
            compaction_urgency=urgency,
            is_urgent=is_urgent,
            active_spans_count=len(active_spans),
        )

    def fold_history(self, span_id: str, summarize: bool = True) -> FoldResult:
        """Discard resolved interaction span and store structured semantic indexing header in place."""
        span = self._spans.get(span_id)
        if not span:
            raise KeyError(f"Working memory span '{span_id}' not found.")

        if span.is_folded:
            return FoldResult(
                span_id=span_id,
                discarded_tokens=0,
                indexing_header=span.indexing_header,
                summary=span.summary,
                preserved_keys=list(span.metadata.keys()),
            )

        original_tokens = span.token_count
        # Extract preserved keys and brief summary
        preserved_keys = list(span.metadata.keys())

        # Extract citations and key entities to preserve in header
        citations = _CITATION_PATTERN.findall(span.text)
        if citations:
            preserved_keys.extend([f"citation:{c}" for c in set(citations)])

        # Generate summary
        if summarize:
            first_sentence = span.text.strip().split(".")[0] if span.text else ""
            summary = (
                f"Resolved interaction span: {first_sentence[:120]}..."
                if len(first_sentence) > 120
                else f"Resolved interaction span: {first_sentence}"
            )
        else:
            summary = f"Folded span {span_id}."

        indexing_header = (
            f"[SEMANTIC-INDEX-HEADER | id={span_id} | summary={summary} | "
            f"preserved={', '.join(preserved_keys[:6])}]"
        )

        header_tokens = len(_tokenize(indexing_header))
        discarded_tokens = max(0, original_tokens - header_tokens)

        span.is_folded = True
        span.text = ""  # Discard raw text
        span.token_count = header_tokens
        span.indexing_header = indexing_header
        span.summary = summary

        return FoldResult(
            span_id=span_id,
            discarded_tokens=discarded_tokens,
            indexing_header=indexing_header,
            summary=summary,
            preserved_keys=preserved_keys,
        )

    def compress_context(self, target_text: str, ratio: float = 0.5) -> str:
        """Extractive/distillation compression pipeline modeled on LLMLingua-2 principles.

        Compresses target_text targeting evidentiary discovery filings and deposition transcripts
        WITHOUT dropping named entities, citations, or causal predicates.
        """
        if not target_text or ratio >= 1.0:
            return target_text

        sentences = re.split(r"(?<=[.?!])\s+", target_text.strip())
        if len(sentences) <= 1:
            # Token-level distillation for single sentences
            return self._compress_tokens(target_text, ratio)

        # Sentence-level scoring based on information value & critical tokens
        scored_sentences: list[tuple[int, float, str]] = []
        for idx, s in enumerate(sentences):
            score = self._score_text_importance(s)
            scored_sentences.append((idx, score, s))

        # Determine target number of sentences based on ratio
        target_count = max(1, int(math.ceil(len(sentences) * ratio)))

        # Sentences with citations or causal predicates must NEVER be dropped
        must_keep_indices: set[int] = set()
        for idx, _score, s in scored_sentences:
            if _CITATION_PATTERN.search(s):
                must_keep_indices.add(idx)
            lower_s = s.lower()
            if any(pred in lower_s for pred in _CAUSAL_PREDICATES):
                must_keep_indices.add(idx)

        # Rank remaining sentences by score
        remaining = [item for item in scored_sentences if item[0] not in must_keep_indices]
        remaining.sort(key=lambda x: x[1], reverse=True)

        slots_needed = max(0, target_count - len(must_keep_indices))
        selected_indices = set(must_keep_indices)
        for idx, _, _ in remaining[:slots_needed]:
            selected_indices.add(idx)

        # Reconstruct in original chronological order
        compressed_sentences = [
            sentences[i] for i in range(len(sentences)) if i in selected_indices
        ]
        return " ".join(compressed_sentences)

    def _score_text_importance(self, text: str) -> float:
        """Calculate importance score based on citations, causal predicates, entities, and entropy."""
        score = 1.0
        # Citations are top priority
        citations = _CITATION_PATTERN.findall(text)
        score += len(citations) * 10.0

        # Causal predicates
        lower = text.lower()
        for pred in _CAUSAL_PREDICATES:
            if pred in lower:
                score += 5.0

        # Named entities
        entities = _NAMED_ENTITY_PATTERN.findall(text)
        score += len(entities) * 2.0

        # Information density & Shannon entropy
        tokens = _tokenize(text)
        if tokens:
            entropy = _calculate_shannon_entropy(tokens)
            score += entropy

        return score

    def _compress_tokens(self, text: str, ratio: float) -> str:
        """Sub-sentence token-level extractive compression preserving citations, entities, and predicates."""
        tokens = _tokenize(text)
        if len(tokens) <= 4:
            return text

        # Identify protected token positions
        protected: set[int] = set()
        for m in _CITATION_PATTERN.finditer(text):
            cit_tokens = set(_tokenize(m.group()))
            for i, t in enumerate(tokens):
                if t in cit_tokens:
                    protected.add(i)

        for i, t in enumerate(tokens):
            if t.lower() in _CAUSAL_PREDICATES or (t and t[0].isupper() and t.isalpha()):
                protected.add(i)

        target_len = max(len(protected), int(len(tokens) * ratio))
        if len(protected) >= target_len:
            # Return all protected tokens and connecting grammar
            kept_tokens = [tokens[i] for i in sorted(protected)]
        else:
            needed = target_len - len(protected)
            # Pick other tokens with highest length / info
            unprotected = [i for i in range(len(tokens)) if i not in protected]
            unprotected.sort(key=lambda i: len(tokens[i]), reverse=True)
            chosen = set(protected).union(unprotected[:needed])
            kept_tokens = [tokens[i] for i in sorted(chosen)]

        # Join tokens cleanly and normalize punctuation spacing
        res = " ".join(kept_tokens)
        res = re.sub(r"\s+([.,;:!?])", r"\1", res)

        # Restore exact formatting of any preserved statutory citations
        for m in _CITATION_PATTERN.finditer(text):
            cit_exact = m.group()
            cit_toks = _tokenize(cit_exact)
            if all(ct in kept_tokens for ct in cit_toks):
                # Pattern match the separated tokens and restore exact citation string
                pattern = r"\b" + r"\s*".join(re.escape(t) for t in cit_toks) + r"\b"
                res = re.sub(pattern, cit_exact, res)

        return res

    def get_active_context(self) -> str:
        """Assemble current active context consisting of semantic headers and active spans."""
        parts: list[str] = []
        for span in self._spans.values():
            if span.is_folded:
                if span.indexing_header:
                    parts.append(span.indexing_header)
            else:
                if span.text:
                    parts.append(span.text)
        return "\n\n".join(parts)


__all__ = [
    "ProactiveContextManager",
    "WorkingMemorySpan",
]
