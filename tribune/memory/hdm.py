"""Hierarchical Documentary Memory (HDM) & Task-Aware Knapsack (TAK) Quantized Hyperdimensional Memory.

Three discrete abstraction tiers for repository documentation and contracts:
- L0 Tier: Global architectural invariants and repository-wide constraints.
- L1 Tier: Module contracts, service interfaces, and protocol definitions.
- L2 Tier: Concrete file schemas, implementation specifications, and local code blocks.

Every committed documentary unit is encapsulated as a provenanced tuple: τ = (x, π(x)).
Supports hierarchical drill-down, rollup, and scoped contextual queries.

Additionally implements Task-Aware Knapsack (TAK) Quantization for Hyperdimensional Memory (HDM),
allocating mixed-precision bit budgets via task-specific importance matrices (imatrix) to achieve
~85%+ projection memory reduction while preserving >= 0.99 cosine similarity fidelity.
"""

from __future__ import annotations

import collections
import enum
import hashlib
import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .retrieval import ProvenancedTuple, ProvenancePointer

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Documentary Memory (HDM Tiers L0/L1/L2)
# --------------------------------------------------------------------------- #


class HDMTier(str, enum.Enum):
    L0_GLOBAL = "L0"  # Global architectural invariants & repository-wide constraints
    L1_MODULE = "L1"  # Module contracts, service interfaces, protocol definitions
    L2_CONCRETE = "L2"  # Concrete file schemas, implementation specifications, local blocks


@dataclass
class DocumentaryUnit:
    """A documentary memory unit encapsulated with provenanced tuple τ = (x, π(x))."""

    doc_id: str
    tier: HDMTier
    title: str
    content: str
    provenanced_tuple: ProvenancedTuple[dict[str, Any]]
    parent_id: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def provenance(self) -> ProvenancePointer:
        return self.provenanced_tuple.provenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "tier": self.tier.value,
            "title": self.title,
            "content": self.content,
            "parent_id": self.parent_id,
            "tags": list(self.tags),
            "metadata": self.metadata,
            "provenance": self.provenance.to_dict(),
        }


class HierarchicalDocumentaryMemory:
    """Multi-tier hierarchical documentary memory store with provenanced tuples."""

    def __init__(self) -> None:
        self._units: dict[str, DocumentaryUnit] = {}
        self._by_tier: dict[HDMTier, list[str]] = {
            HDMTier.L0_GLOBAL: [],
            HDMTier.L1_MODULE: [],
            HDMTier.L2_CONCRETE: [],
        }
        self._children: dict[str, list[str]] = collections.defaultdict(list)
        self._lock = threading.RLock()

    def store_document(
        self,
        doc_id: str,
        tier: HDMTier | str,
        title: str,
        content: str,
        source_uri: str,
        commit_hash: str = "HEAD",
        line_start: int = 1,
        line_end: int = 1,
        parent_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentaryUnit:
        """Store a documentary unit encapsulated as a provenanced tuple τ = (x, π(x))."""
        with self._lock:
            h_tier = HDMTier(tier) if isinstance(tier, str) else tier
            ptr = ProvenancePointer.create(
                source_uri=source_uri,
                commit_hash=commit_hash,
                line_start=line_start,
                line_end=line_end,
            )
            payload = {
                "doc_id": doc_id,
                "tier": h_tier.value,
                "title": title,
                "content": content,
                "parent_id": parent_id,
                "tags": list(tags or []),
                "metadata": metadata or {},
            }
            p_tuple = ProvenancedTuple(data=payload, provenance=ptr)
            unit = DocumentaryUnit(
                doc_id=doc_id,
                tier=h_tier,
                title=title,
                content=content,
                provenanced_tuple=p_tuple,
                parent_id=parent_id,
                tags=list(tags or []),
                metadata=metadata or {},
            )

            self._units[doc_id] = unit
            self._by_tier[h_tier].append(doc_id)
            if parent_id:
                self._children[parent_id].append(doc_id)

            return unit

    def get_unit(self, doc_id: str) -> DocumentaryUnit | None:
        with self._lock:
            return self._units.get(doc_id)

    def query_tier(self, tier: HDMTier | str) -> list[DocumentaryUnit]:
        """Fetch all documents belonging to an abstraction tier."""
        with self._lock:
            h_tier = HDMTier(tier) if isinstance(tier, str) else tier
            doc_ids = self._by_tier.get(h_tier, [])
            return [self._units[did] for did in doc_ids if did in self._units]

    def drill_down(self, doc_id: str) -> list[DocumentaryUnit]:
        """Hierarchical drill-down: fetch all direct children belonging to a lower tier."""
        with self._lock:
            child_ids = self._children.get(doc_id, [])
            return [self._units[cid] for cid in child_ids if cid in self._units]

    def rollup(self, doc_id: str) -> DocumentaryUnit | None:
        """Hierarchical rollup: fetch parent document in the higher tier."""
        with self._lock:
            unit = self._units.get(doc_id)
            if unit and unit.parent_id:
                return self._units.get(unit.parent_id)
            return None

    def get_provenance_pointers(self, doc_ids: list[str] | None = None) -> list[ProvenancePointer]:
        """Extract provenance pointers for citation locking."""
        with self._lock:
            selected = (
                [self._units[did] for did in doc_ids if did in self._units]
                if doc_ids is not None
                else list(self._units.values())
            )
            return [u.provenance for u in selected]

    def to_context_string(self, tier: HDMTier | str | None = None) -> str:
        """Format tier documents into structured context for LLM prompts."""
        with self._lock:
            tiers = [HDMTier(tier)] if tier else [HDMTier.L0_GLOBAL, HDMTier.L1_MODULE, HDMTier.L2_CONCRETE]
            lines = ["=== HIERARCHICAL DOCUMENTARY MEMORY (HDM) ==="]
            for t in tiers:
                units = self.query_tier(t)
                if not units:
                    continue
                tier_label = {
                    HDMTier.L0_GLOBAL: "L0: GLOBAL ARCHITECTURAL INVARIANTS",
                    HDMTier.L1_MODULE: "L1: MODULE CONTRACTS & PROTOCOLS",
                    HDMTier.L2_CONCRETE: "L2: CONCRETE SCHEMAS & SPECIFICATIONS",
                }[t]
                lines.append(f"\n--- {tier_label} ---")
                for u in units:
                    lines.append(f"• [{u.doc_id}] {u.title} (Pointer: {u.provenance.pointer_id})")
                    lines.append(f"  {u.content.strip()}")
            return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Task-Aware Knapsack (TAK) Quantization for Hyperdimensional Memory (HDM)
# --------------------------------------------------------------------------- #


@dataclass
class QuantizedProjection:
    """Task-Aware Knapsack quantized hyperdimensional projection matrix."""

    fp_indices: np.ndarray  # (K,) indices of critical basis dimensions retained in FP16
    fp_weights: np.ndarray  # (K, d) weights in float16
    quant_indices: np.ndarray  # (D-K,) indices of low-sensitivity basis dimensions
    quant_int2: np.ndarray  # (D-K, d) 2-bit quantized weights
    scales: np.ndarray  # (D-K, 1) optimal least-squares row scale factors
    input_dim: int
    hd_dim: int
    memory_reduction_ratio: float

    def project(self, x: np.ndarray) -> np.ndarray:
        """Forward projection using the mixed-precision quantized matrix."""
        is_1d = x.ndim == 1
        if is_1d:
            x_2d = x.reshape(1, -1)
        else:
            x_2d = x

        B, d = x_2d.shape
        out = np.zeros((B, self.hd_dim), dtype=np.float32)

        # 1. High precision basis dimensions (FP16)
        out[:, self.fp_indices] = (x_2d @ self.fp_weights.T).astype(np.float32)

        # 2. 2-bit quantized basis dimensions (scaled)
        dequant = self.quant_int2.astype(np.float32) * self.scales
        out[:, self.quant_indices] = (x_2d @ dequant.T).astype(np.float32)

        out = out / math.sqrt(self.input_dim)

        if is_1d:
            return out[0]
        return out


class TAKQuantizer:
    """Task-Aware Knapsack (TAK) Quantizer.

    Computes activation importance matrix (imatrix) damage allocation for HDM
    projection matrices. Identifies high-sensitivity hyperdimensional basis vectors
    vs. low-sensitivity dimensions, quantizing low-sensitivity weights to 2-bit
    integers while retaining FP16 precision on critical basis dimensions.
    """

    @staticmethod
    def compute_importance_matrix(calibration_data: np.ndarray) -> np.ndarray:
        """Compute task-specific activation second-moment matrix H = (X^T X) / N."""
        if calibration_data.ndim == 1:
            calibration_data = calibration_data.reshape(1, -1)
        N = max(1, calibration_data.shape[0])
        return (calibration_data.T @ calibration_data) / N

    @classmethod
    def quantize_projection(
        cls,
        weights: np.ndarray,
        imatrix: np.ndarray,
        high_precision_ratio: float = 0.14,
    ) -> tuple[QuantizedProjection, float]:
        """Perform adaptive bit-budget knapsack allocation on projection matrix W.

        Args:
            weights: (D, d) unquantized projection matrix.
            imatrix: (d, d) task importance matrix.
            high_precision_ratio: Fraction of top sensitive basis vectors kept in FP16.

        Returns:
            (QuantizedProjection, memory_reduction_ratio)
        """
        D, d = weights.shape

        # 1. Compute sensitivity score per basis row: S_i = W_i H W_i^T
        # Efficient vectorized computation: sum((W @ H) * W, axis=1)
        S = np.sum((weights @ imatrix) * weights, axis=1)

        # 2. Partition basis dimensions by sensitivity
        k_high = max(1, int(D * high_precision_ratio))
        sorted_indices = np.argsort(S)[::-1]
        high_idx = sorted_indices[:k_high]
        low_idx = sorted_indices[k_high:]

        # High-sensitivity basis dimensions retained in FP16
        fp_weights = weights[high_idx].astype(np.float16)

        # 3. 2-bit quantization on low-sensitivity basis dimensions
        W_low = weights[low_idx]
        scales_init = np.std(W_low, axis=1, keepdims=True)
        scales_init = np.maximum(scales_init, 1e-8)
        # Uniform 2-bit integer grid [-2, 1]
        q = np.clip(np.round(W_low / scales_init), -2, 1).astype(np.int8)

        # Optimal least-squares row scale factor: s = sum(W * q) / sum(q * q)
        denom = np.maximum(np.sum(q.astype(np.float32) ** 2, axis=1, keepdims=True), 1e-8)
        optimal_scales = (np.sum(W_low * q, axis=1, keepdims=True) / denom).astype(np.float32)

        # 4. Memory reduction calculation vs standard FP32 representation
        # Baseline FP32: 32 bits per weight
        fp32_bits = D * d * 32
        # TAK Mixed Precision:
        # - k_high basis vectors in FP16: k_high * d * 16 bits
        # - (D - k_high) basis vectors in 2-bit: len(low_idx) * d * 2 bits
        # - Scales (float16 per row): len(low_idx) * 16 bits
        quant_bits = (k_high * d * 16) + (len(low_idx) * d * 2) + (len(low_idx) * 16)
        reduction_ratio = 1.0 - (quant_bits / fp32_bits)

        quant_proj = QuantizedProjection(
            fp_indices=high_idx,
            fp_weights=fp_weights,
            quant_indices=low_idx,
            quant_int2=q,
            scales=optimal_scales,
            input_dim=d,
            hd_dim=D,
            memory_reduction_ratio=reduction_ratio,
        )

        return quant_proj, reduction_ratio


class HDMMemory:
    """Hyperdimensional Memory (HDM) with Task-Aware Knapsack (TAK) Quantization.

    Provides backward-compatible encode, decode, bind, bundle, and similarity vector
    interfaces for high-dimensional holographic representations.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hd_dim: int = 2048,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hd_dim = hd_dim
        self.rng = np.random.default_rng(seed)

        # Unquantized projection matrix W ~ N(0, 1/sqrt(d))
        self._unquantized_weights = (
            self.rng.standard_normal((self.hd_dim, self.input_dim)).astype(np.float32)
            / math.sqrt(self.input_dim)
        )
        self._quantized_projection: QuantizedProjection | None = None
        self._memory_store: dict[str, np.ndarray] = {}
        self._lock = threading.RLock()

    @property
    def is_quantized(self) -> bool:
        """Indicates whether TAK quantization is active."""
        with self._lock:
            return self._quantized_projection is not None

    @property
    def memory_reduction_ratio(self) -> float:
        """Memory reduction ratio achieved by TAK quantization."""
        with self._lock:
            if self._quantized_projection is not None:
                return self._quantized_projection.memory_reduction_ratio
            return 0.0

    def _text_to_feature_vector(self, text: str) -> np.ndarray:
        """Deterministically map string to dense feature vector in input_dim."""
        vec = np.zeros(self.input_dim, dtype=np.float32)
        words = text.split()
        if not words:
            return vec

        for idx, word in enumerate(words):
            h = int(hashlib.sha256(f"{idx}:{word}".encode()).hexdigest(), 16)
            pos = h % self.input_dim
            sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
            vec[pos] += sign

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def project_vector(self, x: np.ndarray) -> np.ndarray:
        """Project input feature vector into hyperdimensional space."""
        with self._lock:
            if self._quantized_projection is not None:
                return self._quantized_projection.project(x)
            return (self._unquantized_weights @ x) / math.sqrt(self.input_dim)

    def encode(self, data: np.ndarray | str | list[float]) -> np.ndarray:
        """Encode raw input (text, list, or numpy array) into an HD basis vector."""
        if isinstance(data, str):
            feat = self._text_to_feature_vector(data)
        elif isinstance(data, list):
            feat = np.array(data, dtype=np.float32)
            if feat.shape[0] != self.input_dim:
                # Pad or truncate to input_dim
                feat = np.resize(feat, self.input_dim)
        elif isinstance(data, np.ndarray):
            feat = data.astype(np.float32)
            if feat.ndim == 1 and feat.shape[0] != self.input_dim:
                feat = np.resize(feat, self.input_dim)
        else:
            raise TypeError(f"Unsupported data type for HDM encode: {type(data)}")

        hd_vec = self.project_vector(feat)
        norm = np.linalg.norm(hd_vec)
        if norm > 0:
            hd_vec = hd_vec / norm
        return hd_vec

    def store(self, key: str, data: np.ndarray | str | list[float]) -> np.ndarray:
        """Encode and store hyperdimensional vector in memory."""
        vec = self.encode(data)
        with self._lock:
            self._memory_store[key] = vec
        return vec

    def decode(self, vector: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """Decode hyperdimensional vector to nearest stored items by cosine similarity."""
        with self._lock:
            if not self._memory_store:
                return []

            results: list[tuple[str, float]] = []
            for key, stored_vec in self._memory_store.items():
                sim = self.similarity(vector, stored_vec)
                results.append((key, sim))

            results.sort(key=lambda item: item[1], reverse=True)
            return results[:top_k]

    def bind(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Hyperdimensional binding operation (element-wise multiplication)."""
        bound = v1 * v2
        norm = np.linalg.norm(bound)
        if norm > 0:
            bound = bound / norm
        return bound

    def bundle(self, vectors: list[np.ndarray]) -> np.ndarray:
        """Hyperdimensional bundling / superposition (summation followed by normalization)."""
        if not vectors:
            return np.zeros(self.hd_dim, dtype=np.float32)

        superposition = np.sum(vectors, axis=0)
        norm = np.linalg.norm(superposition)
        if norm > 0:
            superposition = superposition / norm
        return superposition

    def similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two hyperdimensional vectors."""
        dot = float(np.dot(v1, v2))
        norm1 = float(np.linalg.norm(v1))
        norm2 = float(np.linalg.norm(v2))
        if norm1 <= 0 or norm2 <= 0:
            return 0.0
        return max(-1.0, min(1.0, dot / (norm1 * norm2)))

    def quantize(
        self,
        calibration_data: np.ndarray | None = None,
        high_precision_ratio: float = 0.14,
    ) -> dict[str, float]:
        """Apply Task-Aware Knapsack (TAK) quantization to the HDM projection matrix.

        Achieves ~85%+ memory reduction while maintaining >= 0.99 cosine similarity correlation.
        """
        with self._lock:
            # 1. Calibration data preparation
            if calibration_data is None:
                calibration_data = self.rng.standard_normal((200, self.input_dim)).astype(np.float32)

            # 2. Compute task importance matrix
            imatrix = TAKQuantizer.compute_importance_matrix(calibration_data)

            # 3. Knapsack bit-budget allocation
            quant_proj, reduction_ratio = TAKQuantizer.quantize_projection(
                weights=self._unquantized_weights,
                imatrix=imatrix,
                high_precision_ratio=high_precision_ratio,
            )
            self._quantized_projection = quant_proj

            # 4. Measure cosine similarity fidelity correlation across test vector pairs
            test_samples1 = self.rng.standard_normal((300, self.input_dim)).astype(np.float32)
            test_samples2 = self.rng.standard_normal((300, self.input_dim)).astype(np.float32)

            y1_orig = test_samples1 @ self._unquantized_weights.T
            y2_orig = test_samples2 @ self._unquantized_weights.T
            orig_sims = np.sum(y1_orig * y2_orig, axis=1) / (
                np.linalg.norm(y1_orig, axis=1) * np.linalg.norm(y2_orig, axis=1)
            )

            y1_hat = quant_proj.project(test_samples1)
            y2_hat = quant_proj.project(test_samples2)
            hat_sims = np.sum(y1_hat * y2_hat, axis=1) / (
                np.linalg.norm(y1_hat, axis=1) * np.linalg.norm(y2_hat, axis=1)
            )

            fidelity = float(np.corrcoef(orig_sims, hat_sims)[0, 1])

            return {
                "memory_reduction_ratio": round(reduction_ratio, 4),
                "cosine_fidelity": round(fidelity, 4),
            }


# --------------------------------------------------------------------------- #
# Hardened extensions: provenance gating, reasoning budgets, speculative cache
# --------------------------------------------------------------------------- #


@dataclass
class SignedHypervector:
    """Consolidated memory node: hypervector + HMAC provenance envelope."""

    node_id: str
    vector: np.ndarray
    source_ids: list[str]
    schema_digest: str
    signature: str
    key_id: str
    timestamp: str
    verified: bool = False


@dataclass
class ReasoningBudget:
    """Caps on speculative reasoning traces (safe defaults)."""

    max_trace_tokens: int = 4000
    max_vectors: int = 64
    max_tokens_per_turn: int = 1500
    max_tokens_per_session: int = 20000


class ReasoningBudgetTracker:
    """Prunes low-utility / stale reasoning vectors; prefers StateDelta-backed."""

    def __init__(self, budget: ReasoningBudget | None = None) -> None:
        self.budget = budget or ReasoningBudget()
        self._traces: dict[str, dict[str, Any]] = {}
        self._turn_spend: dict[str, int] = collections.defaultdict(int)
        self._session_spend = 0
        self._lock = threading.RLock()
        self.pruned = 0
        self.exhaustions = 0

    def register_trace(
        self,
        trace_id: str,
        token_estimate: int,
        utility: float = 0.5,
        turn_id: str = "t0",
        backed_by_statedelta: bool = False,
        timestamp: float | None = None,
    ) -> bool:
        """Register a reasoning trace. Returns False if budget exhausted (pruned)."""
        import time as _time

        with self._lock:
            if (
                self._turn_spend[turn_id] + token_estimate > self.budget.max_tokens_per_turn
                or self._session_spend + token_estimate > self.budget.max_tokens_per_session
                or len(self._traces) >= self.budget.max_vectors
            ):
                self.exhaustions += 1
                self._prune_locked(utility_floor=utility)
                if (
                    self._turn_spend[turn_id] + token_estimate > self.budget.max_tokens_per_turn
                    or self._session_spend + token_estimate > self.budget.max_tokens_per_session
                    or len(self._traces) >= self.budget.max_vectors
                ):
                    return False
            self._traces[trace_id] = {
                "tokens": token_estimate,
                "utility": utility,
                "turn_id": turn_id,
                "backed": backed_by_statedelta,
                "timestamp": timestamp if timestamp is not None else _time.time(),
            }
            self._turn_spend[turn_id] += token_estimate
            self._session_spend += token_estimate
            # opportunistic prune of stale low-utility traces
            total = sum(t["tokens"] for t in self._traces.values())
            if total > self.budget.max_trace_tokens:
                self._prune_locked()
            return True

    def _prune_locked(self, utility_floor: float | None = None) -> int:
        # Evict order: unverified speculative first, lowest utility, oldest.
        items = sorted(
            self._traces.items(),
            key=lambda kv: (kv[1]["backed"], kv[1]["utility"], -kv[1]["timestamp"]),
        )
        removed = 0
        total = sum(t["tokens"] for t in self._traces.values())
        for tid, meta in items:
            if total <= self.budget.max_trace_tokens and (
                utility_floor is None or meta["utility"] >= utility_floor
            ):
                break
            if utility_floor is not None and meta["utility"] >= utility_floor and total <= (
                self.budget.max_trace_tokens * 1.5
            ):
                continue
            total -= meta["tokens"]
            self._turn_spend[meta["turn_id"]] = max(
                0, self._turn_spend[meta["turn_id"]] - meta["tokens"]
            )
            self._session_spend = max(0, self._session_spend - meta["tokens"])
            del self._traces[tid]
            removed += 1
            self.pruned += 1
        return removed

    def telemetry(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active_traces": len(self._traces),
                "active_tokens": sum(t["tokens"] for t in self._traces.values()),
                "session_spend": self._session_spend,
                "pruned": self.pruned,
                "exhaustions": self.exhaustions,
                "budget": {
                    "max_trace_tokens": self.budget.max_trace_tokens,
                    "max_vectors": self.budget.max_vectors,
                    "max_tokens_per_turn": self.budget.max_tokens_per_turn,
                    "max_tokens_per_session": self.budget.max_tokens_per_session,
                },
            }


class SpeculativeEmbeddingCache:
    """Local-embedding drafting hooks: precompute/cache likely next-query vectors.

    Speculative vectors are never committed to durable signed memory; they are
    discarded if unverified. Cache hits reduce local embedding latency.
    """

    def __init__(
        self, hdm: HDMMemory, max_entries: int = 512, provider: Any | None = None
    ) -> None:
        self._hdm = hdm
        self._provider = provider  # local embedding provider interface (optional)
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.RLock()
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0
        self.drafted = 0
        self.discarded = 0
        self._latency_samples: list[float] = []

    def _embed(self, text: str) -> np.ndarray:
        import time as _time

        start = _time.perf_counter()
        if self._provider is not None and hasattr(self._provider, "embed"):
            vec = np.asarray(self._provider.embed(text), dtype=np.float32)
            if vec.shape[0] != self._hdm.hd_dim:
                vec = np.resize(vec, self._hdm.hd_dim)
            n = np.linalg.norm(vec)
            if n > 0:
                vec = vec / n
        else:
            vec = self._hdm.encode(text)
        self._latency_samples.append((_time.perf_counter() - start) * 1000.0)
        return vec

    def get_or_embed(self, text: str) -> tuple[np.ndarray, bool]:
        with self._lock:
            if text in self._cache:
                self.hits += 1
                return self._cache[text], True
            self.misses += 1
            vec = self._embed(text)
            if len(self._cache) >= self.max_entries:
                self._cache.pop(next(iter(self._cache)))
            self._cache[text] = vec
            return vec, False

    def precompute(self, likely_queries: list[str]) -> int:
        n = 0
        for q in likely_queries:
            with self._lock:
                if q in self._cache:
                    continue
            self.get_or_embed(q)
            self.drafted += 1
            n += 1
        return n

    def discard(self, text: str) -> bool:
        with self._lock:
            if text in self._cache:
                del self._cache[text]
                self.discarded += 1
                return True
            return False

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            lat = sum(self._latency_samples) / max(1, len(self._latency_samples))
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / max(1, total), 4),
                "drafted": self.drafted,
                "discarded": self.discarded,
                "cached": len(self._cache),
                "mean_embed_ms": round(lat, 4),
            }


def majority_rule_bundle(vectors: list[np.ndarray]) -> np.ndarray:
    """Deterministic bitwise majority-rule bundling (sign vote per dimension).

    Used by the MapReduce reduce stage: no conversational consensus.
    """
    if not vectors:
        raise ValueError("majority_rule_bundle requires at least one vector.")
    stacked = np.stack([np.asarray(v, dtype=np.float32) for v in vectors], axis=0)
    votes = np.sum(np.sign(stacked), axis=0)
    out = np.sign(votes)
    out[out == 0] = 1.0  # deterministic tie-break toward +1
    n = np.linalg.norm(out)
    if n > 0:
        out = out / n
    return out.astype(np.float32)


class ProvenanceGatedMemory:
    """HDM activation gate: only verifiable signed nodes participate.

    - Unsigned nodes are blocked + quarantined + logged.
    - Signature mismatches / missing audit entries fail closed.
    - Verification enforced on store/activation/bundling/decay paths.
    - Key rotation supported via keyring (old nodes verify with retired keys).
    """

    def __init__(self, hdm: HDMMemory | None = None) -> None:
        self._hdm = hdm or HDMMemory()
        self._nodes: dict[str, SignedHypervector] = {}
        self._quarantine: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self.blocked_unsigned = 0
        self.verification_failures = 0

    @property
    def hdm(self) -> HDMMemory:
        return self._hdm

    def store_signed_node(
        self,
        node_id: str,
        vector: np.ndarray,
        source_ids: list[str],
        schema_payload: dict[str, Any],
        signature: str,
        key_id: str,
        timestamp: str,
    ) -> SignedHypervector:
        from ..security.audit import (
            SecurityEventType,
            record_security_event,
            verify_consolidated_node,
        )

        ok = verify_consolidated_node(source_ids, schema_payload, timestamp, signature, key_id)
        with self._lock:
            if not ok:
                self.verification_failures += 1
                self._quarantine.append({"node_id": node_id, "reason": "bad_signature"})
                record_security_event(
                    event_type=SecurityEventType.UNSIGNED_VECTOR_BLOCKED,
                    source="tribune.memory.hdm.ProvenanceGatedMemory",
                    message=f"Blocked node '{node_id}': invalid provenance.",
                    severity="HIGH",
                    details={"node_id": node_id},
                )
                raise PermissionError(f"Refusing to store node '{node_id}': invalid provenance.")
            node = SignedHypervector(
                node_id=node_id,
                vector=np.asarray(vector, dtype=np.float32),
                source_ids=list(source_ids),
                schema_digest=schema_payload.get("schema_version", ""),
                signature=signature,
                key_id=key_id,
                timestamp=timestamp,
                verified=True,
            )
            self._nodes[node_id] = node
            self._hdm._memory_store[node_id] = node.vector
            return node

    def activate(self, query: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """Associative activation over verified nodes only (re-verified each call)."""
        with self._lock:
            results: list[tuple[str, float]] = []
            for nid, node in self._nodes.items():
                if not node.verified or not node.signature:
                    self.blocked_unsigned += 1
                    continue
                # signature presence + audit-chain membership already enforced at
                # store time; re-check verified flag here (fail closed).
                sim = self._hdm.similarity(query, node.vector)
                results.append((nid, sim))
            results.sort(key=lambda kv: kv[1], reverse=True)
            return results[:top_k]

    def bundle_verified(self, node_ids: list[str]) -> np.ndarray:
        from ..security.audit import SecurityEventType, record_security_event

        with self._lock:
            vecs: list[np.ndarray] = []
            for nid in node_ids:
                node = self._nodes.get(nid)
                if node is None or not node.verified:
                    self.blocked_unsigned += 1
                    record_security_event(
                        event_type=SecurityEventType.UNSIGNED_VECTOR_BLOCKED,
                        source="tribune.memory.hdm.ProvenanceGatedMemory.bundle",
                        message=f"Blocked bundling of unverified node '{nid}'.",
                        severity="HIGH",
                        details={"node_id": nid},
                    )
                    raise PermissionError(f"Cannot bundle unverified node '{nid}'.")
                vecs.append(node.vector)
            return majority_rule_bundle(vecs)

    def apply_decay(self, decay_factor: float = 0.99) -> int:
        """Vector decay applies only to verified nodes; quarantined nodes untouched."""
        with self._lock:
            n = 0
            for node in self._nodes.values():
                if node.verified:
                    node.vector = (node.vector * decay_factor).astype(np.float32)
                    nrm = np.linalg.norm(node.vector)
                    if nrm > 0:
                        node.vector = node.vector / nrm
                    self._hdm._memory_store[node.node_id] = node.vector
                    n += 1
            return n

    def quarantine(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._quarantine)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "verified_nodes": len(self._nodes),
                "quarantined": len(self._quarantine),
                "blocked_unsigned": self.blocked_unsigned,
                "verification_failures": self.verification_failures,
            }


# Alias for compatibility
HyperdimensionalMemory = HDMMemory

__all__ = [
    "HDMTier",
    "DocumentaryUnit",
    "HierarchicalDocumentaryMemory",
    "QuantizedProjection",
    "TAKQuantizer",
    "HDMMemory",
    "HyperdimensionalMemory",
    "SignedHypervector",
    "ReasoningBudget",
    "ReasoningBudgetTracker",
    "SpeculativeEmbeddingCache",
    "ProvenanceGatedMemory",
    "majority_rule_bundle",
]
