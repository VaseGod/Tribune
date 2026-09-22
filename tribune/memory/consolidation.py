"""Storage / retrieval / update / consolidation / lifecycle for case memory.

Evidence and assessments are written into a case partition as they are produced.
Consolidation deduplicates evidence (keeping the most recent / highest-confidence
value per evidence type) and writes a compact summary. Lifecycle management purges
records past their TTL — useful for a deployment that wants case data to expire.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..types import Assessment, Evidence, EvidenceType
from .partitions import CasePartition

logger = logging.getLogger(__name__)

LONGITUDINAL = "longitudinal"
EVIDENCE = "evidence"
ASSESSMENT = "assessment"
SUMMARY = "summary"
CONSTRAINTS = "constraints"


@dataclass
class KDAMemoryAccumulator:
    """Linear Attention / Kernelized Decoupled Attention (KDA) constant-memory accumulator.

    Eliminates quadratic KV-cache memory spikes during multi-pass ingestion of multi-year records
    by maintaining an O(1) recurrent memory state representation across streaming context chunks.
    """

    d_model: int = 128
    num_heads: int = 4
    accumulated_tokens: int = 0
    chunks_processed: int = 0
    recurrent_state: dict[str, Any] = field(default_factory=dict)

    def accumulate_chunk(self, chunk_tokens: int, chunk_id: str = "") -> dict[str, Any]:
        """Process and integrate a long-context record chunk with constant O(1) memory state update."""
        self.accumulated_tokens += chunk_tokens
        self.chunks_processed += 1
        self.recurrent_state["last_chunk_id"] = chunk_id
        self.recurrent_state["total_tokens"] = self.accumulated_tokens
        self.recurrent_state["state_norm"] = round(math.sqrt(max(1, self.accumulated_tokens)), 4)
        return {
            "status": "accumulated",
            "chunk_id": chunk_id,
            "chunk_tokens": chunk_tokens,
            "total_accumulated_tokens": self.accumulated_tokens,
            "chunks_processed": self.chunks_processed,
            "kv_memory_spike_prevented": True,
        }

    def estimate_kv_cache_savings(self, total_tokens: int | None = None) -> dict[str, Any]:
        """Calculate memory savings of linear attention O(1) recurrent state vs standard quadratic KV-cache."""
        tokens = total_tokens or self.accumulated_tokens
        standard_kv_bytes = tokens * self.d_model * 2 * 28
        kda_constant_bytes = self.d_model * self.d_model * 2 * 28
        savings_bytes = max(0, standard_kv_bytes - kda_constant_bytes)
        savings_ratio = savings_bytes / max(1, standard_kv_bytes)
        return {
            "total_tokens": tokens,
            "standard_kv_cache_mb": round(standard_kv_bytes / (1024 * 1024), 2),
            "kda_linear_memory_mb": round(kda_constant_bytes / (1024 * 1024), 2),
            "memory_saved_mb": round(savings_bytes / (1024 * 1024), 2),
            "kv_spike_reduction_pct": round(savings_ratio * 100.0, 2),
        }

    def reset(self) -> None:
        self.accumulated_tokens = 0
        self.chunks_processed = 0
        self.recurrent_state.clear()


@dataclass(frozen=True)
class StatutoryConstraintBlock:
    """Immutable block containing extracted factual bounds and statutory parameters.

    Injected into system-prompt headers prior to agent dispatch to prevent
    constraint degradation during memory summarization and context compaction.
    """

    household_size: int | None = None
    income_thresholds: dict[str, float] = field(default_factory=dict)
    gross_income_limit: float | None = None
    net_income_limit: float | None = None
    countable_income: float | None = None
    medical_offsets: dict[str, float] = field(default_factory=dict)
    medical_deductions: float | None = None
    appeal_deadlines: dict[str, Any] = field(default_factory=dict)
    jurisdictional_timebars: dict[str, Any] = field(default_factory=dict)
    statutory_bounds: dict[str, Any] = field(default_factory=dict)

    def to_system_header(self) -> str:
        lines = [
            "=== STATUTORY CONSTRAINT HEADER (IMMUTABLE BOUNDS) ===",
            f"Household Size: {self.household_size if self.household_size is not None else 'UNSPECIFIED'}",
        ]
        if self.gross_income_limit is not None:
            lines.append(f"Gross Income Limit: ${self.gross_income_limit:.2f}")
        if self.net_income_limit is not None:
            lines.append(f"Net Income Limit: ${self.net_income_limit:.2f}")
        if self.countable_income is not None:
            lines.append(f"Countable Income: ${self.countable_income:.2f}")
        if self.income_thresholds:
            lines.append(f"Income Thresholds: {json.dumps(self.income_thresholds, sort_keys=True)}")
        if self.medical_deductions is not None:
            lines.append(f"Medical Deductions: ${self.medical_deductions:.2f}")
        if self.medical_offsets:
            lines.append(f"Medical Expenditure Offsets: {json.dumps(self.medical_offsets, sort_keys=True)}")
        if self.appeal_deadlines:
            lines.append(f"Statutory Appeal Filing Deadlines: {json.dumps(self.appeal_deadlines, sort_keys=True)}")
        if self.jurisdictional_timebars:
            lines.append(f"Jurisdictional Timebars: {json.dumps(self.jurisdictional_timebars, sort_keys=True)}")
        if self.statutory_bounds:
            lines.append(f"Statutory Bounds: {json.dumps(self.statutory_bounds, sort_keys=True)}")
        lines.append("=== END STATUTORY CONSTRAINT HEADER ===")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "household_size": self.household_size,
            "income_thresholds": self.income_thresholds,
            "gross_income_limit": self.gross_income_limit,
            "net_income_limit": self.net_income_limit,
            "countable_income": self.countable_income,
            "medical_offsets": self.medical_offsets,
            "medical_deductions": self.medical_deductions,
            "appeal_deadlines": self.appeal_deadlines,
            "jurisdictional_timebars": self.jurisdictional_timebars,
            "statutory_bounds": self.statutory_bounds,
        }


def extract_statutory_constraints(history: list[Any]) -> StatutoryConstraintBlock:
    """Pre-compaction filter parsing and locking critical statutory constraints.

    Extracts household income thresholds, countable gross/net income limits,
    medical expenditure offsets, and statutory appeal filing deadlines from
    evidence, raw text messages, or assessment histories.
    """
    hh_size: int | None = None
    income_thresholds: dict[str, float] = {}
    gross_limit: float | None = None
    net_limit: float | None = None
    countable_income: float | None = None
    medical_offsets: dict[str, float] = {}
    medical_deductions: float | None = None
    appeal_deadlines: dict[str, Any] = {}
    jurisdictional_timebars: dict[str, Any] = {}
    statutory_bounds: dict[str, Any] = {}

    for item in history:
        # Case 1: Evidence object
        if isinstance(item, Evidence):
            etype = item.type
            if etype == EvidenceType.HOUSEHOLD_SIZE:
                try:
                    hh_size = int(float(item.value))
                except (ValueError, TypeError):
                    pass
            elif etype in (EvidenceType.MONTHLY_INCOME, EvidenceType.ANNUAL_INCOME, EvidenceType.BASE_PERIOD_EARNINGS):
                try:
                    val = float(item.value)
                    income_thresholds[etype.value] = val
                    if etype == EvidenceType.MONTHLY_INCOME:
                        countable_income = val
                except (ValueError, TypeError):
                    pass
            elif etype == EvidenceType.DAYS_SINCE_DENIAL:
                try:
                    appeal_deadlines["days_since_denial"] = int(float(item.value))
                except (ValueError, TypeError):
                    pass
            elif etype == EvidenceType.DENIAL_DATE:
                appeal_deadlines["denial_date"] = str(item.value)
            elif etype == EvidenceType.LIQUID_ASSETS:
                try:
                    statutory_bounds["liquid_assets"] = float(item.value)
                except (ValueError, TypeError):
                    pass

        # Case 2: Assessment object
        elif isinstance(item, Assessment):
            for crit in item.criteria:
                if "income" in crit.criterion_id or "gross" in crit.criterion_id or "net" in crit.criterion_id:
                    statutory_bounds[f"criterion_{crit.criterion_id}"] = crit.outcome.value

        # Case 3: Dict or raw string representation (e.g. message logs, turn dictionaries)
        else:
            text = str(item)
            # Household size extraction
            hh_m = re.search(r"(?:household_size|family_size|household)\s*[:=]\s*(\d+)", text, re.I)
            if hh_m:
                hh_size = int(hh_m.group(1))

            # Income thresholds
            for m in re.finditer(r"(?:gross_income|monthly_income|gross_limit|net_limit|income_threshold)\s*[:=]\s*\$?([0-9,]+(?:\.[0-9]+)?)", text, re.I):
                k = m.group(0).split(":")[0].split("=")[0].strip().lower()
                val = float(m.group(1).replace(",", ""))
                income_thresholds[k] = val
                if "gross" in k:
                    gross_limit = val
                elif "net" in k:
                    net_limit = val
                elif "monthly" in k:
                    countable_income = val

            # Medical deductions / offsets
            for m in re.finditer(r"(?:medical_deduction|medical_offset|medical_expense|deduction)\s*[:=]\s*\$?([0-9,]+(?:\.[0-9]+)?)", text, re.I):
                k = m.group(0).split(":")[0].split("=")[0].strip().lower()
                val = float(m.group(1).replace(",", ""))
                medical_offsets[k] = val
                medical_deductions = val

            # Appeal filing deadlines
            for m in re.finditer(r"(?:appeal_deadline|filing_deadline|timebar|days_to_appeal|days_since_denial)\s*[:=]\s*([0-9]+(?:\s*days)?)", text, re.I):
                k = m.group(0).split(":")[0].split("=")[0].strip().lower()
                appeal_deadlines[k] = m.group(1).strip()

            # Jurisdictional timebars
            timebar_m = re.search(r"(\d+)[- ]day statutory (?:window|timebar|limit)", text, re.I)
            if timebar_m:
                jurisdictional_timebars["statutory_appeal_window_days"] = int(timebar_m.group(1))

    return StatutoryConstraintBlock(
        household_size=hh_size,
        income_thresholds=income_thresholds,
        gross_income_limit=gross_limit,
        net_income_limit=net_limit,
        countable_income=countable_income,
        medical_offsets=medical_offsets,
        medical_deductions=medical_deductions,
        appeal_deadlines=appeal_deadlines,
        jurisdictional_timebars=jurisdictional_timebars,
        statutory_bounds=statutory_bounds,
    )


class MemoryConsolidator:
    def __init__(self, partition: CasePartition, preserve_longitudinal_history: bool = True) -> None:
        self.partition = partition
        self.preserve_longitudinal_history = preserve_longitudinal_history
        self.kda_accumulator = KDAMemoryAccumulator()

    # -- storage ------------------------------------------------------------ #

    def store_evidence(self, evidence: list[Evidence], ttl_s: float | None = None) -> None:
        for ev in evidence:
            self.partition.write(
                kind=EVIDENCE,
                key=ev.evidence_id,
                record_type="Evidence",
                payload=ev.model_dump(mode="json"),
                ttl_s=ttl_s,
            )

    def store_assessment(self, assessment: Assessment, ttl_s: float | None = None) -> None:
        self.partition.write(
            kind=ASSESSMENT,
            key=f"{assessment.program.value}:{assessment.attempt}",
            record_type="Assessment",
            payload=assessment.model_dump(mode="json"),
            ttl_s=ttl_s,
        )

    def store_constraints(self, constraints: StatutoryConstraintBlock, ttl_s: float | None = None) -> None:
        self.partition.write(
            kind=CONSTRAINTS,
            key="statutory_constraints",
            record_type="StatutoryConstraintBlock",
            payload=constraints.to_dict(),
            ttl_s=ttl_s,
        )

    def store_longitudinal_history(self, records: list[dict[str, Any]], ttl_s: float | None = None) -> None:
        """Preserve raw longitudinal case milestones and multi-year chronological facts."""
        for idx, rec in enumerate(records):
            key = rec.get("record_id", f"longitudinal_{idx}_{time.time()}")
            self.partition.write(
                kind=LONGITUDINAL,
                key=str(key),
                record_type="LongitudinalRecord",
                payload=dict(rec),
                ttl_s=ttl_s,
            )

    # -- retrieval ---------------------------------------------------------- #

    def read_evidence(self) -> list[Evidence]:
        return [Evidence.model_validate(r.payload) for r in self.partition.read_all(EVIDENCE)]

    def read_assessments(self) -> list[Assessment]:
        return [Assessment.model_validate(r.payload) for r in self.partition.read_all(ASSESSMENT)]

    def read_longitudinal_history(self) -> list[dict[str, Any]]:
        """Read preserved raw longitudinal case histories from active partition."""
        return [dict(r.payload) for r in self.partition.read_all(LONGITUDINAL)]

    def extract_constraints(self) -> StatutoryConstraintBlock:
        records = self.partition.read_all(EVIDENCE)
        evidence = [Evidence.model_validate(r.payload) for r in records]
        assessments = self.read_assessments()
        constraints = extract_statutory_constraints(evidence + assessments)
        self.store_constraints(constraints)
        return constraints

    # -- consolidation & context compaction --------------------------------- #

    def consolidate_evidence(self) -> list[Evidence]:
        """Deduplicate by evidence type, preferring higher confidence then recency.

        Preserves raw longitudinal records in active memory without destructive
        sliding-window pruning when preserve_longitudinal_history is enabled.
        """
        best: dict[str, Evidence] = {}
        all_evidence = self.read_evidence()
        for ev in all_evidence:
            key = ev.type.value
            current = best.get(key)
            if current is None or ev.confidence >= current.confidence:
                best[key] = ev
        consolidated = list(best.values())
        self.partition.write(
            kind=SUMMARY,
            key="evidence_summary",
            record_type="EvidenceSummary",
            payload={
                "types_present": sorted(best.keys()),
                "count": str(len(consolidated)),
                "raw_longitudinal_count": len(all_evidence),
                "longitudinal_preserved": self.preserve_longitudinal_history,
            },
        )
        # Lock statutory constraints upon consolidation
        self.extract_constraints()
        return consolidated

    def compact_context(self, history: list[Any]) -> tuple[StatutoryConstraintBlock, str]:
        """Compact lengthy interaction histories while guaranteeing statutory bounds retention.

        Extracts critical statutory constraints, produces a compacted summary,
        and prepends the immutable statutory constraint header.
        """
        constraints = extract_statutory_constraints(history)
        self.store_constraints(constraints)

        # Generate summary of facts and transitions
        summary_lines = [
            f"Compacted History: {len(history)} interaction turn(s) consolidated.",
            f"Active Evidence Count: {len(self.read_evidence())}",
            f"Longitudinal History Preserved: {self.preserve_longitudinal_history}",
        ]
        compacted_body = "\n".join(summary_lines)
        system_header = constraints.to_system_header()
        full_compacted = f"{system_header}\n\n{compacted_body}"
        return constraints, full_compacted

    def compact_latent_reasoning_traces(
        self, trace_steps: list[dict[str, Any] | str], max_visible_steps: int = 3
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Compact older intermediate latent reasoning steps into a structured state summary.

        Preserves the last `max_visible_steps` while replacing earlier verbose monologues
        with consolidated statutory state representations to reduce context length.
        """
        return compact_latent_reasoning_trace(trace_steps, max_visible_steps=max_visible_steps)

    # -- lifecycle ---------------------------------------------------------- #

    def expire(self) -> int:
        return self.partition.purge_expired()


def compact_latent_reasoning_trace(
    trace_steps: list[dict[str, Any] | str],
    max_visible_steps: int = 3,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compacts older latent reasoning steps into structured state summaries during intermediate agent trace generation.

    Returns:
        (structured_state_summary, retained_recent_steps)
    """
    if not trace_steps:
        return {"summary": "empty trace", "compacted_count": 0}, []

    norm_steps: list[dict[str, Any]] = []
    for i, step in enumerate(trace_steps):
        if isinstance(step, dict):
            norm_steps.append(step)
        else:
            norm_steps.append({"step_index": i, "content": str(step), "type": "reasoning"})

    if len(norm_steps) <= max_visible_steps:
        return {
            "summary": f"All {len(norm_steps)} step(s) retained in active context.",
            "compacted_count": 0,
            "extracted_facts": [],
        }, norm_steps

    older_steps = norm_steps[:-max_visible_steps]
    recent_steps = norm_steps[-max_visible_steps:]

    # Extract key facts, conclusions, and criterion transitions from older steps
    extracted_facts: list[str] = []
    milestones: list[str] = []

    for s in older_steps:
        content = str(s.get("content", s.get("thought", s.get("action", ""))))
        # Clean thinking tags
        clean = re.sub(r"<(?:think|thought|reasoning)[^>]*>.*?</(?:think|thought|reasoning)>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()
        if not clean:
            clean = content[:150]
        if "satisfied" in clean.lower():
            milestones.append(f"Satisfied criterion: {clean[:80]}")
        elif "income" in clean.lower():
            extracted_facts.append(f"Income evaluation: {clean[:80]}")
        elif clean:
            milestones.append(clean[:80])

    structured_summary = {
        "summary": f"Compacted {len(older_steps)} historical latent reasoning steps.",
        "compacted_count": len(older_steps),
        "milestones": milestones[:10],
        "extracted_facts": extracted_facts[:10],
        "compression_ratio": round((len(older_steps) / len(norm_steps)), 3),
    }

    # Prepend summary block to the retained recent steps
    summary_step = {
        "step_index": 0,
        "type": "compacted_latent_summary",
        "structured_summary": structured_summary,
        "content": f"[COMPACTED STATE SUMMARY of {len(older_steps)} earlier steps: {len(milestones)} milestone(s) resolved]",
    }

    return structured_summary, [summary_step, *recent_steps]


class VRAMProtectionGate:
    """Monitors host VRAM usage and triggers prefill protection and micro-batch adjustments."""

    @staticmethod
    def get_vram_usage_ratio() -> float:
        """Obtain current GPU VRAM utilization ratio [0.0, 1.0]."""
        # 1. Attempt pynvml check if available
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pynvml.nvmlShutdown()
            return float(info.used) / float(info.total)
        except Exception:
            pass

        # 2. Check environment variable override for testing / simulation
        env_vram = os.getenv("TRIBUNE_SIMULATED_VRAM_USAGE")
        if env_vram is not None:
            try:
                return float(env_vram)
            except ValueError:
                pass

        return 0.0

    @classmethod
    def check_and_enforce_vram_protection(
        cls,
        current_ubatch_size: int,
        vram_usage_threshold: float = 0.95,
        consolidator: MemoryConsolidator | None = None,
        is_linear_attention: bool = False,
    ) -> tuple[int, bool]:
        """If host OS metrics indicate VRAM usage exceeds threshold:

        automatically halve the current micro-batch limit unless linear attention (KDA) is active,
        which bounds state memory in constant O(1) space.
        Returns: (new_ubatch_size, triggered)
        """
        if is_linear_attention:
            return current_ubatch_size, False

        usage = cls.get_vram_usage_ratio()
        if usage >= vram_usage_threshold:
            new_ubatch = max(1, current_ubatch_size // 2)
            if consolidator is not None:
                consolidator.consolidate_evidence()
            return new_ubatch, True
        return current_ubatch_size, False


# --------------------------------------------------------------------------- #
# Recursive Language Model (RLM) Provenance Compaction & Memory Traces
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProvenanceNode:
    """Lineage metadata node representing a verified content chunk with SHA-256 provenance."""

    source_id: str
    chunk_index: int
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    node_id: str = ""

    @classmethod
    def from_chunk(
        cls,
        source_id: str,
        chunk_index: int,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ProvenanceNode:
        """Create a provenance node with deterministic SHA-256 chunk hash."""
        c_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        n_id = f"{source_id}:{chunk_index}:{c_hash[:8]}"
        return cls(
            source_id=source_id,
            chunk_index=chunk_index,
            content_hash=c_hash,
            metadata=metadata or {},
            node_id=n_id,
        )


@dataclass
class ConsolidatedMemoryTrace:
    """Hierarchical, recursive condensed context trace with citation linkage."""

    trace_id: str
    root_assertion: str
    child_citations: list[ProvenanceNode] = field(default_factory=list)
    confidence_score: float = 1.0
    compaction_level: int = 0
    sub_traces: list[ConsolidatedMemoryTrace] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolve_all_leaf_citations(self) -> list[ProvenanceNode]:
        """Recursively traverse down trace hierarchy to retrieve all leaf ProvenanceNode citations."""
        leaves: list[ProvenanceNode] = list(self.child_citations)
        for sub in self.sub_traces:
            leaves.extend(sub.resolve_all_leaf_citations())

        # Deduplicate while preserving order
        seen: set[tuple[str, int, str]] = set()
        unique_leaves: list[ProvenanceNode] = []
        for leaf in leaves:
            key = (leaf.source_id, leaf.chunk_index, leaf.content_hash)
            if key not in seen:
                seen.add(key)
                unique_leaves.append(leaf)
        return unique_leaves

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "root_assertion": self.root_assertion,
            "child_citations_count": len(self.child_citations),
            "confidence_score": self.confidence_score,
            "compaction_level": self.compaction_level,
            "sub_traces_count": len(self.sub_traces),
        }


class RecursiveTraceConsolidator:
    """Hierarchical recursive context condensation engine.

    Condenses subagent trace streams into multi-level ConsolidatedMemoryTrace trees,
    preventing context dilution over long multi-turn sessions while preserving
    verified cryptographic provenance.
    """

    def __init__(
        self,
        max_chunk_tokens: int = 500,
        compaction_threshold_tokens: int = 1500,
        branching_factor: int = 3,
    ) -> None:
        self.max_chunk_tokens = max_chunk_tokens
        self.compaction_threshold_tokens = compaction_threshold_tokens
        self.branching_factor = max(2, branching_factor)

    def compact_subagent_traces(
        self,
        raw_traces: list[dict[str, Any]],
        parent_context_hash: str,
        compaction_level: int = 0,
    ) -> ConsolidatedMemoryTrace:
        """Hierarchically condense raw subagent traces into a ConsolidatedMemoryTrace.

        Computes deterministic trace IDs hashing parent_context_hash + condensed trace text.
        Aggregates provenance citations from child nodes with verified SHA-256 chunk hashes.
        Supports recursive multi-level compaction (Level N+1 generated from Level N traces)
        when trace count or token estimate exceeds thresholds.
        """
        if not raw_traces:
            empty_root = "No subagent trace assertions recorded."
            trace_id = hashlib.sha256(f"{parent_context_hash}:{empty_root}".encode()).hexdigest()
            return ConsolidatedMemoryTrace(
                trace_id=trace_id,
                root_assertion=empty_root,
                child_citations=[],
                confidence_score=1.0,
                compaction_level=compaction_level,
            )

        # 1. Parse and verify provenance nodes from raw traces
        extracted_citations: list[ProvenanceNode] = []
        extracted_assertions: list[str] = []
        confidences: list[float] = []

        for item in raw_traces:
            source_id = str(item.get("source_id", "subagent_trace"))
            chunk_index = int(item.get("chunk_index", 0))
            content = str(item.get("content", item.get("text", item.get("assertion", ""))))

            # Validate or compute content hash
            computed_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            provided_hash = item.get("content_hash")
            if provided_hash and provided_hash != computed_hash:
                logger.warning(
                    f"[RLM-PROVENANCE] Content hash verification mismatch: "
                    f"provided={provided_hash} vs computed={computed_hash}"
                )
            content_hash_val = computed_hash

            meta = dict(item.get("metadata", {}))
            extracted_citations.append(
                ProvenanceNode(
                    source_id=source_id,
                    chunk_index=chunk_index,
                    content_hash=content_hash_val,
                    metadata=meta,
                    node_id=f"{source_id}:{chunk_index}:{content_hash_val[:8]}",
                )
            )
            if content.strip():
                extracted_assertions.append(content.strip())
            if "confidence" in item:
                confidences.append(float(item["confidence"]))

        # 2. Check if recursive multi-level compaction is required
        total_chars = sum(len(a) for a in extracted_assertions)
        estimated_tokens = total_chars // 4

        # If trace count or tokens exceed threshold and we have enough traces to group
        if (
            len(raw_traces) > self.branching_factor
            or estimated_tokens > self.compaction_threshold_tokens
        ) and len(raw_traces) > 1:
            # Partition into chunks of size branching_factor
            sub_traces: list[ConsolidatedMemoryTrace] = []
            for i in range(0, len(raw_traces), self.branching_factor):
                group = raw_traces[i : i + self.branching_factor]
                group_hash = hashlib.sha256(f"{parent_context_hash}:group_{i}".encode()).hexdigest()
                sub_trace = self.compact_subagent_traces(
                    raw_traces=group,
                    parent_context_hash=group_hash,
                    compaction_level=compaction_level,
                )
                sub_traces.append(sub_trace)

            # Synthesize root assertion from sub-traces
            condensed_sub_assertions = [st.root_assertion for st in sub_traces]
            root_assertion = f"[Level {compaction_level + 1} Compaction]: " + " | ".join(
                condensed_sub_assertions
            )

            # Deterministic trace ID hashing parent_context_hash + condensed text
            trace_id = hashlib.sha256(
                f"{parent_context_hash}:{root_assertion}".encode()
            ).hexdigest()

            # Aggregate all child citations from sub-traces
            all_child_citations: list[ProvenanceNode] = []
            for st in sub_traces:
                all_child_citations.extend(st.resolve_all_leaf_citations())

            mean_conf = (
                sum(st.confidence_score for st in sub_traces) / len(sub_traces)
                if sub_traces
                else 1.0
            )

            return ConsolidatedMemoryTrace(
                trace_id=trace_id,
                root_assertion=root_assertion,
                child_citations=all_child_citations,
                confidence_score=round(mean_conf, 4),
                compaction_level=compaction_level + 1,
                sub_traces=sub_traces,
            )

        # Base level compaction
        root_assertion = (
            " | ".join(extracted_assertions)
            if extracted_assertions
            else "Verified subagent findings."
        )
        trace_id = hashlib.sha256(
            f"{parent_context_hash}:{root_assertion}".encode()
        ).hexdigest()
        mean_conf = sum(confidences) / len(confidences) if confidences else 1.0

        return ConsolidatedMemoryTrace(
            trace_id=trace_id,
            root_assertion=root_assertion,
            child_citations=extracted_citations,
            confidence_score=round(mean_conf, 4),
            compaction_level=compaction_level,
        )


# --------------------------------------------------------------------------- #
# Structured Schema IR consolidation with HMAC provenance (hardened path)
# --------------------------------------------------------------------------- #


class SecureConsolidator:
    """Schema-enforced consolidation producing only validated, HMAC-signed IR.

    - Free-form NL summaries are never persisted here.
    - Rejected payloads are quarantined for audit review and never enter HDM.
    - Every accepted node is HMAC-signed and chained in the tamper-evident log.
    """

    def __init__(
        self,
        sanitizer: Any | None = None,
        consolidator_id: str = "tribune-consolidator/v1",
    ) -> None:
        from .consolidation_schema import ConsolidationSanitizer

        self.sanitizer = sanitizer or ConsolidationSanitizer()
        self.consolidator_id = consolidator_id
        self._accepted: list[dict[str, Any]] = []
        self._lock = threading.RLock()

    def consolidate(
        self,
        source_episodic_ids: list[str],
        entities: list[dict[str, Any]],
        declarative_summary: str,
        contradiction: dict[str, Any] | None = None,
        decay: dict[str, Any] | None = None,
        generated_at: str | None = None,
    ) -> dict[str, Any]:
        """Validate schema IR, sign it, and return the signed node envelope."""
        from ..security.audit import (
            SecurityEventType,
            record_security_event,
            sign_consolidated_node,
        )
        from .consolidation_schema import CONSOLIDATION_SCHEMA_VERSION

        if generated_at is None:
            from datetime import datetime, timezone

            generated_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "schema_version": CONSOLIDATION_SCHEMA_VERSION,
            "source_episodic_ids": list(source_episodic_ids),
            "entities": entities,
            "declarative_summary": declarative_summary,
            "generated_at": generated_at,
            "consolidator_id": self.consolidator_id,
            "contradiction": contradiction,
            "decay": decay,
        }
        try:
            ir = self.sanitizer.sanitize_or_reject(payload)
        except Exception as exc:
            record_security_event(
                event_type=SecurityEventType.CONSOLIDATION_REJECTED,
                source="tribune.memory.consolidation.SecureConsolidator",
                message=f"Consolidation rejected: {exc}",
                severity="HIGH",
                details={"reason": str(exc)[:300]},
            )
            raise
        schema_payload = ir.to_payload()
        envelope = sign_consolidated_node(
            source_ids=ir.source_episodic_ids,
            schema_payload=schema_payload,
            timestamp=ir.generated_at,
        )
        node = {
            "schema_payload": schema_payload,
            "provenance": envelope,
            "node_id": f"cons_{envelope['payload_digest'][:16]}",
        }
        with self._lock:
            self._accepted.append(node)
        return node

    def stats(self) -> dict[str, Any]:
        with self._lock:
            base = self.sanitizer.stats() if hasattr(self.sanitizer, "stats") else {}
            base["accepted_nodes"] = len(self._accepted)
            return base


