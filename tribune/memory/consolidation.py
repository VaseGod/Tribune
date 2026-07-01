"""Storage / retrieval / update / consolidation / lifecycle for case memory.

Evidence and assessments are written into a case partition as they are produced.
Consolidation deduplicates evidence (keeping the most recent / highest-confidence
value per evidence type) and writes a compact summary. Lifecycle management purges
records past their TTL — useful for a deployment that wants case data to expire.
"""

from __future__ import annotations

from ..types import Assessment, Evidence
from .partitions import CasePartition

EVIDENCE = "evidence"
ASSESSMENT = "assessment"
SUMMARY = "summary"


class MemoryConsolidator:
    def __init__(self, partition: CasePartition) -> None:
        self.partition = partition

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

    # -- retrieval ---------------------------------------------------------- #

    def read_evidence(self) -> list[Evidence]:
        return [Evidence.model_validate(r.payload) for r in self.partition.read_all(EVIDENCE)]

    def read_assessments(self) -> list[Assessment]:
        return [Assessment.model_validate(r.payload) for r in self.partition.read_all(ASSESSMENT)]

    # -- consolidation ------------------------------------------------------ #

    def consolidate_evidence(self) -> list[Evidence]:
        """Deduplicate by evidence type, preferring higher confidence then recency.

        Returns the consolidated set and writes a summary record. The records read
        back from the partition preserve insertion order, so a later, equally
        confident value supersedes an earlier one.
        """
        best: dict[str, Evidence] = {}
        for ev in self.read_evidence():
            key = ev.type.value
            current = best.get(key)
            if current is None or ev.confidence >= current.confidence:
                best[key] = ev
        consolidated = list(best.values())
        self.partition.write(
            kind=SUMMARY,
            key="evidence_summary",
            record_type="EvidenceSummary",
            payload={"types_present": sorted(best.keys()), "count": str(len(consolidated))},
        )
        return consolidated

    # -- lifecycle ---------------------------------------------------------- #

    def expire(self) -> int:
        return self.partition.purge_expired()
