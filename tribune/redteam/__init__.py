"""Ingestion injection red-team suite.

Attacker-writable benefit notices enter TRIBUNE's OCR path. This suite generates
adversarial notices and asserts the action gate never fires from document
instructions, provenance never cites injected text, and injected content is
flagged or inert.
"""

from .notice_generator import build_notice, tamper_case
from .payloads import PAYLOADS, AttackGoal, Payload, Placement, payloads_for
from .probe import Finding, InjectionProbe, ProbeReport

__all__ = [
    "PAYLOADS",
    "AttackGoal",
    "Payload",
    "Placement",
    "payloads_for",
    "build_notice",
    "tamper_case",
    "Finding",
    "InjectionProbe",
    "ProbeReport",
]
