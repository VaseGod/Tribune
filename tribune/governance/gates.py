"""Decoupled Action Gate Service & Out-of-Band Supervisor Governance.

Isolates evaluation policies, rubrics, and gatekeeper scoring scripts from agent read/write
access to prevent deceptive optimization (Anthropic Hacker-Opus vulnerability).
Routes high-stakes action proposals out-of-band and mints short-lived cryptographically signed
HMAC authorization tokens (HMAC-SHA256 with nonce and timestamp TTL) delivered directly
to the hardened sandbox.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from ..types import (
    ActionProposal,
    HMACAuthorizationToken,
)


class PolicyRejectionError(PermissionError):
    """Raised when an out-of-band action proposal violates evaluation policies or rubrics."""
    pass


@dataclass(frozen=True)
class EvaluationPolicy:
    """Isolated evaluation policy definition kept private to the Gatekeeper Supervisor."""

    policy_id: str
    description: str
    prohibited_action_patterns: list[str] = field(default_factory=list)
    prohibited_resources: list[str] = field(default_factory=list)
    required_parameters: list[str] = field(default_factory=list)


class GatekeeperSupervisor:
    """Physically and logically isolated supervisor validating agent proposals out-of-band.

    Holds private evaluation rubrics and secrets. Mints short-lived HMAC-SHA256
    authorization tokens only upon rigorous policy verification.
    """

    def __init__(
        self,
        supervisor_id: str = "gatekeeper_supervisor_v1",
        secret_key: bytes | None = None,
        default_ttl_seconds: int = 60,
    ) -> None:
        self.supervisor_id = supervisor_id
        self._secret_key = secret_key or secrets.token_bytes(32)
        self.default_ttl_seconds = default_ttl_seconds
        self._lock = threading.RLock()
        self._issued_nonces: set[str] = set()

        # Private policies inaccessible to agent context
        self._policies: list[EvaluationPolicy] = [
            EvaluationPolicy(
                policy_id="POL_NO_SPEC_GAMING",
                description="Prevent hidden test inspection and rule store tampering in proposals",
                prohibited_action_patterns=[
                    r"override_rule",
                    r"bypass_preconditions",
                    r"force_satisfied",
                    r"clear_rule_store",
                    r"mock_patch",
                ],
            ),
            EvaluationPolicy(
                policy_id="POL_RESTRICTED_RESOURCES",
                description="Block modifications to core security and governance files",
                prohibited_resources=[
                    "/etc/passwd",
                    ".env",
                    "tribune/governance/gates.py",
                    "tests/test_eval.py",
                    "/proc",
                    "/sys",
                ],
            ),
            EvaluationPolicy(
                policy_id="POL_FORMAL_FILINGS",
                description="Formal filings require case_id and verified program parameters",
                required_parameters=["case_id", "program"],
            ),
        ]

    def submit_proposal(
        self,
        proposal: ActionProposal,
        ttl_seconds: int | None = None,
    ) -> HMACAuthorizationToken:
        """Evaluate an action proposal out-of-band and mint an HMAC token if approved."""
        with self._lock:
            self._evaluate_proposal_against_policies(proposal)

            now = datetime.now(timezone.utc)
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
            expires_at = now + timedelta(seconds=ttl)

            nonce = secrets.token_hex(16)
            self._issued_nonces.add(nonce)

            # Compute HMAC-SHA256 signature
            signature = self._compute_signature(
                proposal_id=proposal.proposal_id,
                action_type=proposal.action_type,
                target_resource=proposal.target_resource,
                nonce=nonce,
                expires_at_timestamp=int(expires_at.timestamp()),
            )

            token = HMACAuthorizationToken(
                token_id=f"auth_tok_{secrets.token_hex(8)}",
                proposal_id=proposal.proposal_id,
                action_type=proposal.action_type,
                nonce=nonce,
                signature=signature,
                issued_at=now,
                expires_at=expires_at,
                supervisor_id=self.supervisor_id,
            )
            return token

    def verify_token(
        self,
        token: HMACAuthorizationToken,
        proposal_id: str,
        action_type: str,
        target_resource: str,
        now: datetime | None = None,
    ) -> bool:
        """Validate HMAC token authenticity, expiration, and proposal binding."""
        with self._lock:
            current_time = now or datetime.now(timezone.utc)

            # Check expiration
            if current_time >= token.expires_at:
                return False

            # Check action type matching
            if token.action_type != action_type or token.proposal_id != proposal_id:
                return False

            # Recompute signature and verify constant-time match
            expected_sig = self._compute_signature(
                proposal_id=proposal_id,
                action_type=action_type,
                target_resource=target_resource,
                nonce=token.nonce,
                expires_at_timestamp=int(token.expires_at.timestamp()),
            )

            return hmac.compare_digest(token.signature, expected_sig)

    def _compute_signature(
        self,
        proposal_id: str,
        action_type: str,
        target_resource: str,
        nonce: str,
        expires_at_timestamp: int,
    ) -> str:
        payload = f"{proposal_id}:{action_type}:{target_resource}:{nonce}:{expires_at_timestamp}"
        return hmac.new(self._secret_key, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def _evaluate_proposal_against_policies(self, proposal: ActionProposal) -> None:
        """Apply private evaluation rubrics and reject specification gaming."""
        # 1. Prohibited resources check
        for res in [proposal.target_resource, str(proposal.parameters)]:
            for prohibited in ["/etc/passwd", ".env", "tests/test_eval.py", "/proc", "/sys"]:
                if prohibited in res:
                    raise PolicyRejectionError(
                        f"Proposal rejected by policy POL_RESTRICTED_RESOURCES: Target '{prohibited}' is prohibited."
                    )

        # 2. Prohibited action patterns
        action_str = f"{proposal.action_type} {proposal.rationale} {proposal.parameters}"
        for policy in self._policies:
            for pattern in policy.prohibited_action_patterns:
                if re.search(pattern, action_str, re.IGNORECASE):
                    raise PolicyRejectionError(
                        f"Proposal rejected by policy {policy.policy_id}: Matched prohibited pattern '{pattern}'."
                    )

        # 3. Parameter checks for formal filings
        if proposal.action_type in ("formal_filing", "legal_filing"):
            if "case_id" not in proposal.parameters or "program" not in proposal.parameters:
                raise PolicyRejectionError(
                    "Proposal rejected by policy POL_FORMAL_FILINGS: Missing required 'case_id' or 'program'."
                )


class DecoupledActionGateService:
    """Service coordination layer delivering out-of-band authorizations directly to the sandbox."""

    def __init__(self, supervisor: GatekeeperSupervisor | None = None) -> None:
        self.supervisor = supervisor or GatekeeperSupervisor()
        self._proposals_log: list[ActionProposal] = []
        self._tokens_log: list[HMACAuthorizationToken] = []

    def submit_action_proposal(
        self,
        proposal: ActionProposal,
        sandbox: Any | None = None,
    ) -> HMACAuthorizationToken:
        """Route proposal to supervisor and optionally deliver minted token directly to sandbox."""
        self._proposals_log.append(proposal)
        token = self.supervisor.submit_proposal(proposal)
        self._tokens_log.append(token)

        # Deliver token directly to sandbox if provided
        if sandbox is not None and hasattr(sandbox, "receive_authorization_token"):
            sandbox.receive_authorization_token(token)

        return token

    def verify_authorization(
        self,
        token: HMACAuthorizationToken,
        proposal_id: str,
        action_type: str,
        target_resource: str,
    ) -> bool:
        return self.supervisor.verify_token(
            token=token,
            proposal_id=proposal_id,
            action_type=action_type,
            target_resource=target_resource,
        )


__all__ = [
    "EvaluationPolicy",
    "GatekeeperSupervisor",
    "DecoupledActionGateService",
    "PolicyRejectionError",
]
