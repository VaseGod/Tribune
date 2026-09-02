"""Tests for Anti-EvoMal Governance, SHA-256 Manifest Enforcement, and Browser Sandboxing."""

from __future__ import annotations

import pytest

from tribune.governance.action_gate import (
    BrowserSandboxManager,
    ManifestEnforcer,
    ManagedProfileIsolation,
    SecurityViolationError,
)
from tribune.memory.partitions import (
    PartitionIntegrityManifest,
    PartitionManager,
    PartitionTamperingError,
)
from tribune.redteam.probe import probe_evomal_defense


def test_sha256_partition_manifest_verification() -> None:
    pm = PartitionManager()
    partition = pm.open("case_legit_100")
    partition.write("evidence", "paystub", "pdf", {"income": 2000})

    # Sign partition
    sig = pm.sign_partition(partition)
    assert sig is not None

    # Verify partition matches signature
    assert pm.verify_partition(partition) is True


def test_anti_evomal_tamper_detection_and_quarantine() -> None:
    pm = PartitionManager()
    partition = pm.open("case_tamper_200")
    partition.write("evidence", "assets", "bank", {"balance": 500})
    pm.sign_partition(partition)

    # Simulate adversarial runtime mutation bypassing write()
    from tribune.memory.store import MemoryRecord
    raw_rec = MemoryRecord(
        case_id="case_tamper_200",
        kind="evidence",
        key="assets",
        record_type="bank",
        payload={"balance": 999999, "malicious_injection": True},
    )
    partition._store.put(raw_rec, requester="case_tamper_200")

    # Verification must detect tampering, quarantine partition, and raise PartitionTamperingError
    with pytest.raises(PartitionTamperingError):
        pm.verify_partition(partition)

    assert partition.is_quarantined is True or partition._is_quarantined is True

    # Subsequent access must be rejected
    with pytest.raises(PartitionTamperingError):
        partition.read("evidence", "assets")


def test_manifest_enforcer_skill_signature() -> None:
    manifest = PartitionIntegrityManifest()
    enforcer = ManifestEnforcer(manifest)

    valid_skill = "def calculate_standard_deduction(income): return max(0, income - 200)"
    manifest.sign_skill("statutory_deduction", valid_skill)

    # Valid skill verification succeeds
    assert enforcer.enforce_skill_integrity("statutory_deduction", valid_skill) is True

    # Modified skill fails verification
    tampered_skill = "def calculate_standard_deduction(income): return 999999"
    with pytest.raises(SecurityViolationError):
        enforcer.enforce_skill_integrity("statutory_deduction", tampered_skill)


def test_browser_sandbox_domain_scoped_acl() -> None:
    sandbox = BrowserSandboxManager()

    # Authorized government statutory domains succeed
    assert sandbox.validate_navigation("https://www.medicaid.gov/policy") is True
    assert sandbox.validate_navigation("https://dhs.state.ny.us/benefits") is True
    assert sandbox.validate_navigation("https://www.fns.usda.gov/snap") is True

    # Unauthorized commercial/exfiltration domains are blocked
    with pytest.raises(SecurityViolationError) as exc_info:
        sandbox.validate_navigation("http://evil-attacker-server.com/steal-pii")
    assert "Browser Sandbox Security Block" in str(exc_info.value)


def test_managed_profile_isolation() -> None:
    profile_manager = ManagedProfileIsolation()
    prof1 = profile_manager.get_isolated_profile("case_1", "subagent_medicaid")
    prof2 = profile_manager.get_isolated_profile("case_1", "subagent_snap")

    assert prof1["profile_path"] != prof2["profile_path"]
    assert prof1["isolated"] is True
    assert prof1["allow_shared_cookies"] is False


def test_full_evomal_redteam_probe_suite() -> None:
    report = probe_evomal_defense()
    assert report["status"] == "PASS"
    assert report["rejection_rate"] == 1.0  # 100% rejection rate for adversarial probes
