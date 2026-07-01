"""Hard rule: one case's memory partition cannot read another's data."""

import pytest

from tribune.memory.partitions import AccessDenied, PartitionManager


def test_cross_case_read_denied():
    mgr = PartitionManager()
    a = mgr.open("caseA")
    b = mgr.open("caseB")

    a.write("evidence", "k", "Note", {"x": "1"})
    b.write("evidence", "k", "Note", {"y": "2"})

    # A can read its own data.
    assert a.read("evidence", "k") is not None
    # A cannot read B's data.
    with pytest.raises(AccessDenied):
        a.try_read_other_case("caseB", "evidence", "k")


def test_partition_only_sees_own_case():
    mgr = PartitionManager()
    a = mgr.open("caseA")
    a.write("assessment", "snap", "Assessment", {"status": "likely_eligible"})
    # B sees nothing for the same kind/key.
    b = mgr.open("caseB")
    assert b.read("assessment", "snap") is None
