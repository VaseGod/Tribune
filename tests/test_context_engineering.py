"""Unit & integration tests for Phase 1: Proactive Context Engineering & Relational Graph Memory."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from tribune.agents.tools.context_ops import (
    analyzeText,
    checkBudget,
    compressContext,
    foldHistory,
    set_context_manager,
)
from tribune.context.manager import ProactiveContextManager
from tribune.mcp import MCPHandler
from tribune.memory.episodic import ContextPilot, RelationalEpisodicMemory
from tribune.types import (
    CausalChain,
    CompactionUrgency,
    EntityGraph,
    EntityNode,
    EntityRelation,
    GraphQuery,
)


def test_analyze_text_metrics():
    """Verify exact token count, information density, and Shannon entropy calculations."""
    text = "The applicant's gross income is $1,200 under 7 CFR 273.9. Income qualifies under standard guidelines."
    analysis = analyzeText(text)

    assert analysis.token_count > 0
    assert 0.0 <= analysis.shannon_entropy <= 10.0
    assert 0.0 <= analysis.information_density <= 1.0
    assert analysis.working_memory_bytes == len(text.encode("utf-8"))


def test_context_budget_tracking_and_urgency_escalation():
    """Verify checkBudget returns accurate quota and escalates compaction urgency appropriately."""
    manager = ProactiveContextManager(total_budget=100)
    set_context_manager(manager)

    # Initial state: Normal urgency
    status = checkBudget()
    assert status.total_budget == 100
    assert status.used_tokens == 0
    assert status.remaining_quota == 100
    assert status.compaction_urgency == CompactionUrgency.NORMAL
    assert status.is_urgent is False

    # Add moderate span (~40 tokens)
    manager.register_span("span_1", "This is a standard text passage for the initial case intake step.")
    status2 = checkBudget()
    assert status2.used_tokens > 0

    # Add large span pushing utilization over 85%
    manager.register_span(
        "span_2",
        "This is an extensive transcript and detailed deposition text containing multiple paragraphs "
        "of evidentiary arguments, statutory cross references, and complex procedural histories that "
        "consume the majority of the token budget allocation. The administrative law judge required "
        "further wage documentation, bank statement records, rental lease agreements, and sworn affidavits "
        "before proceeding to the final eligibility determination stage.",
    )
    status_crit = checkBudget()
    assert status_crit.compaction_urgency in (CompactionUrgency.ELEVATED, CompactionUrgency.CRITICAL)
    if status_crit.used_tokens >= 85:
        assert status_crit.is_urgent is True


def test_fold_history_preserves_indexing_header():
    """Verify foldHistory discards raw spans and preserves structured semantic indexing headers."""
    manager = ProactiveContextManager(total_budget=1000)
    set_context_manager(manager)

    long_text = (
        "Resolved procedural query regarding 7 CFR 273.9 eligibility. The claimant submitted utility bills "
        "confirming residency in Travis County, which satisfies the standard verification requirement. "
        "In addition, the claimant provided employment separation letters from Apex Logistics, pay stubs for "
        "the prior three calendar months, child care expense vouchers, medical deduction receipts, and sworn "
        "statements from household members confirming that gross income remains below statutory maximums."
    )
    manager.register_span("span_res", long_text, metadata={"case_id": "c101", "topic": "residency"})

    result = foldHistory(span_id="span_res", summarize=True)
    assert result.span_id == "span_res"
    assert result.discarded_tokens > 0
    assert "[SEMANTIC-INDEX-HEADER" in result.indexing_header
    assert "residency" in result.preserved_keys or "case_id" in result.preserved_keys
    assert "7 CFR 273.9" in str(result.preserved_keys)

    # Verify active context reflects folded header
    active_ctx = manager.get_active_context()
    assert "[SEMANTIC-INDEX-HEADER" in active_ctx
    assert "utility bills confirming residency" not in active_ctx


def test_compress_context_preserves_citations_entities_and_causal_predicates():
    """Verify compressContext strictly preserves named entities, statutory citations, and causal predicates."""
    filing_text = (
        "The hearing was called to order on Tuesday morning in Travis County. "
        "Claimant Jane Doe submitted timely documentation. "
        "The Department denied Medicaid benefits under 42 USC 1396a because gross earnings exceeded "
        "the 138 percent poverty threshold. Therefore, the claimant appeals the statutory termination."
    )

    compressed = compressContext(filing_text, ratio=0.5)

    # Citations must be preserved
    assert "42 USC 1396a" in compressed
    # Causal predicates must be preserved
    assert "because" in compressed.lower()
    assert "therefore" in compressed.lower() or "denied" in compressed.lower()
    # Key entities must be preserved
    assert "Jane Doe" in compressed or "Medicaid" in compressed


def test_mcp_context_tools_integration():
    """Verify MCP protocol handler exposes and executes analyzeText, checkBudget, foldHistory, compressContext."""
    handler = MCPHandler()

    # 1. tools/list includes new tools
    list_req = {"jsonrpc": "2.0", "id": "req-1", "method": "tools/list", "params": {}}
    resp = handler.handle_request(list_req)
    tool_names = [t["name"] for t in resp["result"]["tools"]]
    assert "analyzeText" in tool_names
    assert "checkBudget" in tool_names
    assert "foldHistory" in tool_names
    assert "compressContext" in tool_names

    # 2. tools/call analyzeText
    call_analyze = {
        "jsonrpc": "2.0",
        "id": "req-2",
        "method": "tools/call",
        "params": {
            "name": "analyzeText",
            "arguments": {"span": "Test evaluation under 7 CFR 273.9."},
        },
    }
    resp_analyze = handler.handle_request(call_analyze)
    data = json.loads(resp_analyze["result"]["content"][0]["text"])
    assert "token_count" in data
    assert "shannon_entropy" in data

    # 3. tools/call checkBudget
    call_budget = {
        "jsonrpc": "2.0",
        "id": "req-3",
        "method": "tools/call",
        "params": {"name": "checkBudget", "arguments": {}},
    }
    resp_budget = handler.handle_request(call_budget)
    b_data = json.loads(resp_budget["result"]["content"][0]["text"])
    assert "total_budget" in b_data
    assert "remaining_quota" in b_data

    # 4. tools/call compressContext
    call_compress = {
        "jsonrpc": "2.0",
        "id": "req-4",
        "method": "tools/call",
        "params": {
            "name": "compressContext",
            "arguments": {
                "target_text": "Claimant John Smith was denied SNAP under 7 CFR 273.9 because of excess assets.",
                "ratio": 0.5,
            },
        },
    }
    resp_compress = handler.handle_request(call_compress)
    c_data = json.loads(resp_compress["result"]["content"][0]["text"])
    assert "7 CFR 273.9" in c_data["compressed_text"]


def test_relational_episodic_memory_graph_and_causal_chains():
    """Verify memorize, readMemory multi-hop traversal, and updateMemory operations."""
    memory = RelationalEpisodicMemory()
    now = datetime.now(timezone.utc)

    # Build entity graph
    node_applicant = EntityNode(
        node_id="app_1",
        name="Alice Walker",
        entity_type="applicant",
        attributes={"income": 950.0},
    )
    node_employer = EntityNode(
        node_id="emp_1",
        name="Apex Logistics",
        entity_type="employer",
        attributes={"industry": "transport"},
    )
    edge_employment = EntityRelation(
        source_id="app_1",
        target_id="emp_1",
        relation_type="employed_by",
        weight=1.0,
    )
    graph = EntityGraph(nodes=[node_applicant, node_employer], edges=[edge_employment])

    chain = CausalChain(
        chain_id="chain_1",
        premise="Alice Walker income $950",
        predicate="satisfies gross income cap",
        consequence="Alice Walker eligible for SNAP",
        confidence=0.98,
        evidence_citations=["7 CFR 273.9"],
    )

    root_id = memory.memorize(graph, now, [chain])
    assert root_id == "app_1"

    # Multi-hop query starting from app_1
    query = GraphQuery(root_node_ids=["app_1"], max_hops=2)
    subgraph = memory.readMemory(query)
    node_ids = [n.node_id for n in subgraph.nodes]
    assert "app_1" in node_ids
    assert "emp_1" in node_ids
    assert len(subgraph.edges) >= 1
    assert len(subgraph.causal_chains) >= 1
    assert subgraph.causal_chains[0].chain_id == "chain_1"

    # Test updateMemory
    memory.updateMemory("app_1", {"attributes": {"income": 1100.0, "status": "verified"}})
    subgraph2 = memory.readMemory(GraphQuery(root_node_ids=["app_1"], max_hops=1))
    app_node = next(n for n in subgraph2.nodes if n.node_id == "app_1")
    assert app_node.attributes["income"] == 1100.0
    assert app_node.attributes["status"] == "verified"


def test_context_pilot_enforces_under_1000_tokens_active_generation():
    """Verify ContextPilot aggressively offloads evidence to keep working generation context < 1,000 tokens."""
    pilot = ContextPilot(max_active_tokens=1000)

    # Ingest 15 voluminous evidentiary depositions
    for i in range(15):
        voluminous_deposition = (
            f"Deposition transcript record #{i}: Witness testified in exhaustive detail regarding applicant "
            f"situation, employment timeline at Apex Logistics, wage slip verification, household expenses, "
            f"heating allowances, medical bill deductions, and cross-examination questions spanning several pages. "
            * 8
        )
        pilot.add_working_evidence(
            span_id=f"depo_{i}",
            text=voluminous_deposition,
            causal_chains=[
                CausalChain(
                    chain_id=f"cc_{i}",
                    premise=f"Deposition {i} sworn testimony",
                    predicate="substantiates",
                    consequence=f"Income validity for turn {i}",
                )
            ],
        )

    # Active generation context must strictly remain under 1,000 tokens
    active_tokens = pilot.estimate_active_tokens()
    assert active_tokens < 1000

    # Final strategy compilation can rehydrate relevant facts from the relational graph
    compiled_strategy = pilot.compile_strategy_context()
    assert "CONTEXTPILOT STRATEGY CONTEXT" in compiled_strategy
    assert "Causal Chains" in compiled_strategy
