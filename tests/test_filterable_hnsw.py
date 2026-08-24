"""Unit & integration tests for Filterable HNSW Graph Index & ColBERT MaxSim Late Interaction."""

import numpy as np

from tribune.context.graph_builder import FilterableHNSWIndex
from tribune.corpus.citations import (
    LateInteractionRetriever,
    embed_text,
)
from tribune.corpus.rule_store import LocalRuleStore
from tribune.types import ProgramId


def test_filterable_hnsw_selective_filter_navigation():
    """Verify that Filterable HNSW finds target nodes under highly selective (1%) metadata filtering."""
    index = FilterableHNSWIndex(dim=32, m=8, m0=16)

    # Insert 100 background nodes with general jurisdiction
    for i in range(100):
        vec = np.random.randn(32)
        vec /= np.linalg.norm(vec)
        index.insert(
            node_id=f"node_general_{i}",
            vector=vec,
            metadata={"jurisdiction": "GENERAL", "effective_year": 2024, "benefit_program": "snap"},
        )

    # Insert exactly 1 highly selective target node (1% selectivity)
    target_vec = np.zeros(32)
    target_vec[0] = 1.0  # distinctive direction
    index.insert(
        node_id="target_selective_node",
        vector=target_vec,
        metadata={"jurisdiction": "SPECIAL_STATE_X", "effective_year": 2026, "benefit_program": "medicaid"},
    )

    query_vec = np.zeros(32)
    query_vec[0] = 0.95
    query_vec[1] = 0.05
    query_vec /= np.linalg.norm(query_vec)

    # Query with strict 1% selective filter
    results = index.search(
        query_vector=query_vec,
        k=5,
        filter_fn=lambda m: m.get("jurisdiction") == "SPECIAL_STATE_X",
    )

    assert len(results) == 1
    assert results[0][0] == "target_selective_node"
    assert results[0][2]["benefit_program"] == "medicaid"
    assert results[0][1] > 0.90  # High similarity


def test_colbert_maxsim_sum_and_mean_scoring():
    """Verify ColBERT token-level MaxSim multi-vector calculation and dense fallback."""
    retriever = LateInteractionRetriever()

    q_text = "gross monthly earned income limit"
    d_text = "gross earned income standard for household"

    q_emb = embed_text(q_text)
    d_emb = embed_text(d_text)

    # 1. Mean MaxSim
    mean_sim = retriever.maxsim(q_emb, d_emb, sum_mode=False)
    assert 0.0 <= mean_sim <= 1.0
    assert mean_sim > 0.5  # Semantically overlapping trigrams

    # 2. Sum MaxSim: Score(Q, D) = sum_{i in Q} max_{j in D} (E_q(i) . E_d(j)^T)
    sum_sim = retriever.maxsim(q_emb, d_emb, sum_mode=True)
    assert sum_sim > mean_sim
    assert abs(sum_sim - (mean_sim * q_emb.shape[0])) < 1e-6

    # 3. Dense single-vector fallback
    dense_score = retriever.score(q_text, d_text, use_dense_fallback=True)
    assert 0.0 <= dense_score <= 1.0


def test_rule_store_retrieve_filtered():
    """Verify LocalRuleStore retrieve_filtered method using FilterableHNSWIndex."""
    store = LocalRuleStore()
    filtered_rules = store.retrieve_filtered(
        query="residency and citizenship requirements",
        program=ProgramId.SNAP,
        jurisdiction="EX",
        k=3,
        filter_fn=lambda m: m.get("statutory_level") == "federal",
    )

    assert len(filtered_rules) >= 1
    for rr in filtered_rules:
        assert rr.rule is not None
        assert rr.citation is not None
        assert rr.score > 0.0
