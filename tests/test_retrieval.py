"""Late-interaction (ColBERT-class) retrieval works."""

from tribune.corpus.citations import LateInteractionRetriever, embed_text
from tribune.corpus.rule_store import LocalRuleStore
from tribune.types import ProgramId


def test_maxsim_prefers_topically_matching_doc():
    r = LateInteractionRetriever()
    income_text = "gross income eligibility standard 130 percent of the federal poverty guidelines"
    residency_text = "a household shall live in the state in which it files an application"
    query = "income limit poverty guideline"
    assert r.score(query, income_text) > r.score(query, residency_text)


def test_maxsim_self_similarity_is_high():
    r = LateInteractionRetriever()
    text = "monetary eligibility base period earnings"
    # MaxSim of identical token sets is 1.0 (each query token matches itself).
    assert abs(r.score(text, text) - 1.0) < 1e-9


def test_rank_orders_by_relevance():
    r = LateInteractionRetriever()
    docs = {
        "income": "gross income eligibility standard 130 percent poverty",
        "residency": "household must live in the state residency requirement",
    }
    ranked = r.rank("income poverty limit", docs, k=2)
    assert ranked[0].doc_id == "income"


def test_store_retrieval_covers_required_criteria_at_full_k():
    store = LocalRuleStore()
    res = store.retrieve("snap food assistance eligibility", ProgramId.SNAP, "EX", k=8)
    ids = {rr.rule.criterion_id for rr in res}
    assert set(store.required_criteria(ProgramId.SNAP)).issubset(ids)


def test_embed_shapes():
    emb = embed_text("hello world")
    assert emb.shape[0] == 2  # two tokens
