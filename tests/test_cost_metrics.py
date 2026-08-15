"""Phase 1 — cost & turn-efficiency accounting.

Covers: multi-turn accumulation, abstention costing (a correct abstention is a
completed task at its actual cost — never a failure, never infinite), cache-hit
pricing, promotional-rate expiry, per-language token counting, and end-to-end
usage attachment through the pipeline.
"""

from datetime import date
from math import isfinite

from tribune.casegen.synthetic import SyntheticCaseGenerator
from tribune.eval.costmodel import BackendPricing, CostModel, Rate
from tribune.eval.costreport import SUCCESS_OUTCOMES, compute_cost_report
from tribune.eval.metrics import EvalRecord, TaskOutcomeType, classify_outcome
from tribune.instrumentation.usage import UsageRecorder, estimate_tokens
from tribune.orchestration.pipeline import CasePipeline
from tribune.types import ModelCallUsage, ProgramId

# --------------------------------------------------------------------------- #
# Cost model fixtures
# --------------------------------------------------------------------------- #

_PROMO = Rate(
    effective_from=date(2025, 11, 24),
    effective_until=date(2026, 8, 31),
    input_per_m=2.0,
    output_per_m=10.0,
    cache_read_per_m=0.2,
    cache_write_per_m=2.5,
)
_LIST = Rate(
    effective_from=date(2026, 9, 1),
    effective_until=None,
    input_per_m=3.0,
    output_per_m=15.0,
    cache_read_per_m=0.3,
    cache_write_per_m=3.75,
)
_SONNET = BackendPricing(
    backend_id="anthropic:claude-sonnet-5",
    kind="api",
    model_patterns=("*claude-sonnet-5*",),
    rates=(_PROMO, _LIST),
)
_SELFHOST = BackendPricing(
    backend_id="selfhost:test",
    kind="self_hosted",
    model_patterns=("openai_compat:*",),
    rates=(
        Rate(
            effective_from=date(2026, 1, 1),
            effective_until=None,
            gpu_hour_usd=3.6,
            throughput_tokens_per_s=1000.0,
        ),
    ),
)


def _call(tokens_in=1_000_000, tokens_out=100_000, cache_read=0, cache_write=0,
          model="claude-sonnet-5"):
    return ModelCallUsage(
        role="proposer",
        model=model,
        tokenizer_id=model,
        tokens_input=tokens_in,
        tokens_output=tokens_out,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )


def test_promo_rate_applies_before_expiry_and_reverts_after():
    cm = CostModel([_SONNET])
    promo_cost, backend = cm.cost_of_call(_call(), on=date(2026, 7, 1))
    assert backend == "anthropic:claude-sonnet-5"
    assert abs(promo_cost - (2.0 + 0.1 * 10.0)) < 1e-9  # 1M in + 0.1M out

    # On the expiry boundary the promo still applies; the day after, list price.
    boundary_cost, _ = cm.cost_of_call(_call(), on=date(2026, 8, 31))
    assert abs(boundary_cost - promo_cost) < 1e-9
    list_cost, _ = cm.cost_of_call(_call(), on=date(2026, 9, 1))
    assert abs(list_cost - (3.0 + 0.1 * 15.0)) < 1e-9


def test_cache_hit_pricing_discounts_cached_input():
    cm = CostModel([_SONNET])
    # 1M input of which 600k came from cache; 200k written to cache; 100k output.
    cost, _ = cm.cost_of_call(
        _call(tokens_in=1_000_000, cache_read=600_000, cache_write=200_000),
        on=date(2026, 7, 1),
    )
    expected = (400_000 * 2.0 + 600_000 * 0.2 + 200_000 * 2.5 + 100_000 * 10.0) / 1e6
    assert abs(cost - expected) < 1e-9


def test_self_hosted_cost_is_gpu_hours_over_throughput():
    cm = CostModel([_SELFHOST])
    # 3.6M tokens at 1000 tok/s = 3600 s = 1 GPU-hour = $3.6
    cost, backend = cm.cost_of_call(
        _call(tokens_in=3_000_000, tokens_out=600_000, model="openai_compat:some/model"),
        on=date(2026, 7, 1),
    )
    assert backend == "selfhost:test"
    assert abs(cost - 3.6) < 1e-9


def test_packaged_pricing_loads_and_prices_local_provider_free():
    cm = CostModel.load()
    cost, backend = cm.cost_of_call(_call(model="local_rules"), on=date(2026, 7, 1))
    assert cost == 0.0
    assert backend == "local:rules"


# --------------------------------------------------------------------------- #
# Recorder: multi-turn accumulation
# --------------------------------------------------------------------------- #


def test_recorder_accumulates_multi_turn_usage():
    rec = UsageRecorder(cost_model=CostModel([_SONNET]), pricing_date=date(2026, 7, 1))
    rec.start_case("c1", language="es")
    rec.start_task(ProgramId.MEDICAID)
    for _ in range(2):  # a replan: two assess/verify rounds
        rec.record_turn("proposer")
        rec.record_call("proposer", "claude-sonnet-5", "claude-sonnet-5", 1000, 200)
        rec.record_turn("verifier")
        rec.record_call("verifier", "claude-sonnet-5", "claude-sonnet-5", 500, 50)
    task = rec.finish_task()
    assert task is not None
    assert task.turns == 4 and task.proposer_turns == 2 and task.verifier_turns == 2
    assert task.tokens_input == 3000 and task.tokens_output == 500
    assert len(task.calls) == 4
    assert task.language == "es"
    assert task.cost_usd is not None and task.cost_usd > 0
    # The scope ended: further recording is inert and a second finish is None.
    rec.record_turn("proposer")
    assert rec.finish_task() is None


def test_estimate_tokens_is_deterministic_and_positive():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 41) == 11
    assert estimate_tokens("hello world") == estimate_tokens("hello world")


# --------------------------------------------------------------------------- #
# Outcome taxonomy + abstention costing (invariant: abstention is a success)
# --------------------------------------------------------------------------- #


def _record(outcome: str, cost: float, language="en", turns=2) -> EvalRecord:
    abstained = outcome in ("correct_abstention", "over_refusal")
    ambiguous = outcome == "correct_abstention"
    predicted = None if abstained else ("eligible" if outcome == "correct_determination" else "ineligible")
    return EvalRecord(
        case_id="c",
        program=ProgramId.SNAP,
        abstained=abstained,
        ground_truth_label="eligible",
        ambiguous=ambiguous,
        predicted_label=predicted,
        cost_usd=cost,
        turns=turns,
        tokens_input=100,
        tokens_output=10,
        language=language,
    )


def test_outcome_classification_covers_all_four_types():
    assert classify_outcome(_record("correct_determination", 0.1)) is TaskOutcomeType.CORRECT_DETERMINATION
    assert classify_outcome(_record("incorrect_determination", 0.1)) is TaskOutcomeType.INCORRECT_DETERMINATION
    assert classify_outcome(_record("correct_abstention", 0.1)) is TaskOutcomeType.CORRECT_ABSTENTION
    assert classify_outcome(_record("over_refusal", 0.1)) is TaskOutcomeType.OVER_REFUSAL


def test_correct_abstention_is_a_completed_success_at_actual_cost():
    records = [
        _record("correct_determination", 0.30),
        _record("correct_abstention", 0.02),  # cheap, correct, complete
        _record("incorrect_determination", 0.40),
        _record("over_refusal", 0.05),
    ]
    report = compute_cost_report(records)
    # Every task is completed and counted.
    assert report.n == 4 and sum(b.n for b in report.buckets.values()) == 4
    bucket = report.buckets[TaskOutcomeType.CORRECT_ABSTENTION.value]
    # Actual low cost — finite, never a penalty, never infinite.
    assert abs(bucket.mean_cost_usd - 0.02) < 1e-9
    assert isfinite(bucket.mean_cost_usd) and isfinite(bucket.median_cost_usd)
    # Correct abstention is in the success set for cost-per-successful-outcome.
    assert TaskOutcomeType.CORRECT_ABSTENTION in SUCCESS_OUTCOMES
    assert abs(report.mean_cost_per_success_usd - (0.30 + 0.02) / 2) < 1e-9


def test_per_language_token_counting():
    records = [
        _record("correct_determination", 0.1, language="en"),
        _record("correct_determination", 0.1, language="en"),
        _record("correct_abstention", 0.01, language="es"),
    ]
    report = compute_cost_report(records)
    assert report.tokens_by_language["en"] == {"n": 2, "tokens_input": 200, "tokens_output": 20}
    assert report.tokens_by_language["es"] == {"n": 1, "tokens_input": 100, "tokens_output": 10}


# --------------------------------------------------------------------------- #
# End-to-end: the pipeline attaches usage to every outcome
# --------------------------------------------------------------------------- #


def test_pipeline_attaches_usage_including_replans_and_abstentions():
    gen = SyntheticCaseGenerator(seed=7)
    cases = gen.generate_demo_set()
    pipe = CasePipeline()
    for case in cases:
        result = pipe.run_case(case)
        for outcome in result.outcomes:
            usage = outcome.usage
            assert usage is not None, f"missing usage on {case.case_id}/{outcome.program}"
            assert usage.turns >= 2  # at least one assess + one verify
            assert usage.tokens_input > 0 and usage.tokens_output > 0
            assert usage.cost_usd == 0.0  # local provider is free
            assert usage.cost_backend_id == "local:rules"
            if outcome.replans:
                assert usage.proposer_turns == outcome.replans + 1
    # The report renders without error and marks estimated tokens.
    from tribune.eval.harness import records_for_case

    records = []
    for case in cases:
        records.extend(records_for_case(case, pipe.run_case(case)))
    rendered = compute_cost_report(records).render()
    assert "by outcome type" in rendered and "deterministic estimates" in rendered


def test_default_pricing_for_grok_and_deepseek_v4_pro():
    from tribune.eval.costmodel import default_cost_model

    cm = default_cost_model()

    # Grok 4.6: $2.00/M in, $6.00/M out
    grok_call = _call(tokens_in=1_000_000, tokens_out=500_000, model="grok-4.6")
    cost_grok, backend_grok = cm.cost_of_call(grok_call, on=date(2026, 8, 1))
    assert backend_grok == "xai:grok-4.6"
    assert abs(cost_grok - (2.00 + 0.5 * 6.00)) < 1e-9  # $5.00

    # DeepSeek V4 Pro: $0.435/M in, $0.87/M out
    deepseek_call = _call(tokens_in=2_000_000, tokens_out=1_000_000, model="deepseek-v4-pro")
    cost_ds, backend_ds = cm.cost_of_call(deepseek_call, on=date(2026, 8, 1))
    assert backend_ds == "deepseek:v4-pro"
    assert abs(cost_ds - (2.0 * 0.435 + 1.0 * 0.87)) < 1e-9  # $1.74


def test_pricing_for_gemini_and_gpt56_ultrafast():
    from tribune.eval.costmodel import default_cost_model

    cm = default_cost_model()

    # Gemini 3.7 Flash: $0.75/M in, $3.75/M out, $0.1875 cache read
    gemini_call = _call(tokens_in=1_000_000, tokens_out=200_000, cache_read=400_000, model="gemini-3.7-flash")
    cost_gemini, backend_gemini = cm.cost_of_call(gemini_call, on=date(2026, 8, 1))
    assert backend_gemini == "google:gemini-3.7-flash"
    expected_gemini = (600_000 * 0.75 + 400_000 * 0.1875 + 200_000 * 3.75) / 1e6  # 0.45 + 0.075 + 0.75 = $1.275
    assert abs(cost_gemini - expected_gemini) < 1e-9

    # GPT-5.6 Sol UltraFast: $2.50/M in, $10.00/M out
    gpt_call = _call(tokens_in=1_000_000, tokens_out=500_000, model="gpt-5.6-sol-ultrafast")
    cost_gpt, backend_gpt = cm.cost_of_call(gpt_call, on=date(2026, 8, 1))
    assert backend_gpt == "openai:gpt-5.6-sol-ultrafast"
    expected_gpt = (1.0 * 2.50 + 0.5 * 10.00)  # $7.50
    assert abs(cost_gpt - expected_gpt) < 1e-9


