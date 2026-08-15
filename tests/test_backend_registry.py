"""Phase 5 — backend registry schema validation.

Covers: the committed registry validates, the seed candidates the plan requires
are present with the right statuses, and the schema's honesty invariants reject
malformed entries. No network (the link check is exercised separately and opt-in).
"""

import importlib.util
import os

import pytest

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "validate_registry.py")
_REGISTRY = os.path.join(os.path.dirname(__file__), "..", "backends", "registry.yaml")

_spec = importlib.util.spec_from_file_location("validate_registry", _SCRIPT)
vr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vr)


def test_committed_registry_validates():
    registry = vr.load_registry(_REGISTRY)
    assert registry.version == 1
    assert registry.seed_set == "tribune-quant-seed-set"
    assert len(registry.candidates) >= 7


def test_all_required_seed_candidates_present():
    registry = vr.load_registry(_REGISTRY)
    ids = {c.candidate_id for c in registry.candidates}
    for required in [
        "glm-5.2",
        "deepseek-v4",
        "openpangu-2-flash-92b",
        "openpangu-2-flash-6b",
        "longcat-2.0",
        "qwen3.6-27b-nvfp4",
        "claude-sonnet-5",
        "grok-4.6",
        "deepseek-v4-pro",
    ]:
        assert required in ids


def test_status_flags_match_plan():
    registry = vr.load_registry(_REGISTRY)
    by_id = {c.candidate_id: c for c in registry.candidates}
    # LongCat is announced-only and must not be deployable/verified.
    assert by_id["longcat-2.0"].weights_status == "announced_only"
    assert by_id["longcat-2.0"].weights_verified is False
    # OpenPangu weights are unverified until confirmed.
    assert by_id["openpangu-2-flash-92b"].weights_status == "unverified"
    # Sonnet 5 is API-only with a promo end date, proposer role.
    sonnet = by_id["claude-sonnet-5"]
    assert sonnet.weights_status == "api_only"
    assert sonnet.promo_end_date == "2026-08-31"
    assert sonnet.roles == ["proposer"]
    # Grok 4.6 is high-tier verifier and proposer
    grok = by_id["grok-4.6"]
    assert grok.weights_status == "api_only"
    assert "verifier" in grok.roles
    assert "proposer" in grok.roles
    # DeepSeek V4 Pro is high-volume low-cost engine
    dsv4pro = by_id["deepseek-v4-pro"]
    assert dsv4pro.weights_status == "api_only"
    assert "proposer" in dsv4pro.roles
    # GLM-5.2 is the incumbent verifier candidate.
    assert "verifier" in by_id["glm-5.2"].roles
    # No measured slots are pre-filled — they come from eval runs.
    assert all(c.measured_kappa is None for c in registry.candidates)
    assert all(c.measured_cost_per_task is None for c in registry.candidates)


def test_downloadable_requires_url():
    with pytest.raises(ValueError, match="requires a weights_url"):
        vr.Candidate(
            candidate_id="x", name="X", vendor="v", params="1B", roles=["proposer"],
            license="open-weights", weights_status="downloadable", weights_url=None,
            serving=["vllm"], context_window=8192,
        )


def test_announced_only_cannot_be_verified():
    with pytest.raises(ValueError, match="weights_verified=true only valid"):
        vr.Candidate(
            candidate_id="x", name="X", vendor="v", params="1B", roles=["verifier"],
            license="unknown", weights_status="announced_only", weights_verified=True,
            serving=["vllm"], context_window=8192,
        )


def test_duplicate_ids_rejected():
    good = vr.load_registry(_REGISTRY)
    payload = {
        "version": 1,
        "seed_set": "s",
        "candidates": [
            good.candidates[0].model_dump(),
            good.candidates[0].model_dump(),
        ],
    }
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        vr.Registry.model_validate(payload)


def test_unknown_field_rejected():
    with pytest.raises(ValueError):
        vr.Candidate(
            candidate_id="x", name="X", vendor="v", params="1B", roles=["proposer"],
            license="open-weights", weights_status="unverified",
            serving=["vllm"], context_window=8192, bogus_field="nope",
        )


def test_main_exits_zero_on_committed_registry(capsys):
    rc = vr.main(["--path", _REGISTRY])
    assert rc == 0
    assert "schema OK" in capsys.readouterr().out


def test_muse_glimmer_local_profile_configured():
    registry = vr.load_registry(_REGISTRY)
    assert registry.muse_glimmer_local is not None
    mgl = registry.muse_glimmer_local
    assert mgl["provider"] == "openai_compat"
    assert mgl["base_url"] == "http://localhost:8080/v1"
    assert mgl["model_name"] == "Muse-Glimmer-30B-GGUF"
    assert mgl["parameters"]["temperature"] == 0.1
    assert mgl["parameters"]["max_tokens"] == 4096
    extra = mgl["parameters"]["extra_body"]
    assert extra["speculative_drafter"] == "DFlash"
    assert extra["sliding_window_attention"] is True
    assert extra["kv_cache_type"] == "q8_0"


def test_openai_compat_extra_body_forwarding():
    from tribune.config import get_settings
    from tribune.providers.openai_compat import OpenAICompatProvider

    settings = get_settings()
    extra_body = {
        "speculative_drafter": "DFlash",
        "sliding_window_attention": True,
        "kv_cache_type": "q8_0",
    }
    provider = OpenAICompatProvider(
        model="Muse-Glimmer-30B-GGUF",
        settings=settings,
        role="proposer",
        extra_body=extra_body,
        temperature=0.1,
        max_tokens=4096,
    )
    assert provider.extra_body == extra_body
    assert provider.temperature == 0.1
    assert provider.max_tokens == 4096


def test_default_workhorse_and_providers_configuration():
    registry = vr.load_registry(_REGISTRY)
    assert registry.default_workhorse == "gemini-3.7-flash"
    assert registry.providers is not None
    assert "gemini-3.7-flash" in registry.providers
    assert "gpt-5.6-sol-ultrafast" in registry.providers

    g37 = registry.providers["gemini-3.7-flash"]
    assert g37.provider_type == "openai_compat"
    assert g37.model_name == "gemini-3.7-flash"
    assert g37.input_cost_per_1m == 0.75
    assert g37.output_cost_per_1m == 3.75
    assert g37.max_context_tokens == 1048576
    assert g37.supports_tools is True

    gpt56 = registry.providers["gpt-5.6-sol-ultrafast"]
    assert gpt56.provider_type == "openai_compat"
    assert gpt56.model_name == "gpt-5.6-sol-ultrafast"
    assert gpt56.input_cost_per_1m == 2.50
    assert gpt56.output_cost_per_1m == 10.00
    assert gpt56.tokens_per_second == 750
    assert gpt56.supports_tools is True


