"""Unit tests for ModelRouter local quantized fallback behavior."""

from __future__ import annotations

import unittest

from tribune.config import TribuneSettings
from tribune.providers.base import (
    Citation,
    CriterionResult,
    ProgramId,
    ReviewRequest,
    ReviewResult,
    SynthesisRequest,
    SynthesisResult,
)
from tribune.providers.local_rules import LocalRulesProvider
from tribune.providers.router import ModelRouter


class FailingMockProvider:
    """Mock provider simulating API failures or 429 rate limit exceptions."""

    def __init__(self, is_rate_limit: bool = False) -> None:
        self.name = "failing_api_provider"
        self.version = "1.0"
        self.is_rate_limit = is_rate_limit

    def synthesize_assessment(self, req: SynthesisRequest) -> SynthesisResult:
        if self.is_rate_limit:
            raise RuntimeError("HTTP 429 Too Many Requests: Rate limit exceeded")
        raise RuntimeError("HTTP 500 Internal Server Error")

    def review_assessment(self, req: ReviewRequest) -> ReviewResult:
        if self.is_rate_limit:
            raise RuntimeError("HTTP 429 Rate Limit Exceeded")
        raise RuntimeError("HTTP 502 Bad Gateway")


class TestLocalFallback(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = TribuneSettings(enable_local_fallback=True)
        self.citation = Citation(
            citation_id="cit-1",
            program=ProgramId.SNAP,
            jurisdiction="EX",
            title="Income Limit Rule",
            source="7 CFR 273.9",
            text="Income limit rule",
        )
        self.req = SynthesisRequest(
            program=ProgramId.SNAP,
            jurisdiction="EX",
            criteria=[
                CriterionResult(
                    criterion_id="c1",
                    description="Income limit",
                    required=True,
                    outcome="satisfied",
                    citation_ids=["cit-1"],
                )
            ],
            required_total=1,
            coverage_complete=True,
            evidence_summary="Verified income documents.",
            citations=[self.citation],
        )

    def test_fallback_on_api_500_error(self) -> None:
        failing_p1 = FailingMockProvider(is_rate_limit=False)
        failing_p2 = FailingMockProvider(is_rate_limit=False)
        fallback_p = LocalRulesProvider(role="local_fallback")

        router = ModelRouter(
            tier1_provider=failing_p1,
            tier2_provider=failing_p2,
            fallback_provider=fallback_p,
            settings=self.settings,
        )

        res = router.synthesize_assessment(self.req)
        self.assertIsInstance(res, SynthesisResult)
        self.assertIn("LOCAL QUANTIZED FALLBACK", res.rationale)
        self.assertEqual(router.stats["local_quantized_fallbacks"], 1)
        self.assertEqual(router.stats["api_failures"], 2)

    def test_fallback_on_api_429_rate_limit(self) -> None:
        failing_p1 = FailingMockProvider(is_rate_limit=True)
        failing_p2 = FailingMockProvider(is_rate_limit=True)
        fallback_p = LocalRulesProvider(role="local_fallback")

        router = ModelRouter(
            tier1_provider=failing_p1,
            tier2_provider=failing_p2,
            fallback_provider=fallback_p,
            settings=self.settings,
        )

        res = router.synthesize_assessment(self.req)
        self.assertIsInstance(res, SynthesisResult)
        self.assertIn("LOCAL QUANTIZED FALLBACK", res.rationale)
        self.assertGreaterEqual(router.stats["rate_limits"], 1)

    def test_generic_route_and_execute_fallback(self) -> None:
        router = ModelRouter(settings=self.settings)

        def fail_t1():
            raise RuntimeError("Primary API crashed")

        def fail_t2():
            raise RuntimeError("Secondary API crashed")

        def fallback_fn():
            return "fallback_result_ok"

        res = router.route_and_execute(
            fn_tier1=fail_t1,
            fn_tier2=fail_t2,
            intent="parsing",
            fn_fallback=fallback_fn,
        )
        self.assertEqual(res, "fallback_result_ok")
        self.assertEqual(router.stats["local_quantized_fallbacks"], 1)


if __name__ == "__main__":
    unittest.main()
