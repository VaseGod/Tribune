"""Runtime configuration.

Every external dependency (LLM serving, vector store, OCR, tracing) is selected
here by environment variable and sits behind an interface with a working local
fallback. The defaults run the *entire* system offline, with no API keys and no
GPU. Flip one variable to point at a self-hosted model, a hosted vector store, or
an OCR endpoint.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TribuneSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRIBUNE_", extra="ignore")

    # -- Model provider ----------------------------------------------------- #
    # "local_rules" is the deterministic, offline default. "openai_compat" talks
    # to any OpenAI-compatible endpoint (vLLM / SGLang / etc.) hosting your model.
    provider: Literal["local_rules", "openai_compat"] = "local_rules"
    openai_base_url: str = "http://localhost:8000/v1"
    openai_api_key: str = "not-needed-for-local-serving"
    openai_model: str = "Qwen/Qwen2.5-7B-Instruct"
    # The verifier can run on a *stronger* model than the proposer.
    verifier_provider: Literal["local_rules", "openai_compat"] = "local_rules"
    verifier_model: str = "Qwen/Qwen2.5-32B-Instruct"
    request_timeout_s: float = 60.0

    # -- Rule store / retrieval --------------------------------------------- #
    rule_store: Literal["local", "hosted"] = "local"
    hosted_vector_url: str = ""
    hosted_vector_collection: str = "tribune-rules"
    retrieval_k: int = 8  # default top-k for late-interaction retrieval

    # -- Document ingestion ------------------------------------------------- #
    doc_ingest: Literal["structured", "ocr"] = "structured"
    ocr_endpoint: str = ""  # e.g. a self-hosted Unlimited-OCR server

    # -- Abstention --------------------------------------------------------- #
    # Below this calibrated confidence, TRIBUNE abstains and routes to a human.
    abstention_threshold: float = 0.70

    # -- Tracing ------------------------------------------------------------ #
    tracing: Literal["none", "weave"] = "none"
    weave_project: str = "tribune"

    # -- Cost accounting ------------------------------------------------------ #
    # Pricing is data, not code. Empty pricing_path uses the packaged
    # tribune/eval/pricing.json. pricing_date (ISO date) pins the accounting date
    # so promotional-rate expiry is reproducible; empty means "today".
    pricing_path: str = ""
    pricing_date: str = ""

    # -- General ------------------------------------------------------------ #
    default_jurisdiction: str = "EX"
    seed: int = 7
    data_dir: str = Field(default_factory=lambda: os.path.join(os.getcwd(), ".tribune"))


@lru_cache(maxsize=1)
def get_settings() -> TribuneSettings:
    return TribuneSettings()


def reset_settings_cache() -> None:
    """Tests flip env vars and re-read settings; this clears the memoized instance."""
    get_settings.cache_clear()
