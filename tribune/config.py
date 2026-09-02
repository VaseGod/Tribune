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
    # "local_rules" is the deterministic, offline default. Supports: "local_rules", "openai_compat", "openai", "anthropic", "deepseek", "vllm", "grok", "xai", "gemini", "glm", "zhipu"
    provider: Literal["local_rules", "openai_compat", "openai", "anthropic", "deepseek", "vllm", "grok", "xai", "gemini", "glm", "zhipu"] = "local_rules"
    openai_base_url: str = "http://localhost:8000/v1"
    openai_api_key: str = "not-needed-for-local-serving"
    openai_model: str = "gemini-3.7-flash"

    # GLM / Zhipu API parameters (including GLM-5.3-Flash)
    glm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    glm_api_key: str = ""
    glm_model: str = "glm-5.3-flash"

    # Gemini API parameters
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.7-flash"

    # Anthropic API parameters
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # DeepSeek API parameters (including DeepSeek-V4-Flash & DeepSeek-V4-Pro)
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-v4-pro"
    deepseek_reasoning_effort: Literal["low", "medium", "high"] = "medium"

    # Grok / xAI API parameters
    grok_base_url: str = "https://api.x.ai/v1"
    grok_api_key: str = ""
    grok_model: str = "grok-4.6"

    # Open-weight / Local vLLM parameters
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "token-vllm-local"
    vllm_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # SGLang serving parameters
    sglang_base_url: str = "http://localhost:30000/v1"
    sglang_api_key: str = "token-sglang-local"
    sglang_model: str = "Qwen/Qwen3.8-Flash-Next"

    # Hybrid-Attention & Engram Offloading parameters
    vllm_ple_cpu_offload: bool = Field(
        default_factory=lambda: str(
            os.getenv("VLLM_PLE_CPU_OFFLOAD", os.getenv("TRIBUNE_VLLM_PLE_CPU_OFFLOAD", "1"))
        ).lower()
        in ("true", "1", "yes")
    )
    num_speculative_tokens: int = Field(
        default_factory=lambda: int(
            os.getenv("TRIBUNE_NUM_SPECULATIVE_TOKENS", os.getenv("NUM_SPECULATIVE_TOKENS", "1"))
        )
    )
    confidence_fallback_threshold: float = Field(
        default_factory=lambda: float(
            os.getenv("TRIBUNE_CONFIDENCE_FALLBACK_THRESHOLD", "0.85")
        )
    )
    hybrid_attention_backend: str = Field(
        default_factory=lambda: os.getenv("TRIBUNE_HYBRID_ATTENTION_BACKEND", "vllm")
    )
    qwen_flash_model: str = Field(
        default_factory=lambda: os.getenv("TRIBUNE_QWEN_FLASH_MODEL", "Qwen3.8-Flash-Next")
    )

    # Local Quantized Fallback Endpoint (3-bit MoE / local server)
    local_fallback_url: str = "http://localhost:8001/v1"
    local_fallback_model: str = "local-quant-moe-3bit"
    enable_local_fallback: bool = True

    # Static Analysis Path & Execution
    semgrep_binary: str = "semgrep"
    static_analysis_timeout_s: float = 30.0

    # The verifier can run on a *stronger* model than the proposer.
    verifier_provider: Literal["local_rules", "openai_compat", "openai", "anthropic", "deepseek", "vllm", "sglang", "grok", "xai", "gemini"] = "local_rules"
    verifier_model: str = "gpt-5.6-sol-ultrafast"
    request_timeout_s: float = 60.0

    # -- Tiered Model Router ------------------------------------------------ #
    tier1_model: str = Field(
        default_factory=lambda: os.getenv(
            "DEFAULT_TIER1_MODEL",
            os.getenv("TRIBUNE_DEFAULT_TIER1_MODEL", "DeepSeek V4 Pro"),
        )
    )
    tier2_model: str = Field(
        default_factory=lambda: os.getenv(
            "DEFAULT_TIER2_MODEL",
            os.getenv("TRIBUNE_DEFAULT_TIER2_MODEL", "Grok 4.6"),
        )
    )
    tier1_provider: Literal["local_rules", "openai_compat"] = "local_rules"
    tier2_provider: Literal["local_rules", "openai_compat"] = "local_rules"
    use_tiered_routing: bool = True
    mcp_auth_token: str = Field(
        default_factory=lambda: os.getenv("TRIBUNE_MCP_AUTH_TOKEN", "")
    )

    # -- ReasonMaxxer & Local Dense Model Configuration -------------------- #
    enable_reasonmaxxer: bool = Field(
        default_factory=lambda: str(
            os.getenv("TRIBUNE_ENABLE_REASONMAXXER", "true")
        ).lower()
        in ("true", "1", "yes")
    )
    local_model_path: str = Field(
        default_factory=lambda: os.getenv("TRIBUNE_LOCAL_MODEL_PATH", "")
    )
    local_model_type: str = Field(
        default_factory=lambda: os.getenv("TRIBUNE_LOCAL_MODEL_TYPE", "qwen3.8-27b")
    )
    route_strategy: str = Field(
        default_factory=lambda: os.getenv("TRIBUNE_ROUTE_STRATEGY", "latency-budget")
    )
    kv_quant_type: str = Field(
        default_factory=lambda: os.getenv("TRIBUNE_KV_QUANT_TYPE", "q4_1")
    )
    spec_type: str = Field(
        default_factory=lambda: os.getenv("TRIBUNE_SPEC_TYPE", "ngram-mod,draft-mtp")
    )
    spec_draft_max: int = Field(
        default_factory=lambda: int(os.getenv("TRIBUNE_SPEC_DRAFT_MAX", "2"))
    )
    batch_size: int = Field(
        default_factory=lambda: int(os.getenv("TRIBUNE_BATCH_SIZE", "1024"))
    )
    ubatch_size: int = Field(
        default_factory=lambda: int(os.getenv("TRIBUNE_UBATCH_SIZE", "512"))
    )
    entropy_threshold: float = Field(
        default_factory=lambda: float(os.getenv("TRIBUNE_ENTROPY_THRESHOLD", "0.35"))
    )
    flash_attention: bool = Field(
        default_factory=lambda: str(
            os.getenv("TRIBUNE_FLASH_ATTENTION", "true")
        ).lower()
        in ("true", "1", "yes")
    )
    local_runtime_quant: str = Field(
        default_factory=lambda: os.getenv("TRIBUNE_LOCAL_RUNTIME_QUANT", "IQ4_XS")
    )
    memory_budget_gb: float = Field(
        default_factory=lambda: float(os.getenv("TRIBUNE_MEMORY_BUDGET_GB", "16.0"))
    )



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
    abstention_threshold: float = 0.85

    # -- Tracing ------------------------------------------------------------ #
    tracing: Literal["none", "weave"] = "none"
    weave_project: str = "tribune"

    # -- Cost accounting ------------------------------------------------------ #
    # Pricing is data, not code. Empty pricing_path uses the packaged
    # tribune/eval/pricing.json. pricing_date (ISO date) pins the accounting date
    # so promotional-rate expiry is reproducible; empty means "today".
    pricing_path: str = ""
    pricing_date: str = ""

    # -- Multilingual parity (equity gate) ------------------------------------ #
    # Empty uses the packaged tribune/eval/parity_thresholds.json.
    parity_thresholds_path: str = ""

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
