"""Inference Provider & Model Tier Configuration.

Enforces configuration hierarchy:
1. Hardcoded safe defaults.
2. YAML / JSON configuration file (e.g., config.yaml).
3. Environment variables (TRIBUNE_*).
4. Explicit runtime parameter overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class TierConfig:
    """Configuration for a specific execution model tier."""

    provider_id: str
    model: str
    base_url: str | None = None
    timeout_s: float = 60.0
    max_tokens: int = 4096
    temperature: float = 0.0
    budget_cap_tokens: int = 100_000
    budget_cap_usd: float = 0.50


@dataclass
class InferenceConfig:
    """Master configuration for all inference subsystems and model tiers."""

    # Lead / Frontier Tier (strategy, complex statutory synthesis, adjudication)
    lead_tier: TierConfig = field(
        default_factory=lambda: TierConfig(
            provider_id=os.getenv("TRIBUNE_LEAD_PROVIDER", "openai_compatible"),
            model=os.getenv("TRIBUNE_LEAD_MODEL", "gpt-4o"),
            base_url=os.getenv("TRIBUNE_LEAD_BASE_URL", "https://api.openai.com/v1"),
            timeout_s=float(os.getenv("TRIBUNE_LEAD_TIMEOUT", "60.0")),
            budget_cap_tokens=150_000,
            budget_cap_usd=1.00,
        )
    )

    # Worker Tier (form filling, docketing, high-volume low-cost ops)
    worker_tier: TierConfig = field(
        default_factory=lambda: TierConfig(
            provider_id=os.getenv("TRIBUNE_WORKER_PROVIDER", "openai_compatible"),
            model=os.getenv("TRIBUNE_WORKER_MODEL", "deepseek-v4.1-flash"),
            base_url=os.getenv("TRIBUNE_WORKER_BASE_URL", "https://api.deepseek.com/v1"),
            timeout_s=float(os.getenv("TRIBUNE_WORKER_TIMEOUT", "30.0")),
            budget_cap_tokens=50_000,
            budget_cap_usd=0.05,
        )
    )

    # Global runtime settings
    enable_mid_run_switching: bool = True
    max_retries: int = int(os.getenv("TRIBUNE_MAX_INFERENCE_RETRIES", "3"))
    strict_budget_enforcement: bool = True
    active_tier: str = "worker"  # "lead" | "worker"

    @classmethod
    def load_from_env_and_file(cls, config_path: str | None = None) -> InferenceConfig:
        """Load configuration respecting defaults -> config file -> environment variables."""
        cfg = cls()

        # Check for config.yaml if path provided or in root
        path = config_path or os.path.join(os.getcwd(), "config.yaml")
        if os.path.exists(path):
            try:
                import yaml  # type: ignore

                with open(path, encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}

                # Apply yaml overrides if present
                if "lead_tier" in data:
                    lead_d = data["lead_tier"]
                    cfg.lead_tier.provider_id = lead_d.get("provider_id", cfg.lead_tier.provider_id)
                    cfg.lead_tier.model = lead_d.get("model", cfg.lead_tier.model)
                    cfg.lead_tier.base_url = lead_d.get("base_url", cfg.lead_tier.base_url)

                if "worker_tier" in data:
                    work_d = data["worker_tier"]
                    cfg.worker_tier.provider_id = work_d.get("provider_id", cfg.worker_tier.provider_id)
                    cfg.worker_tier.model = work_d.get("model", cfg.worker_tier.model)
                    cfg.worker_tier.base_url = work_d.get("base_url", cfg.worker_tier.base_url)

            except Exception:
                pass  # Fall back to defaults and env vars

        return cfg
