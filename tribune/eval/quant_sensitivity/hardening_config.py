"""Centralized configuration for the Hardened Three-Tier Evaluation Architecture.

Supports configuration resolution precedence:
1. Explicit runtime arguments
2. Environment variables
3. YAML/JSON configuration files
4. Safe production defaults
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class HardeningConfig:
    """Configuration settings for hardened execution, Sentinel, and memory sidecar."""

    # Execution & Isolation Tier
    execution_mode: str = "container"  # "container" | "local_fallback" | "legacy" | "auto"
    container_image: str = "tribune-sandbox:latest"
    container_user: str = "tribune-sandbox"
    container_network_mode: str = "none"  # "none" | "sentinel_proxy"
    container_memory_limit: str = "1024m"
    container_cpu_limit: str = "2.0"
    max_command_timeout_seconds: float = 60.0

    # Sentinel Security Daemon
    sentinel_enabled: bool = True
    sentinel_socket_path: str = "/tmp/tribune_sentinel.sock"
    allowlist_path: str = "config/sentinel_allowlist.yaml"
    surrogate_token_prefix: str = "MOCK_"
    redaction_enabled: bool = True

    # Proactive Memory Sidecar
    memory_sidecar_enabled: bool = True
    memory_update_interval_k: int = 2  # Execute every k turns
    memory_max_context_chars: int = 4000
    memory_intervention_cooldown_turns: int = 2
    memory_min_confidence_for_intervention: float = 0.70
    memory_max_interventions_per_task: int = 5
    memory_aux_backend: str = "heuristic_fast"  # "heuristic_fast" | "local_8bit" | "api"
    memory_aux_model_name: str = "fast-eval-aux-8b"
    memory_redaction_enabled: bool = True

    # Cost Accounting & Reporting
    cost_model_pricing_path: str = "tribune/eval/pricing.json"
    report_output_dir: str = "docs/eval_notes"
    evidence_json_path: str = "docs/hardened_roadmap/evidence.json"

    # Additional custom parameters
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        config_path: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> HardeningConfig:
        """Resolve configuration applying: Safe Defaults <- File <- Env <- Runtime Args."""
        data: dict[str, Any] = {}

        # 1. Load from file if provided or if default exists
        file_to_check = config_path or os.environ.get("TRIBUNE_HARDENING_CONFIG")
        if file_to_check and os.path.exists(file_to_check):
            with open(file_to_check, encoding="utf-8") as fh:
                if file_to_check.endswith((".yaml", ".yml")):
                    data = yaml.safe_load(fh) or {}
                else:
                    data = json.load(fh) or {}

        # 2. Overlay environment variables
        env_mappings: dict[str, tuple[str, type]] = {
            "TRIBUNE_EXECUTION_MODE": ("execution_mode", str),
            "TRIBUNE_CONTAINER_IMAGE": ("container_image", str),
            "TRIBUNE_CONTAINER_USER": ("container_user", str),
            "TRIBUNE_CONTAINER_NETWORK_MODE": ("container_network_mode", str),
            "TRIBUNE_MAX_COMMAND_TIMEOUT_SECONDS": ("max_command_timeout_seconds", float),
            "TRIBUNE_SENTINEL_ENABLED": ("sentinel_enabled", lambda v: str(v).lower() in ("true", "1", "yes")),
            "TRIBUNE_SENTINEL_SOCKET_PATH": ("sentinel_socket_path", str),
            "TRIBUNE_ALLOWLIST_PATH": ("allowlist_path", str),
            "TRIBUNE_SURROGATE_TOKEN_PREFIX": ("surrogate_token_prefix", str),
            "TRIBUNE_REDACTION_ENABLED": ("redaction_enabled", lambda v: str(v).lower() in ("true", "1", "yes")),
            "TRIBUNE_MEMORY_SIDECAR_ENABLED": ("memory_sidecar_enabled", lambda v: str(v).lower() in ("true", "1", "yes")),
            "TRIBUNE_MEMORY_UPDATE_INTERVAL_K": ("memory_update_interval_k", int),
            "TRIBUNE_MEMORY_AUX_BACKEND": ("memory_aux_backend", str),
            "TRIBUNE_MEMORY_AUX_MODEL_NAME": ("memory_aux_model_name", str),
            "TRIBUNE_MEMORY_MIN_CONFIDENCE": ("memory_min_confidence_for_intervention", float),
            "TRIBUNE_MEMORY_MAX_INTERVENTIONS": ("memory_max_interventions_per_task", int),
            "TRIBUNE_PRICING_PATH": ("cost_model_pricing_path", str),
            "TRIBUNE_EVIDENCE_JSON_PATH": ("evidence_json_path", str),
        }

        for env_var, (attr, parser) in env_mappings.items():
            if env_var in os.environ:
                try:
                    data[attr] = parser(os.environ[env_var])
                except Exception:
                    pass

        # 3. Overlay explicit runtime overrides
        if overrides:
            for k, v in overrides.items():
                if v is not None:
                    data[k] = v

        # Construct instance with matched fields
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        matched = {k: v for k, v in data.items() if k in fields}
        extra = {k: v for k, v in data.items() if k not in fields}

        cfg = cls(**matched)
        if extra:
            cfg.extra.update(extra)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_DEFAULT_CONFIG: HardeningConfig | None = None


def get_hardening_config() -> HardeningConfig:
    """Return singleton instance of resolved hardening configuration."""
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = HardeningConfig.load()
    return _DEFAULT_CONFIG


def set_hardening_config(config: HardeningConfig) -> None:
    """Override the global hardening configuration."""
    global _DEFAULT_CONFIG
    _DEFAULT_CONFIG = config
