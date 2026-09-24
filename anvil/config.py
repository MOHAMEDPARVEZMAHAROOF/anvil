"""Environment-driven configuration. No secrets are hardcoded; everything comes from env/.env."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


# LLM backend selection. "nebius" (default) is REQUIRED for the hackathon
# submission — the rules demand a runtime call to the Token Factory API.
# "nvidia" (build.nvidia.com) is a dev fallback while Nebius signup is down:
# same OpenAI-compatible chat-completions API, same Nemotron models.
_LLM_BACKEND = _get("ANVIL_LLM_BACKEND", "nebius").lower()


def _model_default(env_name: str, nebius_id: str, nvidia_id: str) -> str:
    v = _get(env_name)
    if v:
        return v
    return nvidia_id if _LLM_BACKEND == "nvidia" else nebius_id


@dataclass
class Settings:
    nebius_api_key: str = _get("NEBIUS_API_KEY")
    # Some accounts require the project id on every Sandboxes call.
    nebius_project_id: str = _get("NEBIUS_PROJECT_ID")
    # Token Factory inference endpoint (OpenAI-compatible chat completions).
    nebius_base_url: str = _get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")

    # Dev fallback backend: NVIDIA build.nvidia.com (OpenAI-compatible).
    # Used only while Nebius signup is unavailable; the submission must use nebius.
    llm_backend: str = _LLM_BACKEND
    nvidia_api_key: str = _get("NVIDIA_API_KEY")
    nvidia_base_url: str = _get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

    # Model routing: heavy reasoning -> Ultra, everyday coding -> Super, fast calls -> Nano.
    # Nebius IDs confirmed against public Token Factory sources (re-verify with GET /v1/models).
    # NVIDIA IDs follow build.nvidia.com catalog naming (re-verify with GET /v1/models).
    model_plan: str = _model_default(
        "ANVIL_MODEL_PLAN", "nvidia/Nemotron-3-Ultra-550b-a55b", "nvidia/nemotron-3-ultra-550b-a55b"
    )
    model_code: str = _model_default(
        "ANVIL_MODEL_CODE", "nvidia/nemotron-3-super-120b-a12b", "nvidia/nemotron-3-super-120b-a12b"
    )
    model_fast: str = _model_default(
        "ANVIL_MODEL_FAST", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", "nvidia/nemotron-nano-3-30b-a3b"
    )

    tavily_api_key: str = _get("TAVILY_API_KEY")

    max_iterations: int = int(_get("ANVIL_MAX_ITERATIONS", "8"))
    sandbox_timeout_s: int = int(_get("ANVIL_SANDBOX_TIMEOUT_S", "180"))

    @property
    def has_nebius_key(self) -> bool:
        return bool(self.nebius_api_key)

    @property
    def has_nvidia_key(self) -> bool:
        return bool(self.nvidia_api_key)

    @property
    def has_tavily_key(self) -> bool:
        return bool(self.tavily_api_key)
