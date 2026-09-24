"""OpenAI-compatible chat-completions client with a selectable backend.

Backends (``ANVIL_LLM_BACKEND``):
  - ``nebius`` (default): Nebius Token Factory inference API. REQUIRED for the
    hackathon submission — the rules demand a runtime call to Token Factory.
  - ``nvidia``: NVIDIA build.nvidia.com API. Dev fallback while Nebius signup
    is under maintenance; same OpenAI-compatible endpoint, same Nemotron
    family. Get a key at https://build.nvidia.com/settings/api-keys.
"""
from __future__ import annotations

import httpx

from .config import Settings


class LLMClient:
    """Thin wrapper over POST {base_url}/chat/completions with a Bearer API key."""

    def __init__(self, settings: Settings, timeout_s: int = 120) -> None:
        self.settings = settings
        self.backend = settings.llm_backend
        if self.backend == "nvidia":
            self.base_url = settings.nvidia_base_url.rstrip("/")
            self.api_key = settings.nvidia_api_key
        else:
            self.base_url = settings.nebius_base_url.rstrip("/")
            self.api_key = settings.nebius_api_key
        # trust_env=False: ignore host proxy env vars (they may be non-URL sockets).
        self._client = httpx.Client(timeout=timeout_s, trust_env=False)

    def chat(
        self,
        model: str,
        system: str,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        url = self.base_url + "/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = self._client.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def make_llm(settings: Settings, timeout_s: int = 120) -> LLMClient:
    """Build the chat client for the configured backend."""
    return LLMClient(settings, timeout_s=timeout_s)


# Backwards compatibility for existing imports.
NebiusLLM = LLMClient
