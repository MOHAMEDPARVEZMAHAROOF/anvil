"""Interfaces the agent loop depends on.

Concrete implementations live in llm.py (Nebius Token Factory inference),
sandbox.py (Token Factory Sandboxes), and websearch.py (Tavily).
Keeping Protocols here lets the agent loop and its tests run without keys.
"""
from __future__ import annotations

from typing import Protocol

from .models import ExecResult


class LLM(Protocol):
    def chat(
        self,
        model: str,
        system: str,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        """Return the assistant's raw text reply."""
        ...


class Sandbox(Protocol):
    def write_file(self, path: str, content: str) -> None: ...
    def read_file(self, path: str) -> str: ...
    def exec(self, command: str, timeout_s: int = 120) -> ExecResult: ...
    def close(self) -> None: ...


class WebSearch(Protocol):
    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Return [{'title':..., 'url':..., 'snippet':...}, ...]."""
        ...
