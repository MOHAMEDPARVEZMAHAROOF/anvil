#!/usr/bin/env python3
"""CLI runner: forge a task and print the agent's event timeline.

Usage:
    python scripts/demo.py "Write a Python rate limiter with pytest tests"
"""
from __future__ import annotations

import sys

from dotenv import load_dotenv

from anvil.agent import CodingAgent
from anvil.config import Settings
from anvil.llm import make_llm
from anvil.models import AgentEvent
from anvil.sandbox import NebiusSandbox
from anvil.websearch import TavilySearch


def main() -> int:
    load_dotenv()
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    task = sys.argv[1]
    settings = Settings()
    has_key = settings.has_nvidia_key if settings.llm_backend == "nvidia" else settings.has_nebius_key
    if not has_key:
        key_name = "NVIDIA_API_KEY" if settings.llm_backend == "nvidia" else "NEBIUS_API_KEY"
        print(f"{key_name} is not set. Copy .env.example to .env and fill it in.")
        return 1

    def on_event(ev: AgentEvent) -> None:
        icon = {
            "plan": "🧭", "write": "✍️", "exec": "▶️", "read": "📖",
            "search": "🔎", "fix": "🔧", "done": "✅", "error": "❌",
        }.get(ev.type, "•")
        print(f"{icon} [{ev.type}] {ev.message}")

    agent = CodingAgent(
        llm=make_llm(settings),
        sandbox_factory=lambda: NebiusSandbox(settings),
        settings=settings,
        websearch=TavilySearch(settings) if settings.has_tavily_key else None,
    )
    result = agent.run(task, on_event=on_event)
    print("\n" + "=" * 60)
    print("SUCCESS" if result.success else "FINISHED WITH ISSUES")
    print(result.summary)
    print(f"\nIterations: {result.iterations}")
    print("\n--- test report ---")
    print(result.test_report)
    print("\n--- diff ---")
    print(result.diff)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
