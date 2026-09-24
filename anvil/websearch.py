"""Tavily web-search tool for the agent's research actions."""
from __future__ import annotations

import httpx

from .config import Settings


class TavilySearch:
    def __init__(self, settings: Settings, timeout_s: int = 30) -> None:
        self.settings = settings
        self._client = httpx.Client(timeout=timeout_s, trust_env=False)

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        resp = self._client.post(
            "https://api.tavily.com/search",
            json={
                "query": query,
                "max_results": max_results,
                "search_depth": "advanced",
            },
            headers={"Authorization": f"Bearer {self.settings.tavily_api_key}"},
        )
        resp.raise_for_status()
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
            }
            for r in resp.json().get("results", [])
        ]
