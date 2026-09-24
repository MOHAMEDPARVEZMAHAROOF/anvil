"""Demo web server: run Anvil from a browser, watch the agent think in real time.

Endpoints:
  GET  /                  -> demo UI
  POST /api/runs          -> {"task": str, "seed_files": {path: content}} -> {"run_id": ...}
  GET  /api/runs/{id}/events (SSE stream of AgentEvent JSON)
  GET  /api/runs/{id}     -> final RunResult JSON (when finished)

Run with:  uvicorn anvil.server:app --reload
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agent import CodingAgent
from .config import Settings
from .models import AgentEvent, RunResult

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="Anvil")

_runs: dict[str, dict] = {}  # run_id -> {"events": Queue, "result": RunResult|None, "done": bool}


def _build_agent(settings: Settings) -> CodingAgent:
    # Concrete llm/sandbox/websearch are wired in llm.py, sandbox.py, websearch.py.
    from .llm import make_llm
    from .sandbox import NebiusSandbox
    from .websearch import TavilySearch

    return CodingAgent(
        llm=make_llm(settings),
        sandbox_factory=lambda: NebiusSandbox(settings),
        settings=settings,
        websearch=TavilySearch(settings) if settings.has_tavily_key else None,
    )


def _worker(run_id: str, task: str, seed_files: dict, settings: Settings) -> None:
    entry = _runs[run_id]
    q: queue.Queue = entry["events"]

    def on_event(ev: AgentEvent) -> None:
        q.put(ev)

    try:
        agent = _build_agent(settings)
        result = agent.run(task, seed_files=seed_files, on_event=on_event)
        entry["result"] = result
    except Exception as e:  # noqa: BLE001
        on_event(AgentEvent(type="error", message=f"Run failed: {e}"))
        entry["result"] = None
    finally:
        entry["done"] = True
        q.put(None)  # sentinel: stream is over


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB_DIR / "index.html").read_text()


@app.post("/api/runs")
def create_run(payload: dict) -> dict:
    task = (payload.get("task") or "").strip()
    if not task:
        raise HTTPException(400, "task is required")
    settings = Settings()
    if not settings.has_nebius_key:
        raise HTTPException(400, "NEBIUS_API_KEY is not set on the server")
    import uuid

    run_id = uuid.uuid4().hex[:12]
    _runs[run_id] = {"events": queue.Queue(), "result": None, "done": False}
    threading.Thread(
        target=_worker,
        args=(run_id, task, payload.get("seed_files") or {}, settings),
        daemon=True,
    ).start()
    return {"run_id": run_id}


@app.get("/api/runs/{run_id}/events")
def stream_events(run_id: str) -> StreamingResponse:
    entry = _runs.get(run_id)
    if entry is None:
        raise HTTPException(404, "unknown run")
    q: queue.Queue = entry["events"]

    def gen():
        while True:
            try:
                ev = q.get(timeout=30)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            if ev is None:
                yield "event: end\ndata: {}\n\n"
                break
            yield f"data: {json.dumps(ev.__dict__)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    entry = _runs.get(run_id)
    if entry is None:
        raise HTTPException(404, "unknown run")
    result: RunResult | None = entry["result"]
    return {
        "done": entry["done"],
        "result": result.to_dict() if result else None,
    }


if os.environ.get("ANVIL_SERVE_STATIC") == "1":
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
