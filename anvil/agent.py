"""The Anvil agent loop.

Pipeline:
  1. Planner (Nemotron 3 Ultra) turns the task into a plan + test command.
  2. Builder loop (Nemotron 3 Super/Nano) acts with one JSON action per turn:
     write / read / exec / search / done.
  3. Every exec result is fed back; failures trigger fix iterations.
  4. Finalize: unified diff of changed files, summary, test report.

All code execution happens inside the sandbox (Token Factory Sandboxes in
production, fakes in tests). The agent never runs code on the host.
"""
from __future__ import annotations

import difflib
import json
import re
import uuid
from typing import Callable

from .config import Settings
from .models import AgentEvent, ExecResult, RunResult
from .tools import LLM, Sandbox, WebSearch

PLANNER_SYSTEM = """You are Anvil's planner, powered by NVIDIA Nemotron 3 Ultra on Nebius.
Given a coding task, output ONE JSON object and nothing else:
{"plan": ["step 1", "step 2", ...], "files": ["relative/path.py", ...], "test_command": "pytest -q"}
Rules:
- Keep the plan to 3-6 concrete steps.
- test_command must verify the work (prefer pytest; fall back to `python -m py_compile` + a smoke run).
- files lists every file the builder should create or modify.
- All paths are relative to the sandbox working directory; never use absolute paths.
"""

BUILDER_SYSTEM = """You are Anvil, an autonomous coding engineer. You solve the task by emitting
ONE JSON action per turn, and nothing else. Available actions:

{"action": "write", "path": "relative/path.py", "content": "<full file content>"}
{"action": "read", "path": "relative/path.py"}
{"action": "exec", "command": "pytest -q"}
{"action": "search", "query": "how to do X with library Y"}
{"action": "done", "summary": "<what was built>", "test_report": "<what was run and the outcome>"}

Rules:
- Think step by step, but output ONLY the JSON action.
- Write complete, working files. No placeholders, no TODOs in shipped code.
- Prefer the standard library; you may `pip install -q` small, well-known packages if needed.
- ALWAYS run the test command before calling done. If tests fail, read the output, fix the code, and rerun.
- Keep commands short and non-interactive. Never use sudo.
- Use the search action when you need current docs or API details.
- Call done exactly once, after tests pass (or you have exhausted reasonable fixes — then say so honestly).
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of model output; raise ValueError if none."""
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    return json.loads(match.group(0))


class CodingAgent:
    def __init__(
        self,
        llm: LLM,
        sandbox_factory: Callable[[], Sandbox],
        settings: Settings,
        websearch: WebSearch | None = None,
    ) -> None:
        self.llm = llm
        self.sandbox_factory = sandbox_factory
        self.settings = settings
        self.websearch = websearch

    # ------------------------------------------------------------------ run
    def run(
        self,
        task: str,
        seed_files: dict[str, str] | None = None,
        on_event: Callable[[AgentEvent], None] | None = None,
    ) -> RunResult:
        run_id = uuid.uuid4().hex[:12]
        events: list[AgentEvent] = []
        seed_files = seed_files or {}

        def emit(type_: str, message: str, **data) -> None:
            ev = AgentEvent(type=type_, message=message, data=data)
            events.append(ev)
            if on_event:
                on_event(ev)

        sandbox = self.sandbox_factory()
        try:
            # Seed initial files.
            initial_files = dict(seed_files)
            for path, content in seed_files.items():
                sandbox.write_file(path, content)

            # 1. Plan with the heavy reasoning model.
            emit("plan", "Planning with Nemotron 3 Ultra…")
            plan_raw = self.llm.chat(
                self.settings.model_plan,
                PLANNER_SYSTEM,
                [{"role": "user", "content": task}],
                temperature=0.2,
                max_tokens=1024,
            )
            try:
                plan = _extract_json(plan_raw)
            except ValueError:
                plan = {"plan": [plan_raw], "files": [], "test_command": "pytest -q"}
            test_command = plan.get("test_command") or "pytest -q"
            emit("plan", "Plan ready", plan=plan.get("plan", []), test_command=test_command)

            # 2. Builder ReAct loop.
            history: list[dict] = [
                {
                    "role": "user",
                    "content": (
                        f"TASK:\n{task}\n\nPLAN:\n"
                        + "\n".join(f"- {s}" for s in plan.get("plan", []))
                        + f"\n\nTEST COMMAND: {test_command}\n"
                        + ("Seed files already exist in the sandbox.\n" if seed_files else "")
                        + "Begin. Output one JSON action."
                    ),
                }
            ]
            last_exec: ExecResult | None = None
            summary, test_report = "", ""
            iterations = 0

            for i in range(self.settings.max_iterations):
                iterations = i + 1
                raw = self.llm.chat(
                    self.settings.model_code, BUILDER_SYSTEM, history, temperature=0.2
                )
                try:
                    action = _extract_json(raw)
                except ValueError:
                    history.append({"role": "assistant", "content": raw})
                    history.append(
                        {
                            "role": "user",
                            "content": "That was not a valid JSON action. Reply with exactly one JSON action object.",
                        }
                    )
                    emit("error", "Builder emitted invalid JSON; asked to retry")
                    continue

                kind = action.get("action")
                history.append({"role": "assistant", "content": json.dumps(action)})

                if kind == "write":
                    path = action["path"]
                    sandbox.write_file(path, action.get("content", ""))
                    emit("write", f"Wrote {path}")
                    history.append({"role": "user", "content": f"Wrote {path} ({len(action.get('content',''))} chars)."})
                elif kind == "read":
                    path = action["path"]
                    try:
                        content = sandbox.read_file(path)
                        emit("read", f"Read {path}")
                        history.append({"role": "user", "content": f"--- {path} ---\n{content[:6000]}"})
                    except Exception as e:  # noqa: BLE001
                        history.append({"role": "user", "content": f"read failed: {e}"})
                elif kind == "exec":
                    cmd = action["command"]
                    emit("exec", f"Running: {cmd}")
                    last_exec = sandbox.exec(cmd, timeout_s=self.settings.sandbox_timeout_s)
                    emit(
                        "exec",
                        f"Exit {last_exec.exit_code}: {cmd}",
                        exit_code=last_exec.exit_code,
                        timed_out=last_exec.timed_out,
                    )
                    if not last_exec.ok:
                        emit("fix", "Command failed — feeding output back to builder")
                    history.append({"role": "user", "content": "RESULT:\n" + last_exec.short()})
                elif kind == "search":
                    query = action.get("query", "")
                    if self.websearch is None:
                        history.append({"role": "user", "content": "Web search is not configured (no TAVILY_API_KEY). Continue without it."})
                        emit("search", "Search unavailable — no Tavily key")
                    else:
                        emit("search", f"Searching: {query}")
                        results = self.websearch.search(query)
                        digest = "\n".join(
                            f"- {r.get('title')}\n  {r.get('url')}\n  {r.get('snippet','')[:400]}"
                            for r in results
                        )
                        history.append({"role": "user", "content": f"SEARCH RESULTS:\n{digest or '(none)'}"})
                elif kind == "done":
                    summary = action.get("summary", "")
                    test_report = action.get("test_report", "")
                    emit("done", "Builder finished")
                    break
                else:
                    history.append(
                        {"role": "user", "content": f"Unknown action {kind!r}. Use write/read/exec/search/done."}
                    )
            else:
                emit("error", "Hit iteration limit without done")

            # 3. Finalize: diff + verdict.
            final_files: dict[str, str] = {}
            for path in self._known_paths(sandbox, seed_files, history):
                try:
                    final_files[path] = sandbox.read_file(path)
                except Exception:  # noqa: BLE001
                    continue

            diff = self._diff(initial_files, final_files)
            success = bool(summary) and last_exec is not None and last_exec.ok
            if not summary:
                summary = f"Completed {iterations} builder iterations without a final summary."
            if not test_report:
                test_report = last_exec.short() if last_exec else "No test command was run."

            emit("done", f"Run {'succeeded' if success else 'finished with issues'}")
            return RunResult(
                run_id=run_id,
                task=task,
                success=success,
                summary=summary,
                test_report=test_report,
                diff=diff,
                iterations=iterations,
                files=final_files,
                events=events,
            )
        finally:
            sandbox.close()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _known_paths(sandbox: Sandbox, seed_files: dict, history: list[dict]) -> list[str]:
        paths = set(seed_files)
        for msg in history:
            try:
                action = json.loads(msg.get("content", ""))
            except (ValueError, AttributeError):
                continue
            if isinstance(action, dict) and action.get("action") == "write" and action.get("path"):
                paths.add(action["path"])
        # Also ask the sandbox for a listing when supported.
        if hasattr(sandbox, "list_files"):
            try:
                paths.update(sandbox.list_files())  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
        return sorted(paths)

    @staticmethod
    def _diff(before: dict[str, str], after: dict[str, str]) -> str:
        chunks: list[str] = []
        for path in sorted(set(before) | set(after)):
            old = before.get(path, "").splitlines(keepends=True)
            new = after.get(path, "").splitlines(keepends=True)
            if old == new:
                continue
            chunks.append(f"--- a/{path}\n+++ b/{path}\n")
            chunks.extend(difflib.unified_diff(old, new, lineterm=""))
            chunks.append("\n")
        return "".join(chunks) or "(no file changes)"
