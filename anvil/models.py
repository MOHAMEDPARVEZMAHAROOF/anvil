"""Core data types for runs, events, and results."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class AgentEvent:
    """One step in the agent's visible timeline. Streamed to the UI as JSON."""

    type: str  # plan | think | write | exec | read | search | fix | done | error
    message: str
    data: dict = field(default_factory=dict)
    at: str = field(default_factory=_now)


@dataclass
class ExecResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def short(self, limit: int = 4000) -> str:
        out = (self.stdout or "")[-limit:]
        err = (self.stderr or "")[-limit:]
        text = f"$ {self.command}\nexit={self.exit_code}\n{out}"
        if err:
            text += f"\n[stderr]\n{err}"
        if self.timed_out:
            text += "\n[TIMED OUT]"
        return text


@dataclass
class RunResult:
    run_id: str
    task: str
    success: bool
    summary: str
    test_report: str
    diff: str
    iterations: int
    files: dict = field(default_factory=dict)  # final path -> content
    events: list = field(default_factory=list)  # list[AgentEvent]

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "success": self.success,
            "summary": self.summary,
            "test_report": self.test_report,
            "diff": self.diff,
            "iterations": self.iterations,
            "files": self.files,
            "events": [e.__dict__ for e in self.events],
        }
