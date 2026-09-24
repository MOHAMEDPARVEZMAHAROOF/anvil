#!/usr/bin/env python3
"""Live end-to-end Anvil test on the NVIDIA backend (dev only).

Uses the `nvidia` skill's stored connector credential via its chat.py CLI for
inference, and a local subprocess sandbox for code execution (Nebius
Sandboxes need a Nebius key, still blocked on signup maintenance).

Usage:
    .venv/bin/python scripts/live_nvidia.py "your task here"
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

# Must be set before anvil.config is imported: model IDs are chosen per backend
# at import time.
os.environ["ANVIL_LLM_BACKEND"] = "nvidia"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anvil.agent import CodingAgent
from anvil.config import Settings
from anvil.models import AgentEvent, ExecResult

CHAT_CLI = os.path.expanduser("~/workspace/skills/nvidia/bin/chat.py")


class SkillLLM:
    """LLM adapter that shells out to the nvidia skill's chat CLI."""

    def __init__(self, model: str) -> None:
        self.model = model

    def chat(self, model, system, messages, temperature=0.2, max_tokens=4096) -> str:
        # The planner passes its own model; the loop passes the routed model.
        use_model = model or self.model
        cmd = [
            sys.executable, CHAT_CLI,
            "--model", use_model,
            "--system", system,
            "--messages", json.dumps(messages),
            "--temperature", str(temperature),
            "--max-tokens", str(max_tokens),
        ]
        # Free-tier inference can 503/429 transiently; retry with backoff.
        last_err = ""
        for attempt in range(4):
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode == 0:
                return proc.stdout
            last_err = proc.stderr.strip()[:300]
            if any(code in last_err for code in ("503", "429", "500", "502")):
                wait = 2 ** attempt
                print(f"   (transient API error, retrying in {wait}s…)", flush=True)
                time.sleep(wait)
                continue
            break
        raise RuntimeError(f"nvidia chat CLI failed: {last_err}")


class LocalSandbox:
    """Dev-only sandbox: runs the agent's code in a temp dir via subprocess."""

    def __init__(self) -> None:
        self.workdir = tempfile.mkdtemp(prefix="anvil-live-")
        self.files: dict[str, str] = {}

    def _safe(self, path: str) -> str:
        p = os.path.normpath(os.path.join(self.workdir, path.lstrip("/")))
        if not p.startswith(self.workdir):
            raise ValueError(f"path escapes workdir: {path}")
        return p

    def write_file(self, path: str, content: str) -> None:
        full = self._safe(path)
        os.makedirs(os.path.dirname(full) or self.workdir, exist_ok=True)
        with open(full, "w") as f:
            f.write(content)
        self.files[path] = content

    def read_file(self, path: str) -> str:
        full = self._safe(path)
        with open(full) as f:
            return f.read()

    def exec(self, command: str, timeout_s: int = 120) -> ExecResult:
        env = dict(os.environ)
        # Make the repo venv (pytest etc.) available inside the sandbox.
        venv_bin = os.path.expanduser("~/workspace/anvil/.venv/bin")
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
        proc = subprocess.run(
            command, shell=True, cwd=self.workdir,
            capture_output=True, text=True, timeout=timeout_s, env=env,
        )
        return ExecResult(command, proc.returncode, proc.stdout, proc.stderr)

    def list_files(self) -> list[str]:
        return sorted(self.files)

    def close(self) -> None:
        pass


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    task = sys.argv[1]

    settings = Settings()
    print(f"backend=nvidia model_code={settings.model_code} model_fast={settings.model_fast}")

    def on_event(ev: AgentEvent) -> None:
        icon = {
            "plan": "🧭", "write": "✍️", "exec": "▶️", "read": "📖",
            "search": "🔎", "fix": "🔧", "done": "✅", "error": "❌",
        }.get(ev.type, "•")
        print(f"{icon} [{ev.type}] {ev.message}", flush=True)

    agent = CodingAgent(
        llm=SkillLLM(settings.model_code),
        sandbox_factory=LocalSandbox,
        settings=settings,
    )
    result = agent.run(task, on_event=on_event)
    print("\n==== RESULT ====")
    print(f"success={result.success}")
    print(f"summary={result.summary}")
    print(f"test_report={result.test_report}")
    print(f"files={list(result.files)}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
