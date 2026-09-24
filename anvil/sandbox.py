"""Sandbox execution backends.

Production: Nebius Token Factory Sandboxes — disposable microVMs.
Every line of agent-written code executes in an isolated Nebius sandbox,
never on the host.
Docs: https://docs.tokenfactory.nebius.com/api-reference/sandboxes/
"""
from __future__ import annotations

import secrets
import time

import httpx

from .config import Settings
from .models import ExecResult

_SANDBOX_BASE = "https://api.tokenfactory.nebius.com/sandboxes"
_TERMINAL_STATES = {"succeeded", "failed", "cancelled", "error", "done", "finished"}


class NebiusSandbox:
    """File-oriented facade over the operation-based Sandboxes REST API.

    Strategy: keep a client-side mirror of files. Each exec() spawns a fresh
    disposable microVM, materializes the mirror into /work via heredocs, runs
    the command, and polls the operation to completion. read_file() downloads
    from the last result image so post-exec modifications are visible.
    """

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        headers = {"Authorization": f"Bearer {settings.nebius_api_key}"}
        if settings.nebius_project_id:
            headers["Project"] = settings.nebius_project_id
        self._headers = headers
        self._client = client or httpx.Client(
            base_url=_SANDBOX_BASE, headers=headers, timeout=30, trust_env=False
        )
        self._files: dict[str, str] = {}
        self._dirty = False
        self._image_uuid: str | None = None

    # ---------------------------------------------------------------- protocol
    def write_file(self, path: str, content: str) -> None:
        self._files[self._norm(path)] = content
        self._dirty = True

    def read_file(self, path: str) -> str:
        path = self._norm(path)
        if self._dirty or not self._image_uuid:
            # Mirror is authoritative: never materialized, or nothing ran yet.
            if path not in self._files:
                raise FileNotFoundError(path)
            return self._files[path]
        # Post-exec: the sandbox may have modified files; download the truth.
        try:
            return self._download(f"/work/{path}")
        except Exception:  # noqa: BLE001
            if path not in self._files:
                raise FileNotFoundError(path) from None
            return self._files[path]

    def exec(self, command: str, timeout_s: int = 120) -> ExecResult:
        script = self._materialize_script() + "\n" + command + "\n"
        op_id = self._spawn(script, timeout_s)
        outcome = self._poll(op_id, command, timeout_s)
        self._dirty = False
        return outcome

    def list_files(self) -> list[str]:
        return sorted(self._files)

    def close(self) -> None:
        self._client.close()

    # ---------------------------------------------------------------- internals
    @staticmethod
    def _norm(path: str) -> str:
        return path.strip().lstrip("./")

    def _materialize_script(self) -> str:
        parts = ["mkdir -p /work && cd /work"]
        for path, content in self._files.items():
            marker = f"ANVIL_EOF_{secrets.token_hex(6)}"
            while marker in content:
                marker = f"ANVIL_EOF_{secrets.token_hex(6)}"
            directory = path.rsplit("/", 1)[0] if "/" in path else ""
            if directory:
                parts.append(f"mkdir -p {self._shq(directory)}")
            parts.append(f"cat > {self._shq(path)} <<'{marker}'\n{content}\n{marker}")
        return "\n".join(parts)

    @staticmethod
    def _shq(s: str) -> str:
        return "'" + s.replace("'", "'\"'\"'") + "'"

    def _spawn(self, script: str, timeout_s: int) -> str:
        resp = self._client.post(
            "/v1/instances",
            headers=self._headers,
            json={
                "image": "tag:ubuntu:latest",
                "command": script,
                "shell": True,
                "cwd": "/work",
                "env": {"PIP_NO_CACHE_DIR": "1", "PYTHONUNBUFFERED": "1"},
                "networking": {"enabled": True},
                "timeout": max(timeout_s + 60, 300),
                "disposable": True,
            },
        )
        if resp.status_code in (401, 403):
            raise RuntimeError(
                f"Sandbox access denied (HTTP {resp.status_code}). This key needs "
                "Sandboxes beta access: request it at "
                "https://tokenfactory.nebius.com/sandboxes/about and grant the "
                "'spawn' permission in the Token Factory console."
            )
        resp.raise_for_status()
        data = resp.json()
        op_id = data.get("uuid") or data.get("id") or data.get("operation_id")
        if not op_id:
            loc = resp.headers.get("Location", "")
            op_id = loc.rstrip("/").rsplit("/", 1)[-1] or None
        if not op_id:
            raise RuntimeError(f"Could not find operation id in spawn response: {data}")
        return op_id

    def _poll(self, op_id: str, command: str, timeout_s: int) -> ExecResult:
        deadline = time.monotonic() + timeout_s + 30
        last: dict = {}
        while time.monotonic() < deadline:
            resp = self._client.get(f"/v1/operations/{op_id}", headers=self._headers)
            resp.raise_for_status()
            last = resp.json()
            if str(last.get("state", "")).lower() in _TERMINAL_STATES:
                break
            time.sleep(2)
        timed_out = str(last.get("state", "")).lower() not in _TERMINAL_STATES
        image_uuid = last.get("result_image_uuid") or last.get("image_uuid")
        if image_uuid:
            self._image_uuid = image_uuid
        return ExecResult(
            command=command,
            exit_code=int(last.get("exit_code", 1 if timed_out else 0)),
            stdout=last.get("stdout", ""),
            stderr=last.get("stderr", ""),
            timed_out=timed_out,
        )

    def _download(self, abs_path: str) -> str:
        if not self._image_uuid:
            raise RuntimeError("no result image to download from")
        resp = self._client.get(
            f"/v1/inspect/{self._image_uuid}/download",
            params={"path": abs_path},
            headers=self._headers,
        )
        resp.raise_for_status()
        return resp.text


class InMemorySandbox:
    """Test/dev double: in-memory FS, fake exec. Never used in production."""

    def __init__(self, exec_handler=None) -> None:
        self.files: dict[str, str] = {}
        self._exec = exec_handler or (lambda cmd, timeout: ExecResult(cmd, 0, "", ""))

    def write_file(self, path: str, content: str) -> None:
        self.files[path] = content

    def read_file(self, path: str) -> str:
        return self.files[path]

    def exec(self, command: str, timeout_s: int = 120) -> ExecResult:
        return self._exec(command, timeout_s)

    def list_files(self) -> list[str]:
        return sorted(self.files)

    def close(self) -> None:
        pass
