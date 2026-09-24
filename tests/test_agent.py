"""Agent-loop tests with fakes: no API keys, no network."""
from __future__ import annotations

import json

import httpx
import pytest

from anvil.agent import CodingAgent
from anvil.config import Settings
from anvil.llm import LLMClient, make_llm
from anvil.models import ExecResult
from anvil.sandbox import InMemorySandbox, NebiusSandbox


class ScriptedLLM:
    """Replies with a canned sequence of JSON actions."""

    def __init__(self, actions: list[dict]) -> None:
        self.actions = list(actions)
        self.calls: list[dict] = []

    def chat(self, model, system, messages, temperature=0.2, max_tokens=4096) -> str:
        self.calls.append({"model": model, "n_messages": len(messages)})
        if "planner" in system.lower() or "Nemotron 3 Ultra" in system:
            return json.dumps(
                {"plan": ["write module", "run tests"], "files": ["calc.py"], "test_command": "pytest -q"}
            )
        action = self.actions.pop(0)
        return json.dumps(action)


def make_settings(**over) -> Settings:
    s = Settings()
    s.max_iterations = over.get("max_iterations", 8)
    s.model_plan = "plan-model"
    s.model_code = "code-model"
    return s


def test_happy_path_writes_runs_and_finishes():
    llm = ScriptedLLM(
        [
            {"action": "write", "path": "calc.py", "content": "def add(a, b):\n    return a + b\n"},
            {"action": "exec", "command": "pytest -q"},
            {"action": "done", "summary": "added calc.add", "test_report": "1 passed"},
        ]
    )
    sandbox = InMemorySandbox(
        exec_handler=lambda cmd, timeout: ExecResult(cmd, 0, "1 passed", "")
    )
    agent = CodingAgent(llm, lambda: sandbox, make_settings())
    events = []
    result = agent.run("write an add function", on_event=events.append)

    assert result.success is True
    assert result.files["calc.py"].startswith("def add")
    assert "calc.py" in result.diff
    assert [m for m in llm.calls if m["model"] == "plan-model"], "planner should run first"
    assert any(e.type == "exec" for e in events)


def test_failure_triggers_fix_iteration():
    attempts = {"n": 0}

    def flaky(cmd, timeout):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return ExecResult(cmd, 1, "", "assert 2 == 3")
        return ExecResult(cmd, 0, "1 passed", "")

    llm = ScriptedLLM(
        [
            {"action": "write", "path": "calc.py", "content": "def add(a, b):\n    return a - b\n"},
            {"action": "exec", "command": "pytest -q"},
            {"action": "write", "path": "calc.py", "content": "def add(a, b):\n    return a + b\n"},
            {"action": "exec", "command": "pytest -q"},
            {"action": "done", "summary": "fixed add", "test_report": "1 passed"},
        ]
    )
    sandbox = InMemorySandbox(exec_handler=flaky)
    agent = CodingAgent(llm, lambda: sandbox, make_settings())
    events = []
    result = agent.run("write an add function", on_event=events.append)

    assert result.success is True
    assert attempts["n"] == 2
    assert any(e.type == "fix" for e in events), "a failing exec should emit a fix event"


def test_invalid_json_gets_recovered():
    class MessyLLM(ScriptedLLM):
        def chat(self, model, system, messages, temperature=0.2, max_tokens=4096):
            if "planner" in system.lower() or "Nemotron 3 Ultra" in system:
                return super().chat(model, system, messages, temperature, max_tokens)
            if not hasattr(self, "_messed"):
                self._messed = True
                return "Sure, here is my plan: write the file..."  # not JSON
            return super().chat(model, system, messages, temperature, max_tokens)

    llm = MessyLLM(
        [
            {"action": "write", "path": "a.py", "content": "x = 1\n"},
            {"action": "exec", "command": "pytest -q"},
            {"action": "done", "summary": "ok", "test_report": "pass"},
        ]
    )
    sandbox = InMemorySandbox()
    agent = CodingAgent(llm, lambda: sandbox, make_settings())
    result = agent.run("task")
    assert result.success is True


def _settings_with_key() -> Settings:
    s = make_settings()
    s.nebius_api_key = "test-key"
    return s


def test_sandbox_materializes_files_as_heredocs():
    sb = NebiusSandbox(_settings_with_key())
    sb.write_file("pkg/mod.py", "print('hi')\n")
    sb.write_file("./rel.py", "x = 1\n")
    script = sb._materialize_script()
    assert "mkdir -p /work && cd /work" in script
    assert "mkdir -p 'pkg'" in script
    assert "cat > 'pkg/mod.py' <<'ANVIL_EOF_" in script
    assert "print('hi')" in script
    assert "cat > 'rel.py' <<'ANVIL_EOF_" in script  # ./ stripped
    assert sb.list_files() == ["pkg/mod.py", "rel.py"]


def test_sandbox_read_before_exec_uses_mirror():
    sb = NebiusSandbox(_settings_with_key())
    sb.write_file("a.py", "x = 1\n")
    assert sb.read_file("a.py") == "x = 1\n"
    with pytest.raises(FileNotFoundError):
        sb.read_file("missing.py")


def test_sandbox_403_raises_helpful_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"detail": "forbidden"})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="https://x")
    sb = NebiusSandbox(_settings_with_key(), client=client)
    with pytest.raises(RuntimeError, match="Sandboxes beta access"):
        sb.exec("echo hi", timeout_s=5)


def test_sandbox_exec_polls_to_success():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/v1/instances"):
            assert request.headers["authorization"] == "Bearer test-key"
            return httpx.Response(200, json={"uuid": "op-123"})
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(200, json={"state": "running"})
        return httpx.Response(
            200,
            json={
                "state": "succeeded",
                "exit_code": 0,
                "stdout": "1 passed",
                "stderr": "",
                "result_image_uuid": "img-1",
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="https://x")
    sb = NebiusSandbox(_settings_with_key(), client=client)
    sb.write_file("t.py", "x=1\n")
    out = sb.exec("pytest -q", timeout_s=30)
    assert out.ok and out.stdout == "1 passed" and out.command == "pytest -q"
    assert sb._image_uuid == "img-1"


def test_llm_backend_selects_nvidia_endpoint(monkeypatch):
    monkeypatch.setenv("ANVIL_LLM_BACKEND", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    # Re-import to pick up the backend-dependent defaults.
    import importlib

    import anvil.config as config_mod

    importlib.reload(config_mod)
    try:
        settings = config_mod.Settings()
        assert settings.llm_backend == "nvidia"
        assert settings.has_nvidia_key is True
        llm = make_llm(settings)
        assert isinstance(llm, LLMClient)
        assert llm.backend == "nvidia"
        assert llm.base_url == "https://integrate.api.nvidia.com/v1"
        assert llm.api_key == "nvapi-test"
        assert "nemotron" in settings.model_fast.lower()
    finally:
        monkeypatch.undo()
        importlib.reload(config_mod)


def test_llm_backend_defaults_to_nebius():
    settings = Settings()
    assert settings.llm_backend != "nvidia"
    llm = make_llm(settings)
    assert llm.backend != "nvidia"
    assert "tokenfactory.nebius.com" in llm.base_url


def test_normalize_content_repairs_double_escaped_newlines():
    from anvil.agent import _normalize_content

    escaped = "line1\\nline2\\nline3\\nline4"
    fixed = _normalize_content(escaped)
    assert fixed == "line1\nline2\nline3\nline4"


def test_normalize_content_leaves_real_newlines_alone():
    from anvil.agent import _normalize_content

    real = "import re\n\nx = re.compile('a\\\\nb')\n"
    assert _normalize_content(real) == real
    assert _normalize_content("") == ""
