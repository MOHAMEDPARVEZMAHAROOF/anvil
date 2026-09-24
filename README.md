# Anvil — autonomous coding engineer on Nebius

**Give Anvil a coding task. It plans, writes, runs, and tests the code — then hands you a green diff.**

Anvil is built for the **Nebius x NVIDIA Global AI Hackathon** (Coding & Agentic
Engineering track). It runs on open infrastructure:

- **Planning** with NVIDIA **Nemotron 3 Ultra** via Nebius Token Factory
- **Coding** with Nemotron 3 Super / Nano via Nebius Token Factory
- **Execution** inside **Nebius Token Factory Sandboxes** — agent-written code never
  touches the host
- **Research** with the Tavily API when the agent needs current docs

## How it works

```
 task ──▶ planner (Nemotron 3 Ultra) ──▶ plan + test command
                                                    │
                    ┌───────────────────────────────┘
                    ▼
           builder loop (Nemotron 3 Super)
              │ write / read / exec / search / done   (one JSON action per turn)
              ▼
     Token Factory Sandbox: files written, tests run
              │ on failure: output fed back → fix → rerun
              ▼
        done → unified diff + summary + test report
```

The builder uses a ReAct-style loop: each turn the model emits exactly one JSON
action (`write`, `read`, `exec`, `search`, `done`). `exec` results stream back
into context, so failing tests trigger fix iterations automatically. Model
routing keeps costs down: Ultra only plans, Super builds, Nano handles fast calls.

## Quickstart

```bash
git clone https://github.com/MOHAMEDPARVEZMAHAROOF/anvil.git && cd anvil
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your NEBIUS_API_KEY (Builder Program: https://dev.nebius.com)
```

CLI:

```bash
python scripts/demo.py "Write a Python rate limiter with pytest tests"
```

Web UI (live agent timeline, streamed over SSE):

```bash
uvicorn anvil.server:app --reload
# open http://127.0.0.1:8000
```

Tests (no keys needed — the agent loop is tested with fakes):

```bash
pytest -q
```

## Configuration

| Variable | Purpose |
|---|---|
| `NEBIUS_API_KEY` | Token Factory API key (**required**) |
| `NEBIUS_BASE_URL` | Token Factory inference base URL (OpenAI-compatible) |
| `ANVIL_MODEL_PLAN` | Planner model (default: Nemotron 3 Ultra) |
| `ANVIL_MODEL_CODE` | Builder model (default: Nemotron 3 Super) |
| `ANVIL_MODEL_FAST` | Fast model (default: Nemotron 3 Nano) |
| `TAVILY_API_KEY` | Enables the agent's web-research tool (optional) |
| `ANVIL_MAX_ITERATIONS` | Builder loop cap (default: 8) |

## Hackathon requirement mapping

- Runs on **Nebius Token Factory**: every inference call hits the Token Factory
  API; every line of generated code executes in a Token Factory Sandbox.
- Uses **NVIDIA open-source models**: Nemotron 3 Ultra / Super / Nano.
- **Coding & Agentic Engineering track**: a coding agent that writes, runs, and
  tests code in Token Factory Sandboxes.

## Roadmap

- [ ] GitHub issue → fix PR flow (paste an issue URL, get a tested patch)
- [ ] Deploy the demo UI on Nebius Serverless Endpoints
- [ ] Multi-file repo awareness (read-only repo snapshot as context)
- [ ] Persistent run history

## License

MIT — see [LICENSE](LICENSE).
