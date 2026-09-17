# llm-site-agent

Multi-tenant LLM chat backend for business websites. One FastAPI service powers
multiple client sites, each with its own persona prompt, behind a shared API.

**Live in production** at [agent.structured64.site](https://agent.structured64.site/api/health),
serving 4 deployed demo sites (e.g. [shadeworks.srv1188665.hstgr.cloud](https://shadeworks.srv1188665.hstgr.cloud)).

## What it does

- `POST /api/chat` — OpenAI-compatible LLM call (any provider that speaks the
  OpenAI chat-completions format) with per-site persona system prompts
- Regex-based lead extraction (name / email / phone / business) merged across
  the conversation and persisted to JSONL on disk
- Booking-intent detection (keyword heuristics on user + assistant messages)
- Per-session in-memory rate limiting (20 msgs/hour)
- Strict CORS allowlist per site; prompt-injection hardening baked into personas
- Dockerized, TLS-terminated behind Traefik

## Run it

```bash
cp .env.example .env   # set OLLAMA_API_KEY
docker compose -f compose.yml up -d --build
curl localhost:8000/api/health
```

## Layout

- `app.py` — the whole service (FastAPI, ~260 lines, no framework bloat)
- `system_prompt.txt` + `*_prompt.txt` — per-site personas
- `Dockerfile` / `compose.yml` — production setup (uvicorn + Traefik labels)