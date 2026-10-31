# memory-core

Shared long-term memory for AI coding agents. One server stores facts, decisions, and runbooks, and every agent (Claude Code, Codex, Antigravity) reads and writes the same memory over MCP or a REST API.

I run it on a home server and use it daily across agents, so a fact one agent learns (a port, a deploy step, a design decision) is there for the next one.

## Features

- **MCP server** at `/mcp` with six tools: `memory_profile`, `memory_recall`, `memory_remember`, `memory_forget`, `memory_timeline`, `memory_search_docs`.
- **REST API** under `/api/v1/*` (remember, recall, forget, browse, stats, usage, events) for scripts and non-MCP clients.
- **Hybrid retrieval**: SQLite FTS5 (BM25) plus dense embeddings (Qwen3-Embedding-0.6B through any OpenAI-compatible endpoint), fused with Reciprocal Rank Fusion. If the embedding service is down, recall falls back to BM25 instead of failing.
- **Write hygiene**: near-duplicate detection on `remember` (cosine ≥ 0.86 within a scope), explicit `supersedes` for replacing old facts, soft-delete with a reason on `forget`, and automatic redaction of API keys and tokens before storage.
- **Per-agent auth and usage log**: each agent gets its own bearer token; every recall, remember, and forget is logged with the agent name, so the dashboard shows who uses what.
- **Dashboard** at `/dashboard` for browsing memories, usage, and history.
- **Backups**: `backup.py` takes a consistent SQLite snapshot and keeps the newest 14.

## Stack

Python, FastAPI, MCP Python SDK (Streamable HTTP), SQLite with FTS5, httpx.

## Run

```bash
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt

cp tokens.example.json tokens.json   # put real random tokens here; never commit it
chmod 600 tokens.json
cp .env.example .env                 # point EMBEDDING_API_URL at your embedding server
set -a; . ./.env; set +a

python server.py                     # serves on :8420
```

Connect an MCP client:

```json
{
  "type": "http",
  "url": "http://<host>:8420/mcp",
  "headers": { "Authorization": "Bearer <token from tokens.json>" }
}
```

## Test

```bash
python test_memory.py
```

The test uses a temporary database and needs a reachable embedding endpoint.
