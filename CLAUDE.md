# memory-core

Shared long-term memory server for agents (MCP + REST, hybrid BM25 + vector recall). Production runs on the homeserver at `~/stacks/memory-core` (systemd unit `memory-core`).

## Verify

Run `make check` before calling any change done. It compiles every module and runs the offline tests (`tests/` with pytest, plus `test_supersede.py`, `test_skills.py`, `test_delta_pack.py`), none of which need an embedding server. `make smoke` runs `test_memory.py` end to end and needs `tokens.json` plus a live embedding server at `EMBEDDING_API_URL`. When a mistake repeats, add a test that catches it instead of another line here.
