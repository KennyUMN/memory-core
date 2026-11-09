import asyncio
import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import pytest
import httpx
from httpx import ASGITransport

_test_db_path = Path("/tmp/test_server_init.db")
import config
config.DB_PATH = _test_db_path
import database
database.DB_PATH = _test_db_path
database.MemoryDatabase.__init__.__defaults__ = (_test_db_path,)

import server
from server import (
    app,
    memory_remember,
    memory_recall,
    backfill_embeddings_once,
    backfill_loop,
    BACKFILL_INTERVAL,
)
from embedder import EmbeddingError, CircuitOpenError
from usage import current_agent, current_source

pytestmark = pytest.mark.asyncio

VALID_TOKEN = "replace-with-a-long-random-token-for-claude-code"
AUTH_HEADER = {"Authorization": f"Bearer {VALID_TOKEN}"}


class FakeDatabase:
    def __init__(self):
        self.memories: List[Dict[str, Any]] = []
        self.remember_calls: List[Dict[str, Any]] = []
        self.recall_calls: List[Dict[str, Any]] = []
        self.search_docs_calls: List[Dict[str, Any]] = []
        self.set_embedding_calls: List[Tuple[int, List[float]]] = []
        self.missing: List[Tuple[int, str]] = []
        self.profiles: Dict[str, str] = {}
        self.forget_calls: List[Tuple[int, str]] = []

    def newest_in_scope(self, scope: Optional[str] = None) -> Optional[str]:
        return None

    def remember(
        self,
        facts: List[str],
        embeddings: List[Optional[List[float]]],
        kind: str = "fact",
        scope: str = "global",
        supersedes: Optional[int] = None,
        tags: Optional[List[str]] = None,
        source: Optional[Dict[str, Any]] = None,
        created_by: str = "system",
        duplicate_threshold: float = 0.90,
        dedup: bool = True
    ) -> Dict[str, Any]:
        self.remember_calls.append({
            "facts": facts,
            "embeddings": embeddings,
            "kind": kind,
            "scope": scope,
            "supersedes": supersedes,
            "tags": tags,
            "created_by": created_by,
            "current_agent": current_agent.get(),
            "current_source": current_source.get(),
        })
        saved_ids = []
        pending_ids = []
        for fact, emb in zip(facts, embeddings):
            new_id = len(self.memories) + 1
            saved_ids.append(new_id)
            if emb is None:
                pending_ids.append(new_id)
            self.memories.append({"id": new_id, "content": fact, "embedding": emb})

        return {
            "saved_ids": saved_ids,
            "near_duplicates": [],
            "skipped": [],
            "count": len(saved_ids),
            "pending_embedding_ids": pending_ids,
        }

    def recall(
        self,
        query: str,
        query_embedding: Optional[List[float]],
        scope: str = "global",
        kinds: Optional[List[str]] = None,
        k: int = 8,
        include_superseded: bool = False
    ) -> List[Dict[str, Any]]:
        self.recall_calls.append({
            "query": query,
            "query_embedding": query_embedding,
            "scope": scope,
            "kinds": kinds,
            "k": k,
            "current_agent": current_agent.get(),
            "current_source": current_source.get(),
        })
        return [{"id": 1, "content": "recalled fact", "rrf_score": 0.8}]

    def search_docs(
        self,
        query_embedding: Optional[List[float]],
        project: Optional[str] = None,
        k: int = 5,
        query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        self.search_docs_calls.append({
            "query": query,
            "query_embedding": query_embedding,
            "project": project,
            "k": k,
        })
        return [{"id": 10, "content": "chunk text"}]

    def missing_embeddings(self, limit: int = 32, exclude_ids=()) -> List[Tuple[int, str]]:
        return [m for m in self.missing if m[0] not in exclude_ids][:limit]

    def set_embedding(self, memory_id: int, embedding: List[float]) -> None:
        self.set_embedding_calls.append((memory_id, embedding))
        for mem in self.memories:
            if mem["id"] == memory_id:
                mem["embedding"] = embedding

    def get_profile(self) -> Dict[str, str]:
        return self.profiles

    def set_profile(self, key: str, value: str) -> None:
        self.profiles[key] = value

    def forget(self, memory_id: int, reason: str = "") -> bool:
        self.forget_calls.append((memory_id, reason))
        return True


@pytest.fixture
def fake_db(monkeypatch):
    db = FakeDatabase()
    monkeypatch.setattr(server, "db", db)
    return db


async def test_remember_api_when_embedder_fails_returns_200_and_embedding_pending(fake_db, monkeypatch):
    async def mock_embed_fail(texts):
        raise httpx.ConnectError("Connection to embedding service failed")

    monkeypatch.setattr(server.embedder, "embed_passages", mock_embed_fail)

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/remember",
            json={"facts": ["Fact 1", "Fact 2"], "kind": "fact", "scope": "project:test"},
            headers=AUTH_HEADER,
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["embedding_pending"] is True
    assert data["saved_ids"] == [1, 2]
    assert data["pending_embedding_ids"] == [1, 2]

    # Verify db was called with embeddings=[None, None]
    assert len(fake_db.remember_calls) == 1
    assert fake_db.remember_calls[0]["embeddings"] == [None, None]
    assert fake_db.remember_calls[0]["facts"] == ["Fact 1", "Fact 2"]


async def test_remember_api_when_embedder_succeeds(fake_db, monkeypatch):
    dummy_vecs = [[0.1] * 1024, [0.2] * 1024]

    async def mock_embed_success(texts):
        return dummy_vecs

    monkeypatch.setattr(server.embedder, "embed_passages", mock_embed_success)

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/remember",
            json={"facts": ["Fact A", "Fact B"]},
            headers=AUTH_HEADER,
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["saved_ids"] == [1, 2]
    assert data.get("embedding_pending") is None or data.get("embedding_pending") is False
    assert fake_db.remember_calls[0]["embeddings"] == dummy_vecs


async def test_mcp_memory_remember_when_embedder_fails(fake_db, monkeypatch):
    async def mock_embed_fail(texts):
        raise CircuitOpenError("Circuit open")

    monkeypatch.setattr(server.embedder, "embed_passages", mock_embed_fail)

    res = await memory_remember(facts=["Stand alone fact"], scope="global")
    assert res["embedding_pending"] is True
    assert res["saved_ids"] == [1]
    assert res["pending_embedding_ids"] == [1]
    assert fake_db.remember_calls[0]["embeddings"] == [None]


async def test_backfill_embeddings_once_fills_missing(fake_db, monkeypatch):
    fake_db.missing = [(42, "unembedded text 1"), (43, "unembedded text 2")]
    dummy_vecs = [[0.5] * 1024, [0.6] * 1024]

    async def mock_embed(texts):
        assert texts == ["unembedded text 1", "unembedded text 2"]
        return dummy_vecs

    monkeypatch.setattr(server.embedder, "embed_passages", mock_embed)
    monkeypatch.setattr(server.embedder, "_circuit_open_until", 0.0)

    count = await backfill_embeddings_once()
    assert count == 2
    assert len(fake_db.set_embedding_calls) == 2
    assert fake_db.set_embedding_calls[0] == (42, dummy_vecs[0])
    assert fake_db.set_embedding_calls[1] == (43, dummy_vecs[1])


async def test_backfill_skips_quietly_when_circuit_open(fake_db, monkeypatch):
    fake_db.missing = [(99, "should not be embedded")]

    embed_called = False

    async def mock_embed(texts):
        nonlocal embed_called
        embed_called = True
        return [[0.1] * 1024]

    monkeypatch.setattr(server.embedder, "embed_passages", mock_embed)
    monkeypatch.setattr(server.embedder, "_circuit_open_until", time.monotonic() + 100.0)

    count = await backfill_embeddings_once()
    assert count == 0
    assert not embed_called
    assert len(fake_db.set_embedding_calls) == 0


async def test_backfill_error_handling_never_crashes(fake_db, monkeypatch):
    fake_db.missing = [(1, "text")]

    async def mock_embed_error(texts):
        raise httpx.ReadTimeout("Timeout reading embedding response")

    monkeypatch.setattr(server.embedder, "embed_passages", mock_embed_error)
    monkeypatch.setattr(server.embedder, "_circuit_open_until", 0.0)

    # Should catch error, log warning, and return 0 (not crash)
    count = await backfill_embeddings_once()
    assert count == 0


async def test_contextvar_propagation_through_to_thread(fake_db, monkeypatch):
    # Direct test: contextvars are propagated across asyncio.to_thread
    current_agent.set("special-agent")
    current_source.set("rest")

    def inspect_context():
        return current_agent.get(), current_source.get()

    agent_val, source_val = await asyncio.to_thread(inspect_context)
    assert agent_val == "special-agent"
    assert source_val == "rest"

    # End-to-end test via remember_api
    dummy_vecs = [[0.1] * 1024]
    monkeypatch.setattr(server.embedder, "embed_passages", lambda texts: asyncio.sleep(0, dummy_vecs))

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/remember",
            json={"facts": ["Context fact"]},
            headers=AUTH_HEADER,
        )
    assert resp.status_code == 200
    assert fake_db.remember_calls[-1]["current_agent"] == "claude-code"
    assert fake_db.remember_calls[-1]["current_source"] == "rest"


async def test_recall_api_falls_back_to_bm25_when_embedder_fails(fake_db, monkeypatch):
    async def mock_embed_query_fail(query, **kwargs):
        raise EmbeddingError("embedder error")

    monkeypatch.setattr(server.embedder, "embed_query", mock_embed_query_fail)

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/recall",
            json={"query": "test query"},
            headers=AUTH_HEADER,
        )

    assert resp.status_code == 200
    assert len(fake_db.recall_calls) == 1
    # query_embedding is None when embedder fails, allowing BM25 fallback
    assert fake_db.recall_calls[0]["query_embedding"] is None


async def test_lifespan_starts_and_cancels_cleanly(monkeypatch):
    aclose_called = False

    async def mock_aclose():
        nonlocal aclose_called
        aclose_called = True

    monkeypatch.setattr(server.embedder, "aclose", mock_aclose)

    async with server.lifespan(app):
        # Lifespan active
        pass

    assert aclose_called


async def test_memory_remember_docstring():
    doc = memory_remember.__doc__
    assert doc is not None
    assert "duplicates at similarity >= 0.90 in the same scope are skipped (not inserted) and reported" in doc.lower()


# --- audit fixes ---

def http_error(status: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://embedder")
    return httpx.HTTPStatusError(f"{status}", request=req, response=httpx.Response(status, request=req))


async def test_backfill_sets_aside_rows_the_embedder_rejects(fake_db, monkeypatch):
    fake_db.missing = [(1, "good one"), (2, "too long"), (3, "good two")]
    monkeypatch.setattr(server, "backfill_rejected", set())
    monkeypatch.setattr(server.embedder, "_circuit_open_until", 0.0)

    async def embed(texts):
        if "too long" in texts:
            raise http_error(413)
        return [[0.1] * 1024 for _ in texts]

    monkeypatch.setattr(server.embedder, "embed_passages", embed)
    assert await backfill_embeddings_once() == 2
    assert [mid for mid, _ in fake_db.set_embedding_calls] == [1, 3]
    assert server.backfill_rejected == {2}

    # The rejected row no longer blocks the queue.
    fake_db.missing = [(2, "too long"), (4, "good three")]
    assert await backfill_embeddings_once() == 1
    assert fake_db.set_embedding_calls[-1][0] == 4


async def test_mcp_remember_rejects_oversized_facts(fake_db):
    from mcp.server.mcpserver.exceptions import ToolError
    with pytest.raises(ToolError):
        await memory_remember(facts=["x" * (server.MAX_FACT_CHARS + 1)])
    assert fake_db.remember_calls == []


async def test_mcp_remember_attributes_to_token_identity(fake_db, monkeypatch):
    async def embed(texts):
        return [[0.1] * 1024 for _ in texts]
    monkeypatch.setattr(server.embedder, "embed_passages", embed)
    token = current_agent.set("claude-code")
    try:
        await memory_remember(facts=["a"])
        await memory_remember(facts=["b"], agent_id="claude-macbook")
        await memory_remember(facts=["c"], agent_id="claude-code")
    finally:
        current_agent.reset(token)
    assert [c["created_by"] for c in fake_db.remember_calls] == [
        "claude-code", "claude-code/claude-macbook", "claude-code"]


async def test_mcp_search_docs_clamps_k_and_skips_empty_query(fake_db, monkeypatch):
    async def no_embedding(query):
        return None
    monkeypatch.setattr(server, "embed_query_or_none", no_embedding)
    assert (await server.memory_search_docs(query="  "))["count"] == 0
    await server.memory_search_docs(query="runbook", k=-1)
    await server.memory_search_docs(query="runbook", k=10_000)
    assert [c["k"] for c in fake_db.search_docs_calls] == [1, server.MAX_K]


async def test_context_pack_query_excludes_bulk_kinds(tmp_path, monkeypatch):
    real = database.MemoryDatabase(tmp_path / "pack.db")
    real.remember(["deploy the homeserver with docker compose"], [None], kind="fact")
    for i in range(5):
        real.remember([f"skill: deploy helper {i} docker homeserver"], [None], kind="skill", scope="skills-archive")
    monkeypatch.setattr(server, "db", real)

    async def no_embedding(query):
        return None
    monkeypatch.setattr(server, "embed_query_or_none", no_embedding)
    pack = await server.build_context_pack("all", "deploy docker homeserver", 8)
    assert [m["kind"] for m in pack["memories"]] == ["fact"]
