import json
from typing import Any, Dict, List

import httpx
import pytest

import server
from reranker import RerankerClient, RerankError, RERANK_CANDIDATES, RERANK_DOC_CHARS

pytestmark = pytest.mark.asyncio


def rerank_transport(scores: List[float], seen: List[Dict[str, Any]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        results = [{"index": i, "relevance_score": s} for i, s in enumerate(scores)]
        return httpx.Response(200, json={"results": sorted(results, key=lambda r: -r["relevance_score"])})
    return httpx.MockTransport(handler)


def failing_transport() -> httpx.MockTransport:
    return httpx.MockTransport(lambda request: httpx.Response(503, json={"error": "down"}))


async def test_rerank_orders_by_score_and_truncates_documents():
    seen: List[Dict[str, Any]] = []
    client = RerankerClient(api_key="k", transport=rerank_transport([0.1, 0.9, 0.5], seen))
    ranked = await client.rerank("q", ["a", "b" * (RERANK_DOC_CHARS + 10), "c"])
    assert [i for i, _ in ranked] == [1, 2, 0]
    assert len(seen[0]["documents"][1]) == RERANK_DOC_CHARS


async def test_rerank_disabled_without_key_and_breaker_opens_after_two_failures():
    assert not RerankerClient(api_key="").enabled
    client = RerankerClient(api_key="k", transport=failing_transport())
    for _ in range(2):
        with pytest.raises(RerankError):
            await client.rerank("q", ["a"])
    assert not client.enabled


async def test_rerank_rejects_incomplete_results():
    client = RerankerClient(api_key="k", transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 1.0}]})))
    with pytest.raises(RerankError):
        await client.rerank("q", ["a", "b"])


class StubDb:
    def __init__(self, n: int):
        self.rows = [{"id": i, "content": f"memory {i}"} for i in range(n)]
        self.calls: List[Dict[str, Any]] = []

    def recall(self, **kwargs: Any) -> List[Dict[str, Any]]:
        self.calls.append(kwargs)
        return self.rows[:kwargs["k"]]


async def test_recall_ranked_widens_pool_and_reorders(monkeypatch):
    stub = StubDb(RERANK_CANDIDATES)
    scores = [i / 100 for i in range(RERANK_CANDIDATES)]  # last candidate is best
    monkeypatch.setattr(server, "db", stub)
    monkeypatch.setattr(server, "reranker", RerankerClient(api_key="k", transport=rerank_transport(scores, [])))
    out = await server.recall_ranked("q", None, 3, scope="all")
    assert stub.calls[0]["k"] == RERANK_CANDIDATES
    assert [r["id"] for r in out] == [RERANK_CANDIDATES - 1, RERANK_CANDIDATES - 2, RERANK_CANDIDATES - 3]
    assert "rerank_score" in out[0]


async def test_recall_ranked_falls_back_to_rrf_order_on_failure(monkeypatch):
    stub = StubDb(RERANK_CANDIDATES)
    monkeypatch.setattr(server, "db", stub)
    monkeypatch.setattr(server, "reranker", RerankerClient(api_key="k", transport=failing_transport()))
    out = await server.recall_ranked("q", None, 3, scope="all")
    assert [r["id"] for r in out] == [0, 1, 2]


async def test_recall_ranked_skips_reranker_when_disabled(monkeypatch):
    stub = StubDb(RERANK_CANDIDATES)
    monkeypatch.setattr(server, "db", stub)
    monkeypatch.setattr(server, "reranker", RerankerClient(api_key=""))
    out = await server.recall_ranked("q", None, 3, scope="all")
    assert stub.calls[0]["k"] == 3 and len(out) == 3
