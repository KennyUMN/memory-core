import os
import time
import pytest
import httpx
from embedder import (
    EmbedderClient,
    EmbeddingError,
    CircuitOpenError,
    EMBEDDING_TIMEOUT,
    CIRCUIT_BREAKER_FAILURES,
    CIRCUIT_BREAKER_RESET_TIMEOUT,
)

pytestmark = pytest.mark.asyncio


def make_embedding_response(texts, dim=1024):
    data = [{"index": i, "embedding": [0.1] * dim} for i, _ in enumerate(texts)]
    return {"data": data}


async def test_timeout_constant_and_config(monkeypatch):
    assert EMBEDDING_TIMEOUT == 8.0
    client = EmbedderClient()
    assert client.timeout == 8.0
    await client.aclose()


async def test_shared_client_reuse():
    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json=make_embedding_response(["dummy"]))

    transport = httpx.MockTransport(mock_handler)
    client = EmbedderClient(transport=transport)

    # First call: client created lazily
    res1 = await client.embed_passages(["first text"])
    assert len(res1) == 1
    shared_client = client._client
    assert shared_client is not None
    assert not shared_client.is_closed

    # Second call: same client instance reused
    res2 = await client.embed_passages(["second text"])
    assert len(res2) == 1
    assert client._client is shared_client
    assert call_count == 2

    # aclose closes the client
    await client.aclose()
    assert client._client is None

    # Next call lazily creates a fresh client
    res3 = await client.embed_passages(["third text"])
    assert len(res3) == 1
    assert client._client is not None
    assert client._client is not shared_client

    await client.aclose()


async def test_circuit_breaker_opens_after_two_failures_and_closes_after_success(monkeypatch):
    should_fail = True
    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if should_fail:
            return httpx.Response(500, json={"error": "embedder down"})
        return httpx.Response(200, json=make_embedding_response(["hello"]))

    transport = httpx.MockTransport(mock_handler)
    client = EmbedderClient(transport=transport, recovery_timeout=60.0)

    # Call 1: fails
    with pytest.raises(httpx.HTTPStatusError):
        await client.embed_passages(["test 1"])
    assert client._consecutive_failures == 1
    assert not client.is_circuit_open

    # Call 2: fails -> opens circuit breaker
    with pytest.raises(httpx.HTTPStatusError):
        await client.embed_passages(["test 2"])
    assert client._consecutive_failures == 2
    assert client.is_circuit_open

    # Call 3: immediate call raises CircuitOpenError without hitting transport
    calls_before = call_count
    with pytest.raises(CircuitOpenError):
        await client.embed_passages(["test 3"])
    assert call_count == calls_before  # No HTTP call made

    # Simulate 61 seconds passing (recovery timeout elapsed)
    current_time = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: current_time + 61.0)
    assert not client.is_circuit_open

    # Embedder recovers: Call 4 succeeds and closes breaker
    should_fail = False
    res = await client.embed_passages(["test recovered"])
    assert len(res) == 1
    assert client._consecutive_failures == 0
    assert not client.is_circuit_open
    assert client._circuit_open_until == 0.0

    # Next call also succeeds normally
    res2 = await client.embed_passages(["test normal"])
    assert len(res2) == 1

    await client.aclose()


async def test_circuit_breaker_reopens_if_recovery_call_fails(monkeypatch):
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "still down"})

    transport = httpx.MockTransport(mock_handler)
    client = EmbedderClient(transport=transport, recovery_timeout=60.0)

    # Fail twice to open circuit
    with pytest.raises(httpx.HTTPStatusError):
        await client.embed_passages(["1"])
    with pytest.raises(httpx.HTTPStatusError):
        await client.embed_passages(["2"])
    assert client.is_circuit_open

    # Advance beyond cooldown
    now = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: now + 65.0)
    assert not client.is_circuit_open

    # Trial call fails
    with pytest.raises(httpx.HTTPStatusError):
        await client.embed_passages(["trial"])

    # Circuit is open again
    assert client.is_circuit_open

    await client.aclose()


async def test_embed_query_fails_fast_when_circuit_open():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "internal error"})

    transport = httpx.MockTransport(mock_handler)
    client = EmbedderClient(transport=transport)

    with pytest.raises(httpx.HTTPStatusError):
        await client.embed_query("q1")
    with pytest.raises(httpx.HTTPStatusError):
        await client.embed_query("q2")

    # Now circuit is open; embed_query raises CircuitOpenError
    assert client.is_circuit_open
    with pytest.raises(CircuitOpenError):
        await client.embed_query("q3")

    await client.aclose()


async def test_empty_texts_returns_empty_list():
    client = EmbedderClient()
    res = await client.embed_passages([])
    assert res == []
    await client.aclose()
