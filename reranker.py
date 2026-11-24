"""Cross-encoder reranking of recall candidates through an OpenRouter-compatible /rerank API."""
import os
import time
from typing import Any, Dict, List, Optional

import httpx

from config import RERANK_API_KEY, RERANK_API_URL, RERANK_MODEL

RERANK_TIMEOUT = float(os.getenv("RERANK_TIMEOUT", "4.0"))
# How many fused candidates go to the reranker. On the 36-query eval, 20 and 40
# scored the same (MRR 0.986), so the smaller, faster request wins.
RERANK_CANDIDATES = 20
# Long skill bodies are cut before sending; the head carries the name and summary.
RERANK_DOC_CHARS = 1500
CIRCUIT_BREAKER_FAILURES = 2
CIRCUIT_BREAKER_RESET_TIMEOUT = 60.0


class RerankError(RuntimeError):
    """The rerank API failed or answered with something unusable."""


class RerankerClient:
    def __init__(
        self,
        endpoint: str = RERANK_API_URL,
        api_key: str = RERANK_API_KEY,
        model: str = RERANK_MODEL,
        timeout: float = RERANK_TIMEOUT,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._transport = transport
        self._client: Optional[httpx.AsyncClient] = None
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) and time.monotonic() >= self._circuit_open_until

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            kwargs: Dict[str, Any] = {"timeout": self.timeout}
            if self._transport is not None:
                kwargs["transport"] = self._transport
            self._client = httpx.AsyncClient(**kwargs)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def rerank(self, query: str, documents: List[str]) -> List[tuple]:
        """(index, relevance_score) pairs for documents, best first."""
        if not documents:
            return []
        try:
            resp = await self._get_client().post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "query": query,
                      "documents": [d[:RERANK_DOC_CHARS] for d in documents]},
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            ranked = [(int(r["index"]), float(r["relevance_score"])) for r in results]
            if sorted(i for i, _ in ranked) != list(range(len(documents))):
                raise RerankError(f"rerank API returned indices for {len(ranked)} of {len(documents)} documents")
        except (httpx.HTTPError, ValueError, KeyError, TypeError, RerankError) as exc:
            self._consecutive_failures += 1
            if self._consecutive_failures >= CIRCUIT_BREAKER_FAILURES:
                self._circuit_open_until = time.monotonic() + CIRCUIT_BREAKER_RESET_TIMEOUT
            if isinstance(exc, RerankError):
                raise
            raise RerankError(f"rerank failed: {exc!r}") from exc
        self._consecutive_failures = 0
        return sorted(ranked, key=lambda r: r[1], reverse=True)
