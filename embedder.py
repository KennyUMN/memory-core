import os
import time
import base64
import array
import math
import httpx
from typing import List, Union, Optional
from config import EMBEDDING_API_URL, EMBEDDING_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM

DEFAULT_TASK = "Given a question, retrieve relevant passages that answer the question"
EMBEDDING_TIMEOUT = float(os.getenv("EMBEDDING_TIMEOUT", "8.0"))
CIRCUIT_BREAKER_FAILURES = 2
CIRCUIT_BREAKER_RESET_TIMEOUT = 60.0


class EmbeddingError(RuntimeError):
    """The embedding API answered, but not with one usable vector per input."""


class CircuitOpenError(EmbeddingError):
    """Circuit breaker is open. Subsequent calls fail fast until recovery timeout expires."""


def normalize_vector(vec: List[float]) -> List[float]:
    sq_sum = math.fsum(x * x for x in vec)
    norm = math.sqrt(sq_sum)
    if norm > 0:
        return [x / norm for x in vec]
    return vec


class EmbedderClient:
    def __init__(
        self,
        endpoint: str = EMBEDDING_API_URL,
        api_key: str = EMBEDDING_API_KEY,
        model: str = EMBEDDING_MODEL,
        timeout: float = EMBEDDING_TIMEOUT,
        failure_threshold: int = CIRCUIT_BREAKER_FAILURES,
        recovery_timeout: float = CIRCUIT_BREAKER_RESET_TIMEOUT,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self._transport = transport
        self._client: Optional[httpx.AsyncClient] = None
        self._consecutive_failures: int = 0
        self._circuit_open_until: float = 0.0

    @property
    def is_circuit_open(self) -> bool:
        return time.monotonic() < self._circuit_open_until

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            kwargs = {"timeout": self.timeout}
            if self._transport is not None:
                kwargs["transport"] = self._transport
            self._client = httpx.AsyncClient(**kwargs)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    def _decode_embedding(self, emb_data: Union[list, str]) -> List[float]:
        if isinstance(emb_data, str):
            raw_bytes = base64.b64decode(emb_data)
            arr = array.array("f")
            arr.frombytes(raw_bytes)
            vec = arr.tolist()
        else:
            vec = [float(x) for x in emb_data]
        
        if len(vec) != EMBEDDING_DIM:
            raise ValueError(f"Expected embedding dimension {EMBEDDING_DIM}, got {len(vec)}")

        return normalize_vector(vec)

    async def embed_passages(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if self.is_circuit_open:
            raise CircuitOpenError(
                f"Circuit breaker is open. Retrying after {self._circuit_open_until - time.monotonic():.1f}s"
            )

        batch_size = 16
        all_embeddings = []
        client = self._get_client()

        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                payload = {
                    "model": self.model,
                    "input": batch,
                    "encoding_format": "base64"
                }
                resp = await client.post(self.endpoint, headers=self.headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                items = sorted(data.get("data", []), key=lambda it: it.get("index", 0))
                if len(items) != len(batch):
                    raise EmbeddingError(f"embedding API returned {len(items)} vectors for {len(batch)} inputs")
                for item in items:
                    all_embeddings.append(self._decode_embedding(item["embedding"]))

            # On success, reset failures and close breaker
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0
            return all_embeddings
        except Exception as exc:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._circuit_open_until = time.monotonic() + self.recovery_timeout
            if isinstance(exc, (httpx.HTTPError, EmbeddingError)):
                raise
            raise EmbeddingError(f"Embedding failed: {exc}") from exc

    async def embed_query(self, query: str, task: str = DEFAULT_TASK) -> List[float]:
        prefixed = f"Instruct: {task}\nQuery:{query.strip()}"
        res = await self.embed_passages([prefixed])
        return res[0]
