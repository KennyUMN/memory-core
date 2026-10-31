import base64
import array
import math
import httpx
from typing import List, Union
from config import EMBEDDING_API_URL, EMBEDDING_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM

DEFAULT_TASK = "Given a question, retrieve relevant passages that answer the question"


class EmbeddingError(RuntimeError):
    """The embedding API answered, but not with one usable vector per input."""

def normalize_vector(vec: List[float]) -> List[float]:
    sq_sum = math.fsum(x * x for x in vec)
    norm = math.sqrt(sq_sum)
    if norm > 0:
        return [x / norm for x in vec]
    return vec

class EmbedderClient:
    def __init__(self, endpoint: str = EMBEDDING_API_URL, api_key: str = EMBEDDING_API_KEY, model: str = EMBEDDING_MODEL):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

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
        
        batch_size = 16
        all_embeddings = []

        async with httpx.AsyncClient(timeout=45.0) as client:
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
        
        return all_embeddings

    async def embed_query(self, query: str, task: str = DEFAULT_TASK) -> List[float]:
        prefixed = f"Instruct: {task}\nQuery:{query.strip()}"
        res = await self.embed_passages([prefixed])
        return res[0]
