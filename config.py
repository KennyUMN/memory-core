import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "memory.db"

# Embedding API
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://localhost:8080/v1/embeddings")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding-0.6b")
EMBEDDING_DIM = 1024

# Server Bindings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8420"))

# Authorized agent bearer tokens -> agent identity
# You can customize these tokens or add more agents
import json
# token -> agent identity; file is mode 600 and must exist or the server will not start
TOKENS = json.loads((BASE_DIR / "tokens.json").read_text())
