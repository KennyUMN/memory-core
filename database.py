import sqlite3
import json
import re
import array
import math
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from config import DB_PATH

SECRET_PATTERNS = [
    re.compile(r'(?i)(?:api_key|token|bearer|secret|password|access_key)[\s:=]+["\']?([a-zA-Z0-9_\-\.]{16,})["\']?'),
    re.compile(r'ghp_[a-zA-Z0-9]{36}'),
    re.compile(r'xox[baprs]-[a-zA-Z0-9\-]+'),
    re.compile(r'sk-(?:ant-|proj-)?[a-zA-Z0-9_\-]{20,}')
]

def redact_secrets(text: str) -> str:
    cleaned = text
    for pattern in SECRET_PATTERNS:
        cleaned = pattern.sub('[REDACTED_SECRET]', cleaned)
    return cleaned

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    # Since Qwen3 embeddings are normalized, dot product = cosine similarity
    return float(math.fsum(x * y for x, y in zip(v1, v2)))

class MemoryDatabase:
    def __init__(self, db_path = DB_PATH):
        self.db_path = str(db_path)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    source TEXT,
                    tags TEXT,
                    created_by TEXT NOT NULL,
                    valid_from TEXT NOT NULL,
                    valid_to TEXT,
                    verified_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope);
                CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
                CREATE INDEX IF NOT EXISTS idx_memories_valid_to ON memories(valid_to);

                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    content,
                    content=memories,
                    content_rowid=id,
                    tokenize='porter unicode61'
                );

                CREATE TRIGGER IF NOT EXISTS trg_memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
                END;

                CREATE TRIGGER IF NOT EXISTS trg_memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.id, old.content);
                END;

                CREATE TRIGGER IF NOT EXISTS trg_memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.id, old.content);
                    INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
                END;

                CREATE TABLE IF NOT EXISTS profile (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)

    def get_profile(self) -> Dict[str, str]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT key, value FROM profile").fetchall()
            return {r["key"]: r["value"] for r in rows}

    def set_profile(self, key: str, value: str):
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO profile (key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """, (key, value, now))

    def remember(
        self,
        facts: List[str],
        embeddings: List[List[float]],
        kind: str = "fact",
        scope: str = "global",
        supersedes: Optional[int] = None,
        tags: Optional[List[str]] = None,
        source: Optional[Dict[str, Any]] = None,
        created_by: str = "system",
        duplicate_threshold: float = 0.86
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        tags_str = json.dumps(tags or [])
        source_str = json.dumps(source or {})
        saved_ids = []
        near_duplicates = []

        with self._get_conn() as conn:
            if supersedes is not None:
                conn.execute(
                    "UPDATE memories SET valid_to = ? WHERE id = ? AND valid_to IS NULL",
                    (now, supersedes)
                )

            # Pre-load active memory embeddings in same scope for near-duplicate check
            active_rows = conn.execute(
                "SELECT id, content, embedding FROM memories WHERE scope = ? AND valid_to IS NULL",
                (scope,)
            ).fetchall()

            existing_records = []
            for r in active_rows:
                arr = array.array("f")
                arr.frombytes(r["embedding"])
                existing_records.append({
                    "id": r["id"],
                    "content": r["content"],
                    "vec": arr.tolist()
                })

            for content, emb in zip(facts, embeddings):
                clean_content = redact_secrets(content.strip())
                if not clean_content:
                    continue

                # Check near-duplicates (similarity >= duplicate_threshold)
                for rec in existing_records:
                    sim = cosine_similarity(rec["vec"], emb)
                    if sim >= duplicate_threshold:
                        near_duplicates.append({
                            "existing_id": rec["id"],
                            "existing_content": rec["content"],
                            "similarity": round(sim, 4),
                            "new_fact": clean_content
                        })

                # Insert fact
                emb_blob = array.array("f", emb).tobytes()
                cur = conn.execute("""
                    INSERT INTO memories (
                        kind, scope, content, embedding, source, tags, created_by, valid_from, valid_to, verified_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                """, (kind, scope, clean_content, emb_blob, source_str, tags_str, created_by, now, now))
                saved_ids.append(cur.lastrowid)

                # Add to existing_records for consecutive batch items
                existing_records.append({
                    "id": cur.lastrowid,
                    "content": clean_content,
                    "vec": emb
                })

        return {
            "saved_ids": saved_ids,
            "near_duplicates": near_duplicates,
            "count": len(saved_ids)
        }

    def recall(
        self,
        query: str,
        query_embedding: Optional[List[float]],
        scope: Optional[str] = None,
        kinds: Optional[List[str]] = None,
        k: int = 8,
        include_superseded: bool = False
    ) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            where_clauses = []
            params = []

            if not include_superseded:
                where_clauses.append("valid_to IS NULL")
            if kinds:
                placeholders = ",".join("?" * len(kinds))
                where_clauses.append(f"kind IN ({placeholders})")
                params.extend(kinds)
            if scope and scope != "all":
                allowed_scopes = ["global"]
                if scope.startswith("project:"):
                    allowed_scopes.append(scope)
                elif scope.startswith("agent:"):
                    allowed_scopes.append(scope)
                elif scope.startswith("session:"):
                    allowed_scopes.append(scope)
                else:
                    allowed_scopes.append(scope)
                
                placeholders = ",".join("?" * len(allowed_scopes))
                where_clauses.append(f"scope IN ({placeholders})")
                params.extend(allowed_scopes)

            where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
            sql = f"SELECT id, kind, scope, content, embedding, source, tags, created_by, valid_from, valid_to, verified_at FROM memories {where_sql}"
            
            rows = conn.execute(sql, params).fetchall()
            if not rows:
                return []

            # 1. Vector Ranking
            cand_rows = {}
            scored_vector_list = []
            for r in rows:
                mid = r["id"]
                cand_rows[mid] = r
                if query_embedding is None:
                    continue  # embedding unavailable: BM25-only fallback
                arr = array.array("f")
                arr.frombytes(r["embedding"])
                sim = cosine_similarity(arr.tolist(), query_embedding)
                scored_vector_list.append((mid, sim))

            # Sort by vector similarity descending
            scored_vector_list.sort(key=lambda x: x[1], reverse=True)
            vector_ranks = {mid: rank + 1 for rank, (mid, _) in enumerate(scored_vector_list[:50])}
            vector_scores = {mid: sim for mid, sim in scored_vector_list[:50]}

            # 2. BM25 / Keyword Search via FTS5
            clean_q = re.sub(r'[^\w\s]', ' ', query).strip()
            terms = [f'"{t}"' for t in clean_q.split() if len(t) > 1]
            fts_ranks = {}
            if terms:
                fts_query = " OR ".join(terms)
                fts_sql = f"""
                    SELECT rowid, rank FROM memories_fts
                    WHERE memories_fts MATCH ?
                      AND rowid IN (SELECT id FROM memories {where_sql})
                    ORDER BY rank
                    LIMIT 50
                """
                try:
                    fts_results = conn.execute(fts_sql, [fts_query] + params).fetchall()
                    for rank, fts_r in enumerate(fts_results):
                        rid = fts_r["rowid"]
                        if rid in cand_rows:
                            fts_ranks[rid] = rank + 1
                except sqlite3.OperationalError:
                    pass

            # 3. Reciprocal Rank Fusion (RRF)
            all_match_ids = set(vector_ranks.keys()).union(set(fts_ranks.keys()))
            scored_candidates = []

            for mid in all_match_ids:
                v_rank = vector_ranks.get(mid, 1000)
                f_rank = fts_ranks.get(mid, 1000)
                rrf_score = (1.0 / (60 + v_rank)) + (1.0 / (60 + f_rank))
                
                r = cand_rows[mid]
                trust_bonus = 0.005 if r["verified_at"] else 0.0

                scored_candidates.append({
                    "id": r["id"],
                    "kind": r["kind"],
                    "scope": r["scope"],
                    "content": r["content"],
                    "source": json.loads(r["source"] or "{}"),
                    "tags": json.loads(r["tags"] or "[]"),
                    "created_by": r["created_by"],
                    "valid_from": r["valid_from"],
                    "valid_to": r["valid_to"],
                    "verified_at": r["verified_at"],
                    "similarity": round(vector_scores.get(mid, 0.0), 4),
                    "rrf_score": round(rrf_score + trust_bonus, 6)
                })

            scored_candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
            return scored_candidates[:k]

    def forget(self, memory_id: int, reason: str = "") -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            cur = conn.execute(
                "UPDATE memories SET valid_to = ? WHERE id = ? AND valid_to IS NULL",
                (now, memory_id)
            )
            return cur.rowcount > 0

    def get_timeline(self, scope: str = "global", since: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            sql = "SELECT id, kind, scope, content, source, created_by, valid_from FROM memories WHERE kind = 'episode'"
            params = []
            if scope != "all":
                sql += " AND (scope = ? OR scope = 'global')"
                params.append(scope)
            if since:
                sql += " AND valid_from >= ?"
                params.append(since)
            sql += " ORDER BY valid_from ASC"

            rows = conn.execute(sql, params).fetchall()
            return [{
                "id": r["id"],
                "scope": r["scope"],
                "content": r["content"],
                "source": json.loads(r["source"] or "{}"),
                "created_by": r["created_by"],
                "valid_from": r["valid_from"]
            } for r in rows]

    def search_docs(self, query_embedding: Optional[List[float]], project: Optional[str] = None, k: int = 5, query: str = "") -> List[Dict[str, Any]]:
        return self.recall(
            query=query,
            query_embedding=query_embedding,
            scope=f"project:{project}" if project else None,
            kinds=["doc_chunk"],
            k=k
        )

    def browse(self, scope=None, kind=None, state="active", q=None, limit=25, offset=0) -> Dict[str, Any]:
        where, params = [], []
        if scope:
            where.append("scope = ?"); params.append(scope)
        if kind:
            where.append("kind = ?"); params.append(kind)
        if state == "active":
            where.append("valid_to IS NULL")
        elif state == "retired":
            where.append("valid_to IS NOT NULL")
        if q:
            where.append("content LIKE ? ESCAPE '\\'")
            params.append("%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM memories" + clause, params).fetchone()[0]
            rows = conn.execute(
                "SELECT id, kind, scope, content, tags, created_by, valid_from, valid_to FROM memories"
                + clause + " ORDER BY id DESC LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()
        items = [{**{k: r[k] for k in r.keys() if k != "tags"}, "tags": json.loads(r["tags"] or "[]")} for r in rows]
        return {"total": total, "items": items}

    def stats(self, days: int = 30) -> Dict[str, Any]:
        with self._get_conn() as conn:
            one = lambda sql, *a: conn.execute(sql, a).fetchone()[0]
            group = lambda col: {r[0]: r[1] for r in conn.execute(
                f"SELECT {col}, COUNT(*) FROM memories WHERE valid_to IS NULL GROUP BY {col}")}
            counts = {r[0]: r[1] for r in conn.execute(
                "SELECT substr(valid_from, 1, 10), COUNT(*) FROM memories GROUP BY 1")}
            from datetime import timedelta
            today = datetime.now(timezone.utc).date()
            per_day = []
            for i in range(days - 1, -1, -1):
                d = (today - timedelta(days=i)).isoformat()
                per_day.append({"day": d, "n": counts.get(d, 0)})
            return {
                "active": one("SELECT COUNT(*) FROM memories WHERE valid_to IS NULL"),
                "retired": one("SELECT COUNT(*) FROM memories WHERE valid_to IS NOT NULL"),
                "by_kind": group("kind"), "by_scope": group("scope"), "by_creator": group("created_by"),
                "per_day": per_day,
            }
