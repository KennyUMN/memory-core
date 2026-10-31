"""Usage logging and analytics for memory-core.

LoggedDatabase records every recall/remember/forget as an event. The dashboard
reads those events to show what is used, what it saves, and the history.
"""
import contextvars
import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from database import MemoryDatabase, redact_secrets

log = logging.getLogger("memory-core.usage")

# Set by the auth layer so the database can attribute each call.
current_agent = contextvars.ContextVar("current_agent", default="unknown")
current_source = contextvars.ContextVar("current_source", default="direct")

CHARS_PER_TOKEN = 4
CONFIDENT_SIM = 0.40
RECALL_TYPES = ("recall", "search_docs")

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    type TEXT NOT NULL,
    agent TEXT NOT NULL,
    source TEXT NOT NULL,
    scope TEXT,
    query TEXT,
    k INTEGER,
    result_ids TEXT NOT NULL DEFAULT '[]',
    n_results INTEGER NOT NULL DEFAULT 0,
    top_score REAL,
    served_chars INTEGER NOT NULL DEFAULT 0,
    baseline_chars INTEGER NOT NULL DEFAULT 0,
    latency_ms REAL NOT NULL DEFAULT 0,
    detail TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
"""

RECALLED_IDS = """
    SELECT CAST(j.value AS INTEGER) AS mid, COUNT(*) AS hits, MAX(e.ts) AS last_ts
    FROM events e, json_each(e.result_ids) j
    WHERE e.type IN ('recall', 'search_docs')
    GROUP BY mid
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scrub(v):
    """Redact secrets in every string of a JSON-like value before it is logged."""
    if isinstance(v, str):
        return redact_secrets(v)
    if isinstance(v, dict):
        return {k: _scrub(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_scrub(x) for x in v]
    return v


class LoggedDatabase(MemoryDatabase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with self._get_conn() as conn:
            conn.executescript(SCHEMA)

    # ---- logging -------------------------------------------------------

    def _log(self, type_: str, *, scope=None, query=None, k=None, ids=(), top_score=None,
             served_chars=0, baseline_chars=0, latency_ms=0.0, detail=None) -> None:
        try:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO events (ts, type, agent, source, scope, query, k, result_ids, n_results,
                       top_score, served_chars, baseline_chars, latency_ms, detail)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (_now(), type_, current_agent.get(), current_source.get(), scope, redact_secrets(query) if query else query, k,
                     json.dumps(list(ids)), len(ids), top_score, served_chars, baseline_chars,
                     round(latency_ms, 2), json.dumps(_scrub(detail or {}))),
                )
        except sqlite3.Error:
            # A logging failure must never break a memory operation.
            log.exception("failed to record %s event", type_)

    def _baseline_chars(self, scope: Optional[str], kinds: Optional[List[str]]) -> int:
        sql, params = "SELECT COALESCE(SUM(length(content)), 0) FROM memories WHERE valid_to IS NULL", []
        if scope and scope != "all":
            sql += " AND scope IN ('global', ?)"
            params.append(scope)
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)
        with self._get_conn() as conn:
            return conn.execute(sql, params).fetchone()[0]

    def recall(self, query, query_embedding, scope=None, kinds=None, k=8, include_superseded=False):
        start = time.perf_counter()
        results = super().recall(query, query_embedding, scope=scope, kinds=kinds, k=k,
                                 include_superseded=include_superseded)
        self._log(
            "search_docs" if kinds == ["doc_chunk"] else "recall",
            scope=scope, query=query, k=k, ids=[r["id"] for r in results],
            top_score=max((r["similarity"] for r in results), default=None),
            served_chars=sum(len(r["content"]) for r in results),
            baseline_chars=self._baseline_chars(scope, kinds),
            latency_ms=(time.perf_counter() - start) * 1000,
            detail={"degraded": "bm25_only"} if query_embedding is None else None,
        )
        return results

    def remember(self, facts, *args, **kwargs):
        start = time.perf_counter()
        res = super().remember(facts, *args, **kwargs)
        self._log(
            "remember", scope=kwargs.get("scope", "global"), query=(facts[0][:200] if facts else None),
            ids=res["saved_ids"], served_chars=sum(len(f) for f in facts),
            latency_ms=(time.perf_counter() - start) * 1000,
            detail={"facts": len(facts), "near_duplicates": len(res["near_duplicates"]),
                    "supersedes": kwargs.get("supersedes")},
        )
        return res

    def forget(self, memory_id, reason=""):
        ok = super().forget(memory_id, reason)
        self._log("forget", ids=[memory_id], detail={"reason": reason, "retired": ok})
        return ok

    # ---- analytics -----------------------------------------------------

    def hit_map(self) -> Dict[int, Dict[str, Any]]:
        with self._get_conn() as conn:
            return {r["mid"]: {"hits": r["hits"], "last_recalled": r["last_ts"]}
                    for r in conn.execute(RECALLED_IDS)}

    def usage_stats(self, days: int = 30) -> Dict[str, Any]:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._get_conn() as conn:
            row = conn.execute(
                """SELECT COUNT(*) n, COALESCE(SUM(top_score >= ?), 0) confident,
                          COALESCE(SUM(served_chars), 0) served, COALESCE(SUM(baseline_chars), 0) baseline,
                          COALESCE(SUM(MAX(baseline_chars - served_chars, 0)), 0) saved,
                          COALESCE(AVG(latency_ms), 0) avg_ms
                   FROM events WHERE type IN ('recall', 'search_docs') AND ts >= ?""",
                (CONFIDENT_SIM, since)).fetchone()
            lat = sorted(r[0] for r in conn.execute(
                "SELECT latency_ms FROM events WHERE type IN ('recall', 'search_docs') AND ts >= ?", (since,)))
            p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else 0

            per_day = {}
            for i in range(days - 1, -1, -1):
                day = (datetime.now(timezone.utc).date() - timedelta(days=i)).isoformat()
                per_day[day] = {"day": day, "recall": 0, "remember": 0, "forget": 0}
            for r in conn.execute(
                    "SELECT substr(ts, 1, 10) d, type, COUNT(*) n FROM events WHERE ts >= ? GROUP BY d, type", (since,)):
                bucket = "recall" if r["type"] in RECALL_TYPES else r["type"]
                if r["d"] in per_day:
                    per_day[r["d"]][bucket] += r["n"]

            by_agent = {r[0]: r[1] for r in conn.execute(
                "SELECT agent, COUNT(*) FROM events WHERE ts >= ? GROUP BY agent ORDER BY 2 DESC", (since,))}
            by_scope = {r[0] or "-": r[1] for r in conn.execute(
                "SELECT scope, COUNT(*) FROM events WHERE type IN ('recall', 'search_docs') AND ts >= ? "
                "GROUP BY scope ORDER BY 2 DESC", (since,))}

            top = [dict(r) for r in conn.execute(
                f"""SELECT m.id, m.kind, m.scope, m.content, m.valid_to, h.hits, h.last_ts AS last_recalled
                    FROM ({RECALLED_IDS}) h JOIN memories m ON m.id = h.mid
                    ORDER BY h.hits DESC, h.last_ts DESC LIMIT 10""")]
            never = [dict(r) for r in conn.execute(
                f"""SELECT id, kind, scope, content, valid_from FROM memories
                    WHERE valid_to IS NULL AND id NOT IN (SELECT mid FROM ({RECALLED_IDS}))
                    ORDER BY id LIMIT 20""")]
            weak = [dict(r) for r in conn.execute(
                """SELECT ts, agent, scope, query, top_score FROM events
                   WHERE type = 'recall' AND (top_score IS NULL OR top_score < ?) AND ts >= ?
                   ORDER BY id DESC LIMIT 10""", (CONFIDENT_SIM, since))]

        return {
            "days": days, "confident_sim": CONFIDENT_SIM, "chars_per_token": CHARS_PER_TOKEN,
            "recalls": row["n"], "confident": row["confident"],
            "served_tokens": row["served"] // CHARS_PER_TOKEN,
            "baseline_tokens": row["baseline"] // CHARS_PER_TOKEN,
            "saved_tokens": row["saved"] // CHARS_PER_TOKEN,
            "avg_ms": round(row["avg_ms"], 1), "p95_ms": round(p95, 1),
            "per_day": list(per_day.values()), "by_agent": by_agent, "by_scope": by_scope,
            "top_memories": top, "never_recalled": never, "weak_queries": weak,
        }

    def events(self, type_=None, agent=None, limit=50, offset=0) -> Dict[str, Any]:
        where, params = [], []
        if type_:
            where.append("type = ?")
            params.append(type_)
        if agent:
            where.append("agent = ?")
            params.append(agent)
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM events" + clause, params).fetchone()[0]
            rows = conn.execute("SELECT * FROM events" + clause + " ORDER BY id DESC LIMIT ? OFFSET ?",
                                params + [limit, offset]).fetchall()
            agents = [r[0] for r in conn.execute("SELECT DISTINCT agent FROM events ORDER BY 1")]
        items = [{**dict(r), "result_ids": json.loads(r["result_ids"]), "detail": json.loads(r["detail"])}
                 for r in rows]
        return {"total": total, "items": items, "agents": agents}

    def memory_detail(self, memory_id: int) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            m = conn.execute(
                "SELECT id, kind, scope, content, source, tags, created_by, valid_from, valid_to, verified_at "
                "FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if not m:
                return None
            evs = conn.execute(
                """SELECT e.id, e.ts, e.type, e.agent, e.query FROM events e
                   WHERE EXISTS (SELECT 1 FROM json_each(e.result_ids) j WHERE CAST(j.value AS INTEGER) = ?)
                   ORDER BY e.id DESC LIMIT 20""", (memory_id,)).fetchall()
        hit = self.hit_map().get(memory_id, {"hits": 0, "last_recalled": None})
        return {**dict(m), "tags": json.loads(m["tags"] or "[]"), "source": json.loads(m["source"] or "{}"),
                **hit, "events": [dict(e) for e in evs]}
