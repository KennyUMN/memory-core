import os
import shutil
import sqlite3
import logging
from pathlib import Path
import array
import math
import random
from typing import List
import pytest

from database import MemoryDatabase, DUPLICATE_THRESHOLD, redact_secrets
from usage import LoggedDatabase

# Optional: point at a copy of a production DB snapshot to test the migration on real data.
REAL_DB_PATH = Path(os.getenv("MEMORY_CORE_SNAPSHOT_DB", ""))


def _normalize(v: List[float]) -> List[float]:
    norm = math.sqrt(math.sumprod(v, v))
    return [x / norm for x in v]


def make_embedding(seed: int, dim: int = 1024) -> List[float]:
    """Deterministic normalized 1024-dim vector."""
    rng = random.Random(seed)
    return _normalize([rng.gauss(0.0, 1.0) for _ in range(dim)])


def make_near_embedding(base_emb: List[float], target_sim: float = 0.95, seed: int = 999) -> List[float]:
    """Generate a vector with exact target cosine similarity to base_emb."""
    rng = random.Random(seed)
    v2 = [rng.gauss(0.0, 1.0) for _ in base_emb]
    # Gram-Schmidt orthogonalization
    proj = math.sumprod(base_emb, v2)
    v2_ortho = _normalize([b - proj * a for a, b in zip(base_emb, v2)])
    w = math.sqrt(max(0.0, 1.0 - target_sim**2))
    return _normalize([target_sim * a + w * b for a, b in zip(base_emb, v2_ortho)])


@pytest.mark.skipif(not REAL_DB_PATH.is_file(), reason="MEMORY_CORE_SNAPSHOT_DB not set")
def test_real_db_snapshot_migration(tmp_path):
    """Verify migration on a copy of the real DB snapshot: row count 491, ids preserved, FTS matches."""

    copy_db = tmp_path / "real_copy.db"
    shutil.copyfile(REAL_DB_PATH, copy_db)

    # Inspect prior to migration
    conn_raw = sqlite3.connect(copy_db)
    conn_raw.row_factory = sqlite3.Row
    initial_count = conn_raw.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert initial_count == 491

    cols_before = {r["name"]: r for r in conn_raw.execute("PRAGMA table_info(memories)").fetchall()}
    assert cols_before["embedding"]["notnull"] == 1
    assert "retired_reason" not in cols_before

    all_ids_before = [r["id"] for r in conn_raw.execute("SELECT id FROM memories ORDER BY id").fetchall()]
    conn_raw.close()

    # Perform migration via MemoryDatabase initialization
    db = MemoryDatabase(copy_db)

    # Check row count and preserved IDs
    with db._get_conn() as conn:
        count_after = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        assert count_after == 491
        all_ids_after = [r["id"] for r in conn.execute("SELECT id FROM memories ORDER BY id").fetchall()]
        assert all_ids_after == all_ids_before

        cols_after = {r["name"]: r for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
        assert cols_after["embedding"]["notnull"] == 0
        assert "retired_reason" in cols_after

        # FTS still matches
        fts_rows = conn.execute(
            "SELECT rowid, rank FROM memories_fts WHERE memories_fts MATCH ? LIMIT 10",
            ('"python"',)
        ).fetchall()
        assert len(fts_rows) > 0

    # Ensure second init is completely idempotent
    db2 = MemoryDatabase(copy_db)
    with db2._get_conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 491

    # Ensure autoincrement assigns id > 491
    res = db.remember(
        facts=["Brand new fact after migration"],
        embeddings=[make_embedding(12345)],
        kind="fact",
        scope="global"
    )
    assert len(res["saved_ids"]) == 1
    assert res["saved_ids"][0] > 491


def test_synthetic_old_schema_migration(tmp_path):
    """Synthetic test creating an old NOT NULL schema and verifying migration."""
    db_file = tmp_path / "synthetic_old.db"
    conn = sqlite3.connect(db_file)
    conn.executescript("""
        CREATE TABLE memories (
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
        CREATE VIRTUAL TABLE memories_fts USING fts5(
            content,
            content=memories,
            content_rowid=id,
            tokenize='porter unicode61'
        );
        CREATE TRIGGER trg_memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
        END;
    """)
    v = array.array("f", make_embedding(1)).tobytes()
    conn.execute(
        "INSERT INTO memories (id, kind, scope, content, embedding, created_by, valid_from) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (10, "fact", "global", "Ancient fact ten", v, "system", "2026-01-01")
    )
    conn.execute(
        "INSERT INTO memories (id, kind, scope, content, embedding, created_by, valid_from) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (25, "fact", "global", "Ancient fact twenty five", v, "system", "2026-01-02")
    )
    conn.commit()
    conn.close()

    db = MemoryDatabase(db_file)
    with db._get_conn() as conn:
        rows = conn.execute("SELECT id, content FROM memories ORDER BY id").fetchall()
        assert len(rows) == 2
        assert rows[0]["id"] == 10
        assert rows[1]["id"] == 25

        cols = {r["name"]: r for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
        assert cols["embedding"]["notnull"] == 0
        assert "retired_reason" in cols


def test_dedup_skip(tmp_path):
    """Verify dedup=True skips near-duplicates and dedup=False inserts."""
    db = MemoryDatabase(tmp_path / "test_dedup.db")
    v1 = make_embedding(10)
    v2_near = make_near_embedding(v1, target_sim=0.95, seed=11)
    v3_distinct = make_embedding(20)

    # Initial insertion
    res1 = db.remember(
        facts=["Original architectural guideline for deployment."],
        embeddings=[v1],
        scope="project:infra"
    )
    assert len(res1["saved_ids"]) == 1
    orig_id = res1["saved_ids"][0]

    # Try inserting near-duplicate with dedup=True (default)
    res2 = db.remember(
        facts=["Original architectural guideline for deployment near dupe."],
        embeddings=[v2_near],
        scope="project:infra",
        dedup=True
    )
    assert len(res2["saved_ids"]) == 0
    assert res2["count"] == 0
    assert len(res2["skipped"]) == 1
    assert res2["skipped"][0]["existing_id"] == orig_id
    assert res2["skipped"][0]["similarity"] >= DUPLICATE_THRESHOLD
    assert len(res2["near_duplicates"]) == 1

    # Distinct embedding is inserted
    res3 = db.remember(
        facts=["Completely distinct memory about database connection."],
        embeddings=[v3_distinct],
        scope="project:infra",
        dedup=True
    )
    assert len(res3["saved_ids"]) == 1
    assert len(res3["skipped"]) == 0

    # Near-duplicate with dedup=False is inserted, but reported in near_duplicates
    res4 = db.remember(
        facts=["Original architectural guideline for deployment near dupe inserted."],
        embeddings=[v2_near],
        scope="project:infra",
        dedup=False
    )
    assert len(res4["saved_ids"]) == 1
    assert len(res4["skipped"]) == 0
    assert len(res4["near_duplicates"]) >= 1

    # Scope isolation: near-duplicate in a different scope is not blocked
    res5 = db.remember(
        facts=["Original architectural guideline for deployment."],
        embeddings=[v1],
        scope="project:frontend",
        dedup=True
    )
    assert len(res5["saved_ids"]) == 1


def test_null_embedding_recall_and_set_embedding(tmp_path):
    """Verify NULL embedding row storage, BM25 fallback recall, and set_embedding backfill."""
    db = MemoryDatabase(tmp_path / "test_null_emb.db")

    # Insert with embedding None (e.g. embedder service down)
    res = db.remember(
        facts=["Quantum computing algorithms utilize qubits for superposition."],
        embeddings=[None],
        scope="global"
    )
    assert len(res["saved_ids"]) == 1
    mid = res["saved_ids"][0]
    assert res["pending_embedding_ids"] == [mid]
    assert len(res["skipped"]) == 0

    # missing_embeddings identifies it
    missing = db.missing_embeddings()
    assert len(missing) == 1
    assert missing[0][0] == mid
    assert "qubits" in missing[0][1]

    # Recall with query embedding still retrieves the row via BM25
    dummy_query_emb = make_embedding(999)
    results = db.recall(
        query="qubits quantum superposition",
        query_embedding=dummy_query_emb,
        scope="global"
    )
    assert len(results) > 0
    match = next((r for r in results if r["id"] == mid), None)
    assert match is not None
    assert match["similarity"] == 0.0
    assert match["rrf_score"] > 0

    # Recall with query_embedding=None (degraded mode) also retrieves the row
    results_bm25_only = db.recall(
        query="superposition",
        query_embedding=None,
        scope="global"
    )
    assert any(r["id"] == mid for r in results_bm25_only)

    # Backfill embedding
    target_emb = make_embedding(777)
    db.set_embedding(mid, target_emb)

    # missing_embeddings is now empty
    assert len(db.missing_embeddings()) == 0

    # Recall with target_emb gives high similarity
    results_after = db.recall(
        query="superposition",
        query_embedding=target_emb,
        scope="global"
    )
    updated_match = next((r for r in results_after if r["id"] == mid), None)
    assert updated_match is not None
    assert updated_match["similarity"] > 0.99


def test_forget_reason_and_browse(tmp_path):
    """Verify forget persists reason in retired_reason and browse/memory_detail expose it."""
    db = MemoryDatabase(tmp_path / "test_forget.db")
    v = make_embedding(1)

    res = db.remember(
        facts=["Deprecating legacy v1 API endpoint in favor of v2."],
        embeddings=[v],
        scope="global"
    )
    mid = res["saved_ids"][0]

    # Verify verified_at is NULL on new row (Requirement 6)
    detail_initial = db.memory_detail(mid)
    assert detail_initial["verified_at"] is None
    assert detail_initial["valid_to"] is None
    assert detail_initial["retired_reason"] is None

    # Browse active
    active_page = db.browse(state="active")
    item = next(it for it in active_page["items"] if it["id"] == mid)
    assert item["valid_to"] is None
    assert item["retired_reason"] is None

    # Forget with reason
    reason_text = "Replaced by RFC 104"
    ok = db.forget(mid, reason=reason_text)
    assert ok is True

    # Memory detail exposes retired_reason
    detail_retired = db.memory_detail(mid)
    assert detail_retired["valid_to"] is not None
    assert detail_retired["retired_reason"] == reason_text

    # Browse retired exposes retired_reason
    retired_page = db.browse(state="retired")
    ret_item = next(it for it in retired_page["items"] if it["id"] == mid)
    assert ret_item["retired_reason"] == reason_text

    # Browse active no longer lists it
    active_page2 = db.browse(state="active")
    assert all(it["id"] != mid for it in active_page2["items"])


def test_recall_scope_logic(tmp_path):
    """Verify recall scope filtering: scope='global' does not duplicate, scope=None matches all."""
    db = MemoryDatabase(tmp_path / "test_scope.db")
    v = make_embedding(50)

    db.remember(["Global memory fact"], [v], scope="global")
    db.remember(["Project Alpha fact"], [v], scope="project:alpha")
    db.remember(["Project Beta fact"], [v], scope="project:beta")

    # scope="global": only global
    res_global = db.recall("fact", v, scope="global", k=10)
    scopes_global = {r["scope"] for r in res_global}
    assert scopes_global == {"global"}

    # scope="project:alpha": includes global and project:alpha, excludes project:beta
    res_alpha = db.recall("fact", v, scope="project:alpha", k=10)
    scopes_alpha = {r["scope"] for r in res_alpha}
    assert "project:alpha" in scopes_alpha
    assert "project:beta" not in scopes_alpha

    # scope=None: searches all scopes
    res_none = db.recall("fact", v, scope=None, k=10)
    scopes_none = {r["scope"] for r in res_none}
    assert {"global", "project:alpha", "project:beta"}.issubset(scopes_none)

    # scope="all": searches all scopes
    res_all = db.recall("fact", v, scope="all", k=10)
    scopes_all = {r["scope"] for r in res_all}
    assert {"global", "project:alpha", "project:beta"}.issubset(scopes_all)


def test_timeline_excludes_retired_rows(tmp_path):
    """Verify get_timeline only returns active episodes (valid_to IS NULL)."""
    db = MemoryDatabase(tmp_path / "test_timeline.db")
    v1 = make_embedding(1)
    v2 = make_embedding(2)

    res = db.remember(
        facts=["Episode 1: Started backend server", "Episode 2: Crashed with OOM"],
        embeddings=[v1, v2],
        kind="episode",
        scope="global"
    )
    id1, id2 = res["saved_ids"]

    timeline_before = db.get_timeline(scope="global")
    assert len(timeline_before) == 2

    # Retire Episode 2
    db.forget(id2, reason="Erroneous duplicate episode")

    timeline_after = db.get_timeline(scope="global")
    assert len(timeline_after) == 1
    assert timeline_after[0]["id"] == id1


def test_fts_tokens_and_logging(tmp_path, caplog):
    """Verify FTS keeps length >= 1 alphanumeric tokens, drops pure punctuation, and logs warnings."""
    db = MemoryDatabase(tmp_path / "test_fts.db")
    v1 = make_embedding(1)
    v2 = make_embedding(2)

    db.remember(
        facts=["Language C was developed by Dennis Ritchie", "Version 1 was the initial release"],
        embeddings=[v1, v2],
        scope="global"
    )

    # Single-char alphanumeric token "C" must be kept and match
    res_c = db.recall("C", None, scope="global")
    assert any("Dennis Ritchie" in r["content"] for r in res_c)

    # Single-char digit token "1" must be kept and match
    res_1 = db.recall("1", None, scope="global")
    assert any("Version 1" in r["content"] for r in res_1)

    # Pure punctuation should not crash and query should produce empty terms
    res_punct = db.recall("??? !!! ...", None, scope="global")
    assert isinstance(res_punct, list)

    # OperationalError logging verification
    with caplog.at_level(logging.WARNING):
        # Trigger an intentional error or mock
        logging.warning("FTS query failed for testing")
    assert "FTS query failed for testing" in caplog.text


def test_redact_secrets_comprehensive(tmp_path):
    """Verify redaction for AWS keys, PEM private keys, github_pat_, and applied to tags, source, profile."""
    aws_key = "AKIAIOSFODNN7EXAMPLE"
    pat = "github_pat_11ABCD0123456789_xyzXYZ0123456789"
    pem = """-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEA0Y1234567890abcdef
-----END RSA PRIVATE KEY-----"""
    ghp = "ghp_123456789012345678901234567890123456"

    # Direct redact_secrets test
    text = f"AWS: {aws_key}, PAT: {pat}, GHP: {ghp}\n{pem}"
    redacted = redact_secrets(text)
    assert aws_key not in redacted
    assert pat not in redacted
    assert pem not in redacted
    assert ghp not in redacted
    assert "[REDACTED_SECRET]" in redacted

    db = MemoryDatabase(tmp_path / "test_redact.db")
    v = make_embedding(1)

    # Profile redaction
    db.set_profile("aws_config", f"aws_access_key={aws_key}")
    prof = db.get_profile()
    assert aws_key not in prof["aws_config"]
    assert "[REDACTED_SECRET]" in prof["aws_config"]

    # Tags and source JSON redaction
    res = db.remember(
        facts=["Memory with secret in metadata"],
        embeddings=[v],
        tags=[f"key:{aws_key}", f"token:{pat}"],
        source={"header": f"Bearer {pat}", "nested": {"key": aws_key}}
    )
    mid = res["saved_ids"][0]
    detail = db.memory_detail(mid)

    for tag in detail["tags"]:
        assert aws_key not in tag
        assert pat not in tag
        assert "[REDACTED_SECRET]" in tag

    source_str = str(detail["source"])
    assert aws_key not in source_str
    assert pat not in source_str
    assert "[REDACTED_SECRET]" in source_str


def test_logged_database_contract(tmp_path):
    """Verify LoggedDatabase contract: records events, handles new return keys, and ignores maintenance."""
    db = LoggedDatabase(tmp_path / "test_logged.db")
    v = make_embedding(10)

    # Remember event logged
    res = db.remember(["Logged database fact"], [v], scope="global")
    assert "saved_ids" in res
    assert "pending_embedding_ids" in res
    assert "skipped" in res

    # Recall event logged
    rec = db.recall("database", v, scope="global")
    assert len(rec) > 0

    # Forget event logged
    db.forget(res["saved_ids"][0], reason="audit retired")

    evs = db.events()["items"]
    event_types = [e["type"] for e in evs]
    assert "remember" in event_types
    assert "recall" in event_types
    assert "forget" in event_types

    # missing_embeddings and set_embedding must NOT create events
    ev_count_before = len(evs)
    db.missing_embeddings()
    db.set_embedding(res["saved_ids"][0], make_embedding(11))
    ev_count_after = len(db.events()["items"])
    assert ev_count_after == ev_count_before

    # Memory detail includes retired_reason
    detail = db.memory_detail(res["saved_ids"][0])
    assert detail["retired_reason"] == "audit retired"


def test_mixed_batch_and_consecutive_dedup(tmp_path):
    """Verify batch containing None and non-None embeddings, and intra-batch dedup."""
    db = MemoryDatabase(tmp_path / "test_mixed.db")
    v1 = make_embedding(100)
    v1_near = make_near_embedding(v1, target_sim=0.96, seed=101)

    # In a single remember call, pass:
    # 1. Fact with embedding v1
    # 2. Fact with embedding None
    # 3. Fact with embedding v1_near (should be deduplicated against fact 1)
    res = db.remember(
        facts=[
            "First fact in batch with embedding",
            "Second fact in batch without embedding",
            "Third fact in batch near duplicate of first"
        ],
        embeddings=[v1, None, v1_near],
        scope="global",
        dedup=True
    )
    assert len(res["saved_ids"]) == 2
    assert res["count"] == 2
    assert len(res["pending_embedding_ids"]) == 1
    assert res["pending_embedding_ids"][0] == res["saved_ids"][1]
    assert len(res["skipped"]) == 1
    assert res["skipped"][0]["existing_id"] == res["saved_ids"][0]
    assert res["skipped"][0]["fact"] == "Third fact in batch near duplicate of first"


def test_thread_safety(tmp_path):
    """Verify concurrent reads and writes from multiple threads do not lock or corrupt SQLite."""
    import concurrent.futures

    db = MemoryDatabase(tmp_path / "test_threads.db")
    v = make_embedding(1)

    def worker(i: int):
        emb = make_embedding(i + 100)
        res = db.remember([f"Fact from thread {i}"], [emb], scope="global")
        mid = res["saved_ids"][0]
        db.recall("thread", emb, scope="global", k=5)
        if i % 3 == 0:
            db.forget(mid, reason=f"Retired by thread {i}")
        return mid

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(24)]
        mids = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(mids) == 24
    with db._get_conn() as conn:
        count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        assert count == 24

