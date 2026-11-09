"""Tests for delta, freshness, provenance, and supersession history. Run with: venv/bin/python test_delta_pack.py"""
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from database import MemoryDatabase, parse_since

AN_HOUR_AGO = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()


def fresh_db() -> MemoryDatabase:
    return MemoryDatabase(Path(tempfile.mkdtemp()) / "t.db")


def save(db, text, scope="global", kind="fact", supersedes=None, source=None):
    return db.remember(facts=[text], embeddings=[None], scope=scope, kind=kind,
                       supersedes=supersedes, source=source, created_by="test")["saved_ids"][0]


def test_parse_since_normalizes_to_utc():
    assert parse_since("2026-11-01T08:00:00Z") == "2026-11-01T08:00:00+00:00"
    assert parse_since("2026-11-01") == "2026-11-01T00:00:00+00:00"
    assert parse_since("2026-11-01T15:00:00+07:00") == "2026-11-01T08:00:00+00:00"
    for bad in ("yesterday", "", "2026-13-01"):
        try:
            parse_since(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_delta_reports_added_and_retired():
    db = fresh_db()
    old = save(db, "port is 8000")
    new = save(db, "port is 8420", supersedes=old)
    out = db.delta(AN_HOUR_AGO)
    assert {r["id"] for r in out["added"]} == {old, new}
    assert [(r["id"], r["retired_reason"]) for r in out["retired"]] == [(old, f"superseded by #{new}")]


def test_delta_skips_bulk_kinds_and_other_scopes():
    db = fresh_db()
    save(db, "a skill", scope="skills-archive", kind="skill")
    save(db, "a doc chunk", scope="project:a", kind="doc_chunk")
    b = save(db, "b fact", scope="project:b")
    g = save(db, "global fact")
    assert db.delta(AN_HOUR_AGO, scope="all")["added"] and \
        {r["id"] for r in db.delta(AN_HOUR_AGO, scope="all")["added"]} == {b, g}
    assert {r["id"] for r in db.delta(AN_HOUR_AGO, scope="project:a")["added"]} == {g}


def test_delta_future_since_is_empty():
    db = fresh_db()
    save(db, "fact")
    out = db.delta("2999-01-01")
    assert out["added"] == [] and out["retired"] == []


def test_recall_history_and_age():
    db = fresh_db()
    old = save(db, "deploy port is 8000")
    save(db, "deploy port is 8420", supersedes=old)
    active = db.recall("deploy port", None)
    assert [r["content"] for r in active] == ["deploy port is 8420"]
    assert active[0]["age_days"] == 0.0
    history = db.recall("deploy port", None, include_superseded=True)
    retired = [r for r in history if r["id"] == old]
    assert retired and retired[0]["retired_reason"].startswith("superseded by #")


def test_source_is_kept():
    db = fresh_db()
    mid = save(db, "uses uv", source={"ref": "pyproject.toml"})
    assert db.memory_detail(mid)["source"] == {"ref": "pyproject.toml"}


def test_recent_and_newest_in_scope():
    db = fresh_db()
    assert db.newest_in_scope("global") is None
    save(db, "a skill", scope="skills-archive", kind="skill")
    assert db.newest_in_scope("all") is None
    f = save(db, "fact one")
    e = save(db, "did a thing", kind="episode")
    assert {r["id"] for r in db.recent("global")} == {f, e}
    assert [r["id"] for r in db.recent("global", ["episode"])] == [e]
    assert db.newest_in_scope("project:x") is not None


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"all {len(tests)} delta/pack tests passed")
