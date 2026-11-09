"""Tests for supersede conflicts. Run with: venv/bin/python test_supersede.py"""
import tempfile
import threading
from pathlib import Path

from database import MemoryDatabase, SupersedeError

RACERS = 2


def fresh_db() -> MemoryDatabase:
    return MemoryDatabase(Path(tempfile.mkdtemp()) / "t.db")


def save(db, text, supersedes=None):
    return db.remember(facts=[text], embeddings=[None], supersedes=supersedes, created_by="test")


def raises_supersede(fn) -> str:
    try:
        fn()
    except SupersedeError as exc:
        return str(exc)
    raise AssertionError("expected SupersedeError")


def active_contents(db):
    return sorted(i["content"] for i in db.browse(state="active", limit=200)["items"])


def test_supersede_retires_old():
    db = fresh_db()
    old = save(db, "port is 8000")["saved_ids"][0]
    res = save(db, "port is 8420", supersedes=old)
    assert res["superseded"] is True
    detail = db.memory_detail(old)
    assert detail["valid_to"] is not None
    assert detail["retired_reason"] == f"superseded by #{res['saved_ids'][0]}"
    assert active_contents(db) == ["port is 8420"]


def test_stale_supersede_rolls_back():
    db = fresh_db()
    old = save(db, "port is 8000")["saved_ids"][0]
    save(db, "port is 8420", supersedes=old)
    msg = raises_supersede(lambda: save(db, "port is 9000", supersedes=old))
    assert "already retired" in msg
    assert active_contents(db) == ["port is 8420"]


def test_missing_target_rolls_back():
    db = fresh_db()
    raises_supersede(lambda: save(db, "orphan", supersedes=999))
    assert active_contents(db) == []


def test_empty_replacement_keeps_old():
    db = fresh_db()
    old = save(db, "port is 8000")["saved_ids"][0]
    raises_supersede(lambda: save(db, "   ", supersedes=old))
    assert db.memory_detail(old)["valid_to"] is None


def test_concurrent_supersede_has_one_winner():
    db = fresh_db()
    old = save(db, "port is 8000")["saved_ids"][0]
    barrier = threading.Barrier(RACERS)
    outcomes = []

    def racer(n):
        barrier.wait()
        try:
            save(db, f"port is {9000 + n}", supersedes=old)
            outcomes.append("won")
        except SupersedeError:
            outcomes.append("lost")

    threads = [threading.Thread(target=racer, args=(n,)) for n in range(RACERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes) == ["lost", "won"], outcomes
    assert len(active_contents(db)) == 1


if __name__ == "__main__":
    test_supersede_retires_old()
    test_stale_supersede_rolls_back()
    test_missing_target_rolls_back()
    test_empty_replacement_keeps_old()
    for _ in range(20):
        test_concurrent_supersede_has_one_winner()
    print("all supersede tests passed")
