"""Tests for skill sources. Run with: venv/bin/python test_skills.py"""
import asyncio
import tarfile
import tempfile
from pathlib import Path

from database import MemoryDatabase
import skills

TMP = Path(tempfile.mkdtemp())
skills.SOURCES_DIR = TMP / "skill-sources"


class FakeEmbedder:
    async def embed_passages(self, texts):
        # Identical vectors: dedup would collapse these rows if sync_rows allowed it.
        return [[1.0, 0.0, 0.0] for _ in texts]


def write_skill(root: Path, rel: str, name: str, desc: str) -> None:
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: >\n  {desc}\n  second line\n---\n\nBody\n")


def active(db, slug):
    return skills._active_rows(db, slug)


def test_parse_repo():
    assert skills.parse_repo("blader/humanizer") == ("blader__humanizer", "https://github.com/blader/humanizer.git")
    assert skills.parse_repo("https://github.com/Kepano/obsidian-skills.git")[0] == "kepano__obsidian-skills"
    assert skills.parse_repo("https://github.com/a/b/tree/main/skills/x")[1] == "https://github.com/a/b.git"
    for bad in ["", "https://gitlab.com/a/b", "a", "file:///etc/passwd", "a/..", "https://github.com/a"]:
        try:
            skills.parse_repo(bad)
            raise AssertionError(f"accepted {bad!r}")
        except ValueError:
            pass


def test_frontmatter():
    fm = skills.parse_frontmatter("---\nname: 'x'\ndescription: |\n  one\n  two\nother: y\n---\nbody")
    assert fm == {"name": "x", "description": "one two", "other": "y"}
    assert skills.parse_frontmatter("no frontmatter") == {}


async def test_sync():
    db = MemoryDatabase(TMP / "t.db")
    skills.ensure_schema(db)
    emb = FakeEmbedder()
    # A legacy archived skill with the same name as one in the repo.
    legacy = db.remember(["skill: alpha | old | path: ~/.claude/backup-2026/skills-archive/alpha/SKILL.md"],
                         [[0.0, 1.0, 0.0]], kind="skill", scope=skills.SCOPE, tags=[skills.TAG])["saved_ids"][0]

    root = skills.SOURCES_DIR / "o__r"
    write_skill(root, "skills/alpha", "alpha", "first")
    write_skill(root, "skills/beta", "beta", "second")
    write_skill(root, ".", "root-skill", "at the root")

    s = await skills.sync_rows(db, emb, "o__r", root, "test")
    assert sorted(s["added"]) == ["beta", "root-skill"] and s["replaced"] == ["alpha"], s
    rows = active(db, "o__r")
    assert set(rows) == {"skills/alpha", "skills/beta", "."}, rows
    assert db.memory_detail(legacy)["valid_to"] is not None
    assert rows["."][1].endswith(f"path: {skills.CLIENT_ROOT}/o__r/SKILL.md")
    assert "first second line" in rows["skills/alpha"][1]

    s = await skills.sync_rows(db, emb, "o__r", root, "test")
    assert s["unchanged"] == 3 and not (s["added"] or s["updated"] or s["removed"]), s

    write_skill(root, "skills/alpha", "alpha", "changed")
    (root / "skills/beta/SKILL.md").unlink()
    old_alpha = rows["skills/alpha"][0]
    s = await skills.sync_rows(db, emb, "o__r", root, "test")
    assert s["updated"] == ["alpha"] and s["removed"] == ["beta"], s
    rows = active(db, "o__r")
    assert set(rows) == {"skills/alpha", "."} and "changed" in rows["skills/alpha"][1]
    assert db.memory_detail(old_alpha)["retired_reason"].startswith("superseded by")

    with db._get_conn() as conn:
        conn.execute("INSERT INTO skill_sources (slug, url, added_at, updated_at) VALUES ('o__r', 'u', 'n', 'n')")
    assert skills.list_sources(db)[0]["skills"] == 2
    r = await skills.remove_source(db, "o__r")
    assert r["retired"] == 2 and not active(db, "o__r") and not root.exists()


def test_bundle():
    root = skills.SOURCES_DIR / "b__b"
    write_skill(root, "skills/x", "x", "d")
    (root / "skills/x/scripts").mkdir()
    (root / "skills/x/scripts/run.sh").write_text("echo hi")
    (root / "skills/x/link").symlink_to("/etc/passwd")
    for bad in ["../", "b__b/../../etc", "", "b__b/missing"]:
        try:
            skills.resolve_skill_dir(bad)
            raise AssertionError(f"accepted {bad!r}")
        except (ValueError, skills.SourceNotFound):
            pass
    out = skills.build_bundle(skills.resolve_skill_dir("b__b/skills/x"))
    with tarfile.open(out) as t:
        names = sorted(t.getnames())
    assert names == ["b__b/skills/x/SKILL.md", "b__b/skills/x/scripts/run.sh"], names


if __name__ == "__main__":
    test_parse_repo()
    test_frontmatter()
    asyncio.run(test_sync())
    test_bundle()
    print("all skill tests passed")
