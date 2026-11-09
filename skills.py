"""Skill sources: GitHub repos cloned on this server and indexed as `kind=skill` memories.

Each SKILL.md in a cloned repo becomes one row in the `skills-archive` scope, in the same
`skill: name | description | path: ...` format the dashboard and find-skill already parse.
Agents fetch a skill's folder through the bundle endpoint, so files never need to exist
on the agent's machine ahead of time.
"""
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import DATA_DIR

log = logging.getLogger("memory-core.skills")

SOURCES_DIR = DATA_DIR / "skill-sources"
SCOPE = "skills-archive"
KIND = "skill"
TAG = "archived-skill"
DESC_MAX = 250
CLIENT_ROOT = "~/.cache/memory-core/skills"
GIT_TIMEOUT_S = 180
MAX_REPO_BYTES = 200 * 1024 * 1024
REMOVED_REASON = "removed from upstream repo"

# owner/repo, github.com/owner/repo, or a full URL with .git, /tree/<branch>/..., etc.
GITHUB_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?(?:github\.com/)?"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/(?P<repo>[A-Za-z0-9._-]{1,100}?)"
    r"(?:\.git)?(?:/.*)?$"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS skill_sources (
    slug TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    commit_sha TEXT,
    added_by TEXT,
    added_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

_lock = asyncio.Lock()


class SkillSourceError(Exception):
    """A git or repository problem the caller should see verbatim."""


class SourceNotFound(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------- parsing ----------

def parse_repo(text: str) -> Tuple[str, str]:
    """Return (slug, clone_url) for a GitHub repo reference, or raise ValueError."""
    m = GITHUB_RE.match((text or "").strip())
    if not m or m["repo"] in (".", ".."):
        raise ValueError("expected a GitHub repo such as https://github.com/owner/repo")
    owner, repo = m["owner"], m["repo"]
    return f"{owner}__{repo}".lower(), f"https://github.com/{owner}/{repo}.git"


def _unquote(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        return v[1:-1]
    return v


def parse_frontmatter(text: str) -> Dict[str, str]:
    """Read top-level scalar keys from YAML frontmatter, including folded `>` / `|` blocks."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    out: Dict[str, str] = {}
    key, block = None, []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        m = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if m:
            if key and block:
                out[key] = " ".join(block)
            key, value = m[1], m[2].strip()
            block = []
            if value in ("", ">", "|", ">-", "|-", ">+", "|+"):
                continue
            out[key] = _unquote(value)
            key = None
        elif key and line.startswith((" ", "\t")):
            block.append(line.strip())
    if key and block:
        out[key] = " ".join(block)
    return out


def _flat(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def row_text(name: str, desc: str, slug: str, rel_dir: str) -> str:
    name = _flat(name).replace("|", "/")
    rel = f"{slug}/{rel_dir}" if rel_dir != "." else slug
    return f"skill: {name} | {_flat(desc)[:DESC_MAX]} | path: {CLIENT_ROOT}/{rel}/SKILL.md"


def discover(root: Path) -> List[Dict[str, str]]:
    """Find every SKILL.md under root, skipping .git and symlinks."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != ".git" and not os.path.islink(os.path.join(dirpath, d)))
        if "SKILL.md" not in filenames:
            continue
        path = Path(dirpath) / "SKILL.md"
        if path.is_symlink():
            continue
        meta = parse_frontmatter(path.read_text(encoding="utf-8", errors="replace"))
        rel = Path(dirpath).relative_to(root).as_posix()
        name = meta.get("name") or (Path(dirpath).name if rel != "." else root.name)
        found.append({"name": name, "desc": meta.get("description", ""), "dir": rel})
    return found


# ---------- git ----------

def _git(args: List[str], cwd: Optional[Path] = None) -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "true"}
    cmd = ["git", "-c", "protocol.allow=never", "-c", "protocol.https.allow=always",
           "-c", "core.symlinks=true", *args]
    try:
        res = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=GIT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise SkillSourceError(f"git {args[0]} timed out after {GIT_TIMEOUT_S}s")
    if res.returncode != 0:
        tail = (res.stderr or res.stdout).strip().splitlines()[-3:]
        raise SkillSourceError(f"git {args[0]} failed: " + " ".join(tail))
    return res.stdout.strip()


def _dir_bytes(root: Path) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            p = os.path.join(dirpath, f)
            if not os.path.islink(p):
                total += os.path.getsize(p)
    return total


def _check_size(root: Path) -> None:
    if _dir_bytes(root) > MAX_REPO_BYTES:
        raise SkillSourceError(f"repo is larger than {MAX_REPO_BYTES // (1024 * 1024)} MB")


def clone(clone_url: str, dest: Path) -> str:
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".clone-", dir=SOURCES_DIR))
    try:
        _git(["clone", "--depth", "1", "--single-branch", "--no-tags", "--no-recurse-submodules",
              "--", clone_url, str(tmp / "repo")])
        _check_size(tmp / "repo")
        if dest.exists():
            shutil.rmtree(dest)
        (tmp / "repo").rename(dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return _git(["rev-parse", "HEAD"], cwd=dest)


def pull(dest: Path) -> str:
    _git(["fetch", "--depth", "1", "--no-tags", "--no-recurse-submodules", "origin", "HEAD"], cwd=dest)
    _git(["reset", "--hard", "FETCH_HEAD"], cwd=dest)
    _git(["clean", "-fdx"], cwd=dest)
    _check_size(dest)
    return _git(["rev-parse", "HEAD"], cwd=dest)


# ---------- database ----------

def ensure_schema(db) -> None:
    with db._get_conn() as conn:
        conn.execute(SCHEMA)


def _source_row(db, slug: str) -> Optional[Dict[str, Any]]:
    with db._get_conn() as conn:
        r = conn.execute("SELECT * FROM skill_sources WHERE slug = ?", (slug,)).fetchone()
    return dict(r) if r else None


def _active_rows(db, slug: str) -> Dict[str, Tuple[int, str]]:
    with db._get_conn() as conn:
        rows = conn.execute(
            "SELECT id, content, json_extract(source, '$.dir') AS dir FROM memories "
            "WHERE kind = ? AND valid_to IS NULL AND json_extract(source, '$.skill_source') = ?",
            (KIND, slug),
        ).fetchall()
    return {r["dir"]: (r["id"], r["content"]) for r in rows}


def _legacy_by_name(db) -> Dict[str, int]:
    """Archived skills with no repo behind them, keyed by name, so a repo copy can replace them."""
    with db._get_conn() as conn:
        rows = conn.execute(
            "SELECT id, content FROM memories WHERE kind = ? AND scope = ? AND valid_to IS NULL "
            "AND json_extract(source, '$.skill_source') IS NULL",
            (KIND, SCOPE),
        ).fetchall()
    out = {}
    for r in rows:
        m = re.match(r"^skill:\s*(.+?) \| ", r["content"])
        if m:
            out.setdefault(m[1], r["id"])
    return out


def _count_rows(db) -> Dict[str, int]:
    with db._get_conn() as conn:
        rows = conn.execute(
            "SELECT json_extract(source, '$.skill_source') AS slug, COUNT(*) AS n FROM memories "
            "WHERE kind = ? AND valid_to IS NULL AND json_extract(source, '$.skill_source') IS NOT NULL "
            "GROUP BY slug",
            (KIND,),
        ).fetchall()
    return {r["slug"]: r["n"] for r in rows}


async def _embed_or_none(embedder, texts: List[str]) -> List[Optional[List[float]]]:
    if not texts:
        return []
    try:
        return await embedder.embed_passages(texts)
    except Exception as exc:
        # The backfill loop embeds NULL rows later, so the sync still succeeds.
        log.warning("skill embedding failed, saving with pending embeddings: %s", exc)
        return [None] * len(texts)


async def sync_rows(db, embedder, slug: str, root: Path, agent: str) -> Dict[str, Any]:
    """Make the active skill rows for `slug` match the SKILL.md files under `root`."""
    skills = await asyncio.to_thread(discover, root)
    existing = await asyncio.to_thread(_active_rows, db, slug)
    legacy = await asyncio.to_thread(_legacy_by_name, db)

    writes, summary = [], {"added": [], "updated": [], "replaced": [], "removed": [], "unchanged": 0}
    for s in skills:
        text = row_text(s["name"], s["desc"], slug, s["dir"])
        if s["dir"] in existing:
            old_id, old_text = existing[s["dir"]]
            if old_text == text:
                summary["unchanged"] += 1
                continue
            writes.append((text, s, old_id, "updated"))
        elif s["name"] in legacy:
            writes.append((text, s, legacy.pop(s["name"]), "replaced"))
        else:
            writes.append((text, s, None, "added"))

    embeddings = await _embed_or_none(embedder, [w[0] for w in writes])
    for (text, s, supersedes, bucket), emb in zip(writes, embeddings):
        # dedup=False: similar skills from other repos must not swallow this row.
        await asyncio.to_thread(
            db.remember, facts=[text], embeddings=[emb], kind=KIND, scope=SCOPE,
            supersedes=supersedes, tags=[TAG], source={"skill_source": slug, "dir": s["dir"]},
            created_by=agent, dedup=False,
        )
        summary[bucket].append(s["name"])

    seen = {s["dir"] for s in skills}
    for rel, (old_id, old_text) in existing.items():
        if rel not in seen:
            await asyncio.to_thread(db.forget, old_id, REMOVED_REASON)
            summary["removed"].append(old_text.split(" | ")[0][7:])
    summary["skills"] = len(skills)
    return summary


# ---------- public operations ----------

def list_sources(db) -> List[Dict[str, Any]]:
    counts = _count_rows(db)
    with db._get_conn() as conn:
        rows = conn.execute("SELECT * FROM skill_sources ORDER BY slug").fetchall()
    return [{**dict(r), "skills": counts.get(r["slug"], 0)} for r in rows]


async def add_source(db, embedder, url: str, agent: str) -> Dict[str, Any]:
    slug, clone_url = parse_repo(url)
    async with _lock:
        if await asyncio.to_thread(_source_row, db, slug):
            return await _update_locked(db, embedder, slug, agent)
        dest = SOURCES_DIR / slug
        commit = await asyncio.to_thread(clone, clone_url, dest)
        if not await asyncio.to_thread(discover, dest):
            shutil.rmtree(dest, ignore_errors=True)
            raise SkillSourceError("no SKILL.md files found in this repo")
        now = _now()
        with db._get_conn() as conn:
            conn.execute(
                "INSERT INTO skill_sources (slug, url, commit_sha, added_by, added_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (slug, clone_url, commit, agent, now, now),
            )
        summary = await sync_rows(db, embedder, slug, dest, agent)
    return {"slug": slug, "commit": commit, "previous_commit": None, **summary}


async def update_source(db, embedder, slug: str, agent: str) -> Dict[str, Any]:
    async with _lock:
        return await _update_locked(db, embedder, slug, agent)


async def _update_locked(db, embedder, slug: str, agent: str) -> Dict[str, Any]:
    src = await asyncio.to_thread(_source_row, db, slug)
    if not src:
        raise SourceNotFound(slug)
    dest = SOURCES_DIR / slug
    if (dest / ".git").is_dir():
        commit = await asyncio.to_thread(pull, dest)
    else:
        commit = await asyncio.to_thread(clone, src["url"], dest)
    with db._get_conn() as conn:
        conn.execute("UPDATE skill_sources SET commit_sha = ?, updated_at = ? WHERE slug = ?", (commit, _now(), slug))
    summary = await sync_rows(db, embedder, slug, dest, agent)
    return {"slug": slug, "commit": commit, "previous_commit": src["commit_sha"], **summary}


async def remove_source(db, slug: str) -> Dict[str, Any]:
    async with _lock:
        if not await asyncio.to_thread(_source_row, db, slug):
            raise SourceNotFound(slug)
        rows = await asyncio.to_thread(_active_rows, db, slug)
        for old_id, _ in rows.values():
            await asyncio.to_thread(db.forget, old_id, f"skill source {slug} removed")
        with db._get_conn() as conn:
            conn.execute("DELETE FROM skill_sources WHERE slug = ?", (slug,))
        await asyncio.to_thread(shutil.rmtree, SOURCES_DIR / slug, True)
    return {"slug": slug, "retired": len(rows)}


def resolve_skill_dir(rel: str) -> Path:
    """Map a `<slug>/<dir>` path from a skill row to its folder, refusing anything outside SOURCES_DIR."""
    base = SOURCES_DIR.resolve()
    target = (base / (rel or "").strip("/")).resolve()
    if target == base or base not in target.parents:
        raise ValueError("path is outside the skill sources")
    if not (target / "SKILL.md").is_file():
        raise SourceNotFound(rel)
    return target


def build_bundle(skill_dir: Path) -> str:
    """Write a .tar.gz of the skill folder (paths relative to SOURCES_DIR) and return its file path."""
    base = SOURCES_DIR.resolve()
    fd, out = tempfile.mkstemp(prefix="skill-", suffix=".tar.gz")
    os.close(fd)
    with tarfile.open(out, "w:gz") as tar:
        for dirpath, dirnames, filenames in os.walk(skill_dir):
            dirnames[:] = [d for d in dirnames if d != ".git" and not os.path.islink(os.path.join(dirpath, d))]
            for f in filenames:
                p = Path(dirpath) / f
                if p.is_symlink() or not p.is_file():
                    continue
                tar.add(p, arcname=p.relative_to(base).as_posix(), recursive=False)
    return out
