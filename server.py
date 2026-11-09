import asyncio
from typing import Annotated, List, Optional, Dict, Any, Tuple
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException, Request, Depends, status
from fastapi.responses import JSONResponse, HTMLResponse, Response, FileResponse
from starlette.background import BackgroundTask
from pathlib import Path
import logging
import re
import httpx
from pydantic import BaseModel, Field, StringConstraints
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings

from config import HOST, PORT, TOKENS
from embedder import EmbedderClient, EmbeddingError, CircuitOpenError
from database import redact_secrets, SupersedeError
from usage import LoggedDatabase, current_agent, current_source
import os
import skills

MAX_FACTS, MAX_FACT_CHARS, MAX_K, MAX_QUERY_CHARS = 100, 20000, 50, 2000
MAX_SOURCE_CHARS = 500
PACK_EPISODES = 5
BACKFILL_INTERVAL = 60.0
BACKFILL_BATCH_SIZE = 32

# Initialize embedder & DB
embedder = EmbedderClient()
db = LoggedDatabase()
skills.ensure_schema(db)
log = logging.getLogger("memory-core")


class RedactTokenFilter(logging.Filter):
    """Keeps ?token=... query secrets out of uvicorn access logs."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.args = tuple(re.sub(r"(token=)[^&\s\"]+", r"\1<redacted>", a) if isinstance(a, str) else a
                            for a in (record.args or ()))
        return True


logging.getLogger("uvicorn.access").addFilter(RedactTokenFilter())


async def embed_query_or_none(query: str) -> Optional[List[float]]:
    """Query embedding, or None when the embedding service is down (recall falls back to BM25)."""
    try:
        return await embedder.embed_query(query)
    except (httpx.HTTPError, EmbeddingError) as exc:
        log.warning("embedding unavailable, recall degraded to BM25-only: %s", exc)
        return None


def source_ref(source: Optional[str]) -> Optional[Dict[str, str]]:
    """Provenance for a remembered fact, stored in the memories.source JSON column."""
    if not source or not source.strip():
        return None
    return {"ref": source.strip()[:MAX_SOURCE_CHARS]}


async def build_context_pack(scope: str, query: Optional[str], k: int) -> Dict[str, Any]:
    """Profile, recent episodes, and the top memories for a scope in one bounded call."""
    k = max(1, min(k, MAX_K))
    if query and query.strip():
        query = redact_secrets(query[:MAX_QUERY_CHARS])
        q_emb = await embed_query_or_none(query)
        memories = await asyncio.to_thread(db.recall, query=query, query_embedding=q_emb, scope=scope, k=k)
    else:
        memories = await asyncio.to_thread(db.recent, scope, None, k)
    episodes = await asyncio.to_thread(db.recent, scope, ["episode"], PACK_EPISODES)
    return {
        "scope": scope,
        "profile": await asyncio.to_thread(db.get_profile),
        "memories": memories,
        "recent_episodes": episodes,
        "newest_in_scope": await asyncio.to_thread(db.newest_in_scope, scope),
    }


async def backfill_embeddings_once() -> int:
    """Run one iteration of backfilling missing embeddings. Returns count of embeddings updated."""
    if getattr(embedder, "is_circuit_open", False):
        log.debug("Skipping backfill: circuit breaker is open")
        return 0

    try:
        missing: List[Tuple[int, str]] = await asyncio.to_thread(db.missing_embeddings, BACKFILL_BATCH_SIZE)
        if not missing:
            return 0

        ids = [item[0] for item in missing]
        texts = [item[1] for item in missing]

        embeddings = await embedder.embed_passages(texts)
        for mem_id, emb in zip(ids, embeddings):
            await asyncio.to_thread(db.set_embedding, mem_id, emb)
        return len(ids)
    except CircuitOpenError:
        log.debug("Skipping backfill: circuit breaker is open")
        return 0
    except (httpx.HTTPError, EmbeddingError) as exc:
        log.warning("Embedding backfill iteration failed: %s", exc)
        return 0
    except Exception as exc:
        log.warning("Unexpected error during embedding backfill: %s", exc)
        return 0


async def backfill_loop():
    """Background task running every BACKFILL_INTERVAL to backfill missing embeddings."""
    while True:
        try:
            await backfill_embeddings_once()
        except asyncio.CancelledError:
            break
        except Exception as exc:
            log.warning("Error in background backfill loop: %s", exc)

        try:
            await asyncio.sleep(BACKFILL_INTERVAL)
        except asyncio.CancelledError:
            break


# Initialize MCP Server
mcp = MCPServer("memory-core")

# --- MCP Resources & Tools ---

@mcp.resource("memory://profile")
def resource_profile() -> str:
    """Core profile block and active context"""
    prof = db.get_profile()
    if not prof:
        return "No profile set yet."
    return "\n".join(f"{k}: {v}" for k, v in prof.items())

@mcp.tool()
def memory_profile() -> Dict[str, str]:
    """Retrieve the core profile, preferences, and active project settings."""
    return db.get_profile()

@mcp.tool()
async def memory_recall(
    query: str,
    scope: str = "global",
    kinds: Optional[List[str]] = None,
    k: int = 8,
    include_superseded: bool = False
) -> Dict[str, Any]:
    """Recall relevant facts, episodes, procedures, or document chunks.
    Performs hybrid BM25 + Vector search fused with Reciprocal Rank Fusion (RRF).
    Each result carries age_days; newest_in_scope shows how fresh this scope's memory is.
    
    Args:
        query: The search query or question
        scope: Target scope ('global', 'project:<name>', 'agent:<name>', or 'all')
        kinds: List of kinds to filter by (e.g. ['fact', 'procedure', 'doc_chunk'])
        k: Maximum number of ranked results to return (default 8)
        include_superseded: Also return retired facts with their retired_reason (correction history)
    """
    if not query.strip():
        return {"results": [], "count": 0}
    
    query, k = redact_secrets(query[:MAX_QUERY_CHARS]), max(1, min(k, MAX_K))
    # Generate query embedding with task instruction
    q_emb = await embed_query_or_none(query)
    results = await asyncio.to_thread(
        db.recall,
        query=query,
        query_embedding=q_emb,
        scope=scope,
        kinds=kinds,
        k=k,
        include_superseded=include_superseded
    )
    return {
        "results": results,
        "count": len(results),
        "scope": scope,
        "newest_in_scope": await asyncio.to_thread(db.newest_in_scope, scope)
    }

@mcp.tool()
async def memory_remember(
    facts: List[str],
    kind: str = "fact",
    scope: str = "global",
    supersedes: Optional[int] = None,
    tags: Optional[List[str]] = None,
    agent_id: Optional[str] = None,
    source: Optional[str] = None
) -> Dict[str, Any]:
    """Remember one or more atomic, standalone facts or procedures.
    Duplicates at similarity >= 0.90 in the same scope are skipped (not inserted) and reported. Redacts secrets automatically.
    
    Args:
        facts: List of short standalone sentences (one fact per string)
        kind: Kind of memory ('fact', 'procedure', 'episode', 'core')
        scope: Context scope ('global', 'project:<name>', etc.)
        supersedes: Optional ID of an older fact being replaced
        tags: Optional category tags
        agent_id: Identifier of the agent writing this memory
        source: Where the fact came from (URL, file path, commit, or "user said"), kept as provenance
    """
    if not facts:
        return {"saved_ids": [], "near_duplicates": [], "count": 0}
    
    facts = [redact_secrets(f) for f in facts[:MAX_FACTS]]
    embedding_failed = False
    try:
        embeddings = await embedder.embed_passages(facts)
    except (httpx.HTTPError, EmbeddingError) as exc:
        log.warning("Embedding failed during memory_remember, saving with pending embeddings: %s", exc)
        embeddings = [None] * len(facts)
        embedding_failed = True

    res = await asyncio.to_thread(
        db.remember,
        facts=facts,
        embeddings=embeddings,
        kind=kind,
        scope=scope,
        supersedes=supersedes,
        tags=tags,
        source=source_ref(source),
        created_by=agent_id or "mcp-agent"
    )
    if embedding_failed:
        res["embedding_pending"] = True
    return res

@mcp.tool()
def memory_forget(id: int, reason: str = "") -> Dict[str, Any]:
    """Soft-delete an outdated or erroneous memory fact (sets valid_to). Mistakes can be audited.
    
    Args:
        id: Memory fact ID to retire
        reason: Explanation for retiring this fact
    """
    success = db.forget(memory_id=id, reason=reason)
    return {"id": id, "retired": success, "reason": reason}

@mcp.tool()
async def memory_search_docs(
    query: str,
    project: Optional[str] = None,
    k: int = 5
) -> Dict[str, Any]:
    """Search ingested document chunks (HANDOVER.md, READMEs, runbooks) by semantic meaning.
    
    Args:
        query: Search query
        project: Optional project filter
        k: Top chunks to return
    """
    q_emb = await embed_query_or_none(query)
    results = await asyncio.to_thread(
        db.search_docs,
        query_embedding=q_emb,
        project=project,
        k=k,
        query=query
    )
    return {"results": results, "count": len(results)}

@mcp.tool()
def memory_timeline(scope: str = "global", since: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve session episode summaries in chronological order.
    
    Args:
        scope: Scope to filter by ('global' or specific project)
        since: Optional ISO datetime string to filter episodes after a specific time
    """
    return db.get_timeline(scope=scope, since=since)

@mcp.tool()
async def memory_context_pack(
    scope: str = "global",
    query: Optional[str] = None,
    k: int = 8
) -> Dict[str, Any]:
    """One-call session bundle: profile, top memories, and recent episodes for a scope.
    Call at session start and after context compaction instead of separate profile + recall calls.

    Args:
        scope: Target scope ('global', 'project:<name>', or 'all'); global memories are always included
        query: Optional task description; ranks memories by relevance instead of recency
        k: Maximum memories to return (default 8)
    """
    return await build_context_pack(scope, query, k)

@mcp.tool()
def memory_delta(since: str, scope: str = "all") -> Dict[str, Any]:
    """What changed since a time: memories added and memories retired (with the reason).
    Skips indexed skills and doc chunks.

    Args:
        since: ISO 8601 time, e.g. '2026-11-01' or '2026-11-01T08:00:00Z'
        scope: 'all' (default), 'global', or 'project:<name>'
    """
    try:
        return db.delta(since=since, scope=scope)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc


# --- REST API & Pydantic Schemas ---

class RememberRequest(BaseModel):
    facts: List[Annotated[str, StringConstraints(max_length=MAX_FACT_CHARS)]] = Field(min_length=1, max_length=MAX_FACTS)
    kind: str = "fact"
    scope: str = "global"
    supersedes: Optional[int] = None
    tags: Optional[List[str]] = None
    source: Optional[Annotated[str, StringConstraints(max_length=MAX_SOURCE_CHARS)]] = None

class RecallRequest(BaseModel):
    query: Annotated[str, StringConstraints(max_length=MAX_QUERY_CHARS)]
    scope: Optional[str] = "global"
    kinds: Optional[List[str]] = None
    k: int = Field(8, ge=1, le=MAX_K)
    include_superseded: bool = False

class ContextPackRequest(BaseModel):
    scope: str = "global"
    query: Optional[Annotated[str, StringConstraints(max_length=MAX_QUERY_CHARS)]] = None
    k: int = Field(8, ge=1, le=MAX_K)

class ProfileRequest(BaseModel):
    key: str
    value: str

class ForgetRequest(BaseModel):
    id: int
    reason: str = ""

class SkillSourceRequest(BaseModel):
    url: Annotated[str, StringConstraints(max_length=300)]

# Served at exactly /mcp; auth is enforced below, so Host-header rebinding checks are off.
mcp_app = mcp.streamable_http_app(
    streamable_http_path="/mcp",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    backfill_task = asyncio.create_task(backfill_loop())
    try:
        async with mcp_app.router.lifespan_context(mcp_app):
            yield
    finally:
        backfill_task.cancel()
        try:
            await backfill_task
        except asyncio.CancelledError:
            pass
        await embedder.aclose()


app = FastAPI(title="Memory Core API", version="1.0.0", lifespan=lifespan,
              docs_url=None, redoc_url=None, openapi_url=None)


@app.exception_handler(httpx.HTTPError)
@app.exception_handler(EmbeddingError)
async def embedding_unavailable(_: Request, exc: Exception):
    return JSONResponse({"detail": "embedding service unavailable"}, status_code=503)


@app.exception_handler(SupersedeError)
async def supersede_conflict(_: Request, exc: SupersedeError):
    return JSONResponse({"detail": str(exc)}, status_code=409)

async def verify_token(authorization: Optional[str] = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Bearer token"
        )
    token = authorization.split(" ")[1]
    if token not in TOKENS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized token"
        )
    current_agent.set(TOKENS[token])
    current_source.set("rest")
    return TOKENS[token]

@app.get("/health")
def health():
    return {"status": "ok", "service": "memory-core"}

@app.get("/api/v1/profile")
def get_profile_api(agent: str = Depends(verify_token)):
    return db.get_profile()

@app.post("/api/v1/profile")
def set_profile_api(req: ProfileRequest, agent: str = Depends(verify_token)):
    db.set_profile(req.key, req.value)
    return {"status": "ok", "key": req.key}

@app.post("/api/v1/remember")
async def remember_api(req: RememberRequest, agent: str = Depends(verify_token)):
    facts = [redact_secrets(f) for f in req.facts]
    embedding_failed = False
    try:
        embeddings = await embedder.embed_passages(facts)
    except (httpx.HTTPError, EmbeddingError) as exc:
        log.warning("Embedding failed during remember_api, saving with pending embeddings: %s", exc)
        embeddings = [None] * len(facts)
        embedding_failed = True

    res = await asyncio.to_thread(
        db.remember,
        facts=facts,
        embeddings=embeddings,
        kind=req.kind,
        scope=req.scope,
        supersedes=req.supersedes,
        tags=req.tags,
        source=source_ref(req.source),
        created_by=agent
    )
    if embedding_failed:
        res["embedding_pending"] = True
    return res

@app.post("/api/v1/recall")
async def recall_api(req: RecallRequest, agent: str = Depends(verify_token)):
    query = redact_secrets(req.query)
    q_emb = await embed_query_or_none(query)
    results = await asyncio.to_thread(
        db.recall,
        query=query,
        query_embedding=q_emb,
        scope=req.scope,
        kinds=req.kinds,
        k=req.k,
        include_superseded=req.include_superseded
    )
    return {"results": results, "count": len(results),
            "newest_in_scope": await asyncio.to_thread(db.newest_in_scope, req.scope)}


@app.post("/api/v1/context-pack")
async def context_pack_api(req: ContextPackRequest, agent: str = Depends(verify_token)):
    return await build_context_pack(req.scope, req.query, req.k)


@app.get("/api/v1/delta")
def delta_api(since: str, scope: str = "all", agent: str = Depends(verify_token)):
    try:
        return db.delta(since=since, scope=scope)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/api/v1/forget")
def forget_api(req: ForgetRequest, agent: str = Depends(verify_token)):
    res = db.forget(req.id, req.reason)
    return {"id": req.id, "retired": res}

@app.get("/api/v1/browse")
def browse_api(scope: Optional[str] = None, kind: Optional[str] = None, state: str = "active",
               q: Optional[str] = None, limit: int = 25, offset: int = 0, agent: str = Depends(verify_token)):
    page = db.browse(scope=scope, kind=kind, state=state, q=q, limit=min(max(limit, 1), 200), offset=max(offset, 0))
    hits = db.hit_map()
    for item in page["items"]:
        item.update(hits.get(item["id"], {"hits": 0, "last_recalled": None}))
    return page


@app.get("/api/v1/stats")
def stats_api(agent: str = Depends(verify_token)):
    return db.stats()


@app.get("/api/v1/usage")
def usage_api(days: int = 30, agent: str = Depends(verify_token)):
    return db.usage_stats(days=min(max(days, 1), 365))


@app.get("/api/v1/events")
def events_api(type: Optional[str] = None, agent_name: Optional[str] = None, limit: int = 50, offset: int = 0,
               agent: str = Depends(verify_token)):
    return db.events(type_=type, agent=agent_name, limit=min(max(limit, 1), 200), offset=max(offset, 0))


@app.get("/api/v1/memory/{memory_id}")
def memory_api(memory_id: int, agent: str = Depends(verify_token)):
    detail = db.memory_detail(memory_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return detail


# --- Skill sources (GitHub repos indexed as archived skills) ---

def _skill_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, skills.SourceNotFound):
        return HTTPException(status_code=404, detail=f"skill source not found: {exc}")
    return HTTPException(status_code=502, detail=str(exc))


SKILL_ERRORS = (ValueError, skills.SourceNotFound, skills.SkillSourceError)


@app.get("/api/v1/skills/sources")
def skill_sources_api(agent: str = Depends(verify_token)):
    return {"sources": skills.list_sources(db)}


@app.post("/api/v1/skills/sources")
async def add_skill_source_api(req: SkillSourceRequest, agent: str = Depends(verify_token)):
    try:
        return await skills.add_source(db, embedder, req.url, agent)
    except SKILL_ERRORS as exc:
        raise _skill_error(exc)


@app.post("/api/v1/skills/sources/{slug}/update")
async def update_skill_source_api(slug: str, agent: str = Depends(verify_token)):
    try:
        return await skills.update_source(db, embedder, slug, agent)
    except SKILL_ERRORS as exc:
        raise _skill_error(exc)


@app.delete("/api/v1/skills/sources/{slug}")
async def remove_skill_source_api(slug: str, agent: str = Depends(verify_token)):
    try:
        return await skills.remove_source(db, slug)
    except SKILL_ERRORS as exc:
        raise _skill_error(exc)


@app.get("/api/v1/skills/bundle")
async def skill_bundle_api(path: str, agent: str = Depends(verify_token)):
    """A .tar.gz of one skill folder; extract it into ~/.cache/memory-core/skills."""
    try:
        skill_dir = skills.resolve_skill_dir(path)
    except SKILL_ERRORS as exc:
        raise _skill_error(exc)
    out = await asyncio.to_thread(skills.build_bundle, skill_dir)
    return FileResponse(out, media_type="application/gzip", filename="skill.tar.gz",
                        background=BackgroundTask(os.unlink, out))

# Public shell; every data call from the page still needs a Bearer token.
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return (Path(__file__).parent / "dashboard.html").read_text()


def bearer_guarded(inner):
    """ASGI wrapper that rejects requests without a valid Bearer token."""
    async def asgi(scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            auth = headers.get(b"authorization", b"").decode()
            token = auth[7:] if auth.startswith("Bearer ") else None
            if not token:
                import urllib.parse
                qs = urllib.parse.parse_qs(scope.get("query_string", b"").decode())
                token = qs.get("token", [None])[0]
            if not (token and token in TOKENS):
                resp = JSONResponse({"detail": "Missing or invalid Bearer token"}, status_code=401)
                await resp(scope, receive, send)
                return
            current_agent.set(TOKENS[token])
            current_source.set("mcp")
        await inner(scope, receive, send)
    return asgi


# Mounted last so the REST routes above take precedence; MCP lives at /mcp.
app.mount("/", bearer_guarded(mcp_app))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=HOST, port=PORT, reload=False)
