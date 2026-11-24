"""Offline retrieval eval for memory-core recall.

Each case pairs a natural query with a substring that a relevant memory must contain.
Reports hit@1, hit@5 and MRR for vector-only, BM25-only and RRF variants, plus the
cosine-similarity spread of relevant vs. irrelevant memories.

Usage: venv/bin/python eval_recall.py
"""
import asyncio
import re
import sqlite3
import statistics
import time

from config import DB_PATH
from database import MemoryDatabase, cosine_similarity, vector_from_blob
from embedder import EmbedderClient

CASES = [
    ("homeserver tailscale IP address", "Tailscale 100.69.70.66"),
    ("kenapa numpy crash di server", "NumPy 2.x crash"),
    ("port berapa memory-core jalan", "port 8420"),
    ("embedding model yang dipakai memory-core", "qwen3-embedding-0.6b dari API"),
    ("apakah memory-core otomatis nyimpen memori", "tidak menangkap otomatis"),
    ("where is the admin token for memory-core stored", "Token admin tersimpan"),
    ("which port does BandarTracker backend use", "BandarTracker runs as a backend"),
    ("caddy reverse proxy bandar domain", "Caddy routes bandar.home"),
    ("how do I open the bili dashboard", "bili-dashboard"),
    ("rtk hook disabled when", "rtk PreToolUse hook skips"),
    ("my linkedin profile link", "linkedin.com/in/kenny-sembiring"),
    ("personal website url", "kennyvws.my.id"),
    ("startup decision pricing model", "global self-serve SaaS"),
    ("can I get paid in dollars from Indonesia", "Polar and Creem"),
    ("best marketplace to sell scrapers for first revenue", "Apify Store (pay-per-event"),
    ("tokopedia scraper actor test result", "tokopedia-search"),
    ("whisper slow on apify container threads", "faster-whisper with the default cpu_threads"),
    ("instagram reel audio without login", "DASH manifest"),
    ("published apify actors", "went public on the Apify Store"),
    ("deadline tugas akhir proposal dan poster", "due Wednesday 25 November 2026"),
    ("nama lengkap untuk dokumen akademik", "Kenny Valent Winalda Sembiring"),
    ("nomor induk mahasiswa", "NIM is"),
    ("siapa dosen pembimbing skripsi", "Winarno"),
    ("which billion-context image version runs on the homeserver", "bili-local:0.1.216"),
    ("gemini cli can't run shell commands headless", "cannot execute shell commands headless"),
    ("remove debug bridge from iOS app", "skill: ios-clean |"),
    ("save my progress so I can resume later", "skill: context-save |"),
    ("bloom and depth of field effects in three.js", "skill: threejs-postprocessing |"),
    ("reverse engineer router firmware for vulnerabilities", "skill: firmware-pentest |"),
    ("analyze a suspicious binary and write YARA rules", "skill: malware-analysis |"),
    ("improve search engine ranking of my site", "skill: seo |"),
    ("reproducible dev environment with nix", "skill: flox-environments |"),
    ("check router config before deploying", "skill: network-config-validation |"),
    ("animation jank optimize gsap", "skill: gsap-performance |"),
    ("spring boot unit tests with mockito", "skill: springboot-tdd |"),
    ("prove it works before saying done", "skill: verification-before-completion |"),
]
VARIANTS = {"vector": (1.0, 0.0), "bm25": (0.0, 1.0), "rrf_1_1": (1.0, 1.0),
            "rrf_2_1": (2.0, 1.0), "rrf_1_2": (1.0, 2.0)}
RRF_K = 60
MISS_RANK = 1000
TOP_N = 50


def rank_lists(conn, query, q_emb):
    rows = conn.execute("SELECT id, content, embedding FROM memories WHERE valid_to IS NULL").fetchall()
    content = {r[0]: r[1] for r in rows}
    sims = sorted(((r[0], cosine_similarity(vector_from_blob(r[2]), q_emb)) for r in rows if r[2]),
                  key=lambda x: x[1], reverse=True)
    terms = [f'"{t}"' for t in re.sub(r"[^\w\s]", " ", query).split()]
    fts = conn.execute("SELECT rowid FROM memories_fts WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?",
                       (" OR ".join(terms), TOP_N)).fetchall() if terms else []
    v_rank = {mid: i + 1 for i, (mid, _) in enumerate(sims[:TOP_N])}
    f_rank = {r[0]: i + 1 for i, r in enumerate(fts) if r[0] in content}
    return content, dict(sims), v_rank, f_rank


def fused(v_rank, f_rank, wv, wf):
    ids = set(v_rank) | set(f_rank)
    score = {m: wv / (RRF_K + v_rank.get(m, MISS_RANK)) + wf / (RRF_K + f_rank.get(m, MISS_RANK)) for m in ids}
    return sorted(ids, key=lambda m: score[m], reverse=True)


async def main():
    emb = EmbedderClient()
    conn = sqlite3.connect(DB_PATH)
    stats = {name: [] for name in VARIANTS}
    gold_sims, best_other_sims, all_sims = [], [], []
    for query, needle in CASES:
        q = await emb.embed_query(query)
        content, sims, v_rank, f_rank = rank_lists(conn, query, q)
        gold = {m for m, c in content.items() if needle in c}
        if not gold:
            print(f"!! no memory contains {needle!r}")
            continue
        gold_sims.append(max(sims.get(m, 0.0) for m in gold))
        best_other_sims.append(max(s for m, s in sims.items() if m not in gold))
        all_sims.extend(sims.values())
        for name, (wv, wf) in VARIANTS.items():
            order = fused(v_rank, f_rank, wv, wf)
            rank = next((i + 1 for i, m in enumerate(order) if m in gold), None)
            stats[name].append(rank)
    n = len(gold_sims)
    print(f"cases: {n}")
    print(f"{'variant':10} hit@1  hit@5  MRR")
    for name, ranks in stats.items():
        h1 = sum(r == 1 for r in ranks if r) / n
        h5 = sum(r <= 5 for r in ranks if r) / n
        mrr = sum(1 / r for r in ranks if r) / n
        print(f"{name:10} {h1:.2f}   {h5:.2f}   {mrr:.3f}")
    print(f"cosine  relevant: mean {statistics.mean(gold_sims):.3f} min {min(gold_sims):.3f} max {max(gold_sims):.3f}")
    print(f"cosine  best irrelevant: mean {statistics.mean(best_other_sims):.3f}")
    print(f"cosine  all memories: mean {statistics.mean(all_sims):.3f} median {statistics.median(all_sims):.3f}")
    db = MemoryDatabase()
    t0 = time.perf_counter()
    db.recall(query=CASES[0][0], query_embedding=q, scope="all", k=8)
    print(f"db.recall latency (no embedding call): {(time.perf_counter() - t0) * 1000:.0f} ms")
    await emb.aclose()


if __name__ == "__main__":
    asyncio.run(main())
