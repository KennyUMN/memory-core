import asyncio
import os
from pathlib import Path
import tempfile
DB_PATH = Path(tempfile.mkdtemp()) / "test_memory.db"
from embedder import EmbedderClient
from database import MemoryDatabase

async def run_tests():
    print("=== Testing Memory Core (Pure Python / No NumPy) ===")
    
    # Fresh test DB
    if DB_PATH.exists():
        DB_PATH.unlink()

    embedder = EmbedderClient()
    db = MemoryDatabase(DB_PATH)

    # 1. Profile test
    print("\n1. Testing Profile...")
    db.set_profile("owner", "Kenny")
    db.set_profile("active_projects", "BandarTracker, umn-elearning-assistant, rakazo")
    profile = db.get_profile()
    assert profile.get("owner") == "Kenny"
    print(" Profile test passed:", profile)

    # 2. Remember test
    print("\n2. Testing Remember...")
    facts = [
        "BandarTracker adalah tool analisis transaksi bandar saham Indonesia.",
        "BandarTracker berjalan sebagai service backend di port 8765.",
        "Caddy me-route bandar.home ke port 8765."
    ]
    embs = await embedder.embed_passages(facts)
    res = db.remember(
        facts=facts,
        embeddings=embs,
        kind="fact",
        scope="project:bandar",
        created_by="test-runner"
    )
    print(" Saved IDs:", res["saved_ids"])
    assert len(res["saved_ids"]) == 3

    # 3. Duplicate check test
    print("\n3. Testing Duplicate Detection...")
    dup_fact = ["BandarTracker jalan sebagai service backend di port 8765."]
    dup_emb = await embedder.embed_passages(dup_fact)
    dup_res = db.remember(
        facts=dup_fact,
        embeddings=dup_emb,
        kind="fact",
        scope="project:bandar",
        created_by="test-runner"
    )
    print(" Near-duplicates detected:", dup_res["near_duplicates"])
    assert len(dup_res["near_duplicates"]) > 0
    print(f" Detected near-duplicate with similarity: {dup_res['near_duplicates'][0]['similarity']}")

    # 4. Recall test
    print("\n4. Testing Recall (Hybrid BM25 + Vector + RRF)...")
    q = "Di port berapa BandarTracker berjalan?"
    q_emb = await embedder.embed_query(q)
    recall_res = db.recall(
        query=q,
        query_embedding=q_emb,
        scope="project:bandar",
        k=3
    )
    print(" Top recall results:")
    for r in recall_res:
        print(f"  - [RRF={r['rrf_score']}] (cos_sim={r['similarity']}) {r['content']}")
    assert len(recall_res) > 0
    assert "8765" in recall_res[0]["content"]

    # 5. Forget test
    print("\n5. Testing Forget...")
    to_delete_id = res["saved_ids"][0]
    success = db.forget(to_delete_id, reason="testing forget")
    assert success
    recalled_ids = [r["id"] for r in db.recall("analisis transaksi bandar", q_emb, scope="project:bandar")]
    assert to_delete_id not in recalled_ids
    print(f" Memory ID {to_delete_id} successfully retired.")

    print("\n=== All Tests Passed Successfully! ===")

if __name__ == "__main__":
    asyncio.run(run_tests())
