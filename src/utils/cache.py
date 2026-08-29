# src/utils/cache.py

import json
from typing import Any, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import redis
    r = redis.Redis(host="localhost", port=6379, db=0, socket_connect_timeout=1)
    r.ping()
    REDIS_AVAILABLE = True
except Exception:
    REDIS_AVAILABLE = False
    _IN_MEMORY_CACHE = {}

# Embedding model for semantic similarity calculation
_model = SentenceTransformer("all-MiniLM-L6-v2")


def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """Computes cosine similarity between two 1D vector arrays."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_semantic_cache(query: str, threshold: float = 0.80) -> Optional[Dict[str, Any]]:
    """Checks cache for semantically similar previous audit reports."""
    query_vector = _model.encode(query).tolist()

    if REDIS_AVAILABLE:
        try:
            results = r.ft("idx:cache").search(query_vector)
            if results.docs and float(results.docs[0].score) >= threshold:
                return json.loads(results.docs[0].compliance_report)
        except Exception:
            pass
    else:
        for cached_query, entry in _IN_MEMORY_CACHE.items():
            sim = cosine_similarity(query_vector, entry["vector"])
            if sim >= threshold:
                print(f"🟢 [Semantic Cache HIT] Match: '{cached_query}' (Similarity: {sim:.4f})")
                return entry["report"]
            else:
                print(f"ℹ️ [Cache Compare] Similarity with '{cached_query}': {sim:.4f} (Threshold: {threshold})")

    print("🔴 [Semantic Cache MISS] Proceeding to live audit execution.")
    return None


def set_semantic_cache(query: str, report: Dict[str, Any]):
    """Stores generated compliance report with its semantic query embedding."""
    query_vector = _model.encode(query).tolist()

    if REDIS_AVAILABLE:
        try:
            r.hset(
                f"audit:{hash(query)}",
                mapping={
                    "query": query,
                    "vector": json.dumps(query_vector),
                    "compliance_report": json.dumps(report),
                },
            )
        except Exception:
            pass
    else:
        _IN_MEMORY_CACHE[query] = {
            "vector": query_vector,
            "report": report,
        }
        print(f"💾 [Semantic Cache Stored] In-memory cache entries: {len(_IN_MEMORY_CACHE)}")


if __name__ == "__main__":
    print("🧪 Testing Semantic Cache System...\n")

    dummy_report = {
        "assessment_status": "COMPLETE",
        "risk_rating": "HIGH",
        "flagged_wires": ["TXN-984211-X"],
        "applicable_regulations": ["FINRA Rule 3310"],
        "audit_summary": "Structuring under $10,000 threshold detected.",
        "source_document_hashes": ["finra_rule_3310.pdf"],
    }

    # Store initial record
    original_query = "Audit wire TXN-984211-X for structuring under FINRA Rule 3310."
    set_semantic_cache(original_query, dummy_report)

    # Test 1: Near-identical phrasing (Should HIT with threshold 0.80)
    print("\n--- Test 1: Near-Identical Phrasing ---")
    test_query_hit = "Check wire TXN-984211-X for structuring under FINRA 3310 regulations."
    hit_result = get_semantic_cache(test_query_hit, threshold=0.80)

    # Test 2: Unrelated query (Should MISS)
    print("\n--- Test 2: Unrelated Query ---")
    test_query_miss = "What is the capital requirement for margin debt accounts?"
    miss_result = get_semantic_cache(test_query_miss, threshold=0.80)
