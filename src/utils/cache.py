import json
import os
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional

import numpy as np


VALID_CACHE_MODES = frozenset({"disabled", "memory", "redis"})
DEFAULT_MEMORY_CACHE_MAX_ENTRIES = 128

_IN_MEMORY_CACHE: OrderedDict[str, Dict[str, Any]] = OrderedDict()
_CACHE_LOCK = threading.RLock()
_MODEL = None
_MODEL_LOCK = threading.Lock()
_REDIS_CLIENT = None
_REDIS_LOCK = threading.Lock()


def get_cache_mode() -> str:
    """Return the explicitly configured cache mode."""
    mode = os.getenv("FINGUARD_CACHE_MODE", "memory").strip().lower()
    if mode not in VALID_CACHE_MODES:
        supported = ", ".join(sorted(VALID_CACHE_MODES))
        raise RuntimeError(
            f"Invalid FINGUARD_CACHE_MODE '{mode}'. Expected one of: {supported}."
        )
    return mode


def _memory_cache_max_entries() -> int:
    raw_value = os.getenv(
        "FINGUARD_MEMORY_CACHE_MAX_ENTRIES",
        str(DEFAULT_MEMORY_CACHE_MAX_ENTRIES),
    )
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            "FINGUARD_MEMORY_CACHE_MAX_ENTRIES must be a positive integer."
        ) from exc
    if value < 1:
        raise RuntimeError(
            "FINGUARD_MEMORY_CACHE_MAX_ENTRIES must be a positive integer."
        )
    return value


def _local_models_only() -> bool:
    return os.getenv("FINGUARD_MODEL_LOCAL_ONLY", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _get_embedding_model():
    """Load the semantic-cache model once, only when caching needs it."""
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                from sentence_transformers import SentenceTransformer

                _MODEL = SentenceTransformer(
                    "all-MiniLM-L6-v2",
                    local_files_only=_local_models_only(),
                )
    return _MODEL


def _validate_local_embedding_asset() -> None:
    """Check the semantic-cache model cache without loading or downloading it."""
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError("cache_embedding_model_unavailable") from exc


def _encode_query(query: str) -> list[float]:
    return _get_embedding_model().encode(query).tolist()


def _get_redis_client():
    """Create a Redis client only when Redis mode is explicitly used."""
    global _REDIS_CLIENT
    if _REDIS_CLIENT is None:
        with _REDIS_LOCK:
            if _REDIS_CLIENT is None:
                import redis

                _REDIS_CLIENT = redis.Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", "6379")),
                    db=int(os.getenv("REDIS_DB", "0")),
                    socket_connect_timeout=1,
                )
    return _REDIS_CLIENT


def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """Compute cosine similarity between two one-dimensional vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_semantic_cache(
    query: str,
    threshold: float = 0.80,
) -> Optional[Dict[str, Any]]:
    """Return a semantically similar final report when caching is enabled."""
    mode = get_cache_mode()
    if mode == "disabled":
        return None

    if mode == "redis":
        query_vector = _encode_query(query)
        try:
            results = _get_redis_client().ft("idx:cache").search(query_vector)
            if results.docs and float(results.docs[0].score) >= threshold:
                return json.loads(results.docs[0].compliance_report)
        except Exception:
            return None
        return None

    with _CACHE_LOCK:
        if not _IN_MEMORY_CACHE:
            return None

    query_vector = _encode_query(query)
    with _CACHE_LOCK:
        for cached_query, entry in list(_IN_MEMORY_CACHE.items()):
            similarity = cosine_similarity(query_vector, entry["vector"])
            if similarity >= threshold:
                _IN_MEMORY_CACHE.move_to_end(cached_query)
                return entry["report"]
    return None


def set_semantic_cache(query: str, report: Dict[str, Any]) -> None:
    """Store final report data when the configured cache mode supports it."""
    mode = get_cache_mode()
    if mode == "disabled":
        return

    query_vector = _encode_query(query)

    if mode == "redis":
        try:
            _get_redis_client().hset(
                f"audit:{hash(query)}",
                mapping={
                    "query": query,
                    "vector": json.dumps(query_vector),
                    "compliance_report": json.dumps(report),
                },
            )
        except Exception:
            pass
        return

    with _CACHE_LOCK:
        _IN_MEMORY_CACHE.pop(query, None)
        _IN_MEMORY_CACHE[query] = {
            "vector": query_vector,
            "report": report,
        }
        while len(_IN_MEMORY_CACHE) > _memory_cache_max_entries():
            _IN_MEMORY_CACHE.popitem(last=False)


def validate_cache_readiness() -> Dict[str, Any]:
    """Validate cache configuration without loading embedding models."""
    mode = get_cache_mode()
    if mode == "memory":
        _memory_cache_max_entries()
        try:
            _validate_local_embedding_asset()
        except RuntimeError:
            return {
                "ready": False,
                "mode": mode,
                "reason": "cache_embedding_model_unavailable",
            }
        return {"ready": True, "mode": mode}
    if mode == "disabled":
        return {"ready": True, "mode": mode}

    try:
        _get_redis_client().ping()
    except Exception:
        return {"ready": False, "mode": mode, "reason": "redis_unavailable"}
    return {"ready": True, "mode": mode}
