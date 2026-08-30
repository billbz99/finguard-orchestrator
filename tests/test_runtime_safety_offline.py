import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.graph import nodes
from src.graph.schemas import AMLAssessment
from src.ingestion import retriever as retriever_module
from src.ingestion.retriever import RuntimeAssetError
from src.utils import cache


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeStructuredLLM:
    def invoke(self, prompt):
        return AMLAssessment(
            risk_rating="Low",
            suspicious_patterns=[],
            flagged_transactions=[],
            applicable_regulations=[],
            required_evidence_gaps=[],
            reasoning_summary="Offline assessment.",
            insufficient_evidence=False,
        )


class FakeLLM:
    def with_structured_output(self, schema):
        assert schema is AMLAssessment
        return FakeStructuredLLM()


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        return []


def test_dockerignore_excludes_sensitive_context_but_keeps_runtime_chroma():
    rules = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")

    for required_rule in (
        ".env",
        ".venv/",
        ".git/",
        "artifacts/evaluations/",
        "data/raw/",
        "data/test_chroma/",
        ".pytest_cache/",
        "*.log",
    ):
        assert required_rule in rules

    assert "data/chroma/" not in {
        line.strip()
        for line in rules.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_cache_import_is_lazy_and_does_not_create_redis_client(monkeypatch):
    monkeypatch.delenv("FINGUARD_CACHE_MODE", raising=False)

    reloaded = importlib.reload(cache)

    assert reloaded.get_cache_mode() == "memory"
    assert reloaded._MODEL is None
    assert reloaded._REDIS_CLIENT is None


def test_disabled_cache_does_not_load_embedding_model(monkeypatch):
    monkeypatch.setenv("FINGUARD_CACHE_MODE", "disabled")
    monkeypatch.setattr(
        cache,
        "_get_embedding_model",
        lambda: pytest.fail("embedding model loaded"),
    )

    assert cache.get_semantic_cache("query") is None
    cache.set_semantic_cache("query", {"assessment_status": "COMPLETE"})


def test_memory_cache_is_bounded_and_evicts_oldest_entry(monkeypatch):
    monkeypatch.setenv("FINGUARD_CACHE_MODE", "memory")
    monkeypatch.setenv("FINGUARD_MEMORY_CACHE_MAX_ENTRIES", "2")
    monkeypatch.setattr(cache, "_encode_query", lambda query: [float(len(query))])
    cache._IN_MEMORY_CACHE.clear()

    cache.set_semantic_cache("a", {"value": 1})
    cache.set_semantic_cache("bb", {"value": 2})
    cache.set_semantic_cache("ccc", {"value": 3})

    assert list(cache._IN_MEMORY_CACHE) == ["bb", "ccc"]


def test_invalid_cache_mode_fails_clearly(monkeypatch):
    monkeypatch.setenv("FINGUARD_CACHE_MODE", "unknown")

    with pytest.raises(RuntimeError, match="Invalid FINGUARD_CACHE_MODE"):
        cache.get_cache_mode()


def test_memory_cache_readiness_reports_missing_model_without_loading_it(
    monkeypatch,
):
    monkeypatch.setenv("FINGUARD_CACHE_MODE", "memory")
    monkeypatch.setattr(
        cache,
        "_validate_local_embedding_asset",
        lambda: (_ for _ in ()).throw(
            RuntimeError("cache_embedding_model_unavailable")
        ),
    )
    monkeypatch.setattr(
        cache,
        "_get_embedding_model",
        lambda: pytest.fail("embedding model loaded during readiness"),
    )

    assert cache.validate_cache_readiness() == {
        "ready": False,
        "mode": "memory",
        "reason": "cache_embedding_model_unavailable",
    }


def test_retriever_is_lazy_and_reused_across_aml_passes(monkeypatch):
    constructed = []

    def factory(**kwargs):
        constructed.append(kwargs)
        return FakeRetriever()

    monkeypatch.setattr(nodes, "FinGuardRetriever", factory)
    monkeypatch.setattr(nodes, "get_llm", lambda: FakeLLM())
    nodes._RETRIEVER = None
    nodes._RETRIEVER_FACTORY = None

    state = {
        "raw_query": "Review synthetic transaction.",
        "extracted_entities": {},
        "loop_count": 0,
    }
    nodes.aml_audit_node(state)
    nodes.aml_audit_node({**state, "loop_count": 1})

    assert len(constructed) == 1
    assert len(nodes._RETRIEVER.calls) == 2
    assert constructed[0] == {
        "chroma_path": "./data/chroma",
        "collection_name": "finguard_knowledge_base",
        "reranker_model": "BAAI/bge-reranker-large",
    }


def test_retrieval_readiness_does_not_create_collection(tmp_path, monkeypatch):
    collection = type("Collection", (), {"count": lambda self: 3})()

    class FakeClient:
        def get_collection(self, *, name):
            assert name == "finguard_knowledge_base"
            return collection

        def get_or_create_collection(self, **kwargs):
            pytest.fail("readiness attempted to create a collection")

    monkeypatch.setattr(
        retriever_module.chromadb,
        "PersistentClient",
        lambda **kwargs: FakeClient(),
    )
    monkeypatch.setattr(
        retriever_module,
        "_validate_local_model_assets",
        lambda model: None,
    )

    result = retriever_module.validate_retrieval_assets(str(tmp_path))

    assert result == {"document_count": 3}


@pytest.mark.parametrize(
    ("path_exists", "count", "expected_reason"),
    [
        (False, None, "chroma_path_unavailable"),
        (True, 0, "chroma_collection_empty"),
    ],
)
def test_retrieval_readiness_rejects_missing_or_empty_index(
    tmp_path,
    monkeypatch,
    path_exists,
    count,
    expected_reason,
):
    path = tmp_path / "chroma"
    if path_exists:
        path.mkdir()
        collection = type("Collection", (), {"count": lambda self: count})()
        client = type(
            "Client",
            (),
            {"get_collection": lambda self, *, name: collection},
        )()
        monkeypatch.setattr(
            retriever_module.chromadb,
            "PersistentClient",
            lambda **kwargs: client,
        )

    with pytest.raises(RuntimeAssetError, match=expected_reason):
        retriever_module.validate_retrieval_assets(str(path))


def test_importing_main_does_not_initialize_models_redis_or_retriever():
    environment = os.environ.copy()
    environment["FINGUARD_CACHE_MODE"] = "disabled"
    command = (
        "import src.main; "
        "import src.graph.nodes as n; "
        "import src.utils.cache as c; "
        "assert n._RETRIEVER is None; "
        "assert c._MODEL is None; "
        "assert c._REDIS_CLIENT is None"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert completed.returncode == 0, completed.stderr


def test_liveness_and_readiness_are_distinct(monkeypatch):
    from src import main

    monkeypatch.setenv("XAI_API_KEY", "configured-for-offline-test")
    monkeypatch.setattr(
        main,
        "validate_retrieval_assets",
        lambda: {"document_count": 102},
    )
    monkeypatch.setattr(
        main,
        "validate_cache_readiness",
        lambda: {"ready": True, "mode": "memory"},
    )
    client = TestClient(main.app)

    assert client.get("/health").status_code == 200
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["checks"]["retrieval_document_count"] == 102


def test_readiness_reports_missing_assets_without_internal_paths(monkeypatch):
    from src import main

    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(
        main,
        "validate_retrieval_assets",
        lambda: (_ for _ in ()).throw(
            RuntimeAssetError("chroma_collection_unavailable")
        ),
    )
    monkeypatch.setattr(
        main,
        "validate_cache_readiness",
        lambda: {"ready": True, "mode": "disabled"},
    )
    response = TestClient(main.app).get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["checks"]["retrieval_reason"] == (
        "chroma_collection_unavailable"
    )
    assert str(PROJECT_ROOT) not in response.text


def test_audit_error_response_does_not_expose_raw_exception(monkeypatch, caplog):
    from src import main

    class FailingGraph:
        async def ainvoke(self, state, config):
            raise RuntimeError("provider secret SENTINEL-RAW-ERROR")

    monkeypatch.setattr(main, "graph", FailingGraph())
    monkeypatch.setattr(main, "get_semantic_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "set_semantic_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "route_incoming_audit", lambda **kwargs: "AGENTIC_GRAPH")

    response = TestClient(main.app).post(
        "/api/v1/audit",
        json={"query": "Audit synthetic transaction.", "audit_id": "safe-id"},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Audit execution failed."}
    assert "SENTINEL-RAW-ERROR" not in response.text
    assert "SENTINEL-RAW-ERROR" not in caplog.text
    assert "RuntimeError" in caplog.text
