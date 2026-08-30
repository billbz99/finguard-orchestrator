import json
from pathlib import Path

import pytest

import deployment.prepare_model_assets as model_assets
from deployment.start_api import prepare_chroma_runtime_copy
from deployment.prepare_model_assets import record_main_revision
from src.graph import nodes
from src.ingestion import retriever


ROOT = Path(__file__).resolve().parents[1]


def test_chroma_path_defaults_to_existing_local_path(monkeypatch):
    monkeypatch.delenv("FINGUARD_CHROMA_PATH", raising=False)
    assert retriever.get_chroma_path() == "./data/chroma"


def test_chroma_path_uses_runtime_configuration(monkeypatch, tmp_path):
    runtime_path = tmp_path / "runtime-chroma"
    monkeypatch.setenv("FINGUARD_CHROMA_PATH", str(runtime_path))
    assert retriever.get_chroma_path() == str(runtime_path)


def test_production_retriever_uses_configured_chroma_path(monkeypatch, tmp_path):
    captured = {}

    class FakeRetriever:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("FINGUARD_CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(nodes, "FinGuardRetriever", FakeRetriever)
    monkeypatch.setattr(nodes, "_RETRIEVER", None)
    monkeypatch.setattr(nodes, "_RETRIEVER_FACTORY", None)

    nodes.get_production_retriever()

    assert captured["chroma_path"] == str(tmp_path / "chroma")
    assert captured["collection_name"] == "finguard_knowledge_base"
    assert captured["reranker_model"] == "BAAI/bge-reranker-large"


def test_runtime_copy_replaces_stale_data_without_mutating_seed(monkeypatch, tmp_path):
    seed = tmp_path / "seed"
    runtime = tmp_path / "runtime"
    seed.mkdir()
    (seed / "chroma.sqlite3").write_bytes(b"validated-seed")
    runtime.mkdir()
    (runtime / "stale").write_text("old", encoding="utf-8")
    monkeypatch.setenv("FINGUARD_CHROMA_SEED_PATH", str(seed))
    monkeypatch.setenv("FINGUARD_CHROMA_PATH", str(runtime))

    result = prepare_chroma_runtime_copy()

    assert result == runtime
    assert (runtime / "chroma.sqlite3").read_bytes() == b"validated-seed"
    assert not (runtime / "stale").exists()
    assert (seed / "chroma.sqlite3").read_bytes() == b"validated-seed"


def test_runtime_copy_requires_seed(monkeypatch, tmp_path):
    monkeypatch.setenv("FINGUARD_CHROMA_SEED_PATH", str(tmp_path / "missing"))
    monkeypatch.setenv("FINGUARD_CHROMA_PATH", str(tmp_path / "runtime"))

    with pytest.raises(RuntimeError, match="seed directory"):
        prepare_chroma_runtime_copy()


def test_model_manifest_pins_all_required_assets():
    manifest = json.loads(
        (ROOT / "deployment" / "model-manifest.json").read_text(encoding="utf-8")
    )
    models = {entry["purpose"]: entry for entry in manifest["models"]}

    assert models["semantic_cache_embeddings"]["identifier"] == (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    assert models["retrieval_reranking"]["identifier"] == "BAAI/bge-reranker-large"
    assert models["chroma_query_embeddings"]["revision"].startswith("sha256:")
    assert all(entry["revision"] for entry in models.values())


def test_pinned_snapshot_is_resolvable_as_local_main(tmp_path):
    revision = "abc123"
    snapshot = tmp_path / "models--example--model" / "snapshots" / revision
    snapshot.mkdir(parents=True)

    record_main_revision(str(snapshot), revision)

    assert (snapshot.parents[1] / "refs" / "main").read_text(encoding="utf-8") == revision


def test_model_downloads_are_pinned_and_limited_to_runtime_files(monkeypatch, tmp_path):
    calls = []

    def fake_snapshot_download(*, repo_id, revision, allow_patterns):
        snapshot = (
            tmp_path
            / f"models--{repo_id.replace('/', '--')}"
            / "snapshots"
            / revision
        )
        snapshot.mkdir(parents=True)
        calls.append((repo_id, revision, tuple(allow_patterns), snapshot))
        return str(snapshot)

    class FakeEmbeddingFunction:
        def __call__(self, documents):
            assert documents == ["FinGuard runtime asset verification"]
            return [[0.0]]

    monkeypatch.setattr(model_assets, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(
        model_assets,
        "DefaultEmbeddingFunction",
        FakeEmbeddingFunction,
    )

    model_assets.main()

    assert len(calls) == 2
    for (model_id, revision, required_files), call in zip(model_assets.MODELS, calls):
        assert call[:3] == (model_id, revision, required_files)
        assert "model.safetensors" in required_files
        assert not any("pytorch_model" in path for path in required_files)
        assert not any(path.startswith(("onnx/", "openvino/")) for path in required_files)
        assert (call[3].parents[1] / "refs" / "main").read_text(
            encoding="utf-8"
        ) == revision


def test_dockerfile_enforces_locked_offline_single_worker_runtime():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "uv sync --frozen --no-dev --no-install-project" in dockerfile
    assert "FINGUARD_MODEL_LOCAL_ONLY=1" in dockerfile
    assert "HF_HUB_OFFLINE=1" in dockerfile
    assert "TRANSFORMERS_OFFLINE=1" in dockerfile
    assert "ANONYMIZED_TELEMETRY=FALSE" in dockerfile
    assert "USER finguard" in dockerfile
    assert "EXPOSE 8000" in dockerfile
    assert "deployment/start_api.py" in dockerfile
    assert "COPY --chown=finguard:finguard data/chroma" in dockerfile
    assert (
        "COPY --from=builder --chown=finguard:finguard "
        "/opt/finguard/home/.cache/chroma /opt/finguard/home/.cache/chroma"
        in dockerfile
    )
    assert "COPY --from=builder --chown=finguard:finguard /opt/finguard/home " not in dockerfile
    assert "COPY . ." not in dockerfile


def test_dockerignore_excludes_sensitive_inputs_but_keeps_seed():
    patterns = {
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }

    assert {".env", ".env.*", "!.env.example", ".venv/", ".git/"} <= patterns
    assert {"data/raw/", "data/test_chroma/", "artifacts/evaluations/"} <= patterns
    assert ".cache/" in patterns
    assert "data/chroma/" not in patterns
