"""Populate only the model assets required by the production image."""

from pathlib import Path

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from huggingface_hub import snapshot_download


MODELS = (
    (
        "sentence-transformers/all-MiniLM-L6-v2",
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        (
            "1_Pooling/config.json",
            "config_sentence_transformers.json",
            "config.json",
            "model.safetensors",
            "modules.json",
            "sentence_bert_config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.txt",
        ),
    ),
    (
        "BAAI/bge-reranker-large",
        "55611d7bca2a7133960a6d3b71e083071bbfc312",
        (
            "config.json",
            "model.safetensors",
            "sentencepiece.bpe.model",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
        ),
    ),
)


def record_main_revision(snapshot_path: str, revision: str) -> None:
    """Make a pinned snapshot resolvable by local-only model-id lookups."""
    model_cache = Path(snapshot_path).parents[1]
    refs_dir = model_cache / "refs"
    refs_dir.mkdir(exist_ok=True)
    (refs_dir / "main").write_text(revision, encoding="utf-8")


def main() -> None:
    for model_id, revision, required_files in MODELS:
        snapshot_path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            allow_patterns=list(required_files),
        )
        record_main_revision(snapshot_path, revision)

    # Chroma's ONNX cache is independent of Hugging Face. A real embedding call
    # downloads, verifies, extracts, and loads the exact asset used at query time.
    DefaultEmbeddingFunction()(["FinGuard runtime asset verification"])


if __name__ == "__main__":
    main()
