# src/ingestion/retriever.py

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.utils import embedding_functions


class RuntimeAssetError(RuntimeError):
    """Raised when required local retrieval assets are unavailable."""


def _local_models_only() -> bool:
    return os.getenv("FINGUARD_MODEL_LOCAL_ONLY", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def get_chroma_path() -> str:
    """Return the configured runtime Chroma path with the local default."""
    return os.getenv("FINGUARD_CHROMA_PATH", "./data/chroma")


def _validate_local_model_assets(reranker_model: str) -> None:
    """Check model caches without downloading or loading model weights."""
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(repo_id=reranker_model, local_files_only=True)
    except Exception as exc:
        raise RuntimeAssetError("reranker_model_unavailable") from exc

    onnx_dir = (
        Path.home()
        / ".cache"
        / "chroma"
        / "onnx_models"
        / "all-MiniLM-L6-v2"
        / "onnx"
    )
    required_onnx_files = {
        "config.json",
        "model.onnx",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt",
    }
    if not all((onnx_dir / filename).is_file() for filename in required_onnx_files):
        raise RuntimeAssetError("embedding_model_unavailable")


def validate_retrieval_assets(
    chroma_path: Optional[str] = None,
    collection_name: str = "finguard_knowledge_base",
    reranker_model: str = "BAAI/bge-reranker-large",
) -> Dict[str, Any]:
    """Validate serving assets without creating collections or loading models."""
    path = Path(chroma_path or get_chroma_path())
    if not path.is_dir():
        raise RuntimeAssetError("chroma_path_unavailable")

    try:
        client = chromadb.PersistentClient(path=str(path))
        collection = client.get_collection(name=collection_name)
        document_count = collection.count()
    except Exception as exc:
        raise RuntimeAssetError("chroma_collection_unavailable") from exc

    if document_count < 1:
        raise RuntimeAssetError("chroma_collection_empty")

    _validate_local_model_assets(reranker_model)
    return {"document_count": document_count}


class FinGuardRetriever:
    """Hybrid Retrieval Logic for FinGuard Orchestrator.
    Combines ChromaDB vector retrieval with metadata pre-filtering 
    and Cross-Encoder reranking.
    """
    def __init__(
        self, 
        chroma_path: str = "./finguard_chroma_db",
        collection_name: str = "finguard_grounding",
        reranker_model: str = "BAAI/bge-reranker-large"
    ):
        path = Path(chroma_path)
        if not path.is_dir():
            raise RuntimeAssetError("chroma_path_unavailable")

        try:
            self.chroma_client = chromadb.PersistentClient(path=str(path))
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
            self.collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_fn,
            )
        except Exception as exc:
            raise RuntimeAssetError("chroma_collection_unavailable") from exc

        if self.collection.count() < 1:
            raise RuntimeAssetError("chroma_collection_empty")

        if _local_models_only():
            _validate_local_model_assets(reranker_model)

        try:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(
                reranker_model,
                local_files_only=_local_models_only(),
            )
        except Exception as exc:
            raise RuntimeAssetError("reranker_model_unavailable") from exc

    def _build_where_clause(
        self, 
        doc_type: Optional[str] = None, 
        entity_bic: Optional[str] = None,
        jurisdiction: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        conditions = []
        if doc_type:
            conditions.append({"doc_type": {"$eq": doc_type}})
        if entity_bic:
            conditions.append({"entity_bic": {"$eq": entity_bic}})
        if jurisdiction:
            conditions.append({"jurisdiction": {"$eq": jurisdiction}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def retrieve(
        self, 
        query: str, 
        top_k_vector: int = 30, 
        top_n_final: int = 5,
        doc_type: Optional[str] = None,
        entity_bic: Optional[str] = None,
        jurisdiction: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        where_filter = self._build_where_clause(
            doc_type=doc_type, 
            entity_bic=entity_bic, 
            jurisdiction=jurisdiction
        )

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k_vector,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        ids = results["ids"][0] if results["ids"] else []

        if not documents:
            return []

        sentence_pairs = [[query, doc] for doc in documents]
        rerank_scores = self.reranker.predict(sentence_pairs)

        candidate_pool = []
        for i in range(len(documents)):
            candidate_pool.append({
                "id": ids[i],
                "content": documents[i],
                "metadata": metadatas[i],
                "rerank_score": float(rerank_scores[i])
            })

        candidate_pool.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidate_pool[:top_n_final]
