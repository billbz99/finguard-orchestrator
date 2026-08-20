# src/ingestion/retriever.py

import os
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder


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
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        self.reranker = CrossEncoder(reranker_model)

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