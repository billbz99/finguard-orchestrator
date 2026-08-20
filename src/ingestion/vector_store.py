"""
Vector Store Management Module
Handles initialization, deterministic upsert, and metadata-filtered 
retrieval against ChromaDB for the FinGuard Knowledge Base.
"""

import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_chroma import Chroma
#from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStoreManager:
    """
    Manages persistent vector storage operations using ChromaDB and OpenAI Embeddings.
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "finguard_knowledge_base",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embeddings: Optional[HuggingFaceEmbeddings] = None,
    ):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        
        # Ensure the persistence directory exists on disk
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Use injected embeddings or instantiate HuggingFaceEmbeddings
        self.embeddings = embeddings or HuggingFaceEmbeddings(model_name=embedding_model_name)

        # Initialize Chroma vector store with HuggingFace embeddings
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory),
        )

    def _generate_deterministic_id(self, doc: Document, index: int) -> str:
        """
        Helper method: Creates a unique, deterministic ID for a chunk to ensure idempotency.
        
        Example Key Strategy:
        - For SWIFT logs: "swift_log:swift_transactions.txt:record_001"
        - For Regulatory PDFs: "regulatory_pdf:finra_rule_3310.pdf:chunk_0"
        """
        metadata = doc.metadata
        doc_type = metadata.get("doc_type", "unknown")
        source = metadata.get("source", "unknown")
        
        # Fallback to record_id or chunk_id if present, else index
        unique_key = metadata.get("record_id") or metadata.get("chunk_id") or index
        
        raw_identifier = f"{doc_type}:{source}:{unique_key}"
        return hashlib.md5(raw_identifier.encode("utf-8")).hexdigest()

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Upserts a list of documents into ChromaDB with deterministic IDs.
        """
        if not documents:
            print("⚠️ No documents provided to add_documents.")
            return []

        # Generate deterministic IDs for every document chunk
        ids = [self._generate_deterministic_id(doc, idx) for idx, doc in enumerate(documents)]

        # Add / Upsert into ChromaDB
        inserted_ids = self.vector_store.add_documents(documents=documents, ids=ids)
        print(f"✅ Successfully ingested {len(inserted_ids)} chunks into Chroma collection '{self.collection_name}'")
        return inserted_ids

    def similarity_search_with_filter(
        self,
        query: str,
        k: int = 4,
        doc_type: Optional[str] = None,
    ) -> List[Document]:
        """
        Performs vector similarity search with optional metadata filtering.
        """
        filter_dict: Dict[str, Any] = {}
        if doc_type:
            filter_dict["doc_type"] = doc_type

        # Execute search with Chroma metadata filter
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter_dict if filter_dict else None,
        )
        return results

    def get_retriever(self, k: int = 4, doc_type: Optional[str] = None):
        """
        Exposes a LangChain VectorStoreRetriever object for integration into LangGraph nodes.
        """
        search_kwargs: Dict[str, Any] = {"k": k}
        if doc_type:
            search_kwargs["filter"] = {"doc_type": doc_type}

        return self.vector_store.as_retriever(search_kwargs=search_kwargs)