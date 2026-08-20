# tests/test_retrieval.py

from pathlib import Path
from src.ingestion.retriever import FinGuardRetriever


def test_end_to_end_retrieval():
    project_root = Path(__file__).resolve().parent.parent
    chroma_dir = str(project_root / "data" / "chroma")

    print("🚀 Initializing FinGuardRetriever...")
    retriever = FinGuardRetriever(
        chroma_path=chroma_dir,
        collection_name="finguard_knowledge_base",
        reranker_model="BAAI/bge-reranker-large"
    )

    query = "What are the rules regarding structuring and CTR threshold evasions?"

    print(f"\n🔍 Executing Hybrid Retrieval for Query: '{query}'")
    results = retriever.retrieve(
        query=query,
        doc_type="regulatory_pdf",
        top_k_vector=10,
        top_n_final=3
    )

    print(f"\nRetrieved {len(results)} reranked results:\n")
    for idx, doc in enumerate(results, 1):
        print(f"--- Rank {idx} [Rerank Score: {doc['rerank_score']:.4f}] ---")
        print(f"ID: {doc['id']}")
        print(f"Source: {doc['metadata'].get('source', 'N/A')}")
        print(f"Content Sample: {doc['content'][:200]}...\n")


if __name__ == "__main__":
    test_end_to_end_retrieval()