"""
Ingestion Pipeline Runner Script
Orchestrates parsing of raw financial transaction logs and regulatory PDFs,
generates semantic embeddings locally via HuggingFace, and persists chunks into ChromaDB.
"""

from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

from src.ingestion.loader import parse_swift_log_file, parse_regulatory_pdf
from src.ingestion.vector_store import VectorStoreManager

load_dotenv()


def main():
    print("🚀 Starting FinGuard Ingestion Pipeline...")

    project_root = Path(__file__).resolve().parent.parent.parent
    swift_path = project_root / "data" / "processed" / "swift_transactions.txt"
    pdf_path = project_root / "data" / "raw" / "finra_rule_3310.pdf"

    # Step 0: Instantiate the local HuggingFace Embedding Model once
    print("🧠 Loading local embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    documents = []

    # Step 1: Parse SWIFT Transaction Logs (Delimiter-based)
    if swift_path.exists():
        swift_docs = parse_swift_log_file(str(swift_path))
        documents.extend(swift_docs)
    else:
        print(f"⚠️ Warning: SWIFT file not found at {swift_path}")

    # Step 2: Parse Regulatory PDF (Semantic Cosine Distance Chunking)
    if pdf_path.exists():
        pdf_docs = parse_regulatory_pdf(str(pdf_path), embeddings=embeddings)
        documents.extend(pdf_docs)
    else:
        print(f"⚠️ Warning: PDF file not found at {pdf_path}")

    if not documents:
        print("❌ No documents loaded. Exiting pipeline.")
        return

    # Step 3: Persist into ChromaDB
    print("\n📦 Persisting documents into ChromaDB...")
    manager = VectorStoreManager(
        persist_directory=str(project_root / "data" / "chroma"),
        collection_name="finguard_knowledge_base",
        embeddings=embeddings,
    )
    
    manager.add_documents(documents)

    # Step 4: Run Sanity Verification Query
    print("\n🔍 Running Sanity Retrieval Test...")
    test_query = "What are the requirements for Anti-Money Laundering compliance monitoring?"
    results = manager.similarity_search_with_filter(
        query=test_query,
        k=2,
        doc_type="regulatory_pdf",
    )

    print(f"\nFound {len(results)} matching chunks for test query:")
    for idx, doc in enumerate(results, 1):
        print(f"\n--- Result {idx} (Source: {doc.metadata.get('source')}, Doc Type: {doc.metadata.get('doc_type')}) ---")
        print(doc.page_content[:250] + "...")

    print("\n✅ Ingestion Pipeline Complete!")


if __name__ == "__main__":
    main()