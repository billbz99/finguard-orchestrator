from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
"""
Semantic Grounding & Loader Module
Parses unstructured SWIFT transaction logs and regulatory PDFs into context-aware, metadata-enriched LangChain Document chunks ready for ChromaDB vector storage.
"""

from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
def parse_swift_log_file(file_path: str) -> List[Document]:
    """
    Parses a SWIFT transaction log file using delimeter-aware record boundaries. Guarantees that individual SWIFT block messages remain atomic and intact.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"SWIFT log file not found at: {file_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()
        
    # Split on record delimeters
    raw_records = raw_text.split("--- TRANSACTION RECORD #")
    documents: List[Document] = []
    
    for record in raw_records:
        cleaned_record = record.strip()
        if not cleaned_record:
            continue
        
        # Extract record number header line if present
        lines = cleaned_record.splitlines()
        first_line = lines[0] if lines else ""
        record_id = first_line.split("---")[0].strip() if "---" in first_line else "unknown"
        
        # Construct clean page content restoring full delimiter structure
        full_content = f"--- TRANSACTION RECORD #{cleaned_record}"
        
        doc = Document(
            page_content=full_content,
            metadata={
                "source": path.name,
                "doc_type": "swift_log",
                "record_id": record_id,
            },
        )
        documents.append(doc)
        
    print(
        f"✅ Parsed {len(documents)} atomic SWIFT transaction records from {path.name}"
    )
    return documents
  
def parse_regulatory_pdf(
    file_path: str,
    embeddings: Optional[HuggingFaceEmbeddings] = None,
    breakpoint_threshold_type: str = "percentile",
) -> List[Document]:
    """
    Parses a regulatory PDF file using LangChain's SemanticChunker to split text
    at points of semantic drift based on embedding cosine distance.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found at: {file_path}")

    # Inject default embeddings if not supplied (useful for unit testing dependency injection)
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Step 1: Load raw document using PyPDFLoader
    loader = PyPDFLoader(str(path))
    pages = loader.load()

    if not pages:
        print(f"⚠️ Warning: No text content extracted from {path.name}")
        return []

    # Step 2: Combine page contents to prevent artificial page-boundary splits
    full_pdf_text = "\n\n".join([page.page_content for page in pages])

    # Step 3: Instantiate SemanticChunker with selected threshold type
    semantic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
    )

    # Step 4: Generate semantic text chunks
    raw_chunks = semantic_splitter.split_text(full_pdf_text)

    # Step 5: Convert text chunks into metadata-enriched Document objects
    documents: List[Document] = []
    for idx, chunk_text in enumerate(raw_chunks):
        cleaned_text = chunk_text.strip()
        if not cleaned_text:
            continue

        doc = Document(
            page_content=cleaned_text,
            metadata={
                "source": path.name,
                "doc_type": "regulatory_pdf",
                "chunk_id": idx,
                "total_chunks": len(raw_chunks),
            },
        )
        documents.append(doc)

    print(f"✅ Parsed {len(documents)} semantic chunks from {path.name}")
    return documents
    