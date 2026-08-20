import pytest
from langchain_core.documents import Document
from src.ingestion.vector_store import VectorStoreManager


def test_generate_deterministic_id():
    manager = VectorStoreManager(persist_directory="data/test_chroma", collection_name="test_coll")
    
    doc1 = Document(
        page_content="Test SWIFT record",
        metadata={"doc_type": "swift_log", "source": "test.txt", "record_id": "001"}
    )
    doc2 = Document(
        page_content="Test SWIFT record",
        metadata={"doc_type": "swift_log", "source": "test.txt", "record_id": "001"}
    )

    id1 = manager._generate_deterministic_id(doc1, 0)
    id2 = manager._generate_deterministic_id(doc2, 0)

    # Assert that identical metadata produces the exact same hash ID (Idempotency)
    assert id1 == id2
    assert len(id1) == 32  # MD5 hex digest length