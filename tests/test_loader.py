import pytest
from pathlib import Path
from langchain_core.documents import Document
from src.ingestion.loader import parse_swift_log_file, parse_regulatory_pdf

def test_parse_swift_log_file(tmp_path: Path):
    # Setup: Create a temporary mock SWIFT log file
    sample_swift_content = (
        "--- TRANSACTION RECORD #001 ---\n"
        ":20:TRX10001\n"
        ":32A:240315USD50000,\n"
        "--- TRANSACTION RECORD #002 ---\n"
        ":20:TRX10002\n"
        ":32A:240315USD75000,\n"
    )
    test_file = tmp_path / "mock_swift.txt"
    test_file.write_text(sample_swift_content, encoding="utf-8")

    # Execute
    docs = parse_swift_log_file(str(test_file))

    # Assertions
    assert len(docs) == 2
    assert docs[0].metadata["doc_type"] == "swift_log"
    assert docs[0].metadata["record_id"] == "001"
    assert ":20:TRX10001" in docs[0].page_content
    assert docs[1].metadata["record_id"] == "002"

@pytest.mark.integration
def test_parse_regulatory_pdf_file_not_found():
    with pytest.raises(FileNotFoundError):
        parse_regulatory_pdf("non_existent_file.pdf")