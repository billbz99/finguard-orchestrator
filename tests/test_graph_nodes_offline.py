from copy import deepcopy

import pytest

from src.graph import nodes
from src.graph.nodes import aml_audit_node, extraction_node
from src.graph.schemas import AMLAssessment, TransactionExtraction


class FakeStructuredLLM:
    def __init__(self, response, calls):
        self.response = response
        self.calls = calls

    def invoke(self, prompt):
        self.calls["prompt"] = prompt
        return self.response


class FakeLLM:
    def __init__(self, expected_schema, response, calls):
        self.expected_schema = expected_schema
        self.response = response
        self.calls = calls

    def with_structured_output(self, schema):
        assert schema is self.expected_schema
        self.calls["schema"] = schema
        return FakeStructuredLLM(self.response, self.calls)


class FakeRetriever:
    def __init__(self, results, calls):
        self.results = results
        self.calls = calls

    def retrieve(self, **kwargs):
        self.calls["retrieve"] = kwargs
        return list(self.results)


def install_fake_llm(monkeypatch, schema, response):
    calls = {}
    monkeypatch.setattr(
        nodes,
        "get_llm",
        lambda: FakeLLM(schema, response, calls),
    )
    return calls


def install_fake_retriever(monkeypatch, results):
    calls = {}

    def build_retriever(**kwargs):
        calls["init"] = kwargs
        return FakeRetriever(results, calls)

    monkeypatch.setattr(nodes, "FinGuardRetriever", build_retriever)
    return calls


def test_extraction_node_propagates_structured_transaction_fields(monkeypatch):
    extraction = TransactionExtraction(
        transaction_ids=["TXN-100", "TXN-101"],
        amount=8500.0,
        transaction_type="wire",
        regulations=[],
        suspected_patterns=[],
        jurisdiction=None,
    )
    calls = install_fake_llm(monkeypatch, TransactionExtraction, extraction)
    state = {
        "raw_query": "Review SWIFT wire TXN-100 and TXN-101 for processing.",
        "loop_count": 7,
        "unrelated": {"keep": True},
    }
    original_state = deepcopy(state)

    update = extraction_node(state)

    assert update == {
        "doc_type": "swift_log",
        "jurisdiction": None,
        "extracted_entities": extraction.model_dump(),
    }
    assert update["extracted_entities"]["transaction_ids"] == ["TXN-100", "TXN-101"]
    assert update["extracted_entities"]["regulations"] == []
    assert update["extracted_entities"]["suspected_patterns"] == []
    assert state == original_state
    assert "loop_count" not in update
    assert "unrelated" not in update
    assert state["raw_query"] in calls["prompt"]


@pytest.mark.parametrize(
    ("query", "expected_doc_type", "expected_jurisdiction"),
    [
        ("Inspect SWIFT message TXN-200.", "swift_log", None),
        ("Summarize FINRA Rule 3310 requirements.", "regulatory_pdf", None),
        (
            "Audit wire TXN-300 for AML concerns under FINRA Rule 3310.",
            None,
            None,
        ),
        ("Review an OFAC screening alert.", None, "US_OFAC"),
    ],
)
def test_extraction_node_classifies_scope_and_jurisdiction(
    monkeypatch,
    query,
    expected_doc_type,
    expected_jurisdiction,
):
    extraction = TransactionExtraction()
    install_fake_llm(monkeypatch, TransactionExtraction, extraction)

    update = extraction_node({"raw_query": query})

    assert update["doc_type"] == expected_doc_type
    assert update["jurisdiction"] == expected_jurisdiction
    assert update["extracted_entities"] == {
        "transaction_ids": [],
        "amount": None,
        "transaction_type": None,
        "regulations": [],
        "suspected_patterns": [],
        "jurisdiction": None,
    }


@pytest.mark.parametrize(
    ("query", "expected_jurisdiction"),
    [
        ("Review this OFAC screening alert.", "US_OFAC"),
        ("Review this transaction under US jurisdiction.", "US_OFAC"),
        ("Review this transaction under U.S. jurisdiction.", "US_OFAC"),
        ("Review this transaction in the United States.", "US_OFAC"),
        ("Review suspicious transaction activity.", None),
        ("Review unusual transaction activity.", None),
        ("Review the customer transaction history.", None),
        ("The business requested an account review.", None),
        ("Review this transaction for compliance.", None),
    ],
)
def test_extraction_node_detects_only_explicit_us_jurisdiction(
    monkeypatch,
    query,
    expected_jurisdiction,
):
    install_fake_llm(monkeypatch, TransactionExtraction, TransactionExtraction())

    update = extraction_node({"raw_query": query})

    assert update["jurisdiction"] == expected_jurisdiction


def test_extraction_node_does_not_invent_fields_missing_from_llm_response(monkeypatch):
    extraction = TransactionExtraction(transaction_ids=["TXN-400"])
    install_fake_llm(monkeypatch, TransactionExtraction, extraction)

    update = extraction_node(
        {"raw_query": "Inspect transaction TXN-400 without additional details."}
    )

    assert update["extracted_entities"] == {
        "transaction_ids": ["TXN-400"],
        "amount": None,
        "transaction_type": None,
        "regulations": [],
        "suspected_patterns": [],
        "jurisdiction": None,
    }
    assert update["jurisdiction"] is None


def aml_assessment(*, insufficient_evidence=False):
    return AMLAssessment(
        risk_rating="Low" if insufficient_evidence else "High",
        suspicious_patterns=[] if insufficient_evidence else ["structuring"],
        flagged_transactions=[] if insufficient_evidence else ["TXN-500"],
        applicable_regulations=[] if insufficient_evidence else ["FINRA Rule 3310"],
        reasoning_summary=(
            "Transaction evidence is insufficient."
            if insufficient_evidence
            else "Retrieved evidence supports additional review."
        ),
        insufficient_evidence=insufficient_evidence,
    )


def test_aml_audit_retrieves_filters_context_and_propagates_assessment(monkeypatch):
    accepted = {
        "id": "accepted",
        "content": "FINRA Rule 3310 requires an AML monitoring program.",
        "metadata": {"source": "finra.pdf", "doc_type": "regulatory_pdf"},
        "rerank_score": 0.15,
    }
    high = {
        "id": "high",
        "content": "Structuring can involve transactions below reporting thresholds.",
        "metadata": {"source": "fincen.pdf", "section": "structuring"},
        "rerank_score": 0.91,
    }
    rejected = {
        "id": "rejected",
        "content": "Irrelevant low-scoring material must not reach the LLM.",
        "metadata": {"source": "irrelevant.pdf"},
        "rerank_score": 0.149,
    }
    retriever_calls = install_fake_retriever(monkeypatch, [accepted, rejected, high])
    assessment = aml_assessment()
    llm_calls = install_fake_llm(monkeypatch, AMLAssessment, assessment)
    state = {
        "raw_query": "Audit TXN-500 for structuring.",
        "doc_type": "regulatory_pdf",
        "jurisdiction": "US_OFAC",
        "extracted_entities": {"transaction_ids": ["TXN-500"]},
        "loop_count": 0,
        "unrelated": ["preserve", "me"],
    }
    original_state = deepcopy(state)

    update = aml_audit_node(state)

    assert retriever_calls["init"] == {
        "chroma_path": "./data/chroma",
        "collection_name": "finguard_knowledge_base",
        "reranker_model": "BAAI/bge-reranker-large",
    }
    assert retriever_calls["retrieve"] == {
        "query": state["raw_query"],
        "doc_type": "regulatory_pdf",
        "jurisdiction": "US_OFAC",
        "top_k_vector": 10,
        "top_n_final": 3,
    }
    assert update["retrieved_context"] == [accepted, high]
    assert update["retrieved_context"][0]["metadata"] == accepted["metadata"]
    assert update["aml_assessment"] == assessment.model_dump()
    assert accepted["content"] in llm_calls["prompt"]
    assert high["content"] in llm_calls["prompt"]
    assert rejected["content"] not in llm_calls["prompt"]
    assert state["raw_query"] in llm_calls["prompt"]
    assert '"TXN-500"' in llm_calls["prompt"]
    assert "audit request and extracted entities as evidence" in llm_calls["prompt"]
    assert "retrieved context as evidence about" in llm_calls["prompt"]
    assert "Regulatory documents do not need to" in llm_calls["prompt"]
    assert "A missing field is not automatically fatal" in llm_calls["prompt"]
    assert "supported LOW/no-indicator conclusion" in llm_calls["prompt"]
    normalized_prompt = " ".join(llm_calls["prompt"].split())
    assert "Assert an applicable regulation only when" in normalized_prompt
    assert "does not support finalizing the requested AML conclusion" in normalized_prompt
    assert "do not confirm illegal activity" in llm_calls["prompt"]
    assert state == original_state
    assert "unrelated" not in update


def test_aml_audit_handles_empty_retrieval_and_insufficient_evidence(monkeypatch):
    retriever_calls = install_fake_retriever(monkeypatch, [])
    assessment = aml_assessment(insufficient_evidence=True)
    llm_calls = install_fake_llm(monkeypatch, AMLAssessment, assessment)
    state = {
        "raw_query": "Review TXN-600.",
        "doc_type": None,
        "jurisdiction": None,
        "extracted_entities": {"transaction_ids": ["TXN-600"]},
        "loop_count": 0,
    }

    update = aml_audit_node(state)

    assert retriever_calls["retrieve"]["doc_type"] is None
    assert retriever_calls["retrieve"]["jurisdiction"] is None
    assert update["retrieved_context"] == []
    assert update["aml_assessment"]["insufficient_evidence"] is True
    assert update["aml_assessment"]["flagged_transactions"] == []
    context_section = llm_calls["prompt"].split("Retrieved regulatory context:", 1)[1]
    assert context_section.strip() == ""


def test_aml_audit_refinement_query_appends_context_without_replacing_facts(monkeypatch):
    retriever_calls = install_fake_retriever(monkeypatch, [])
    assessment = aml_assessment(insufficient_evidence=True)
    llm_calls = install_fake_llm(monkeypatch, AMLAssessment, assessment)
    state = {
        "raw_query": "Audit wire TXN-700 amount $9,500 from ACME Bank.",
        "doc_type": None,
        "jurisdiction": None,
        "extracted_entities": {
            "transaction_ids": ["TXN-700"],
            "amount": 9500.0,
        },
        "loop_count": 1,
        "unrelated": "unchanged",
    }
    original_state = deepcopy(state)

    update = aml_audit_node(state)

    refined_query = retriever_calls["retrieve"]["query"]
    assert refined_query.startswith(state["raw_query"])
    assert "TXN-700 amount $9,500 from ACME Bank" in refined_query
    assert "FINRA Rule 3310 structuring" in refined_query
    assert "Currency Transaction Reporting thresholds" in refined_query
    assert state["raw_query"] in llm_calls["prompt"]
    assert '"amount": 9500.0' in llm_calls["prompt"]
    assert update["aml_assessment"] == assessment.model_dump()
    assert state == original_state
