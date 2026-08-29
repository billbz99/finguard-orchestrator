from collections import defaultdict, deque

import pytest

from src.graph import nodes
from src.graph.schemas import (
    AMLAssessment,
    ComplianceReport,
    CriticAssessment,
    TransactionExtraction,
)
from src.graph.workflow import build_finguard_graph


class FakeStructuredLLM:
    def __init__(self, controller, schema):
        self.controller = controller
        self.schema = schema

    def invoke(self, prompt):
        self.controller.prompts[self.schema].append(prompt)
        return self.controller.responses[self.schema].popleft()


class FakeLLM:
    def __init__(self, controller):
        self.controller = controller

    def with_structured_output(self, schema):
        assert schema in self.controller.responses
        return FakeStructuredLLM(self.controller, schema)


class FakeLLMController:
    def __init__(self, responses):
        self.responses = {
            schema: deque(schema_responses)
            for schema, schema_responses in responses.items()
        }
        self.prompts = defaultdict(list)

    def get_llm(self):
        return FakeLLM(self)


class FakeRetrieverController:
    def __init__(self, result_batches):
        self.result_batches = deque(result_batches)
        self.init_calls = []
        self.retrieve_calls = []

    def build(self, **kwargs):
        self.init_calls.append(kwargs)
        return self

    def retrieve(self, **kwargs):
        self.retrieve_calls.append(kwargs)
        return list(self.result_batches.popleft())


@pytest.fixture(autouse=True)
def disable_remote_tracing(monkeypatch):
    for variable in (
        "LANGCHAIN_TRACING",
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_API_KEY",
        "LANGSMITH_TRACING",
    ):
        monkeypatch.delenv(variable, raising=False)


def initial_state(query, *, max_loops=2):
    return {
        "raw_query": query,
        "doc_type": None,
        "jurisdiction": None,
        "extracted_entities": {},
        "retrieved_context": [],
        "aml_assessment": None,
        "critic_assessment": None,
        "compliance_draft": None,
        "confidence_score": 0.0,
        "loop_count": 0,
        "max_loops": max_loops,
        "is_audit_complete": False,
        "final_report": None,
    }


def extraction(transaction_id="TXN-900"):
    return TransactionExtraction(
        transaction_ids=[transaction_id],
        amount=8500.0,
        transaction_type="wire",
        regulations=["FINRA Rule 3310"],
        suspected_patterns=["structuring"],
    )


def assessment(
    *,
    risk="High",
    transaction_id="TXN-900",
    insufficient=False,
    summary="Evidence supports a structuring concern.",
):
    return AMLAssessment(
        risk_rating=risk,
        suspicious_patterns=[] if insufficient else ["structuring"],
        flagged_transactions=[] if insufficient else [transaction_id],
        applicable_regulations=[] if insufficient else ["FINRA Rule 3310"],
        reasoning_summary=summary,
        insufficient_evidence=insufficient,
    )


def critic(action, *, failure_type="NONE", sufficient=None):
    return CriticAssessment(
        is_sufficient=(action == "GENERATE" if sufficient is None else sufficient),
        missing_evidence=[] if action == "GENERATE" else ["additional evidence"],
        failure_type=failure_type,
        recommended_action=action,
        critique=f"Critic selected {action}.",
    )


def chunk(source, content="Applicable AML regulatory guidance.", score=0.9):
    return {
        "id": source,
        "content": content,
        "metadata": {"source": source, "doc_type": "regulatory_pdf"},
        "rerank_score": score,
    }


def run_graph(monkeypatch, *, llm_responses, retrieval_batches, state):
    llm = FakeLLMController(llm_responses)
    retriever = FakeRetrieverController(retrieval_batches)
    monkeypatch.setattr(nodes, "get_llm", llm.get_llm)
    monkeypatch.setattr(nodes, "FinGuardRetriever", retriever.build)

    graph = build_finguard_graph()
    result = graph.invoke(state)

    return result, llm, retriever


def test_graph_sufficient_evidence_generates_without_retry(monkeypatch):
    extracted = extraction()
    aml_result = assessment()
    result, llm, retriever = run_graph(
        monkeypatch,
        llm_responses={
            TransactionExtraction: [extracted],
            AMLAssessment: [aml_result],
            CriticAssessment: [critic("GENERATE")],
        },
        retrieval_batches=[[chunk("finra_rule_3310.pdf")]],
        state=initial_state(
            "Audit wire TXN-900 for structuring under FINRA Rule 3310."
        ),
    )

    assert result["extracted_entities"] == extracted.model_dump()
    assert result["aml_assessment"] == aml_result.model_dump()
    assert result["critic_assessment"]["recommended_action"] == "GENERATE"
    assert result["loop_count"] == 1
    assert result["is_audit_complete"] is True
    assert len(retriever.retrieve_calls) == 1
    assert len(llm.prompts[AMLAssessment]) == 1
    assert result["compliance_draft"] == aml_result.reasoning_summary
    report = ComplianceReport.model_validate(result["final_report"])
    assert report.assessment_status == "COMPLETE"
    assert report.risk_rating == "HIGH"
    assert report.flagged_wires == ["TXN-900"]
    assert report.applicable_regulations == ["FINRA Rule 3310"]
    assert report.source_document_hashes == ["finra_rule_3310.pdf"]


def test_graph_missing_transaction_evidence_stops_without_retry(monkeypatch):
    extracted = TransactionExtraction(transaction_ids=["TXN-901"])
    aml_result = assessment(
        risk="Low",
        transaction_id="TXN-901",
        insufficient=True,
        summary="Transaction facts are insufficient for an AML conclusion.",
    )
    result, _, retriever = run_graph(
        monkeypatch,
        llm_responses={
            TransactionExtraction: [extracted],
            AMLAssessment: [aml_result],
            CriticAssessment: [
                critic(
                    "STOP_INSUFFICIENT",
                    failure_type="MISSING_TRANSACTION_DATA",
                    sufficient=False,
                )
            ],
        },
        retrieval_batches=[[chunk("finra_rule_3310.pdf")]],
        state=initial_state("Review wire TXN-901 for suspicious activity."),
    )

    assert len(retriever.retrieve_calls) == 1
    assert result["critic_assessment"]["failure_type"] == "MISSING_TRANSACTION_DATA"
    assert result["critic_assessment"]["recommended_action"] == "STOP_INSUFFICIENT"
    assert result["loop_count"] == 1
    assert result["is_audit_complete"] is True
    assert result["aml_assessment"]["insufficient_evidence"] is True
    assert result["compliance_draft"] == aml_result.reasoning_summary
    report = ComplianceReport.model_validate(result["final_report"])
    assert report.assessment_status == "INSUFFICIENT_EVIDENCE"
    assert report.risk_rating == "LOW"
    assert report.flagged_wires == []
    assert report.applicable_regulations == []
    assert report.audit_summary == aml_result.reasoning_summary


def test_graph_performs_exactly_one_retrieval_refinement_cycle(monkeypatch):
    first_assessment = assessment(
        risk="Low",
        insufficient=True,
        summary="More regulatory context is required.",
    )
    second_assessment = assessment(summary="Refined context supports review.")
    result, llm, retriever = run_graph(
        monkeypatch,
        llm_responses={
            TransactionExtraction: [extraction("TXN-902")],
            AMLAssessment: [first_assessment, second_assessment],
            CriticAssessment: [
                critic(
                    "RETRIEVE_MORE",
                    failure_type="MISSING_REGULATORY_CONTEXT",
                    sufficient=False,
                ),
                critic("GENERATE"),
            ],
        },
        retrieval_batches=[
            [chunk("initial_guidance.pdf")],
            [chunk("refined_guidance.pdf", "Refined FINRA structuring guidance.")],
        ],
        state=initial_state(
            "Audit wire TXN-902 amount $8,500 for structuring.",
            max_loops=2,
        ),
    )

    assert len(retriever.retrieve_calls) == 2
    assert len(llm.prompts[AMLAssessment]) == 2
    assert len(llm.prompts[CriticAssessment]) == 2
    assert retriever.retrieve_calls[0]["query"] == (
        "Audit wire TXN-902 amount $8,500 for structuring."
    )
    refined_query = retriever.retrieve_calls[1]["query"]
    assert refined_query.startswith(retriever.retrieve_calls[0]["query"])
    assert "TXN-902 amount $8,500" in refined_query
    assert "FINRA Rule 3310 structuring" in refined_query
    assert result["retrieved_context"] == [
        chunk("refined_guidance.pdf", "Refined FINRA structuring guidance.")
    ]
    assert result["aml_assessment"] == second_assessment.model_dump()
    assert result["critic_assessment"]["recommended_action"] == "GENERATE"
    assert result["loop_count"] == 2
    assert result["is_audit_complete"] is True
    assert result["final_report"] is not None
    assert result["final_report"]["assessment_status"] == "COMPLETE"
    assert result["final_report"]["source_document_hashes"] == [
        "refined_guidance.pdf"
    ]


def test_graph_enforces_max_loops_for_repeated_retrieve_more(monkeypatch):
    first_assessment = assessment(
        risk="Low",
        insufficient=True,
        summary="Regulatory evidence remains incomplete.",
    )
    second_assessment = assessment(
        risk="Low",
        insufficient=True,
        summary="Regulatory evidence is still incomplete.",
    )
    result, llm, retriever = run_graph(
        monkeypatch,
        llm_responses={
            TransactionExtraction: [extraction("TXN-903")],
            AMLAssessment: [first_assessment, second_assessment],
            CriticAssessment: [
                critic(
                    "RETRIEVE_MORE",
                    failure_type="MISSING_REGULATORY_CONTEXT",
                    sufficient=False,
                ),
                critic(
                    "RETRIEVE_MORE",
                    failure_type="MISSING_REGULATORY_CONTEXT",
                    sufficient=False,
                ),
            ],
        },
        retrieval_batches=[
            [chunk("first_attempt.pdf")],
            [chunk("second_attempt.pdf")],
        ],
        state=initial_state("Audit wire TXN-903.", max_loops=2),
    )

    assert len(retriever.retrieve_calls) == 2
    assert len(llm.prompts[CriticAssessment]) == 2
    assert result["loop_count"] == 2
    assert result["is_audit_complete"] is True
    assert result["critic_assessment"]["recommended_action"] == "STOP_INSUFFICIENT"
    assert result["critic_assessment"]["failure_type"] == (
        "MISSING_REGULATORY_CONTEXT"
    )
    assert result["critic_assessment"]["critique"].endswith(
        "Maximum refinement loops reached."
    )
    assert result["aml_assessment"] == second_assessment.model_dump()
    report = ComplianceReport.model_validate(result["final_report"])
    assert report.assessment_status == "INSUFFICIENT_EVIDENCE"
