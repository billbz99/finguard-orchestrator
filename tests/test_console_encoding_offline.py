import pytest

from src.graph.pre_router import route_incoming_audit
from src.graph.schemas import AMLAssessment, CriticAssessment, TransactionExtraction
from src.graph.workflow import should_continue_audit
from tests.test_graph_integration_offline import (
    chunk,
    initial_state,
    run_graph,
)


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


@pytest.fixture
def cp1252_print(monkeypatch):
    messages = []

    def strict_print(*values, sep=" ", end="\n", **kwargs):
        message = sep.join(str(value) for value in values) + end
        message.encode("cp1252", errors="strict")
        messages.append(message)

    monkeypatch.setattr("builtins.print", strict_print)
    return messages


def test_compiled_graph_diagnostics_are_cp1252_safe(monkeypatch, cp1252_print):
    extraction = TransactionExtraction(
        transaction_ids=["TXN-ENC-001"],
        amount=1250.0,
        transaction_type="国際送金",
        suspected_patterns=["activité inhabituelle"],
    )
    assessment = AMLAssessment(
        risk_rating="低",
        suspicious_patterns=["activité inhabituelle"],
        flagged_transactions=[],
        applicable_regulations=[],
        required_evidence_gaps=[],
        reasoning_summary="顧客 evidence supports an ordinary payment.",
        insufficient_evidence=False,
    )
    critic = CriticAssessment(
        is_sufficient=True,
        missing_evidence=["追加資料なし"],
        failure_type="NONE",
        recommended_action="GENERATE",
        critique="Evidence is sufficient.",
    )

    result, _, _ = run_graph(
        monkeypatch,
        llm_responses={
            TransactionExtraction: [extraction],
            AMLAssessment: [assessment],
            CriticAssessment: [critic],
        },
        retrieval_batches=[[chunk("synthetic_source.txt")]],
        state=initial_state("Review wire TXN-ENC-001 for processing."),
    )

    diagnostic_output = "".join(cp1252_print)
    assert result["final_report"] is not None
    assert "[Extraction Node]" in diagnostic_output
    assert "[AML Audit Node]" in diagnostic_output
    assert "[AML Reasoning]" in diagnostic_output
    assert "[Auditor Critic]" in diagnostic_output
    assert "[Generation Node]" in diagnostic_output
    assert "[Routing]" in diagnostic_output


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"critic_assessment": {"recommended_action": "RETRIEVE_MORE"}}, "refine"),
        ({"critic_assessment": {"recommended_action": "STOP_INSUFFICIENT"}}, "generate"),
        ({"critic_assessment": {"recommended_action": "GENERATE"}}, "generate"),
    ],
)
def test_all_routing_diagnostics_are_cp1252_safe(cp1252_print, state, expected):
    assert should_continue_audit(state) == expected


@pytest.mark.parametrize(
    "query",
    [
        "Process a routine local ACH payment.",
        "Audit wire TXN-ENC-002 for structuring under FINRA Rule 3310.",
    ],
)
def test_pre_router_diagnostics_are_cp1252_safe(cp1252_print, query):
    route_incoming_audit(query, amount=100.0, is_cross_border=False)
