import pytest
from pydantic import ValidationError

from src.graph import nodes
from src.graph.nodes import auditor_critic_node, structured_generation_node
from src.graph.schemas import (
    AMLAssessment,
    ComplianceReport,
    CriticAssessment,
    TransactionExtraction,
)
from src.graph.workflow import should_continue_audit


class FakeStructuredLLM:
    def __init__(self, response: CriticAssessment):
        self.response = response
        self.prompt = None

    def invoke(self, prompt: str) -> CriticAssessment:
        assert "independent critic" in prompt
        self.prompt = prompt
        return self.response


class FakeLLM:
    def __init__(self, response: CriticAssessment):
        self.response = response
        self.structured_llm = FakeStructuredLLM(response)

    def with_structured_output(self, schema):
        assert schema is CriticAssessment
        return self.structured_llm


def critic_response(action: str) -> CriticAssessment:
    return CriticAssessment(
        is_sufficient=action == "GENERATE",
        missing_evidence=[] if action == "GENERATE" else ["supporting evidence"],
        failure_type="NONE" if action == "GENERATE" else "MISSING_REGULATORY_CONTEXT",
        recommended_action=action,
        critique=f"Critic selected {action}.",
    )


@pytest.mark.parametrize(
    ("action", "expected_route"),
    [
        ("GENERATE", "generate"),
        ("RETRIEVE_MORE", "refine"),
        ("STOP_INSUFFICIENT", "generate"),
    ],
)
def test_should_continue_audit_routes_supported_actions(action, expected_route):
    state = {"critic_assessment": {"recommended_action": action}}

    assert should_continue_audit(state) == expected_route


def test_should_continue_audit_defaults_to_generate_when_critic_is_missing():
    assert should_continue_audit({}) == "generate"


@pytest.mark.parametrize(
    ("action", "expected_complete"),
    [
        ("GENERATE", True),
        ("RETRIEVE_MORE", False),
        ("STOP_INSUFFICIENT", True),
    ],
)
def test_auditor_critic_updates_state(monkeypatch, action, expected_complete):
    monkeypatch.setattr(nodes, "get_llm", lambda: FakeLLM(critic_response(action)))
    state = {
        "raw_query": "Audit TXN-100.",
        "extracted_entities": {"transaction_ids": ["TXN-100"]},
        "aml_assessment": {
            "risk_rating": "Low",
            "required_evidence_gaps": (
                []
                if action == "GENERATE"
                else [
                    "REGULATORY_CONTEXT"
                    if action == "RETRIEVE_MORE"
                    else "AMOUNT"
                ]
            ),
        },
        "loop_count": 0,
        "max_loops": 3,
    }

    update = auditor_critic_node(state)

    assert update["critic_assessment"]["recommended_action"] == action
    assert update["loop_count"] == 1
    assert update["is_audit_complete"] is expected_complete


def test_auditor_critic_stops_retrieval_at_max_loops(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "get_llm",
        lambda: FakeLLM(critic_response("RETRIEVE_MORE")),
    )
    state = {
        "raw_query": "Audit TXN-100.",
        "extracted_entities": {"transaction_ids": ["TXN-100"]},
        "aml_assessment": {
            "risk_rating": "Low",
            "required_evidence_gaps": ["REGULATORY_CONTEXT"],
        },
        "loop_count": 1,
        "max_loops": 2,
    }

    update = auditor_critic_node(state)

    assert update["critic_assessment"]["recommended_action"] == "STOP_INSUFFICIENT"
    assert update["critic_assessment"]["critique"].endswith(
        "Maximum refinement loops reached."
    )
    assert update["loop_count"] == 2
    assert update["is_audit_complete"] is True


def test_auditor_critic_prompt_uses_conclusion_relative_sufficiency(monkeypatch):
    fake_llm = FakeLLM(critic_response("GENERATE"))
    monkeypatch.setattr(nodes, "get_llm", lambda: fake_llm)

    auditor_critic_node(
        {
            "raw_query": "Review benign wire TXN-100 without a jurisdiction.",
            "extracted_entities": {"transaction_ids": ["TXN-100"]},
            "aml_assessment": {
                "risk_rating": "Low",
                "required_evidence_gaps": [],
                "insufficient_evidence": False,
            },
            "loop_count": 0,
            "max_loops": 2,
        }
    )

    prompt = fake_llm.structured_llm.prompt
    assert "not a universal mandatory checklist" in prompt
    assert "supported LOW/no-indicator conclusion" in prompt
    assert "regulatory documents do not need to repeat transaction facts" in prompt
    assert "insufficient_evidence=true" in prompt
    assert "Your critique does not rewrite the AML assessment" in prompt


@pytest.mark.parametrize(
    ("gaps", "expected_failure", "expected_action"),
    [
        ([], "NONE", "GENERATE"),
        (["AMOUNT"], "MISSING_TRANSACTION_DATA", "STOP_INSUFFICIENT"),
        (
            ["JURISDICTION"],
            "MISSING_TRANSACTION_DATA",
            "STOP_INSUFFICIENT",
        ),
        (
            ["MATERIAL_CONFLICT"],
            "INCONSISTENT_ANALYSIS",
            "STOP_INSUFFICIENT",
        ),
        (
            ["REGULATORY_CONTEXT"],
            "MISSING_REGULATORY_CONTEXT",
            "RETRIEVE_MORE",
        ),
    ],
)
def test_auditor_critic_enforces_typed_evidence_policy(
    monkeypatch,
    gaps,
    expected_failure,
    expected_action,
):
    monkeypatch.setattr(
        nodes,
        "get_llm",
        lambda: FakeLLM(critic_response("STOP_INSUFFICIENT")),
    )

    update = auditor_critic_node(
        {
            "raw_query": "Review synthetic wire TXN-100.",
            "extracted_entities": {"transaction_ids": ["TXN-100"]},
            "aml_assessment": {
                "risk_rating": "Low",
                "required_evidence_gaps": gaps,
            },
            "loop_count": 0,
            "max_loops": 3,
        }
    )

    critic = update["critic_assessment"]
    assert critic["failure_type"] == expected_failure
    assert critic["recommended_action"] == expected_action


def test_auditor_critic_preserves_typed_inconsistent_analysis(monkeypatch):
    response = CriticAssessment(
        is_sufficient=False,
        missing_evidence=[],
        failure_type="INCONSISTENT_ANALYSIS",
        recommended_action="STOP_INSUFFICIENT",
        critique="Assessment contradicts the supplied evidence.",
    )
    monkeypatch.setattr(nodes, "get_llm", lambda: FakeLLM(response))

    update = auditor_critic_node(
        {
            "raw_query": "Review synthetic wire TXN-100.",
            "extracted_entities": {"transaction_ids": ["TXN-100"]},
            "aml_assessment": {
                "risk_rating": "Low",
                "required_evidence_gaps": [],
            },
            "loop_count": 0,
            "max_loops": 2,
        }
    )

    critic = update["critic_assessment"]
    assert critic["failure_type"] == "INCONSISTENT_ANALYSIS"
    assert critic["recommended_action"] == "STOP_INSUFFICIENT"


def test_structured_generation_builds_normalized_deduplicated_report():
    state = {
        "aml_assessment": {
            "risk_rating": "High",
            "flagged_transactions": ["TXN-100", "TXN-200"],
            "applicable_regulations": ["FINRA Rule 3310", "31 U.S.C. 5324"],
            "reasoning_summary": "Evidence supports a structuring concern.",
            "insufficient_evidence": False,
        },
        "retrieved_context": [
            {"metadata": {"source": "finra_rule_3310.pdf"}},
            {"metadata": {"source": "fincen_advisory.pdf"}},
            {"metadata": {"source": "finra_rule_3310.pdf"}},
        ],
        "critic_assessment": {"recommended_action": "GENERATE"},
    }

    update = structured_generation_node(state)
    report = update["final_report"]

    assert set(report) == {
        "assessment_status",
        "risk_rating",
        "flagged_wires",
        "applicable_regulations",
        "audit_summary",
        "source_document_hashes",
    }
    assert report["assessment_status"] == "COMPLETE"
    assert report["risk_rating"] == "HIGH"
    assert report["flagged_wires"] == ["TXN-100", "TXN-200"]
    assert report["applicable_regulations"] == [
        "FINRA Rule 3310",
        "31 U.S.C. 5324",
    ]
    assert report["audit_summary"] == "Evidence supports a structuring concern."
    assert set(report["source_document_hashes"]) == {
        "finra_rule_3310.pdf",
        "fincen_advisory.pdf",
    }
    assert update["compliance_draft"] == report["audit_summary"]
    assert ComplianceReport.model_validate(report).model_dump() == report


def test_structured_generation_handles_missing_optional_state():
    update = structured_generation_node({})

    assert update == {
        "final_report": {
            "assessment_status": "INSUFFICIENT_EVIDENCE",
            "risk_rating": "LOW",
            "flagged_wires": [],
            "applicable_regulations": [],
            "audit_summary": "No AML assessment available.",
            "source_document_hashes": [],
        },
        "compliance_draft": "No AML assessment available.",
    }


@pytest.mark.parametrize(
    ("critic_action", "insufficient_evidence", "expected_status"),
    [
        ("GENERATE", True, "INSUFFICIENT_EVIDENCE"),
        ("GENERATE", False, "COMPLETE"),
        ("STOP_INSUFFICIENT", False, "INSUFFICIENT_EVIDENCE"),
        (None, True, "INSUFFICIENT_EVIDENCE"),
        (None, False, "INSUFFICIENT_EVIDENCE"),
    ],
)
def test_structured_generation_derives_assessment_status(
    critic_action,
    insufficient_evidence,
    expected_status,
):
    state = {
        "aml_assessment": {
            "risk_rating": "Low",
            "reasoning_summary": "Assessment summary.",
            "insufficient_evidence": insufficient_evidence,
            "required_evidence_gaps": (
                ["AMOUNT"] if insufficient_evidence else []
            ),
        }
    }
    if critic_action is not None:
        state["critic_assessment"] = {"recommended_action": critic_action}

    update = structured_generation_node(state)

    assert update["final_report"]["assessment_status"] == expected_status


def test_insufficient_report_preserves_supported_review_findings():
    update = structured_generation_node(
        {
            "aml_assessment": {
                "risk_rating": "Medium",
                "suspicious_patterns": ["structuring"],
                "flagged_transactions": ["TXN-REVIEW"],
                "applicable_regulations": [],
                "reasoning_summary": "Transaction facts require review.",
                "insufficient_evidence": True,
                "required_evidence_gaps": ["MATERIAL_CONFLICT"],
            },
            "critic_assessment": {"recommended_action": "GENERATE"},
        }
    )

    report = update["final_report"]
    assert report["assessment_status"] == "INSUFFICIENT_EVIDENCE"
    assert report["risk_rating"] == "MEDIUM"
    assert report["flagged_wires"] == ["TXN-REVIEW"]
    assert report["applicable_regulations"] == []


def test_flagged_contracts_describe_review_not_confirmed_illegality():
    transaction_description = AMLAssessment.model_fields[
        "flagged_transactions"
    ].description
    wire_description = ComplianceReport.model_fields["flagged_wires"].description

    for description in (transaction_description, wire_description):
        assert "requiring AML review" in description
        assert "does not confirm illegal activity" in description


def test_transaction_extraction_schema_accepts_valid_contract():
    extraction = TransactionExtraction(
        transaction_ids=["TXN-100"],
        amount=8500.0,
        transaction_type="wire",
        regulations=["FINRA Rule 3310"],
        suspected_patterns=["structuring"],
        jurisdiction="US",
    )

    assert extraction.model_dump()["transaction_ids"] == ["TXN-100"]


@pytest.mark.parametrize(
    ("schema", "payload"),
    [
        (TransactionExtraction, {"transaction_ids": "TXN-100"}),
        (
            AMLAssessment,
            {
                "risk_rating": "High",
                "reasoning_summary": "Supported.",
                "insufficient_evidence": "not-a-boolean",
            },
        ),
        (
            CriticAssessment,
            {
                "is_sufficient": True,
                "failure_type": "NONE",
                "recommended_action": "GENERATE",
            },
        ),
        (ComplianceReport, {"risk_rating": "LOW"}),
    ],
)
def test_structured_contracts_reject_invalid_payloads(schema, payload):
    with pytest.raises(ValidationError):
        schema.model_validate(payload)


def test_structured_contract_list_defaults_are_isolated():
    first = TransactionExtraction()
    second = TransactionExtraction()

    first.transaction_ids.append("TXN-100")

    assert second.transaction_ids == []
