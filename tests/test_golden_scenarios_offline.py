import pytest

from tests.evaluation.loader import load_golden_dataset
from tests.evaluation.matchers import match_value
from tests.evaluation.offline_runner import (
    evaluate_prohibited_outcomes,
    run_offline_replay,
)
from tests.evaluation.scenario_models import (
    AllowedMatcher,
    CriticActionMatcher,
    ExactMatcher,
    RangeMatcher,
    SubsetMatcher,
)


EXPECTED_STATUSES = {
    "structuring-clear-001": "COMPLETE",
    "ordinary-wire-001": "COMPLETE",
    "insufficient-facts-001": "INSUFFICIENT_EVIDENCE",
    "missing-amount-001": "COMPLETE",
    "multiple-ids-001": "COMPLETE",
    "ofac-us-001": "COMPLETE",
    "jurisdiction-ambiguous-001": "COMPLETE",
    "jurisdiction-substrings-001": "COMPLETE",
    "missing-regulation-001": "INSUFFICIENT_EVIDENCE",
    "single-refinement-001": "COMPLETE",
    "max-loop-001": "INSUFFICIENT_EVIDENCE",
    "multiple-regulations-001": "COMPLETE",
    "relevant-rule-benign-facts-001": "COMPLETE",
    "conflicting-evidence-001": "INSUFFICIENT_EVIDENCE",
    "hallucination-trap-001": "INSUFFICIENT_EVIDENCE",
}


@pytest.fixture(scope="module")
def scenarios_by_id():
    _, scenarios = load_golden_dataset()
    return {scenario.scenario_id: scenario for scenario in scenarios}


def test_exact_match_is_scalar_and_order_insensitive_for_lists():
    scalar = match_value(
        "assessment_status",
        ExactMatcher(match="exact", value="COMPLETE"),
        " COMPLETE ",
    )
    transaction_ids = match_value(
        "transaction_ids",
        ExactMatcher(match="exact", value=["TXN-SYN-1", "TXN-SYN-2"]),
        ["TXN-SYN-2", "TXN-SYN-1"],
    )

    assert scalar.passed is True
    assert transaction_ids.passed is True


def test_subset_match_requires_all_expected_values():
    matcher = SubsetMatcher(match="subset", values=["FINRA Rule 3310"])

    assert match_value(
        "applicable_regulations",
        matcher,
        ["31 U.S.C. 5324", "finra rule 3310"],
    ).passed
    assert not match_value("applicable_regulations", matcher, []).passed


def test_allowed_match_uses_controlled_case_normalization():
    matcher = AllowedMatcher(match="allowed", values=["High", "HIGH"])

    assert match_value("risk_rating", matcher, " high ").passed
    assert not match_value("risk_rating", matcher, "LOW").passed


@pytest.mark.parametrize("actual", [1, 2, 3])
def test_range_match_includes_declared_boundaries(actual):
    matcher = RangeMatcher(match="range", min=1, max=3)

    assert match_value("retrieval_count", matcher, actual).passed


def test_deliberate_mismatch_has_clear_failure_message():
    result = match_value(
        "risk_rating",
        ExactMatcher(match="exact", value="HIGH"),
        "LOW",
    )

    assert result.passed is False
    assert "risk_rating" in result.message
    assert "HIGH" in result.message
    assert "LOW" in result.message


def test_critic_action_sequence_is_order_sensitive():
    matcher = ExactMatcher(
        match="exact",
        value=["RETRIEVE_MORE", "GENERATE"],
    )

    assert match_value(
        "critic_actions",
        matcher,
        ["RETRIEVE_MORE", "GENERATE"],
        order_sensitive=True,
    ).passed
    assert not match_value(
        "critic_actions",
        matcher,
        ["GENERATE", "RETRIEVE_MORE"],
        order_sensitive=True,
    ).passed


@pytest.mark.parametrize(
    ("scenario_id", "expected_status"),
    EXPECTED_STATUSES.items(),
)
def test_all_golden_scenarios_execute_and_pass(
    scenarios_by_id,
    scenario_id,
    expected_status,
):
    result = run_offline_replay(scenarios_by_id[scenario_id])

    assert result.execution_mode == "offline_replay"
    assert result.terminated is True
    assert result.passed is True, result.failed_assertions
    assert result.failed_assertions == []
    assert result.prohibited_violations == []
    assert result.final_status == expected_status


@pytest.mark.parametrize(
    ("scenario_id", "actions", "final_action"),
    [
        (
            "missing-regulation-001",
            ["RETRIEVE_MORE", "RETRIEVE_MORE"],
            "STOP_INSUFFICIENT",
        ),
        (
            "single-refinement-001",
            ["RETRIEVE_MORE", "GENERATE"],
            "GENERATE",
        ),
        (
            "ofac-us-001",
            ["RETRIEVE_MORE", "GENERATE"],
            "GENERATE",
        ),
        (
            "max-loop-001",
            ["RETRIEVE_MORE", "RETRIEVE_MORE"],
            "STOP_INSUFFICIENT",
        ),
    ],
)
def test_retry_scenarios_use_real_graph_cycles(
    scenarios_by_id,
    scenario_id,
    actions,
    final_action,
):
    result = run_offline_replay(scenarios_by_id[scenario_id])

    assert result.passed is True
    assert result.retrieval_count == 2
    assert result.critic_count == 2
    assert result.final_loop_count == 2
    assert result.critic_actions == actions
    assert result.final_critic_action == final_action


def test_single_refinement_retries_once_then_generates(scenarios_by_id):
    result = run_offline_replay(scenarios_by_id["single-refinement-001"])

    assert result.critic_actions == ["RETRIEVE_MORE", "GENERATE"]
    assert result.final_status == "COMPLETE"
    assert result.final_critic_action == "GENERATE"


def test_max_loop_terminates_as_insufficient_evidence(scenarios_by_id):
    result = run_offline_replay(scenarios_by_id["max-loop-001"])

    assert result.terminated is True
    assert result.final_status == "INSUFFICIENT_EVIDENCE"
    assert result.final_critic_action == "STOP_INSUFFICIENT"
    assert result.final_risk_rating in {"MEDIUM", "HIGH"}
    assert result.flagged_wires == ["TXN-SYN-1100", "TXN-SYN-1101"]
    assert "structuring" in result.suspicious_patterns


def test_repaired_ofac_contract_retrieves_then_generates(scenarios_by_id):
    result = run_offline_replay(scenarios_by_id["ofac-us-001"])

    assert result.passed is True
    assert result.critic_actions == ["RETRIEVE_MORE", "GENERATE"]
    assert result.retrieval_count == 2
    assert result.final_status == "COMPLETE"
    assert result.flagged_wires == ["TXN-SYN-600"]


def test_final_stored_action_matches_independently_from_raw_actions(scenarios_by_id):
    scenario = scenarios_by_id["max-loop-001"].model_copy(deep=True)
    scenario.schema_version = "1.1"
    scenario.expected.critic.final_stored_action = CriticActionMatcher(
        match="exact",
        value="STOP_INSUFFICIENT",
    )

    result = run_offline_replay(scenario)

    assert result.passed is True
    assert result.critic_actions == ["RETRIEVE_MORE", "RETRIEVE_MORE"]
    assert result.final_critic_action == "STOP_INSUFFICIENT"
    assert result.metric_contributions[
        "final_stored_critic_action_accuracy"
    ].numerator == 1


def test_final_stored_action_mismatch_has_distinct_failure(scenarios_by_id):
    scenario = scenarios_by_id["max-loop-001"].model_copy(deep=True)
    scenario.schema_version = "1.1"
    scenario.expected.critic.final_stored_action = CriticActionMatcher(
        match="exact",
        value="GENERATE",
    )

    result = run_offline_replay(scenario)

    assert result.passed is False
    assert result.critic_actions == ["RETRIEVE_MORE", "RETRIEVE_MORE"]
    assert any(
        failure.field == "final_stored_action"
        for failure in result.failed_assertions
    )
    assert not any(
        failure.field == "critic_actions"
        for failure in result.failed_assertions
    )


def test_raw_action_mismatch_does_not_imply_stored_action_mismatch(
    scenarios_by_id,
):
    from src.graph.schemas import CriticAssessment
    from tests.evaluation.evaluation_core import evaluate_scenario

    scenario = scenarios_by_id["max-loop-001"].model_copy(deep=True)
    scenario.schema_version = "1.1"
    scenario.expected.critic.actions = ExactMatcher(
        match="exact",
        value=["RETRIEVE_MORE", "GENERATE"],
    )
    scenario.expected.critic.final_stored_action = CriticActionMatcher(
        match="exact",
        value="STOP_INSUFFICIENT",
    )
    state = {
        "jurisdiction": None,
        "doc_type": None,
        "extracted_entities": {
            "transaction_ids": ["TXN-SYN-1100"],
            "amount": 9700.0,
            "transaction_type": "wire",
            "regulations": ["FINRA Rule 3310"],
            "suspected_patterns": ["structuring"],
            "jurisdiction": None,
        },
        "aml_assessment": {
            "risk_rating": "Low",
            "suspicious_patterns": [],
            "flagged_transactions": [],
            "applicable_regulations": [],
            "reasoning_summary": "Synthetic assessment.",
            "insufficient_evidence": True,
        },
        "critic_assessment": {"recommended_action": "STOP_INSUFFICIENT"},
        "loop_count": 2,
        "final_report": {
            "assessment_status": "INSUFFICIENT_EVIDENCE",
            "risk_rating": "LOW",
            "flagged_wires": [],
            "applicable_regulations": [],
            "source_document_hashes": ["synthetic_max_loop_b.txt"],
            "audit_summary": "Synthetic assessment.",
        },
    }
    critics = [
        CriticAssessment(
            is_sufficient=False,
            missing_evidence=[],
            failure_type="MISSING_REGULATORY_CONTEXT",
            recommended_action="RETRIEVE_MORE",
            critique="More context required.",
        ),
        CriticAssessment(
            is_sufficient=False,
            missing_evidence=[],
            failure_type="MISSING_REGULATORY_CONTEXT",
            recommended_action="RETRIEVE_MORE",
            critique="More context still required.",
        ),
    ]
    queries = [
        scenario.input.query,
        scenario.input.query
        + " FINRA Rule 3310 structuring Currency Transaction Reporting thresholds",
    ]

    result = evaluate_scenario(
        scenario,
        state,
        critics,
        queries,
        execution_mode="offline_replay",
    )

    assert any(failure.field == "critic_actions" for failure in result.failed_assertions)
    assert not any(
        failure.field == "final_stored_action"
        for failure in result.failed_assertions
    )


def test_jurisdiction_substrings_do_not_produce_us_ofac(scenarios_by_id):
    result = run_offline_replay(scenarios_by_id["jurisdiction-substrings-001"])

    assert result.passed is True
    assert result.jurisdiction is None

    assert scenarios_by_id[
        "jurisdiction-substrings-001"
    ].expected.extraction.amount.value is None


def test_conflicting_amount_is_unresolved_and_flagged_for_review(scenarios_by_id):
    scenario = scenarios_by_id["conflicting-evidence-001"]
    result = run_offline_replay(scenario)

    assert scenario.expected.extraction.amount.value is None
    assert result.passed is True
    assert result.final_status == "INSUFFICIENT_EVIDENCE"
    assert result.flagged_wires == ["TXN-SYN-1400"]
    assert result.suspicious_patterns == []


def test_hallucination_trap_has_no_prohibited_violations(scenarios_by_id):
    result = run_offline_replay(scenarios_by_id["hallucination-trap-001"])

    assert result.passed is True
    assert result.prohibited_violations == []


def test_relevant_rule_does_not_make_benign_facts_suspicious(scenarios_by_id):
    result = run_offline_replay(
        scenarios_by_id["relevant-rule-benign-facts-001"]
    )

    assert result.passed is True
    assert result.final_status == "COMPLETE"
    assert result.final_risk_rating == "LOW"
    assert result.flagged_wires == []
    assert result.suspicious_patterns == []


def test_retrieval_query_expectations_are_enforced(scenarios_by_id):
    scenario = scenarios_by_id["ordinary-wire-001"].model_copy(deep=True)
    scenario.retrieval.passes[0].expected_query_contains.append(
        "TERM-NOT-PRESENT-IN-QUERY"
    )

    result = run_offline_replay(scenario)

    assert result.passed is False
    assert any(
        failure.field == "retrieval_queries[0]"
        and failure.matcher == "contains"
        for failure in result.failed_assertions
    )


def test_prohibited_structured_value_produces_violation(scenarios_by_id):
    scenario = scenarios_by_id["ordinary-wire-001"]
    state = {
        "jurisdiction": "US_OFAC",
        "extracted_entities": {},
        "aml_assessment": {},
        "final_report": {},
    }

    violations = evaluate_prohibited_outcomes(scenario, state, [])

    assert any(
        violation.category == "jurisdictions"
        and violation.value == "US_OFAC"
        for violation in violations
    )


def test_unsupported_fact_terms_are_detected_in_generated_text(scenarios_by_id):
    scenario = scenarios_by_id["ordinary-wire-001"]
    state = {
        "jurisdiction": None,
        "extracted_entities": {},
        "aml_assessment": {
            "reasoning_summary": "The payment involved a shell company."
        },
        "final_report": {"audit_summary": "Synthetic summary."},
    }

    violations = evaluate_prohibited_outcomes(scenario, state, [])

    assert any(
        violation.category == "unsupported_fact_terms"
        and violation.value == "shell company"
        for violation in violations
    )
