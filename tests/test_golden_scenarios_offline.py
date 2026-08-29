import pytest

from tests.evaluation.loader import load_golden_dataset
from tests.evaluation.matchers import match_value
from tests.evaluation.offline_runner import (
    evaluate_prohibited_outcomes,
    run_offline_replay,
)
from tests.evaluation.scenario_models import (
    AllowedMatcher,
    ExactMatcher,
    RangeMatcher,
    SubsetMatcher,
)


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
    [
        ("structuring-clear-001", "COMPLETE"),
        ("ordinary-wire-001", "COMPLETE"),
        ("insufficient-facts-001", "INSUFFICIENT_EVIDENCE"),
    ],
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
    assert result.retrieval_count == 1
    assert result.critic_count == 1
    assert result.final_status == expected_status


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
