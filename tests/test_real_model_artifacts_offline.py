from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from types import SimpleNamespace

from src.graph.schemas import AMLAssessment, CriticAssessment, TransactionExtraction
from tests.evaluation.loader import load_golden_dataset
from tests.evaluation.offline_runner import _response_plan
from tests.evaluation.real_model_runner import (
    FailureClass,
    build_run_artifact,
    run_controlled_scenario,
)


class FakeStructuredClient:
    def __init__(self, responses, schema):
        self.responses = responses
        self.schema = schema

    def invoke(self, prompt):
        response = self.responses[self.schema].popleft()
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    def __init__(self, responses):
        self.responses = {
            schema: deque(values) for schema, values in responses.items()
        }

    def with_structured_output(self, schema, *, include_raw=False):
        assert include_raw is True
        return FakeStructuredClient(self.responses, schema)


def raw_result(parsed):
    return {
        "raw": SimpleNamespace(
            usage_metadata={
                "input_tokens": 2,
                "output_tokens": 3,
                "total_tokens": 5,
            },
            response_metadata={},
            id="safe-request-id",
        ),
        "parsed": parsed,
        "parsing_error": None,
    }


def scenario_client(scenario):
    return FakeClient(
        {
            schema: [raw_result(response) for response in responses]
            for schema, responses in _response_plan(scenario).items()
        }
    )


def scenarios_by_id():
    manifest, scenarios = load_golden_dataset()
    return manifest, {scenario.scenario_id: scenario for scenario in scenarios}


def build_artifact(manifest, results, *, requested_ids=None):
    return build_run_artifact(
        manifest,
        results,
        provider="offline-fake",
        model="fake-structured-client",
        started_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        completed_at=datetime(2026, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
        selected_scenario_ids=requested_ids,
    )


def test_single_pass_artifact_preserves_all_structured_histories():
    manifest, scenarios = scenarios_by_id()
    scenario = scenarios["ordinary-wire-001"]

    result = run_controlled_scenario(scenario, scenario_client(scenario))
    artifact = build_artifact(manifest, [result])

    assert len(result.transaction_extractions) == 1
    assert len(result.aml_assessments) == 1
    assert len(result.critic_assessments) == 1
    assert result.transaction_extractions[0] == result.actual_extraction
    assert result.aml_assessments[0] == result.actual_aml_assessment
    assert result.critic_assessments[0]["recommended_action"] == "GENERATE"
    assert artifact.aggregate_metrics.counts.scenarios_passed == 1


def test_retry_artifact_preserves_aml_and_critic_outputs_in_order():
    _, scenarios = scenarios_by_id()
    scenario = scenarios["single-refinement-001"]

    result = run_controlled_scenario(scenario, scenario_client(scenario))

    assert [value["insufficient_evidence"] for value in result.aml_assessments] == [
        True,
        False,
    ]
    assert [value["recommended_action"] for value in result.critic_assessments] == [
        "RETRIEVE_MORE",
        "GENERATE",
    ]
    assert result.requested_critic_actions == ["RETRIEVE_MORE", "GENERATE"]
    assert result.final_critic_action == "GENERATE"


def test_loop_boundary_keeps_raw_requests_separate_from_stored_action():
    _, scenarios = scenarios_by_id()
    scenario = scenarios["max-loop-001"]

    result = run_controlled_scenario(scenario, scenario_client(scenario))

    assert [value["recommended_action"] for value in result.critic_assessments] == [
        "RETRIEVE_MORE",
        "RETRIEVE_MORE",
    ]
    assert result.requested_critic_actions == ["RETRIEVE_MORE", "RETRIEVE_MORE"]
    assert result.final_critic_action == "STOP_INSUFFICIENT"


def test_run_timestamps_cover_execution_lifecycle():
    manifest, scenarios = scenarios_by_id()
    scenario = scenarios["ordinary-wire-001"]
    result = run_controlled_scenario(scenario, scenario_client(scenario))

    artifact = build_artifact(manifest, [result])

    assert artifact.started_at < artifact.completed_at
    assert (artifact.completed_at - artifact.started_at).total_seconds() == 5


def test_aggregate_metrics_all_pass_and_semantic_mismatch():
    manifest, scenarios = scenarios_by_id()
    scenario = scenarios["ordinary-wire-001"]
    passing = run_controlled_scenario(scenario, scenario_client(scenario))

    plan = _response_plan(scenario)
    plan[TransactionExtraction][0] = plan[TransactionExtraction][0].model_copy(
        update={"transaction_ids": ["TXN-SYN-WRONG"]}
    )
    mismatch = run_controlled_scenario(
        scenario,
        FakeClient(
            {
                schema: [raw_result(response) for response in responses]
                for schema, responses in plan.items()
            }
        ),
    )
    artifact = build_artifact(manifest, [passing, mismatch])
    metrics = artifact.aggregate_metrics

    assert metrics.counts.semantically_evaluated == 2
    assert metrics.counts.scenarios_passed == 1
    assert metrics.scenario_pass_rate.model_dump() == {
        "numerator": 1,
        "denominator": 2,
        "value": 0.5,
    }
    assert metrics.transaction_id_exact_match.numerator == 1
    assert metrics.transaction_id_exact_match.denominator == 2


def test_partial_failures_keep_semantic_denominators_separate():
    manifest, scenarios = scenarios_by_id()
    scenario = scenarios["ordinary-wire-001"]
    passing = run_controlled_scenario(scenario, scenario_client(scenario))

    plan = _response_plan(scenario)
    infrastructure_responses = {
        TransactionExtraction: [raw_result(plan[TransactionExtraction][0])],
        AMLAssessment: [RuntimeError("provider unavailable")],
        CriticAssessment: [],
    }
    infrastructure = run_controlled_scenario(
        scenario, FakeClient(infrastructure_responses)
    )
    artifact = build_artifact(
        manifest,
        [passing, infrastructure],
        requested_ids=[scenario.scenario_id, "not-started", "also-not-started"],
    )
    metrics = artifact.aggregate_metrics

    assert infrastructure.failure_class == FailureClass.INFRASTRUCTURE_ERROR
    assert infrastructure.transaction_extractions
    assert infrastructure.aml_assessments == []
    assert metrics.counts.model_dump() == {
        "scenarios_requested": 3,
        "scenarios_started": 2,
        "infrastructure_completed": 1,
        "semantically_evaluated": 1,
        "scenarios_passed": 1,
    }
    assert metrics.scenario_pass_rate.value == 1.0
    assert metrics.infrastructure_success_rate.value == 0.5


def test_zero_semantic_evaluations_use_null_metric_values():
    manifest, scenarios = scenarios_by_id()
    scenario = scenarios["ordinary-wire-001"]
    result = run_controlled_scenario(
        scenario,
        FakeClient(
            {
                TransactionExtraction: [RuntimeError("provider unavailable")],
                AMLAssessment: [],
                CriticAssessment: [],
            }
        ),
    )

    metrics = build_artifact(manifest, [result]).aggregate_metrics

    assert metrics.counts.semantically_evaluated == 0
    assert metrics.scenario_pass_rate.denominator == 0
    assert metrics.scenario_pass_rate.value is None
    assert metrics.risk_accuracy.value is None
    assert metrics.schema_valid_structured_call_rate.value == 0.0


def test_full_artifact_round_trips_without_secrets(monkeypatch):
    manifest, scenarios = scenarios_by_id()
    scenario = scenarios["single-refinement-001"]
    monkeypatch.setenv("XAI_API_KEY", "SENTINEL-SECRET-KEY")
    artifact = build_artifact(
        manifest,
        [run_controlled_scenario(scenario, scenario_client(scenario))],
    )

    payload = artifact.model_dump_json()
    restored = type(artifact).model_validate_json(payload)

    assert restored == artifact
    assert "SENTINEL-SECRET-KEY" not in payload
    assert "authorization" not in payload.casefold()
    json.loads(payload)
