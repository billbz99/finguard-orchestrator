"""Explicitly opt-in paid entry point for controlled-retrieval evaluation."""

import os
from datetime import datetime, timezone

import pytest

from tests.evaluation.loader import load_golden_dataset
from tests.evaluation.real_model_runner import paid_run_selection, run_controlled_scenario
from tests.evaluation.real_model_runner import build_run_artifact, write_artifact


pytestmark = pytest.mark.real_model_eval


def test_selected_golden_scenarios_with_real_model():
    manifest, scenarios = load_golden_dataset()
    try:
        selected_ids = paid_run_selection(os.environ, [s.scenario_id for s in scenarios])
    except RuntimeError as exc:
        pytest.skip(str(exc))

    from src.llm.client import get_llm

    by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    model = os.environ.get("XAI_MODEL", "grok-4.3")
    print(
        "PAID REAL-MODEL EVALUATION: "
        f"provider=xAI model={model} scenarios={selected_ids} "
        f"maximum_calls={len(selected_ids) * 5}"
    )
    started_at = datetime.now(timezone.utc)
    results = [run_controlled_scenario(by_id[scenario_id], get_llm()) for scenario_id in selected_ids]
    artifact = build_run_artifact(
        manifest,
        results,
        provider="xAI",
        model=model,
        started_at=started_at,
        selected_scenario_ids=selected_ids,
    )
    write_artifact(artifact, f"artifacts/evaluations/{artifact.run_id}.json")
    failures = [result for result in results if result.status != "passed"]
    assert not failures, [result.model_dump() for result in failures]
