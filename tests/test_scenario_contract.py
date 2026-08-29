import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.evaluation.loader import (
    DEFAULT_DATASET_DIR,
    FORBIDDEN_CREDENTIAL_KEYS,
    load_golden_dataset,
)
from tests.evaluation.scenario_models import GoldenScenario
from tests.evaluation.scenario_models import CriticActionMatcher


EXPECTED_SCENARIO_IDS = {
    "structuring-clear-001",
    "ordinary-wire-001",
    "insufficient-facts-001",
    "missing-amount-001",
    "multiple-ids-001",
    "ofac-us-001",
    "jurisdiction-ambiguous-001",
    "jurisdiction-substrings-001",
    "missing-regulation-001",
    "single-refinement-001",
    "max-loop-001",
    "multiple-regulations-001",
    "relevant-rule-benign-facts-001",
    "conflicting-evidence-001",
    "hallucination-trap-001",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def copied_dataset(tmp_path):
    destination = tmp_path / "dataset"
    shutil.copytree(DEFAULT_DATASET_DIR, destination)
    return destination


def test_manifest_and_exactly_fifteen_scenarios_load_successfully():
    manifest, scenarios = load_golden_dataset()

    assert manifest.dataset_id == "finguard-synthetic-aml-golden"
    assert manifest.dataset_version == "1.1.0"
    assert manifest.schema_version == "1.1"
    assert manifest.expectation_profile == "aml-golden-v1"
    assert len(manifest.scenarios) == 15
    assert {scenario.scenario_id for scenario in scenarios} == EXPECTED_SCENARIO_IDS


def test_every_manifest_reference_exists_and_matches_a_valid_scenario():
    manifest, scenarios = load_golden_dataset()

    loaded_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    for reference in manifest.scenarios:
        assert (DEFAULT_DATASET_DIR / reference.file).is_file()
        assert reference.scenario_id in loaded_by_id
        assert loaded_by_id[reference.scenario_id].schema_version == "1.0"
        assert loaded_by_id[reference.scenario_id].scenario_version > 0


def test_scenarios_require_expected_semantic_sections_and_synthetic_ids():
    _, scenarios = load_golden_dataset()

    for scenario in scenarios:
        assert scenario.synthetic_data is True
        assert scenario.expected.extraction
        assert scenario.expected.aml_assessment
        assert scenario.expected.critic
        assert scenario.expected.report
        assert scenario.expected.execution
        assert scenario.expected.report.assessment_status.value in {
            "COMPLETE",
            "INSUFFICIENT_EVIDENCE",
        }
        assert all(
            transaction_id.startswith("TXN-SYN-")
            for transaction_id in scenario.synthetic_facts.transaction_ids
        )


def test_scenario_ids_are_unique():
    manifest, scenarios = load_golden_dataset()

    manifest_ids = [reference.scenario_id for reference in manifest.scenarios]
    loaded_ids = [scenario.scenario_id for scenario in scenarios]
    assert len(manifest_ids) == len(set(manifest_ids))
    assert len(loaded_ids) == len(set(loaded_ids))


@pytest.mark.parametrize("target", ["manifest", "scenario"])
def test_unsupported_schema_version_fails(tmp_path, target):
    dataset = copied_dataset(tmp_path)
    path = (
        dataset / "manifest.json"
        if target == "manifest"
        else dataset / "cases" / "ordinary-wire-001.json"
    )
    payload = read_json(path)
    payload["schema_version"] = "2.0"
    write_json(path, payload)

    with pytest.raises(ValueError, match="unsupported schema_version"):
        load_golden_dataset(dataset)


@pytest.mark.parametrize("synthetic_value", [None, False])
def test_synthetic_data_declaration_is_required_and_true(tmp_path, synthetic_value):
    dataset = copied_dataset(tmp_path)
    path = dataset / "cases" / "ordinary-wire-001.json"
    payload = read_json(path)
    if synthetic_value is None:
        del payload["synthetic_data"]
    else:
        payload["synthetic_data"] = synthetic_value
    write_json(path, payload)

    with pytest.raises(ValidationError):
        load_golden_dataset(dataset)


def test_unsupported_matcher_type_fails_validation():
    payload = read_json(
        DEFAULT_DATASET_DIR / "cases" / "ordinary-wire-001.json"
    )
    payload["expected"]["extraction"]["doc_type"] = {
        "match": "fuzzy",
        "value": "swift_log",
    }

    with pytest.raises(ValidationError):
        GoldenScenario.model_validate(payload)


def test_invalid_expected_assessment_status_fails_validation():
    payload = read_json(
        DEFAULT_DATASET_DIR / "cases" / "ordinary-wire-001.json"
    )
    payload["expected"]["report"]["assessment_status"]["value"] = "UNKNOWN"

    with pytest.raises(ValidationError):
        GoldenScenario.model_validate(payload)


def test_final_stored_action_is_optional_for_legacy_scenarios():
    payload = read_json(
        DEFAULT_DATASET_DIR / "cases" / "ordinary-wire-001.json"
    )

    scenario = GoldenScenario.model_validate(payload)

    assert scenario.expected.critic.final_stored_action is None


def test_schema_1_1_final_stored_action_round_trips():
    payload = read_json(DEFAULT_DATASET_DIR / "cases" / "max-loop-001.json")
    payload["schema_version"] = "1.1"
    payload["expected"]["critic"]["final_stored_action"] = {
        "match": "exact",
        "value": "STOP_INSUFFICIENT",
    }

    scenario = GoldenScenario.model_validate(payload)
    restored = GoldenScenario.model_validate_json(scenario.model_dump_json())

    assert restored == scenario
    assert restored.expected.critic.final_stored_action == CriticActionMatcher(
        match="exact",
        value="STOP_INSUFFICIENT",
    )


@pytest.mark.parametrize("value", ["UNKNOWN", "retrieve_more", 1])
def test_final_stored_action_rejects_invalid_values(value):
    payload = read_json(DEFAULT_DATASET_DIR / "cases" / "max-loop-001.json")
    payload["schema_version"] = "1.1"
    payload["expected"]["critic"]["final_stored_action"] = {
        "match": "exact",
        "value": value,
    }

    with pytest.raises(ValidationError):
        GoldenScenario.model_validate(payload)


def test_nonpositive_scenario_version_fails_validation():
    payload = read_json(
        DEFAULT_DATASET_DIR / "cases" / "ordinary-wire-001.json"
    )
    payload["scenario_version"] = 0

    with pytest.raises(ValidationError):
        GoldenScenario.model_validate(payload)


def test_missing_expected_semantic_section_fails_validation():
    payload = read_json(
        DEFAULT_DATASET_DIR / "cases" / "ordinary-wire-001.json"
    )
    del payload["expected"]["aml_assessment"]

    with pytest.raises(ValidationError):
        GoldenScenario.model_validate(payload)


def test_non_synthetic_transaction_id_fails_validation():
    payload = read_json(
        DEFAULT_DATASET_DIR / "cases" / "ordinary-wire-001.json"
    )
    payload["synthetic_facts"]["transaction_ids"] = ["REAL-ACCOUNT-123"]

    with pytest.raises(ValidationError, match="TXN-SYN"):
        GoldenScenario.model_validate(payload)


def test_missing_manifest_referenced_file_fails_clearly(tmp_path):
    dataset = copied_dataset(tmp_path)
    manifest_path = dataset / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["scenarios"][0]["file"] = "cases/missing-scenario.json"
    write_json(manifest_path, manifest)

    with pytest.raises(FileNotFoundError, match="manifest-referenced scenario"):
        load_golden_dataset(dataset)


def test_duplicate_manifest_scenario_id_fails(tmp_path):
    dataset = copied_dataset(tmp_path)
    manifest_path = dataset / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["scenarios"][1]["scenario_id"] = manifest["scenarios"][0][
        "scenario_id"
    ]
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="duplicate scenario IDs"):
        load_golden_dataset(dataset)


def test_obvious_credential_fields_are_rejected(tmp_path):
    dataset = copied_dataset(tmp_path)
    path = dataset / "cases" / "ordinary-wire-001.json"
    payload = read_json(path)
    payload["retrieval"]["passes"][0]["documents"][0]["metadata"][
        "api_key"
    ] = "synthetic-but-forbidden"
    write_json(path, payload)

    with pytest.raises(ValueError, match="forbidden credential field"):
        load_golden_dataset(dataset)


def test_committed_scenarios_contain_no_obvious_credential_fields_or_secrets():
    _, scenarios = load_golden_dataset()
    forbidden_fragments = {"sk-", "xai_api_key", "openai_api_key", "password="}

    for scenario in scenarios:
        serialized = json.dumps(scenario.model_dump()).lower()
        assert not any(fragment in serialized for fragment in forbidden_fragments)

        def assert_no_forbidden_keys(value):
            if isinstance(value, dict):
                assert not (set(map(str.lower, value)) & FORBIDDEN_CREDENTIAL_KEYS)
                for child in value.values():
                    assert_no_forbidden_keys(child)
            elif isinstance(value, list):
                for child in value:
                    assert_no_forbidden_keys(child)

        assert_no_forbidden_keys(scenario.model_dump())
