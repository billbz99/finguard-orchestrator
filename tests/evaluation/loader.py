from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.evaluation.scenario_models import (
    DatasetManifest,
    GoldenScenario,
    SUPPORTED_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
)


DEFAULT_DATASET_DIR = (
    Path(__file__).resolve().parents[1] / "scenarios" / "aml_golden" / "v1"
)
FORBIDDEN_CREDENTIAL_KEYS = {
    "api_key",
    "apikey",
    "password",
    "secret",
    "token",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _reject_credential_fields(value: Any, *, location: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in FORBIDDEN_CREDENTIAL_KEYS:
                raise ValueError(f"forbidden credential field '{key}' in {location}")
            _reject_credential_fields(child, location=location)
    elif isinstance(value, list):
        for child in value:
            _reject_credential_fields(child, location=location)


def _require_supported_schema(version: str, *, location: str) -> None:
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported schema_version '{version}' in {location}; "
            f"expected one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )


def load_golden_dataset(
    dataset_dir: str | Path | None = None,
) -> tuple[DatasetManifest, list[GoldenScenario]]:
    """Loads and validates the versioned synthetic AML golden dataset."""
    root = Path(dataset_dir) if dataset_dir is not None else DEFAULT_DATASET_DIR
    root = root.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"golden dataset manifest not found: {manifest_path}")

    manifest_payload = _load_json(manifest_path)
    _reject_credential_fields(manifest_payload, location=str(manifest_path))
    manifest = DatasetManifest.model_validate(manifest_payload)
    _require_supported_schema(manifest.schema_version, location=str(manifest_path))

    referenced_ids = [reference.scenario_id for reference in manifest.scenarios]
    if len(referenced_ids) != len(set(referenced_ids)):
        raise ValueError("manifest contains duplicate scenario IDs")

    scenarios = []
    loaded_ids = set()
    for reference in manifest.scenarios:
        scenario_path = (root / reference.file).resolve()
        if root not in scenario_path.parents:
            raise ValueError(
                f"scenario path escapes dataset directory: {reference.file}"
            )
        if not scenario_path.is_file():
            raise FileNotFoundError(
                f"manifest-referenced scenario not found: {scenario_path}"
            )

        scenario_payload = _load_json(scenario_path)
        _reject_credential_fields(scenario_payload, location=str(scenario_path))
        scenario = GoldenScenario.model_validate(scenario_payload)
        _require_supported_schema(
            scenario.schema_version,
            location=str(scenario_path),
        )
        if scenario.scenario_id != reference.scenario_id:
            raise ValueError(
                "manifest scenario ID does not match scenario file: "
                f"{reference.scenario_id} != {scenario.scenario_id}"
            )
        if scenario.scenario_id in loaded_ids:
            raise ValueError(f"duplicate loaded scenario ID: {scenario.scenario_id}")
        loaded_ids.add(scenario.scenario_id)
        scenarios.append(scenario)

    return manifest, scenarios
