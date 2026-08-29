from __future__ import annotations

import subprocess
import time
import uuid
from collections import defaultdict, deque
from contextlib import ExitStack
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict, Field

from src.graph import nodes
from src.graph.schemas import CriticAssessment
from src.graph.workflow import build_finguard_graph
from tests.evaluation.evaluation_core import ScenarioEvaluationResult, evaluate_scenario, initial_state
from tests.evaluation.offline_runner import _remote_tracing_disabled
from tests.evaluation.scenario_models import DatasetManifest, GoldenScenario, RetrievalDocument

EXECUTION_MODE = "real_model_controlled_retrieval"
ARTIFACT_SCHEMA_VERSION = "1.0"
DEFAULT_MAX_SCENARIOS = 3


class FailureClass(StrEnum):
    EVALUATED = "evaluated"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    STRUCTURED_OUTPUT_ERROR = "structured_output_error"
    SEMANTIC_MISMATCH = "semantic_mismatch"


class EvaluationAdapterError(RuntimeError): pass
class ControlledRetrievalExhausted(EvaluationAdapterError): pass
class StructuredOutputFailure(EvaluationAdapterError): pass


class LLMCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_name: str
    call_index: int = Field(ge=1)
    started_monotonic: float
    completed_monotonic: float
    latency_seconds: float = Field(ge=0)
    parsed_output_success: bool
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    provider_request_id: str | None = None
    usage_status: str
    cost_usd: float | None = None
    cost_status: str = "not_reported"


class RetrievalCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    doc_type: str | None
    jurisdiction: str | None
    top_k_vector: int
    top_n_final: int


class AggregateUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    usage_status: str
    cost_usd: float | None = None
    cost_status: str = "not_reported"


class ScenarioRunArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario_id: str
    scenario_version: int
    status: str
    failure_class: FailureClass
    failure_message: str | None = None
    actual_extraction: dict[str, Any] | None = None
    actual_aml_assessment: dict[str, Any] | None = None
    requested_critic_actions: list[str] = Field(default_factory=list)
    final_critic_action: str | None = None
    final_report: dict[str, Any] | None = None
    loop_count: int = 0
    retrieval_queries: list[str] = Field(default_factory=list)
    supplied_retrieval_passes: int = 0
    matcher_result: ScenarioEvaluationResult | None = None
    prohibited_violations: list[dict[str, Any]] = Field(default_factory=list)
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
    usage: AggregateUsage
    latency_seconds: float = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


class EvaluationRunArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_schema_version: str = ARTIFACT_SCHEMA_VERSION
    run_id: str
    started_at: str
    completed_at: str
    git_commit: str | None = None
    dataset_id: str
    dataset_version: str
    schema_version: str
    expectation_profile: str
    execution_mode: str = EXECUTION_MODE
    provider: str
    model: str
    selected_scenario_ids: list[str]
    scenario_versions: dict[str, int]
    scenarios: list[ScenarioRunArtifact]
    usage: AggregateUsage
    latency_seconds: float = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    run_errors: list[str] = Field(default_factory=list)


def _safe_usage(raw: Any) -> tuple[int | None, int | None, int | None, str, str | None]:
    usage = getattr(raw, "usage_metadata", None) or (getattr(raw, "response_metadata", {}) or {}).get("token_usage")
    request_id = getattr(raw, "id", None) or (getattr(raw, "response_metadata", {}) or {}).get("request_id")
    if not usage:
        return None, None, None, "not_reported", request_id
    def value(*keys):
        for key in keys:
            candidate = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
            if candidate is not None: return int(candidate)
        return None
    return value("input_tokens", "prompt_tokens"), value("output_tokens", "completion_tokens"), value("total_tokens"), "reported", request_id


class _InstrumentedStructuredOutput:
    def __init__(self, owner, schema, runnable): self.owner, self.schema, self.runnable = owner, schema, runnable
    def invoke(self, prompt):
        started = time.monotonic()
        index = len(self.owner.calls) + 1
        try:
            result = self.runnable.invoke(prompt)
            parsed = result.get("parsed") if isinstance(result, dict) else None
            parsing_error = result.get("parsing_error") if isinstance(result, dict) else None
            raw = result.get("raw") if isinstance(result, dict) else None
            success = parsed is not None and parsing_error is None
            if not success:
                raise StructuredOutputFailure(str(parsing_error or "structured output did not contain a parsed object"))
            self.owner.returned[self.schema].append(parsed)
            return parsed
        finally:
            completed = time.monotonic()
            raw_value = locals().get("raw")
            input_tokens, output_tokens, total_tokens, usage_status, request_id = _safe_usage(raw_value)
            self.owner.calls.append(LLMCallRecord(schema_name=self.schema.__name__, call_index=index, started_monotonic=started, completed_monotonic=completed, latency_seconds=completed-started, parsed_output_success=locals().get("success", False), input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens, provider_request_id=request_id, usage_status=usage_status))


class InstrumentedLLM:
    def __init__(self, client):
        self.client = client
        self.calls: list[LLMCallRecord] = []
        self.returned = defaultdict(list)
    def with_structured_output(self, schema):
        return _InstrumentedStructuredOutput(self, schema, self.client.with_structured_output(schema, include_raw=True))


class ControlledScenarioRetriever:
    def __init__(self, document_batches: list[list[RetrievalDocument | dict[str, Any]]]):
        self._batches = deque([[d.model_dump() if isinstance(d, RetrievalDocument) else dict(d) for d in batch] for batch in document_batches])
        self.calls: list[RetrievalCall] = []
        self.supplied_passes = 0
    def retrieve(self, **kwargs):
        self.calls.append(RetrievalCall(**kwargs))
        if not self._batches:
            raise ControlledRetrievalExhausted("graph requested more retrieval passes than declared")
        self.supplied_passes += 1
        return self._batches.popleft()


def _aggregate_usage(calls):
    if not calls or any(call.usage_status != "reported" for call in calls):
        return AggregateUsage(usage_status="not_reported")
    return AggregateUsage(input_tokens=sum(c.input_tokens or 0 for c in calls), output_tokens=sum(c.output_tokens or 0 for c in calls), total_tokens=sum(c.total_tokens or 0 for c in calls), usage_status="reported")


def _classify(error, result):
    if isinstance(error, ControlledRetrievalExhausted): return FailureClass.SEMANTIC_MISMATCH
    if isinstance(error, StructuredOutputFailure): return FailureClass.STRUCTURED_OUTPUT_ERROR
    if error is not None: return FailureClass.INFRASTRUCTURE_ERROR
    if result is not None and not result.passed: return FailureClass.SEMANTIC_MISMATCH
    return FailureClass.EVALUATED


def run_controlled_scenario(scenario: GoldenScenario, llm_client: Any) -> ScenarioRunArtifact:
    instrumented = InstrumentedLLM(llm_client)
    retriever = ControlledScenarioRetriever([[d.model_copy(deep=True) for d in p.documents] for p in scenario.retrieval.passes])
    state: dict[str, Any] = initial_state(scenario)
    result = None
    error = None
    started = time.monotonic()
    try:
        with ExitStack() as stack:
            stack.enter_context(_remote_tracing_disabled())
            stack.enter_context(patch.object(nodes, "get_llm", lambda: instrumented))
            stack.enter_context(patch.object(nodes, "FinGuardRetriever", lambda **kwargs: retriever))
            state = build_finguard_graph().invoke(state)
        result = evaluate_scenario(scenario, state, instrumented.returned[CriticAssessment], [c.query for c in retriever.calls], execution_mode=EXECUTION_MODE)
    except Exception as exc:
        error = exc
    elapsed = time.monotonic() - started
    failure_class = _classify(error, result)
    warnings = ["LLM usage metadata was not reported"] if any(c.usage_status == "not_reported" for c in instrumented.calls) else []
    return ScenarioRunArtifact(
        scenario_id=scenario.scenario_id, scenario_version=scenario.scenario_version,
        status="passed" if failure_class == FailureClass.EVALUATED else "failed", failure_class=failure_class,
        failure_message=str(error) if error else None, actual_extraction=state.get("extracted_entities"),
        actual_aml_assessment=state.get("aml_assessment"), requested_critic_actions=[r.recommended_action for r in instrumented.returned[CriticAssessment]],
        final_critic_action=(state.get("critic_assessment") or {}).get("recommended_action"), final_report=state.get("final_report"),
        loop_count=state.get("loop_count", 0), retrieval_queries=[c.query for c in retriever.calls], supplied_retrieval_passes=retriever.supplied_passes,
        matcher_result=result, prohibited_violations=[v.model_dump() for v in result.prohibited_violations] if result else [],
        llm_calls=instrumented.calls, usage=_aggregate_usage(instrumented.calls), latency_seconds=elapsed, warnings=warnings,
    )


def paid_run_selection(environ: dict[str, str], available_ids: list[str]) -> list[str]:
    if environ.get("FINGUARD_RUN_REAL_MODEL_EVAL") != "1": raise EvaluationAdapterError("real-model evaluation requires FINGUARD_RUN_REAL_MODEL_EVAL=1")
    raw = environ.get("FINGUARD_REAL_MODEL_SCENARIOS", "").strip()
    if not raw: raise EvaluationAdapterError("explicit FINGUARD_REAL_MODEL_SCENARIOS selection is required")
    selected = [item.strip() for item in raw.split(",") if item.strip()]
    if len(selected) != len(set(selected)): raise EvaluationAdapterError("scenario selection contains duplicate IDs")
    unknown = sorted(set(selected) - set(available_ids))
    if unknown: raise EvaluationAdapterError(f"unknown scenario IDs: {unknown}")
    maximum = int(environ.get("FINGUARD_REAL_MODEL_MAX_SCENARIOS", DEFAULT_MAX_SCENARIOS))
    if maximum < 1 or maximum > len(available_ids): raise EvaluationAdapterError("scenario maximum must be between 1 and dataset size")
    if len(selected) > maximum: raise EvaluationAdapterError(f"selected {len(selected)} scenarios but maximum is {maximum}")
    return selected


def write_artifact(artifact: EvaluationRunArtifact, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return destination


def build_run_artifact(manifest: DatasetManifest, results: list[ScenarioRunArtifact], *, provider: str, model: str, started_at: datetime | None = None) -> EvaluationRunArtifact:
    started = started_at or datetime.now(timezone.utc)
    completed = datetime.now(timezone.utc)
    calls = [call for result in results for call in result.llm_calls]
    try:
        git_commit = subprocess.run(
            ["git", "-c", "safe.directory=C:/dev/projects/finguard-orchestrator", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        git_commit = None
    return EvaluationRunArtifact(
        run_id=str(uuid.uuid4()), started_at=started.isoformat(), completed_at=completed.isoformat(), git_commit=git_commit,
        dataset_id=manifest.dataset_id, dataset_version=manifest.dataset_version, schema_version=manifest.schema_version,
        expectation_profile=manifest.expectation_profile, provider=provider, model=model,
        selected_scenario_ids=[result.scenario_id for result in results],
        scenario_versions={result.scenario_id: result.scenario_version for result in results}, scenarios=results,
        usage=_aggregate_usage(calls), latency_seconds=sum(result.latency_seconds for result in results),
    )
