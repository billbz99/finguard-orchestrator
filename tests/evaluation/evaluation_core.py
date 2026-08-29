from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.graph.schemas import CriticAssessment
from tests.evaluation.matchers import MatchResult, match_value
from tests.evaluation.scenario_models import GoldenScenario


class EvaluationFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    matcher: str
    expected: Any
    actual: Any
    message: str


class ProhibitedViolation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: str
    value: str
    location: str


class ScenarioEvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario_id: str
    scenario_version: int
    execution_mode: Literal["offline_replay", "real_model_controlled_retrieval"]
    passed: bool
    terminated: bool
    assertion_count: int = Field(ge=0)
    failed_assertions: list[EvaluationFailure]
    prohibited_violations: list[ProhibitedViolation]
    retrieval_queries: list[str]
    retrieval_count: int = Field(ge=0)
    critic_actions: list[str]
    critic_count: int = Field(ge=0)
    final_loop_count: int = Field(ge=0)
    final_status: str | None
    final_critic_action: str | None
    jurisdiction: str | None
    final_risk_rating: str | None
    flagged_wires: list[str]
    suspicious_patterns: list[str]


def initial_state(scenario: GoldenScenario) -> dict[str, Any]:
    return {
        "raw_query": scenario.input.query,
        "doc_type": None,
        "jurisdiction": None,
        "extracted_entities": {},
        "retrieved_context": [],
        "aml_assessment": None,
        "critic_assessment": None,
        "compliance_draft": None,
        "confidence_score": 0.0,
        "loop_count": 0,
        "max_loops": scenario.input.max_loops,
        "is_audit_complete": False,
        "final_report": None,
    }


def evaluate_prohibited_outcomes(scenario, state, critic_responses):
    extraction = state.get("extracted_entities") or {}
    assessment = state.get("aml_assessment") or {}
    report = state.get("final_report") or {}
    structured = {
        "transaction_ids": extraction.get("transaction_ids", []) + assessment.get("flagged_transactions", []) + report.get("flagged_wires", []),
        "regulations": extraction.get("regulations", []) + assessment.get("applicable_regulations", []) + report.get("applicable_regulations", []),
        "jurisdictions": [v for v in (state.get("jurisdiction"), extraction.get("jurisdiction")) if v is not None],
        "suspicious_patterns": extraction.get("suspected_patterns", []) + assessment.get("suspicious_patterns", []),
    }
    violations = []
    for category, actual in structured.items():
        normalized = {str(value).strip().casefold() for value in actual}
        for prohibited in getattr(scenario.prohibited, category):
            if prohibited.strip().casefold() in normalized:
                violations.append(ProhibitedViolation(category=category, value=prohibited, location="structured_output"))
    text = "\n".join([assessment.get("reasoning_summary", ""), report.get("audit_summary", ""), *[r.critique for r in critic_responses]]).casefold()
    for prohibited in scenario.prohibited.unsupported_fact_terms:
        if prohibited.casefold() in text:
            violations.append(ProhibitedViolation(category="unsupported_fact_terms", value=prohibited, location="generated_text"))
    return violations


def _add_match(failures, field, matcher, actual, *, order_sensitive=False):
    result: MatchResult = match_value(field, matcher, actual, order_sensitive=order_sensitive)
    if not result.passed:
        failures.append(EvaluationFailure(field=field, matcher=matcher.match, expected=result.expected, actual=result.actual, message=result.message))
    return 1


def evaluate_scenario(scenario, state, critic_responses, retrieval_queries, *, execution_mode):
    failures: list[EvaluationFailure] = []
    count = 0
    extraction = state.get("extracted_entities") or {}
    for field in ("transaction_ids", "amount", "transaction_type", "regulations", "suspected_patterns"):
        count += _add_match(failures, field, getattr(scenario.expected.extraction, field), extraction.get(field))
    count += _add_match(failures, "jurisdiction", scenario.expected.extraction.jurisdiction, state.get("jurisdiction"))
    count += _add_match(failures, "doc_type", scenario.expected.extraction.doc_type, state.get("doc_type"))
    assessment = state.get("aml_assessment") or {}
    for field in ("risk_rating", "suspicious_patterns", "flagged_transactions", "applicable_regulations", "insufficient_evidence"):
        count += _add_match(failures, field, getattr(scenario.expected.aml_assessment, field), assessment.get(field))
    actions = [r.recommended_action for r in critic_responses]
    failure_types = [r.failure_type for r in critic_responses]
    count += _add_match(failures, "critic_actions", scenario.expected.critic.actions, actions, order_sensitive=True)
    count += _add_match(failures, "failure_types", scenario.expected.critic.failure_types, failure_types, order_sensitive=True)
    report = state.get("final_report") or {}
    for field in ("assessment_status", "risk_rating", "flagged_wires", "applicable_regulations", "source_document_hashes"):
        count += _add_match(failures, field, getattr(scenario.expected.report, field), report.get(field))
    execution = {"retrieval_count": len(retrieval_queries), "critic_count": len(actions), "final_loop_count": state.get("loop_count", 0), "terminates": bool(report)}
    for field, actual in execution.items():
        count += _add_match(failures, field, getattr(scenario.expected.execution, field), actual)
    for index, retrieval_pass in enumerate(scenario.retrieval.passes):
        for term in retrieval_pass.expected_query_contains:
            count += 1
            query = retrieval_queries[index] if index < len(retrieval_queries) else None
            if query is None or term not in query:
                failures.append(EvaluationFailure(field=f"retrieval_queries[{index}]", matcher="contains", expected=term, actual=query, message=f"retrieval query {index} did not contain {term!r}"))
    violations = evaluate_prohibited_outcomes(scenario, state, critic_responses)
    return ScenarioEvaluationResult(
        scenario_id=scenario.scenario_id, scenario_version=scenario.scenario_version,
        execution_mode=execution_mode, passed=not failures and not violations,
        terminated=bool(report), assertion_count=count, failed_assertions=failures,
        prohibited_violations=violations, retrieval_queries=retrieval_queries,
        retrieval_count=len(retrieval_queries), critic_actions=actions,
        critic_count=len(actions), final_loop_count=state.get("loop_count", 0),
        final_status=report.get("assessment_status"),
        final_critic_action=(state.get("critic_assessment") or {}).get("recommended_action"),
        jurisdiction=state.get("jurisdiction"), final_risk_rating=report.get("risk_rating"),
        flagged_wires=report.get("flagged_wires", []), suspicious_patterns=assessment.get("suspicious_patterns", []),
    )
