from __future__ import annotations

import os
from collections import defaultdict, deque
from contextlib import ExitStack, contextmanager
from typing import Any, Literal
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict, Field

from src.graph import nodes
from src.graph.schemas import (
    AMLAssessment,
    CriticAssessment,
    TransactionExtraction,
)
from src.graph.workflow import build_finguard_graph
from tests.evaluation.matchers import MatchResult, match_value
from tests.evaluation.scenario_models import (
    AllowedMatcher,
    ExactMatcher,
    GoldenScenario,
    SubsetMatcher,
)


TRACING_ENVIRONMENT_VARIABLES = (
    "LANGCHAIN_TRACING",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGSMITH_TRACING",
)


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
    execution_mode: Literal["offline_replay"] = "offline_replay"
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


def _matcher_value(matcher):
    if isinstance(matcher, ExactMatcher):
        return matcher.value
    if isinstance(matcher, (AllowedMatcher, SubsetMatcher)):
        return matcher.values[0] if isinstance(matcher, AllowedMatcher) else matcher.values
    raise ValueError(
        f"matcher {type(matcher).__name__} cannot drive a structured fake response"
    )


def _response_plan(scenario: GoldenScenario):
    expected = scenario.expected
    extraction = TransactionExtraction(
        transaction_ids=_matcher_value(expected.extraction.transaction_ids),
        amount=_matcher_value(expected.extraction.amount),
        transaction_type=_matcher_value(expected.extraction.transaction_type),
        regulations=_matcher_value(expected.extraction.regulations),
        suspected_patterns=_matcher_value(expected.extraction.suspected_patterns),
        jurisdiction=_matcher_value(expected.extraction.jurisdiction),
    )
    final_aml_assessment = AMLAssessment(
        risk_rating=_matcher_value(expected.aml_assessment.risk_rating),
        suspicious_patterns=_matcher_value(
            expected.aml_assessment.suspicious_patterns
        ),
        flagged_transactions=_matcher_value(
            expected.aml_assessment.flagged_transactions
        ),
        applicable_regulations=_matcher_value(
            expected.aml_assessment.applicable_regulations
        ),
        reasoning_summary="Synthetic offline replay assessment.",
        insufficient_evidence=_matcher_value(
            expected.aml_assessment.insufficient_evidence
        ),
    )
    actions = _matcher_value(expected.critic.actions)
    failure_types = _matcher_value(expected.critic.failure_types)
    aml_assessments = []
    for index, action in enumerate(actions):
        if action == "RETRIEVE_MORE" and index < len(actions) - 1:
            aml_assessments.append(
                final_aml_assessment.model_copy(
                    update={
                        "risk_rating": "Low",
                        "suspicious_patterns": [],
                        "flagged_transactions": [],
                        "applicable_regulations": [],
                        "reasoning_summary": (
                            "Synthetic offline replay requires more regulatory context."
                        ),
                        "insufficient_evidence": True,
                    }
                )
            )
        else:
            aml_assessments.append(final_aml_assessment)
    critics = [
        CriticAssessment(
            is_sufficient=action == "GENERATE",
            missing_evidence=[] if action == "GENERATE" else ["synthetic evidence"],
            failure_type=failure_type,
            recommended_action=action,
            critique=f"Synthetic offline replay selected {action}.",
        )
        for action, failure_type in zip(actions, failure_types, strict=True)
    ]
    return {
        TransactionExtraction: [extraction],
        AMLAssessment: aml_assessments,
        CriticAssessment: critics,
    }


class _FakeStructuredLLM:
    def __init__(self, controller, schema):
        self.controller = controller
        self.schema = schema

    def invoke(self, prompt):
        response = self.controller.responses[self.schema].popleft()
        self.controller.returned[self.schema].append(response)
        return response


class _FakeLLM:
    def __init__(self, controller):
        self.controller = controller

    def with_structured_output(self, schema):
        return _FakeStructuredLLM(self.controller, schema)


class _FakeLLMController:
    def __init__(self, response_plan):
        self.responses = {
            schema: deque(responses) for schema, responses in response_plan.items()
        }
        self.returned = defaultdict(list)

    def get_llm(self):
        return _FakeLLM(self)


class _FakeRetrieverController:
    def __init__(self, scenario):
        self.batches = deque(
            [
                [document.model_dump() for document in retrieval_pass.documents]
                for retrieval_pass in scenario.retrieval.passes
            ]
        )
        self.queries = []

    def build(self, **kwargs):
        return self

    def retrieve(self, **kwargs):
        self.queries.append(kwargs["query"])
        if not self.batches:
            raise AssertionError("graph requested more retrieval passes than declared")
        return self.batches.popleft()


@contextmanager
def _remote_tracing_disabled():
    saved = {
        variable: os.environ.pop(variable)
        for variable in TRACING_ENVIRONMENT_VARIABLES
        if variable in os.environ
    }
    try:
        yield
    finally:
        os.environ.update(saved)


def _initial_state(scenario):
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


def _add_match(failures, field, matcher, actual, *, order_sensitive=False):
    result: MatchResult = match_value(
        field,
        matcher,
        actual,
        order_sensitive=order_sensitive,
    )
    if not result.passed:
        failures.append(
            EvaluationFailure(
                field=field,
                matcher=matcher.match,
                expected=result.expected,
                actual=result.actual,
                message=result.message,
            )
        )
    return 1


def evaluate_prohibited_outcomes(
    scenario: GoldenScenario,
    state: dict[str, Any],
    critic_responses: list[CriticAssessment],
) -> list[ProhibitedViolation]:
    extraction = state.get("extracted_entities") or {}
    assessment = state.get("aml_assessment") or {}
    report = state.get("final_report") or {}
    structured = {
        "transaction_ids": (
            extraction.get("transaction_ids", [])
            + assessment.get("flagged_transactions", [])
            + report.get("flagged_wires", [])
        ),
        "regulations": (
            extraction.get("regulations", [])
            + assessment.get("applicable_regulations", [])
            + report.get("applicable_regulations", [])
        ),
        "jurisdictions": [
            value
            for value in (
                state.get("jurisdiction"),
                extraction.get("jurisdiction"),
            )
            if value is not None
        ],
        "suspicious_patterns": (
            extraction.get("suspected_patterns", [])
            + assessment.get("suspicious_patterns", [])
        ),
    }
    violations = []
    for category in (
        "transaction_ids",
        "regulations",
        "jurisdictions",
        "suspicious_patterns",
    ):
        actual_values = {str(value).strip().casefold() for value in structured[category]}
        for prohibited in getattr(scenario.prohibited, category):
            if prohibited.strip().casefold() in actual_values:
                violations.append(
                    ProhibitedViolation(
                        category=category,
                        value=prohibited,
                        location="structured_output",
                    )
                )

    text_outputs = [
        assessment.get("reasoning_summary", ""),
        report.get("audit_summary", ""),
        *[response.critique for response in critic_responses],
    ]
    combined_text = "\n".join(text_outputs).casefold()
    for prohibited in scenario.prohibited.unsupported_fact_terms:
        if prohibited.casefold() in combined_text:
            violations.append(
                ProhibitedViolation(
                    category="unsupported_fact_terms",
                    value=prohibited,
                    location="generated_text",
                )
            )
    return violations


def run_offline_replay(scenario: GoldenScenario) -> ScenarioEvaluationResult:
    """Runs one golden scenario through the real graph with deterministic fakes."""
    llm = _FakeLLMController(_response_plan(scenario))
    retriever = _FakeRetrieverController(scenario)
    with ExitStack() as stack:
        stack.enter_context(_remote_tracing_disabled())
        stack.enter_context(patch.object(nodes, "get_llm", llm.get_llm))
        stack.enter_context(
            patch.object(nodes, "FinGuardRetriever", retriever.build)
        )
        state = build_finguard_graph().invoke(_initial_state(scenario))

    critic_responses = llm.returned[CriticAssessment]
    critic_actions = [response.recommended_action for response in critic_responses]
    critic_failure_types = [response.failure_type for response in critic_responses]
    failures = []
    assertion_count = 0

    extraction = state["extracted_entities"]
    extraction_expected = scenario.expected.extraction
    for field in (
        "transaction_ids",
        "amount",
        "transaction_type",
        "regulations",
        "suspected_patterns",
    ):
        assertion_count += _add_match(
            failures,
            field,
            getattr(extraction_expected, field),
            extraction.get(field),
        )
    assertion_count += _add_match(
        failures, "jurisdiction", extraction_expected.jurisdiction, state["jurisdiction"]
    )
    assertion_count += _add_match(
        failures, "doc_type", extraction_expected.doc_type, state["doc_type"]
    )

    aml_expected = scenario.expected.aml_assessment
    for field in (
        "risk_rating",
        "suspicious_patterns",
        "flagged_transactions",
        "applicable_regulations",
        "insufficient_evidence",
    ):
        assertion_count += _add_match(
            failures,
            field,
            getattr(aml_expected, field),
            state["aml_assessment"].get(field),
        )

    assertion_count += _add_match(
        failures,
        "critic_actions",
        scenario.expected.critic.actions,
        critic_actions,
        order_sensitive=True,
    )
    assertion_count += _add_match(
        failures,
        "failure_types",
        scenario.expected.critic.failure_types,
        critic_failure_types,
        order_sensitive=True,
    )

    report_expected = scenario.expected.report
    for field in (
        "assessment_status",
        "risk_rating",
        "flagged_wires",
        "applicable_regulations",
        "source_document_hashes",
    ):
        assertion_count += _add_match(
            failures,
            field,
            getattr(report_expected, field),
            state["final_report"].get(field),
        )

    execution_values = {
        "retrieval_count": len(retriever.queries),
        "critic_count": len(critic_actions),
        "final_loop_count": state["loop_count"],
        "terminates": state.get("final_report") is not None,
    }
    for field, actual in execution_values.items():
        assertion_count += _add_match(
            failures,
            field,
            getattr(scenario.expected.execution, field),
            actual,
        )

    for index, retrieval_pass in enumerate(scenario.retrieval.passes):
        for expected_term in retrieval_pass.expected_query_contains:
            assertion_count += 1
            actual_query = retriever.queries[index] if index < len(retriever.queries) else None
            if actual_query is None or expected_term not in actual_query:
                failures.append(
                    EvaluationFailure(
                        field=f"retrieval_queries[{index}]",
                        matcher="contains",
                        expected=expected_term,
                        actual=actual_query,
                        message=(
                            f"retrieval query {index} did not contain {expected_term!r}"
                        ),
                    )
                )

    violations = evaluate_prohibited_outcomes(scenario, state, critic_responses)
    return ScenarioEvaluationResult(
        scenario_id=scenario.scenario_id,
        scenario_version=scenario.scenario_version,
        passed=not failures and not violations,
        terminated=state.get("final_report") is not None,
        assertion_count=assertion_count,
        failed_assertions=failures,
        prohibited_violations=violations,
        retrieval_queries=retriever.queries,
        retrieval_count=len(retriever.queries),
        critic_actions=critic_actions,
        critic_count=len(critic_actions),
        final_loop_count=state["loop_count"],
        final_status=(state.get("final_report") or {}).get("assessment_status"),
        final_critic_action=(state.get("critic_assessment") or {}).get(
            "recommended_action"
        ),
        jurisdiction=state.get("jurisdiction"),
        final_risk_rating=(state.get("final_report") or {}).get("risk_rating"),
        flagged_wires=(state.get("final_report") or {}).get("flagged_wires", []),
        suspicious_patterns=(state.get("aml_assessment") or {}).get(
            "suspicious_patterns", []
        ),
    )
