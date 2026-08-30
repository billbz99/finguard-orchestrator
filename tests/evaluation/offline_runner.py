from __future__ import annotations

import os
from collections import defaultdict, deque
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from src.graph import nodes
from src.graph.schemas import AMLAssessment, CriticAssessment, TransactionExtraction
from src.graph.workflow import build_finguard_graph
from tests.evaluation.evaluation_core import ScenarioEvaluationResult, evaluate_prohibited_outcomes, evaluate_scenario, initial_state
from tests.evaluation.scenario_models import AllowedMatcher, ExactMatcher, GoldenScenario, SubsetMatcher

TRACING_ENVIRONMENT_VARIABLES = ("LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGSMITH_API_KEY", "LANGSMITH_TRACING")


def _matcher_value(matcher):
    if isinstance(matcher, ExactMatcher):
        return matcher.value
    if isinstance(matcher, AllowedMatcher):
        return matcher.values[0]
    if isinstance(matcher, SubsetMatcher):
        return matcher.values
    raise ValueError(f"matcher {type(matcher).__name__} cannot drive a structured fake response")


def _response_plan(scenario: GoldenScenario):
    expected = scenario.expected
    extraction = TransactionExtraction(
        transaction_ids=_matcher_value(expected.extraction.transaction_ids), amount=_matcher_value(expected.extraction.amount),
        transaction_type=_matcher_value(expected.extraction.transaction_type), regulations=_matcher_value(expected.extraction.regulations),
        suspected_patterns=_matcher_value(expected.extraction.suspected_patterns), jurisdiction=_matcher_value(expected.extraction.jurisdiction),
    )
    actions = _matcher_value(expected.critic.actions)
    failure_types = _matcher_value(expected.critic.failure_types)
    final_failure_type = failure_types[-1]
    if final_failure_type == "MISSING_REGULATORY_CONTEXT":
        final_gaps = ["REGULATORY_CONTEXT"]
    elif final_failure_type == "MISSING_TRANSACTION_DATA":
        final_gaps = ["AMOUNT"]
    else:
        final_gaps = []
    final = AMLAssessment(
        risk_rating=_matcher_value(expected.aml_assessment.risk_rating), suspicious_patterns=_matcher_value(expected.aml_assessment.suspicious_patterns),
        flagged_transactions=_matcher_value(expected.aml_assessment.flagged_transactions), applicable_regulations=_matcher_value(expected.aml_assessment.applicable_regulations),
        required_evidence_gaps=final_gaps,
        reasoning_summary="Synthetic offline replay assessment.", insufficient_evidence=_matcher_value(expected.aml_assessment.insufficient_evidence),
    )
    assessments = [final.model_copy(update={"risk_rating": "Low", "suspicious_patterns": [], "flagged_transactions": [], "applicable_regulations": [], "required_evidence_gaps": ["REGULATORY_CONTEXT"], "reasoning_summary": "Synthetic offline replay requires more regulatory context.", "insufficient_evidence": True}) if action == "RETRIEVE_MORE" and index < len(actions) - 1 else final for index, action in enumerate(actions)]
    critics = [CriticAssessment(is_sufficient=action == "GENERATE", missing_evidence=[] if action == "GENERATE" else ["synthetic evidence"], failure_type=failure_type, recommended_action=action, critique=f"Synthetic offline replay selected {action}.") for action, failure_type in zip(actions, failure_types, strict=True)]
    return {TransactionExtraction: [extraction], AMLAssessment: assessments, CriticAssessment: critics}


class _FakeStructuredLLM:
    def __init__(self, controller, schema): self.controller, self.schema = controller, schema
    def invoke(self, prompt):
        response = self.controller.responses[self.schema].popleft()
        self.controller.returned[self.schema].append(response)
        return response


class _FakeLLM:
    def __init__(self, controller): self.controller = controller
    def with_structured_output(self, schema): return _FakeStructuredLLM(self.controller, schema)


class _FakeLLMController:
    def __init__(self, plan):
        self.responses = {schema: deque(values) for schema, values in plan.items()}
        self.returned = defaultdict(list)
    def get_llm(self): return _FakeLLM(self)


class _FakeRetrieverController:
    def __init__(self, scenario):
        self.batches = deque([[d.model_dump() for d in p.documents] for p in scenario.retrieval.passes])
        self.queries = []
    def build(self, **kwargs): return self
    def retrieve(self, **kwargs):
        self.queries.append(kwargs["query"])
        if not self.batches: raise AssertionError("graph requested more retrieval passes than declared")
        return self.batches.popleft()


@contextmanager
def _remote_tracing_disabled():
    saved = {key: os.environ.pop(key) for key in TRACING_ENVIRONMENT_VARIABLES if key in os.environ}
    try: yield
    finally: os.environ.update(saved)


def run_offline_replay(scenario: GoldenScenario) -> ScenarioEvaluationResult:
    """Runs one golden scenario through the real graph with deterministic fakes."""
    llm = _FakeLLMController(_response_plan(scenario))
    retriever = _FakeRetrieverController(scenario)
    with ExitStack() as stack:
        stack.enter_context(_remote_tracing_disabled())
        stack.enter_context(patch.object(nodes, "get_llm", llm.get_llm))
        stack.enter_context(patch.object(nodes, "FinGuardRetriever", retriever.build))
        state = build_finguard_graph().invoke(initial_state(scenario))
    return evaluate_scenario(scenario, state, llm.returned[CriticAssessment], retriever.queries, execution_mode="offline_replay")
