from __future__ import annotations

import importlib
import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.graph.schemas import AMLAssessment, CriticAssessment, TransactionExtraction
from tests.evaluation.loader import load_golden_dataset
from tests.evaluation.real_model_runner import (
    ControlledScenarioRetriever,
    EvaluationAdapterError,
    FailureClass,
    InstrumentedLLM,
    StructuredOutputFailure,
    build_run_artifact,
    paid_run_selection,
    run_controlled_scenario,
    write_artifact,
)


class FakeRunnable:
    def __init__(self, owner, schema): self.owner, self.schema = owner, schema
    def invoke(self, prompt):
        self.owner.prompts.append(prompt)
        response = self.owner.responses[self.schema].popleft()
        if isinstance(response, Exception): raise response
        return response


class FakeClient:
    def __init__(self, responses):
        self.responses = {schema: deque(values) for schema, values in responses.items()}
        self.prompts = []
        self.include_raw = []
    def with_structured_output(self, schema, *, include_raw=False):
        self.include_raw.append(include_raw)
        return FakeRunnable(self, schema)


def raw_result(parsed, *, usage=None, request_id="req-safe", parsing_error=None):
    raw = SimpleNamespace(usage_metadata=usage, response_metadata={}, id=request_id)
    return {"raw": raw, "parsed": parsed, "parsing_error": parsing_error}


def normal_responses(*, critic_actions=("GENERATE",), mismatch=False, usage=True):
    token_usage = {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5} if usage else None
    extraction = TransactionExtraction(transaction_ids=[] if mismatch else ["TXN-SYN-200"], amount=1250, transaction_type="wire", regulations=[], suspected_patterns=[], jurisdiction=None)
    aml = AMLAssessment(risk_rating="High" if mismatch else "Low", suspicious_patterns=[], flagged_transactions=[], applicable_regulations=[], reasoning_summary="Evidence-grounded result.", insufficient_evidence=False)
    critics = [CriticAssessment(is_sufficient=action == "GENERATE", missing_evidence=[], failure_type="NONE" if action == "GENERATE" else "MISSING_REGULATORY_CONTEXT", recommended_action=action, critique="Controlled result.") for action in critic_actions]
    return {
        TransactionExtraction: [raw_result(extraction, usage=token_usage)],
        AMLAssessment: [raw_result(aml, usage=token_usage) for _ in critic_actions],
        CriticAssessment: [raw_result(value, usage=token_usage) for value in critics],
    }


@pytest.fixture(scope="module")
def dataset(): return load_golden_dataset()


@pytest.fixture
def ordinary(dataset): return next(s.model_copy(deep=True) for s in dataset[1] if s.scenario_id == "ordinary-wire-001")


def test_import_has_no_external_side_effects(monkeypatch):
    monkeypatch.setattr("src.llm.client.get_llm", lambda: pytest.fail("get_llm called"))
    monkeypatch.setattr("src.graph.nodes.FinGuardRetriever", lambda **kwargs: pytest.fail("retriever constructed"))
    assert importlib.import_module("tests.evaluation.real_model_runner") is not None


def test_expected_outcomes_are_not_visible_to_model(ordinary):
    sentinels = ["EXPECTED-ONLY-EXTRACTION", "EXPECTED-ONLY-AML", "EXPECTED-ONLY-CRITIC", "PROHIBITED-ONLY", "QUERY-MATCHER-ONLY", "ABSENT-FACT-ONLY"]
    ordinary.expected.extraction.transaction_ids.value = [sentinels[0]]
    ordinary.expected.aml_assessment.suspicious_patterns.value = [sentinels[1]]
    ordinary.expected.critic.actions.value = [sentinels[2]]
    ordinary.prohibited.unsupported_fact_terms.append(sentinels[3])
    ordinary.retrieval.passes[0].expected_query_contains.append(sentinels[4])
    ordinary.synthetic_facts.explicitly_absent.append(sentinels[5])
    client = FakeClient(normal_responses())
    run_controlled_scenario(ordinary, client)
    prompts = "\n".join(client.prompts)
    assert all(value not in prompts for value in sentinels)
    assert ordinary.input.query in prompts
    assert ordinary.retrieval.passes[0].documents[0].content in prompts


def test_controlled_retriever_is_lazy_and_preserves_documents(ordinary):
    batches = [[document.model_copy(deep=True) for document in ordinary.retrieval.passes[0].documents], []]
    retriever = ControlledScenarioRetriever(batches)
    first = retriever.retrieve(query="actual query", doc_type="swift_log", jurisdiction=None, top_k_vector=10, top_n_final=3)
    assert retriever.supplied_passes == 1
    assert first[0] == ordinary.retrieval.passes[0].documents[0].model_dump()
    assert retriever.calls[0].query == "actual query"
    assert retriever.retrieve(query="retry", doc_type=None, jurisdiction=None, top_k_vector=10, top_n_final=3) == []
    with pytest.raises(Exception, match="more retrieval passes"):
        retriever.retrieve(query="third", doc_type=None, jurisdiction=None, top_k_vector=10, top_n_final=3)


def test_graph_keeps_production_threshold_filtering(ordinary):
    ordinary.retrieval.passes[0].documents[0].rerank_score = 0.14
    client = FakeClient(normal_responses())
    run_controlled_scenario(ordinary, client)
    assert ordinary.retrieval.passes[0].documents[0].content not in client.prompts[1]


def test_instrumentation_aggregates_usage_and_returns_parsed():
    parsed = TransactionExtraction()
    client = FakeClient({TransactionExtraction: [raw_result(parsed, usage={"input_tokens": 4, "output_tokens": 6, "total_tokens": 10})]})
    proxy = InstrumentedLLM(client)
    assert proxy.with_structured_output(TransactionExtraction).invoke("prompt") is parsed
    call = proxy.calls[0]
    assert (call.input_tokens, call.output_tokens, call.total_tokens) == (4, 6, 10)
    assert call.schema_name == "TransactionExtraction" and call.latency_seconds >= 0
    assert client.include_raw == [True]


def test_missing_usage_is_warning_not_failure(ordinary):
    result = run_controlled_scenario(ordinary, FakeClient(normal_responses(usage=False)))
    assert result.failure_class == FailureClass.EVALUATED
    assert result.usage.usage_status == "not_reported"
    assert result.warnings


@pytest.mark.parametrize("payload", [raw_result(None), raw_result(None, parsing_error=ValueError("invalid"))])
def test_structured_output_failures_serialize(ordinary, payload):
    responses = normal_responses(); responses[TransactionExtraction] = [payload]
    result = run_controlled_scenario(ordinary, FakeClient(responses))
    assert result.failure_class == FailureClass.STRUCTURED_OUTPUT_ERROR
    json.loads(result.model_dump_json())


def test_provider_failure_preserves_completed_usage(ordinary):
    responses = normal_responses(); responses[AMLAssessment] = [RuntimeError("provider unavailable")]
    result = run_controlled_scenario(ordinary, FakeClient(responses))
    assert result.failure_class == FailureClass.INFRASTRUCTURE_ERROR
    assert result.llm_calls[0].usage_status == "reported"
    assert len(result.llm_calls) == 2


def test_schema_valid_wrong_answer_is_semantic_mismatch(ordinary):
    result = run_controlled_scenario(ordinary, FakeClient(normal_responses(mismatch=True)))
    assert result.failure_class == FailureClass.SEMANTIC_MISMATCH


def test_unexpected_extra_retrieval_is_semantic_mismatch(ordinary):
    ordinary.input.max_loops = 3
    result = run_controlled_scenario(ordinary, FakeClient(normal_responses(critic_actions=("RETRIEVE_MORE", "RETRIEVE_MORE", "GENERATE"))))
    assert result.failure_class == FailureClass.SEMANTIC_MISMATCH
    assert "more retrieval passes" in result.failure_message


def test_artifact_contains_metadata_and_excludes_secrets(tmp_path, dataset, ordinary, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "SENTINEL-SECRET-KEY")
    result = run_controlled_scenario(ordinary, FakeClient(normal_responses()))
    artifact = build_run_artifact(
        dataset[0],
        [result],
        provider="xAI",
        model="fake",
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    path = write_artifact(artifact, tmp_path / "run.json")
    payload = path.read_text(encoding="utf-8")
    assert artifact.dataset_id in payload and ordinary.scenario_id in payload
    assert "actual_extraction" in payload and "llm_calls" in payload
    assert "SENTINEL-SECRET-KEY" not in payload and "authorization" not in payload.casefold()


def test_paid_run_guards(dataset):
    ids = [s.scenario_id for s in dataset[1]]
    with pytest.raises(EvaluationAdapterError): paid_run_selection({}, ids)
    with pytest.raises(EvaluationAdapterError): paid_run_selection({"FINGUARD_RUN_REAL_MODEL_EVAL": "1"}, ids)
    env = {"FINGUARD_RUN_REAL_MODEL_EVAL": "1", "FINGUARD_REAL_MODEL_SCENARIOS": ",".join(ids[:4])}
    with pytest.raises(EvaluationAdapterError): paid_run_selection(env, ids)
    env["FINGUARD_REAL_MODEL_MAX_SCENARIOS"] = "4"
    assert paid_run_selection(env, ids) == ids[:4]
    env["FINGUARD_REAL_MODEL_MAX_SCENARIOS"] = str(len(ids) + 1)
    with pytest.raises(EvaluationAdapterError): paid_run_selection(env, ids)


def test_runner_invokes_compiled_graph(monkeypatch, ordinary):
    import tests.evaluation.real_model_runner as runner
    actual_builder = runner.build_finguard_graph
    called = []
    def spy_builder(): called.append(True); return actual_builder()
    monkeypatch.setattr(runner, "build_finguard_graph", spy_builder)
    run_controlled_scenario(ordinary, FakeClient(normal_responses()))
    assert called == [True]
