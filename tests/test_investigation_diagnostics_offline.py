from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from threading import Barrier
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from src.graph import nodes
from src.graph.schemas import AMLAssessment, CriticAssessment, TransactionExtraction
from src.graph.workflow import build_finguard_graph
from src.ingestion.retriever import FinGuardRetriever
from src.observability.investigation import InvestigationCollector
from src.observability.llm_usage import LLMCallUsage, LLMUsageCollector
from src.ui.api_client import prepare_diagnostics


@pytest.fixture(autouse=True)
def disable_tracing(monkeypatch):
    for name in ("LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING", "LANGSMITH_TRACING_V2"):
        monkeypatch.setenv(name, "false")


def retriever():
    instance = FinGuardRetriever.__new__(FinGuardRetriever)
    instance.collection = SimpleNamespace(query=lambda **kwargs: {
        "documents": [["private-a", "private-b", "private-c", "private-d"]],
        "ids": [["a", "b", "c", "d"]],
        "metadatas": [[{"source": "fixture", "secret": "private"}] * 4],
        "distances": [[0.1, 0.2, 0.3, 0.4]],
    })
    instance.reranker = SimpleNamespace(predict=lambda pairs: [0.149, 0.15, 0.8, 0.01])
    return instance


def completed(count=1):
    return {"loop_count": count, "is_audit_complete": True}


def test_latency_complete_partial_missing_and_zero():
    collector = LLMUsageCollector(provider="xAI", model="configured")
    assert collector.snapshot().total_latency_ms == 0
    assert collector.snapshot().latency_status == "not_applicable"
    collector._next_call_index = 3
    collector._calls = [LLMCallUsage(call_index=1, usage_status="unavailable", latency_ms=12.5),
                        LLMCallUsage(call_index=2, usage_status="failed", latency_ms=7.5)]
    assert collector.snapshot().total_latency_ms == 20
    assert collector.snapshot().latency_status == "reported"
    run = uuid4()
    collector.on_chat_model_start({}, [[]], run_id=run)
    assert collector.snapshot().total_latency_ms == 20
    assert collector.snapshot().latency_status == "partial"
    collector._active.clear()
    collector._next_call_index = 3
    collector._calls[1].latency_ms = None
    assert collector.snapshot().total_latency_ms == 12.5
    assert collector.snapshot().latency_status == "partial"
    collector._calls[0].latency_ms = None
    assert collector.snapshot().total_latency_ms is None
    assert collector.snapshot().latency_status == "unavailable"
    run = uuid4()
    collector.on_chat_model_start({}, [[]], run_id=run)
    assert collector.snapshot().latency_status == "unavailable"


@pytest.mark.parametrize("location", ["message", "result", "missing"])
def test_reported_model_separate_from_configured(location):
    collector = LLMUsageCollector(provider="xAI", model="configured")
    run = uuid4()
    collector.on_chat_model_start({}, [[]], run_id=run)
    response = LLMResult(generations=[[ChatGeneration(message=AIMessage(
        content="private", response_metadata={"model_name": "reported"} if location == "message" else {}
    ))]], llm_output={"model": "reported"} if location == "result" else {})
    collector.on_llm_end(response, run_id=run)
    result = collector.snapshot()
    assert result.model == "configured"
    assert result.calls[0].reported_model == (None if location == "missing" else "reported")


def test_retrieval_alignment_and_admission():
    instance = retriever()
    collector = InvestigationCollector()
    chunks = instance.retrieve("query", top_k_vector=10, top_n_final=3, observer=collector)
    assert [c["id"] for c in chunks] == ["c", "b", "a"]
    collector.record_admission(1, [c["id"] for c in chunks if c["rerank_score"] >= 0.15])
    observation = collector.snapshot(completed()).retrieval_passes[0]
    assert (observation.candidate_count, observation.reranked_count, observation.shortlist_count,
            observation.admitted_count) == (4, 4, 3, 2)
    assert [c.vector_distance for c in observation.candidates] == [0.1, 0.2, 0.3, 0.4]
    assert [c.rerank_score for c in observation.candidates] == [0.149, 0.15, 0.8, 0.01]
    assert [c.admitted for c in observation.candidates] == [False, True, True, False]
    assert [c.shortlisted for c in observation.candidates] == [True, True, True, False]
    assert "private" not in observation.model_dump_json()
    instance.collection.query = lambda **kwargs: {"documents": [[]], "ids": [[]], "metadatas": [[]]}
    assert instance.retrieve("empty", observer=collector, pass_index=2) == []
    collector.record_admission(2, [])
    assert collector.snapshot(completed(2)).retrieval_passes[1].candidate_count == 0


def test_missing_distances_and_observer_failure():
    instance = retriever()
    original_query = instance.collection.query
    instance.collection.query = lambda **kwargs: {k: v for k, v in original_query(**kwargs).items() if k != "distances"}
    collector = InvestigationCollector()
    baseline = instance.retrieve("query", top_n_final=3)
    instance.retrieve("query", top_n_final=3, observer=collector)
    assert all(c.vector_distance is None for c in collector.snapshot(completed()).retrieval_passes[0].candidates)
    class Broken:
        def record_retrieval(self, *args, **kwargs):
            raise RuntimeError("telemetry")
    assert instance.retrieve("query", top_n_final=3, observer=Broken()) == baseline


def test_shared_retriever_concurrent_request_isolation():
    instance = retriever()
    barrier = Barrier(2)
    def interleaved_scores(pairs):
        barrier.wait(timeout=5)
        return [0.149, 0.15, 0.8, 0.01] if pairs[0][0] == "first" else [0.9, 0.1, 0.01, 0.02]
    instance.reranker.predict = interleaved_scores
    first, second = InvestigationCollector(), InvestigationCollector()
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(instance.retrieve, query, observer=collector)
                   for query, collector in (("first", first), ("second", second))]
        for future in futures:
            future.result(timeout=10)
    first.record_admission(1, ["b", "c"])
    assert first.snapshot(completed()).retrieval_passes[0].admitted_count == 2
    assert second.snapshot(completed()).retrieval_passes[0].admitted_count is None
    assert all(c.admitted is None for c in second.snapshot(completed()).retrieval_passes[0].candidates)
    assert not hasattr(instance, "observer")
    assert second.snapshot(completed()).retrieval_passes[0].candidates[0].rerank_score == 0.9


def test_admission_observes_ids_even_when_scores_disagree_and_validates():
    from pydantic import ValidationError
    collector = InvestigationCollector()
    collector.record_retrieval(1, candidate_ids=["a", "b"], candidate_count=2,
        reranked_count=2, shortlist_count=2, candidates=[
            {"candidate_index": 1, "rerank_score": 0.001, "shortlisted": True},
            {"candidate_index": 2, "rerank_score": 100, "shortlisted": True}])
    collector.record_admission(1, ["a"])
    observation = collector.snapshot(completed()).retrieval_passes[0]
    assert [c.admitted for c in observation.candidates] == [True, False]
    assert observation.admitted_count == 1
    collector._passes[1].candidate_count = -1
    with pytest.raises(ValidationError):
        collector.record_admission(1, [])


def test_ties_and_fewer_than_top_n():
    instance = retriever()
    instance.collection.query = lambda **kwargs: {"documents": [["x", "y"]],
        "ids": [["x", "y"]], "metadatas": [[{}, {}]], "distances": [[0.1, 0.2]]}
    instance.reranker.predict = lambda pairs: [0.15, 0.15]
    collector = InvestigationCollector()
    selected = instance.retrieve("query", top_n_final=3, observer=collector)
    collector.record_admission(1, [c["id"] for c in selected])
    observation = collector.snapshot(completed()).retrieval_passes[0]
    assert observation.shortlist_count == observation.admitted_count == 2
    assert all(c.admitted for c in observation.candidates)


@pytest.mark.parametrize("value", [None, "invalid", {}, 1, [None, {}, "invalid"]])
def test_null_and_non_list_ui_fields(value):
    payload = {"observability": {"llm_usage": {"calls": value},
        "investigation": {"retrieval_passes": value}}}
    result = prepare_diagnostics(payload)
    assert result["input_tokens"] == "Unavailable"
    result = prepare_diagnostics({"observability": {"investigation": {
        "retrieval_passes": [{"candidates": value}]}}})
    assert result["retrieval_passes"][0]["admitted_count"] is None


def test_empty_observer_failure_logs_safely(caplog):
    instance = retriever()
    instance.collection.query = lambda **kwargs: {"documents": [[]], "ids": [[]], "metadatas": [[]]}
    class Broken:
        def record_retrieval(self, *args, **kwargs):
            raise RuntimeError("PRIVATE QUERY SECRET")
    assert instance.retrieve("PRIVATE QUERY SECRET", observer=Broken()) == []
    assert "RuntimeError" in caplog.text
    assert "PRIVATE" not in caplog.text
    assert len(caplog.records) == 1


def test_broken_logging_handler_cannot_fail_observer(monkeypatch):
    from src.observability import investigation
    def fail(*args, **kwargs):
        raise RuntimeError("private")
    monkeypatch.setattr(investigation.logger, "warning", fail)
    investigation.observe(SimpleNamespace(record_admission=fail), "record_admission", 1, [])


@pytest.mark.parametrize("refine", [False, True, "max_loop"])
def test_graph_history_and_unchanged_report_on_observer_failure(monkeypatch, refine):
    instance = retriever()
    monkeypatch.setattr(nodes, "get_production_retriever", lambda: instance)
    def execute(observer):
        calls = {AMLAssessment: 0}
        def structured(schema):
            def invoke(prompt):
                if schema is TransactionExtraction:
                    return TransactionExtraction()
                if schema is AMLAssessment:
                    calls[schema] += 1
                    gaps = ["REGULATORY_CONTEXT"] if refine == "max_loop" or (refine and calls[schema] == 1) else []
                    return AMLAssessment(risk_rating="Low", required_evidence_gaps=gaps,
                                         reasoning_summary="fixture", insufficient_evidence=bool(gaps))
                return CriticAssessment(is_sufficient=False, failure_type="MISSING_REGULATORY_CONTEXT", recommended_action="RETRIEVE_MORE", critique="fixture") if refine == "max_loop" else CriticAssessment(is_sufficient=True, failure_type="NONE", recommended_action="GENERATE", critique="fixture")
            return SimpleNamespace(invoke=invoke)
        monkeypatch.setattr(nodes, "get_llm", lambda: SimpleNamespace(with_structured_output=structured))
        state = {"raw_query": "wire", "loop_count": 0, "max_loops": 2}
        return build_finguard_graph().invoke(state, config={"configurable": {"investigation_collector": observer}})
    collector = InvestigationCollector()
    result = execute(collector)
    diagnostics = collector.snapshot(result)
    assert diagnostics.status == "reported"
    assert diagnostics.critic_pass_count == (2 if refine else 1)
    assert diagnostics.refinement_count == int(bool(refine))
    assert len(diagnostics.retrieval_passes) == (2 if refine else 1)
    assert all(p.admitted_count == 2 for p in diagnostics.retrieval_passes)
    if refine == "max_loop":
        assert result["critic_assessment"]["recommended_action"] == "STOP_INSUFFICIENT"
        assert result["final_report"]["assessment_status"] == "INSUFFICIENT_EVIDENCE"
    class Broken:
        def record_retrieval(self, *args, **kwargs): raise RuntimeError("telemetry")
        def record_admission(self, *args, **kwargs): raise RuntimeError("telemetry")
    assert execute(Broken())["final_report"] == result["final_report"]


def test_api_additive_and_report_only_cache(monkeypatch):
    from src import main
    report = {"assessment_status": "COMPLETE", "risk_rating": "LOW", "audit_summary": "fixture"}
    class Graph:
        async def ainvoke(self, state, config):
            collector = config["configurable"]["investigation_collector"]
            collector.record_retrieval(1, candidate_count=0, reranked_count=0, shortlist_count=0, candidates=[], candidate_ids=[])
            collector.record_admission(1, [])
            return {**completed(), "final_report": report}
    saved = []
    monkeypatch.setattr(main, "graph", Graph())
    monkeypatch.setattr(main, "get_semantic_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "set_semantic_cache", lambda query, value: saved.append(value))
    monkeypatch.setattr(main, "route_incoming_audit", lambda **kwargs: "AGENTIC_GRAPH")
    body = TestClient(main.app).post("/api/v1/audit", json={"query": "fixture"}).json()
    assert body["observability"]["investigation"]["critic_pass_count"] == 1
    assert saved == [report]
    monkeypatch.setattr(main.InvestigationCollector, "snapshot", lambda *args: (_ for _ in ()).throw(RuntimeError()))
    body = TestClient(main.app).post("/api/v1/audit", json={"query": "fixture"}).json()
    assert body["report"] == report
    assert body["observability"]["investigation"] is None
    assert body["observability"]["llm_usage"] is not None


def test_ui_legacy_fallback_zero_and_safe_tables():
    missing = prepare_diagnostics({})
    assert missing["critic_passes"] == missing["input_tokens"] == "Unavailable"
    payload = {"observability": {"llm_usage": {"input_tokens": 0, "total_latency_ms": 0,
        "calls": [{"node": "aml_audit", "content": "private", "metadata": {"secret": "private"}}]},
        "investigation": {"critic_pass_count": 0, "refinement_count": 0}}}
    result = prepare_diagnostics(payload)
    assert result["input_tokens"] == result["critic_passes"] == "0"
    assert result["llm_latency"] == "0.0 s"
    assert "private" not in str(result)


def test_unknown_workflow_not_zero_filled():
    assert InvestigationCollector().snapshot({}).critic_pass_count is None
    assert InvestigationCollector().snapshot({"loop_count": 1}).refinement_count is None
    assert InvestigationCollector().snapshot(None).refinement_count == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_api_candidate_exposure_setting(monkeypatch, enabled):
    from src import main
    collector = InvestigationCollector()
    selected = retriever().retrieve("private", observer=collector)
    collector.record_admission(1, [c["id"] for c in selected])
    monkeypatch.delenv("FINGUARD_EXPOSE_RETRIEVAL_DETAILS", raising=False)
    if enabled:
        monkeypatch.setenv("FINGUARD_EXPOSE_RETRIEVAL_DETAILS", "1")
    envelope = main._safe_observability(LLMUsageCollector(provider="xAI", model="fixture"), collector, completed())
    observation = envelope.investigation.retrieval_passes[0]
    assert observation.candidate_count == 4
    assert observation.admitted_count == 4
    assert bool(observation.candidates) == enabled
    assert "private" not in envelope.model_dump_json()
    ui = prepare_diagnostics({"observability": envelope.model_dump()})
    assert bool(ui["scores"]) == enabled


@pytest.mark.parametrize("cache_hit", [True, False])
def test_api_zero_work_paths(monkeypatch, cache_hit):
    from src import main
    report = {"assessment_status": "COMPLETE", "risk_rating": "LOW", "audit_summary": "fixture"}
    monkeypatch.setattr(main, "get_semantic_cache", lambda *a, **k: report if cache_hit else None)
    monkeypatch.setattr(main, "route_incoming_audit", lambda **k: "DETERMINISTIC_PASS")
    monkeypatch.setattr(main, "run_deterministic_ach_check", lambda state: report)
    saved = []
    monkeypatch.setattr(main, "set_semantic_cache", lambda query, value: saved.append(value))
    body = TestClient(main.app).post("/api/v1/audit", json={"query": "fixture"}).json()
    diagnostics = body["observability"]
    assert diagnostics["investigation"] == {"status": "not_applicable", "critic_pass_count": 0,
        "refinement_count": 0, "retrieval_passes": []}
    assert diagnostics["llm_usage"]["logical_call_count"] == 0
    assert diagnostics["llm_usage"]["total_latency_ms"] == 0
    assert saved == ([] if cache_hit else [report])


def test_non_string_model_does_not_mask_reported_identity():
    collector = LLMUsageCollector(provider="xAI", model="configured")
    run = uuid4()
    collector.on_chat_model_start({}, [[]], run_id=run)
    collector.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(
        content="fixture", response_metadata={"model_name": 123, "model": "reported"}
    ))]]), run_id=run)
    assert collector.snapshot().calls[0].reported_model == "reported"
