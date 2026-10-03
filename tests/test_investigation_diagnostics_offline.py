from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
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
    collector.record_admission(1, len([c for c in chunks if c["rerank_score"] >= 0.15]))
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
    collector.record_admission(2, 0)
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
    first, second = InvestigationCollector(), InvestigationCollector()
    with ThreadPoolExecutor() as executor:
        list(executor.map(lambda collector: instance.retrieve("query", observer=collector), [first, second]))
    first.record_admission(1, 2)
    assert first.snapshot(completed()).retrieval_passes[0].admitted_count == 2
    assert second.snapshot(completed()).retrieval_passes[0].admitted_count is None
    assert all(c.admitted is None for c in second.snapshot(completed()).retrieval_passes[0].candidates)
    assert not hasattr(instance, "observer")


@pytest.mark.parametrize("refine", [False, True])
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
                    gaps = ["REGULATORY_CONTEXT"] if refine and calls[schema] == 1 else []
                    return AMLAssessment(risk_rating="Low", required_evidence_gaps=gaps,
                                         reasoning_summary="fixture", insufficient_evidence=bool(gaps))
                return CriticAssessment(is_sufficient=True, failure_type="NONE", recommended_action="GENERATE", critique="fixture")
            return SimpleNamespace(invoke=invoke)
        monkeypatch.setattr(nodes, "get_llm", lambda: SimpleNamespace(with_structured_output=structured))
        state = {"raw_query": "wire", "loop_count": 0, "max_loops": 2}
        return build_finguard_graph().invoke(state, config={"configurable": {"investigation_collector": observer}})
    collector = InvestigationCollector()
    result = execute(collector)
    diagnostics = collector.snapshot(result)
    assert diagnostics.status == "reported"
    assert diagnostics.critic_pass_count == (2 if refine else 1)
    assert diagnostics.refinement_count == int(refine)
    assert len(diagnostics.retrieval_passes) == (2 if refine else 1)
    assert all(p.admitted_count == 2 for p in diagnostics.retrieval_passes)
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
            collector.record_retrieval(1, candidate_count=0, reranked_count=0, shortlist_count=0, candidates=[])
            collector.record_admission(1, 0)
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
