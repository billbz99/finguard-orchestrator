from decimal import Decimal
from typing import TypedDict
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langgraph.graph import END, StateGraph

from src.graph.schemas import ComplianceReport
from src.observability.llm_usage import LLMPricing, LLMUsageCollector, load_xai_pricing


@pytest.fixture(autouse=True)
def _disable_remote_tracing(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGSMITH_TRACING_V2", "false")
    for name in (
        "XAI_PRICE_MODEL",
        "XAI_INPUT_PRICE_PER_MILLION",
        "XAI_OUTPUT_PRICE_PER_MILLION",
        "XAI_CACHED_INPUT_PRICE_PER_MILLION",
        "XAI_PRICING_REVISION",
    ):
        monkeypatch.delenv(name, raising=False)


def _result(*, usage=None, raw_usage=None, request_id="request-1"):
    message = AIMessage(content="offline", id=request_id)
    if usage is not None:
        message.usage_metadata = usage
    return LLMResult(
        generations=[[ChatGeneration(message=message)]],
        llm_output={"token_usage": raw_usage} if raw_usage is not None else None,
    )


def _record(collector, *, usage=None, raw_usage=None, node="aml_audit"):
    run_id = uuid4()
    collector.on_chat_model_start(
        {}, [[]], run_id=run_id, metadata={"langgraph_node": node}
    )
    collector.on_llm_end(
        _result(usage=usage, raw_usage=raw_usage), run_id=run_id
    )


def _normalized(input_tokens=10, output_tokens=4, total_tokens=14):
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _report():
    return {
        "assessment_status": "COMPLETE",
        "risk_rating": "LOW",
        "flagged_wires": [],
        "applicable_regulations": [],
        "audit_summary": "Offline result.",
        "source_document_hashes": [],
    }


def test_normalized_usage_cached_reasoning_and_request_id():
    collector = LLMUsageCollector(provider="xAI", model="offline")
    usage = _normalized()
    usage["input_token_details"] = {"cache_read": 3}
    usage["output_token_details"] = {"reasoning": 2}
    _record(collector, usage=usage)

    snapshot = collector.snapshot()
    assert snapshot.usage_status == "reported"
    assert (snapshot.input_tokens, snapshot.output_tokens, snapshot.total_tokens) == (10, 4, 14)
    assert snapshot.cached_input_tokens == 3
    assert snapshot.reasoning_tokens == 2
    assert snapshot.calls[0].provider_request_id == "request-1"
    assert snapshot.calls[0].node == "aml_audit"


def test_openai_token_usage_fallback():
    collector = LLMUsageCollector(provider="xAI", model="offline")
    _record(
        collector,
        raw_usage={
            "prompt_tokens": 7,
            "completion_tokens": 5,
            "total_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 2},
            "completion_tokens_details": {"reasoning_tokens": 1},
        },
    )
    snapshot = collector.snapshot()
    assert snapshot.input_tokens == 7
    assert snapshot.output_tokens == 5
    assert snapshot.total_tokens == 12
    assert snapshot.cached_input_tokens == 2
    assert snapshot.reasoning_tokens == 1


def test_missing_and_partial_usage_are_never_zero_filled():
    unavailable = LLMUsageCollector(provider="xAI", model="offline")
    _record(unavailable)
    result = unavailable.snapshot()
    assert result.usage_status == "unavailable"
    assert result.logical_call_count == result.completed_call_count == 1
    assert result.total_tokens is None
    assert result.calls[0].usage_status == "unavailable"

    partial = LLMUsageCollector(provider="xAI", model="offline")
    _record(partial, usage=_normalized())
    _record(partial)
    result = partial.snapshot()
    assert result.usage_status == "partial"
    assert result.total_tokens is None
    assert [call.usage_status for call in result.calls] == ["reported", "unavailable"]


@pytest.mark.parametrize("call_count", [3, 5])
def test_successful_multi_call_aggregation(call_count):
    collector = LLMUsageCollector(provider="xAI", model="offline")
    nodes = ["extraction", "aml_audit", "auditor_critic", "aml_audit", "auditor_critic"]
    for index in range(call_count):
        _record(collector, usage=_normalized(2, 1, 3), node=nodes[index])
    result = collector.snapshot()
    assert result.logical_call_count == result.completed_call_count == call_count
    assert result.failed_call_count == 0
    assert result.total_tokens == call_count * 3


def test_failed_call_after_success_is_recorded_without_tokens():
    collector = LLMUsageCollector(provider="xAI", model="offline")
    _record(collector, usage=_normalized())
    run_id = uuid4()
    collector.on_chat_model_start({}, [[]], run_id=run_id)
    collector.on_llm_error(RuntimeError("secret response"), run_id=run_id)
    result = collector.snapshot()
    assert result.usage_status == "partial"
    assert result.logical_call_count == 2
    assert result.completed_call_count == 1
    assert result.failed_call_count == 1
    assert result.calls[1].total_tokens is None


class GraphState(TypedDict):
    value: str


class OfflineChatModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "offline"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="offline",
                        usage_metadata=_normalized(3, 2, 5),
                    )
                )
            ]
        )


def test_callback_propagates_through_compiled_langgraph_with_node_attribution():
    model = OfflineChatModel()
    builder = StateGraph(GraphState)

    def aml_audit(state):
        model.invoke("offline")
        return state

    builder.add_node("aml_audit", aml_audit)
    builder.set_entry_point("aml_audit")
    builder.add_edge("aml_audit", END)
    graph = builder.compile()
    collector = LLMUsageCollector(provider="xAI", model="offline")

    graph.invoke({"value": "safe"}, config={"callbacks": [collector]})

    result = collector.snapshot()
    assert result.logical_call_count == 1
    assert result.total_tokens == 5
    assert result.calls[0].node == "aml_audit"


def test_collectors_are_request_scoped_and_callback_errors_do_not_escape(monkeypatch):
    first = LLMUsageCollector(provider="xAI", model="offline")
    second = LLMUsageCollector(provider="xAI", model="offline")
    _record(first, usage=_normalized())
    assert first.snapshot().logical_call_count == 1
    assert second.snapshot().logical_call_count == 0

    monkeypatch.setattr(first, "_start", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("safe")))
    first.on_chat_model_start({}, [[]], run_id=uuid4())


def test_cache_hit_reports_zero_new_usage_and_does_not_replay_or_persist(monkeypatch):
    from src import main

    historical = {**_report(), "observability": {"llm_usage": {"total_tokens": 999}}}
    monkeypatch.setattr(main, "get_semantic_cache", lambda *args, **kwargs: historical)
    saved = []
    monkeypatch.setattr(main, "set_semantic_cache", lambda *args: saved.append(args))
    response = TestClient(main.app).post("/api/v1/audit", json={"query": "cached"})
    usage = response.json()["observability"]["llm_usage"]
    assert usage["usage_status"] == "not_applicable"
    assert usage["logical_call_count"] == 0
    assert usage["total_tokens"] == 0
    assert saved == []


def test_deterministic_path_reports_zero_and_caches_report_only(monkeypatch):
    from src import main

    monkeypatch.setattr(main, "get_semantic_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "route_incoming_audit", lambda **kwargs: "DETERMINISTIC_PASS")
    monkeypatch.setattr(main, "run_deterministic_ach_check", lambda state: _report())
    saved = []
    monkeypatch.setattr(main, "set_semantic_cache", lambda query, report: saved.append((query, report)))
    response = TestClient(main.app).post("/api/v1/audit", json={"query": "routine ACH"})
    body = response.json()
    assert body["observability"]["llm_usage"]["usage_status"] == "not_applicable"
    assert body["observability"]["llm_usage"]["total_tokens"] == 0
    assert saved == [("routine ACH", _report())]
    assert "observability" not in saved[0][1]


def test_agentic_api_response_and_observability_failure_do_not_change_report(monkeypatch):
    from src import main

    class FakeGraph:
        async def ainvoke(self, state, config):
            collector = config["callbacks"][0]
            for node in ("extraction", "aml_audit", "auditor_critic"):
                _record(collector, usage=_normalized(2, 1, 3), node=node)
            return {"final_report": _report()}

    monkeypatch.setattr(main, "graph", FakeGraph())
    monkeypatch.setattr(main, "get_semantic_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "set_semantic_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "route_incoming_audit", lambda **kwargs: "AGENTIC_GRAPH")
    response = TestClient(main.app).post("/api/v1/audit", json={"query": "audit"})
    assert response.status_code == 200
    assert response.json()["report"] == _report()
    assert response.json()["observability"]["llm_usage"]["logical_call_count"] == 3

    monkeypatch.setattr(main.LLMUsageCollector, "snapshot", lambda self: (_ for _ in ()).throw(RuntimeError("telemetry")))
    response = TestClient(main.app).post("/api/v1/audit", json={"query": "audit"})
    assert response.status_code == 200
    assert response.json()["report"] == _report()
    assert response.json()["observability"] is None


def test_api_serializes_configured_decimal_cost(monkeypatch):
    from src import main

    class FakeGraph:
        async def ainvoke(self, state, config):
            _record(config["callbacks"][0], usage=_normalized(100, 20, 120))
            return {"final_report": _report()}

    monkeypatch.setenv("XAI_MODEL", "offline")
    monkeypatch.setenv("XAI_PRICE_MODEL", "offline")
    monkeypatch.setenv("XAI_INPUT_PRICE_PER_MILLION", "2")
    monkeypatch.setenv("XAI_OUTPUT_PRICE_PER_MILLION", "4")
    monkeypatch.setattr(main, "graph", FakeGraph())
    monkeypatch.setattr(main, "get_semantic_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "set_semantic_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "route_incoming_audit", lambda **kwargs: "AGENTIC_GRAPH")

    response = TestClient(main.app).post("/api/v1/audit", json={"query": "audit"})
    usage = response.json()["observability"]["llm_usage"]
    assert response.status_code == 200
    assert usage["cost_status"] == "estimated"
    assert usage["estimated_cost_usd"] == "0.00028"


def test_compliance_report_schema_is_unchanged_and_sensitive_content_not_logged(caplog):
    assert set(ComplianceReport.model_fields) == {
        "assessment_status",
        "risk_rating",
        "flagged_wires",
        "applicable_regulations",
        "audit_summary",
        "source_document_hashes",
    }
    collector = LLMUsageCollector(provider="xAI", model="offline")
    secret = "SENTINEL-PROMPT-RESPONSE-SECRET"
    run_id = uuid4()
    collector.on_chat_model_start({}, [[secret]], run_id=run_id)
    collector.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=AIMessage(content=secret))]]),
        run_id=run_id,
    )
    assert secret not in caplog.text


def _pricing(*, model="offline", cached=None):
    return LLMPricing(
        model=model,
        input_per_million=Decimal("2"),
        output_per_million=Decimal("4"),
        cached_input_per_million=(Decimal(str(cached)) if cached is not None else None),
        revision="fixture-1",
    )


def test_decimal_cost_estimation_uses_configured_input_and_output_rates():
    collector = LLMUsageCollector(provider="xAI", model="offline", pricing=_pricing())
    _record(collector, usage=_normalized(100, 20, 120))
    usage = collector.snapshot()
    assert usage.cost_status == "estimated"
    assert usage.estimated_cost_usd == Decimal("0.00028")
    assert usage.pricing_revision == "fixture-1"


def test_cached_input_uses_cached_rate_without_double_counting():
    collector = LLMUsageCollector(
        provider="xAI", model="offline", pricing=_pricing(cached="0.5")
    )
    metadata = _normalized(100, 20, 120)
    metadata["input_token_details"] = {"cache_read": 20}
    metadata["output_token_details"] = {"reasoning": 10}
    _record(collector, usage=metadata)
    usage = collector.snapshot()
    assert usage.cached_input_tokens == 20
    assert usage.reasoning_tokens == 10
    assert usage.estimated_cost_usd == Decimal("0.00025")


def test_cached_usage_without_cached_rate_has_no_estimate():
    collector = LLMUsageCollector(provider="xAI", model="offline", pricing=_pricing())
    metadata = _normalized(100, 20, 120)
    metadata["input_token_details"] = {"cache_read": 20}
    _record(collector, usage=metadata)
    usage = collector.snapshot()
    assert usage.estimated_cost_usd is None
    assert usage.cost_status == "pricing_not_configured"


@pytest.mark.parametrize("value", ["-1", "invalid", "NaN", "Infinity"])
def test_invalid_pricing_is_rejected_without_raising(monkeypatch, value):
    monkeypatch.setenv("XAI_PRICE_MODEL", "offline")
    monkeypatch.setenv("XAI_INPUT_PRICE_PER_MILLION", value)
    monkeypatch.setenv("XAI_OUTPUT_PRICE_PER_MILLION", "4")
    assert load_xai_pricing() is None


def test_missing_pricing_and_model_mismatch_are_explicit():
    missing = LLMUsageCollector(provider="xAI", model="offline")
    _record(missing, usage=_normalized())
    assert missing.snapshot().cost_status == "pricing_not_configured"

    mismatch = LLMUsageCollector(
        provider="xAI", model="offline", pricing=_pricing(model="another-model")
    )
    _record(mismatch, usage=_normalized())
    usage = mismatch.snapshot()
    assert usage.cost_status == "model_mismatch"
    assert usage.estimated_cost_usd is None


def test_unavailable_and_partial_usage_cannot_be_costed():
    unavailable = LLMUsageCollector(
        provider="xAI", model="offline", pricing=_pricing()
    )
    _record(unavailable)
    assert unavailable.snapshot().cost_status == "usage_unavailable"

    partial = LLMUsageCollector(provider="xAI", model="offline", pricing=_pricing())
    _record(partial, usage=_normalized())
    _record(partial)
    usage = partial.snapshot()
    assert usage.usage_status == "partial"
    assert usage.cost_status == "usage_unavailable"
    assert usage.estimated_cost_usd is None


def test_zero_call_path_has_zero_not_applicable_cost():
    usage = LLMUsageCollector(provider="xAI", model="offline").snapshot()
    assert usage.estimated_cost_usd == Decimal("0")
    assert usage.cost_status == "not_applicable"
