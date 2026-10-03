"""Domain attribution contracts with synthetic evidence and no provider calls."""

import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.graph import nodes
from src.graph.explainability import build_explainability, evidence_registry, labelled_context
from src.graph.schemas import AMLAssessment, ComplianceReport, CriticAssessment, TransactionExtraction
from src.graph.workflow import build_finguard_graph
from src.ingestion.retriever import FinGuardRetriever
from src.ui.api_client import prepare_explanation, risk_display


@pytest.fixture(autouse=True)
def disable_tracing(monkeypatch):
    for name in ("LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING", "LANGSMITH_TRACING_V2"):
        monkeypatch.setenv(name, "false")


def chunk(identity="record-a", role="swift_log", score=0.9, **metadata):
    return {"id": identity, "content": "PRIVATE_DOCUMENT_SENTINEL", "rerank_score": score,
            "metadata": {"source": "synthetic.txt", "doc_type": role, "private": "SECRET_METADATA", **metadata}}


def link(claim="structuring", references=None):
    return {"claim": claim, "evidence_references": references or ["P1-E1"],
            "support_summary": "Related transaction pattern requires review."}


def assessment(attribution=None, gaps=None):
    return AMLAssessment(risk_rating="Medium", suspicious_patterns=["structuring"],
        flagged_transactions=["TXN-SYN-1"], applicable_regulations=["FINRA Rule 3310"],
        required_evidence_gaps=gaps or [], reasoning_summary="Existing concise summary.",
        insufficient_evidence=bool(gaps), evidence_attribution=attribution)


def state(value=None, context=None, count=1):
    return {"aml_assessment": (value or assessment()).model_dump(),
            "retrieved_context": context if context is not None else [chunk()],
            "loop_count": count, "is_audit_complete": True,
            "critic_assessment": {"recommended_action": "GENERATE", "failure_type": "NONE"}}


@pytest.mark.parametrize("reference, accepted", [("P1-E1", True), ("P9-E1", False), ("E1", False), ("P1-E2", False)])
def test_reference_membership(reference, accepted):
    explanation = build_explainability(state(assessment({"findings": [link(references=[reference])]})))
    assert bool(explanation.finding_attributions) == accepted
    assert explanation.attribution_status == ("available" if accepted else "unavailable")


def test_mixed_unknown_reference_rejects_whole_link_and_claims_must_exist():
    explanation = build_explainability(state(assessment({"findings": [
        link(references=["P1-E1", "UNKNOWN"]), link(claim="invented")],
        "regulations": [link(claim="FINRA Rule 3310")]})))
    assert explanation.finding_attributions == []
    assert len(explanation.regulation_attributions) == 1
    assert explanation.attribution_status == "partial"


def test_missing_attribution_not_fabricated_and_metadata_honest():
    context = [{"id": "a", "content": "private"}, {"content": "private"}, chunk("b", "unknown", chunk_id=0)]
    explanation = build_explainability(state(context=context))
    assert explanation.finding_attributions == explanation.regulation_attributions == []
    assert explanation.admitted_evidence[0].source_label is None
    assert explanation.admitted_evidence[0].evidence_role == "unknown"
    assert explanation.admitted_evidence[1].evidence_reference == "P1-E3"
    assert explanation.admitted_evidence[1].chunk_locator == 0
    assert "not citable" in labelled_context(context, 1)


def test_same_source_chunks_roles_and_allowlist():
    registry = evidence_registry([chunk("a", record_id="record-1"),
        chunk("b", "regulatory_pdf", chunk_id=2, jurisdiction="US_OFAC")], 1)
    assert registry[0].source_label == registry[1].source_label
    assert registry[0].indexed_document_id != registry[1].indexed_document_id
    assert registry[0].evidence_reference != registry[1].evidence_reference
    assert [item.evidence_role for item in registry] == ["transaction_record", "regulatory_guidance"]
    payload = json.dumps([item.model_dump() for item in registry])
    assert "PRIVATE_DOCUMENT" not in payload and "SECRET_METADATA" not in payload
    assert "rerank_score" not in payload and "vector_distance" not in payload


def run_graph(monkeypatch, *, attribution=None, refine=False, gaps=None):
    responses = {TransactionExtraction: deque([TransactionExtraction()]),
                 AMLAssessment: deque(([assessment(gaps=["REGULATORY_CONTEXT"])] if refine else [])
                     + [assessment(attribution, gaps)]),
                 CriticAssessment: deque([CriticAssessment(is_sufficient=True, failure_type="NONE",
                     recommended_action="GENERATE", critique="fixture")] * (2 if refine else 1))}
    prompts, retrieve_calls = [], []
    def structured(schema):
        def invoke(prompt):
            prompts.append((schema, prompt))
            return responses[schema].popleft()
        return SimpleNamespace(invoke=invoke)
    monkeypatch.setattr(nodes, "get_llm", lambda: SimpleNamespace(with_structured_output=structured))
    batches = deque([[chunk("prior")] , [chunk("final"), chunk("low", score=0.149)]] if refine
                    else [[chunk(), chunk("low", score=0.149), chunk("at", "regulatory_pdf", score=0.15)]])
    def retrieve(**kwargs):
        retrieve_calls.append(kwargs)
        return batches.popleft()
    monkeypatch.setattr(nodes, "get_production_retriever", lambda: SimpleNamespace(retrieve=retrieve))
    result = build_finguard_graph().invoke({"raw_query": "wire synthetic", "loop_count": 0, "max_loops": 2})
    return result, prompts, retrieve_calls


def test_below_threshold_cannot_support_and_prompt_has_only_admitted_labels(monkeypatch):
    result, prompts, calls = run_graph(monkeypatch, attribution={"findings": [link(references=["P1-E3"])]})
    # The node numbers the admitted list: P1-E2 is the threshold-equal document, not the rejected one.
    explanation = result["final_report"]["explainability"]
    assert [item["indexed_document_id"] for item in explanation["admitted_evidence"]] == ["record-a", "at"]
    assert explanation["finding_attributions"] == []
    aml_prompt = next(prompt for schema, prompt in prompts if schema is AMLAssessment)
    assert "Evidence P1-E1 | role: transaction_record" in aml_prompt
    assert "Evidence P1-E2 | role: regulatory_guidance" in aml_prompt
    assert "Evidence P1-E3" not in aml_prompt
    assert calls[0]["top_k_vector"] == 10 and calls[0]["top_n_final"] == 3
    critic_prompt = next(prompt for schema, prompt in prompts if schema is CriticAssessment)
    assert "evidence_attribution" not in critic_prompt and "support_summary" not in critic_prompt


def test_unshortlisted_evidence_cannot_support(monkeypatch):
    retriever = FinGuardRetriever.__new__(FinGuardRetriever)
    retriever.collection = SimpleNamespace(query=lambda **kwargs: {"documents": [["d"] * 4],
        "ids": [["a", "b", "c", "unshortlisted"]], "metadatas": [[{}] * 4], "distances": [[0.1] * 4]})
    retriever.reranker = SimpleNamespace(predict=lambda pairs: [0.9, 0.8, 0.7, 0.6])
    admitted = retriever.retrieve("fixture", top_k_vector=10, top_n_final=3)
    explanation = build_explainability(state(assessment({"findings": [link(references=["P1-E4"])]}), admitted))
    assert explanation.finding_attributions == []
    assert "unshortlisted" not in explanation.model_dump_json()


@pytest.mark.parametrize("reference, accepted", [("P1-E1", False), ("P2-E1", True)])
def test_refinement_replaces_prior_pass_provenance(monkeypatch, reference, accepted):
    result, prompts, calls = run_graph(monkeypatch, refine=True, attribution={"findings": [link(references=[reference])]})
    explanation = result["final_report"]["explainability"]
    assert explanation["refinement_occurred"] is True
    assert explanation["admitted_evidence"][0]["indexed_document_id"] == "final"
    assert bool(explanation["finding_attributions"]) == accepted
    assert result["loop_count"] == 2 and len(calls) == 2 and len(prompts) == 5
    assert "prior" not in json.dumps(explanation)


@pytest.mark.parametrize("attribution", [None, {"findings": [link(references=["invalid"])]},
    {"findings": [{"claim": "structuring", "evidence_references": None}]}, "bad"])
def test_attribution_failure_preserves_core_decisions(monkeypatch, attribution):
    baseline, baseline_prompts, _ = run_graph(monkeypatch)
    result, prompts, _ = run_graph(monkeypatch, attribution=attribution)
    assert {k: v for k, v in result["final_report"].items() if k != "explainability"} == {
        k: v for k, v in baseline["final_report"].items() if k != "explainability"}
    assert result["critic_assessment"] == baseline["critic_assessment"]
    assert result["loop_count"] == baseline["loop_count"] == 1
    assert [p for schema, p in prompts if schema is CriticAssessment] == [p for schema, p in baseline_prompts if schema is CriticAssessment]


def test_insufficient_evidence_preserves_findings_and_gaps(monkeypatch):
    result, _, _ = run_graph(monkeypatch, gaps=["AMOUNT"])
    report = result["final_report"]
    assert report["assessment_status"] == "INSUFFICIENT_EVIDENCE"
    assert report["risk_rating"] == "MEDIUM"
    assert report["flagged_wires"] == ["TXN-SYN-1"]
    assert report["explainability"]["suspicious_patterns"] == ["structuring"]
    assert report["explainability"]["required_evidence_gaps"] == ["AMOUNT"]


def test_explanation_generation_failure_does_not_fail_report(monkeypatch):
    monkeypatch.setattr(nodes, "build_explainability", lambda *a: (_ for _ in ()).throw(ValueError("private")))
    report = nodes.structured_generation_node(state())["final_report"]
    assert report["explainability"] is None and report["assessment_status"] == "COMPLETE"


def test_report_api_ui_privacy_and_cache_identity(monkeypatch):
    from src import main
    report = nodes.structured_generation_node(state(assessment({"findings": [link()]})))["final_report"]
    assert "PRIVATE_DOCUMENT_SENTINEL" not in json.dumps(report)
    assert "SECRET_METADATA" not in json.dumps(report)
    assert "PRIVATE_DOCUMENT_SENTINEL" not in json.dumps(prepare_explanation(report))
    monkeypatch.setattr(main, "get_semantic_cache", lambda *a, **k: report)
    monkeypatch.setattr(main, "route_incoming_audit", lambda **k: pytest.fail("cache hit routed"))
    monkeypatch.setattr(main, "set_semantic_cache", lambda *a: pytest.fail("cache hit stored"))
    body = TestClient(main.app).post("/api/v1/audit", json={"query": "DIFFERENT_PRIVATE_REQUEST"}).json()
    assert body["report"] == report
    assert "DIFFERENT_PRIVATE_REQUEST" not in json.dumps(body)
    assert body["observability"]["llm_usage"]["logical_call_count"] == 0
    assert body["observability"]["investigation"]["status"] == "not_applicable"


def test_legacy_ui_medium_and_review_terminology():
    legacy = {"assessment_status": "COMPLETE", "risk_rating": "Medium", "audit_summary": "summary"}
    assert ComplianceReport.model_validate(legacy).explainability is None
    assert prepare_explanation(legacy) is None
    assert risk_display(legacy) == "MEDIUM"
    source = (Path(__file__).resolve().parents[1] / "src/ui/app.py").read_text(encoding="utf-8")
    assert "FLAGGED FOR SAR" not in source
    assert "REQUIRES AML REVIEW" in source
    assert "Verified Citations" not in source and "Cryptographic Hashes" not in source
    assert "Evidence detail unavailable for this report." in source


def test_ui_explanation_drops_arbitrary_nested_metadata_and_null_lists():
    prepared = prepare_explanation({"explainability": {
        "suspicious_patterns": None, "required_evidence_gaps": None,
        "finding_attributions": None, "regulation_attributions": None,
        "admitted_evidence": [{"evidence_reference": "P1-E1", "source_label": {"secret": "PRIVATE"},
            "metadata": {"secret": "PRIVATE"}, "content": "PRIVATE", "chunk_locator": 0}],
        "critic_action": {"secret": "PRIVATE"}}})
    assert "PRIVATE" not in json.dumps(prepared)
    assert prepared["admitted_evidence"][0]["source_label"] is None
    assert prepared["admitted_evidence"][0]["chunk_locator"] == 0


def test_golden_relationship_matcher_rejects_wrong_link():
    from tests.evaluation.loader import load_golden_dataset
    from tests.evaluation.evaluation_core import evaluate_scenario
    from tests.evaluation.offline_runner import run_offline_replay
    # Golden replay is contract/routing verification, not model grounding evidence.
    _, scenarios = load_golden_dataset()
    scenario = next(s for s in scenarios if s.scenario_id == "structuring-clear-001")
    assert run_offline_replay(scenario).passed
    incorrect = state()
    incorrect["final_report"] = {"explainability": {"regulation_attributions": [
        link(claim="FINRA Rule 3310", references=["UNKNOWN"])]}}
    result = evaluate_scenario(scenario, incorrect, [], [], execution_mode="offline_replay")
    assert any(failure.field == "regulation_evidence" for failure in result.failed_assertions)
