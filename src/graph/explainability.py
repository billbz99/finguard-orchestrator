"""Deterministic final-pass provenance and reference membership validation."""

import json
import logging
from typing import Any

from pydantic import ValidationError

from src.graph.schemas import (
    AssessmentAttribution, EvidenceAttribution, EvidenceProvenance, ReportExplainability,
)


logger = logging.getLogger(__name__)


class InternalEvidenceProvenance(EvidenceProvenance):
    """Indexed identity/locators stay internal; never serialize this into reports."""
    indexed_document_id: str
    record_locator: str | None = None
    chunk_locator: str | int | None = None
    jurisdiction: str | None = None


def log_explainability_failure(error: Exception) -> None:
    try:
        logger.warning("Explainability construction failed error_type=%s", type(error).__name__)
    except Exception:
        pass


def resolve_attributions(items: list[EvidenceAttribution], claims: list[str],
                         registry: list[EvidenceProvenance], role: str) -> list[EvidenceAttribution]:
    """Exact claim/reference matching; conflicts fail closed, duplicates collapse."""
    references = {item.evidence_reference: item for item in registry}
    groups: dict[str, list[EvidenceAttribution]] = {}
    for item in items:
        payload = item.model_dump()
        payload["evidence_references"] = list(dict.fromkeys(item.evidence_references))
        normalized = EvidenceAttribution.model_validate(payload)
        groups.setdefault(item.claim, []).append(normalized)
    accepted = []
    for claim, entries in groups.items():
        item = entries[0]
        if claim not in claims or any(entry != item for entry in entries[1:]):
            continue
        refs = item.evidence_references
        if all(ref in references for ref in refs) and any(references[ref].evidence_role == role for ref in refs):
            accepted.append(item)
    return accepted


def coverage_status(findings: list[EvidenceAttribution], regulations: list[EvidenceAttribution],
                    patterns: list[str], rules: list[str]) -> str:
    total = len(set(patterns)) + len(set(rules))
    covered = len({item.claim for item in findings}) + len({item.claim for item in regulations})
    return "available" if total and covered == total else "partial" if covered else "unavailable"


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def evidence_registry(context: list[dict[str, Any]], pass_index: int) -> list[InternalEvidenceProvenance]:
    """References include pass identity; missing indexed IDs cannot be cited."""
    registry = []
    for index, chunk in enumerate(context, 1):
        identity = _string(chunk.get("id"))
        if identity is None:
            continue
        metadata = chunk.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        document_type = _string(metadata.get("doc_type"))
        locator = metadata.get("chunk_id")
        registry.append(InternalEvidenceProvenance(
            evidence_reference=f"P{pass_index}-E{index}",
            indexed_document_id=identity,
            source_label=_string(metadata.get("source")),
            document_type=document_type,
            evidence_role={"swift_log": "transaction_record", "regulatory_pdf": "regulatory_guidance"}.get(document_type, "unknown"),
            record_locator=_string(metadata.get("record_id")),
            chunk_locator=locator if isinstance(locator, (str, int)) and not isinstance(locator, bool) else None,
            jurisdiction=_string(metadata.get("jurisdiction")),
        ))
    return registry


def labelled_context(context: list[dict[str, Any]], pass_index: int) -> str:
    """Label admitted strings; unexpected content types contribute no prompt text."""
    registry = {item.evidence_reference: item for item in evidence_registry(context, pass_index)}
    blocks = []
    for index, chunk in enumerate(context, 1):
        reference = f"P{pass_index}-E{index}"
        item = registry.get(reference)
        label = f"Evidence {reference} | role: {item.evidence_role}" if item else "Evidence without indexed identity | role: unknown | not citable"
        # A JSON string has one physical line; content brackets cannot forge headers.
        raw_content = chunk.get("content", "")
        content = json.dumps(raw_content if isinstance(raw_content, str) else "", ensure_ascii=True)
        content = content.replace("[", r"\u005b").replace("]", r"\u005d")
        blocks.append(f"[{label}]\nDocument text (JSON string): {content}")
    return "\n\n".join(blocks)


def build_explainability(state: dict[str, Any]) -> ReportExplainability:
    """Resolve links, never infer support from relevance or alter AML decisions."""
    assessment = state.get("aml_assessment") or {}
    critic = state.get("critic_assessment") or {}
    count = state.get("loop_count")
    valid_count = isinstance(count, int) and not isinstance(count, bool) and count >= 1
    registry = evidence_registry(state.get("retrieved_context", []), count) if valid_count else []
    try:
        attribution = AssessmentAttribution.model_validate(assessment.get("evidence_attribution"))
    except (ValidationError, TypeError, ValueError):
        attribution = None

    patterns = assessment.get("suspicious_patterns", [])
    rules = assessment.get("applicable_regulations", [])
    findings = resolve_attributions(attribution.findings, patterns, registry, "transaction_record") if attribution else []
    regulations = resolve_attributions(attribution.regulations, rules, registry, "regulatory_guidance") if attribution else []
    public_registry = [EvidenceProvenance.model_validate({
        key: value for key, value in item.model_dump().items() if key in EvidenceProvenance.model_fields
    }) for item in registry]
    attributed_refs = {ref for item in findings + regulations for ref in item.evidence_references}
    public_registry = [EvidenceProvenance.model_validate({**item.model_dump(), "source_label":
        item.source_label if item.evidence_role == "regulatory_guidance" or item.evidence_reference in attributed_refs else None})
        for item in public_registry]
    return ReportExplainability(
        suspicious_patterns=assessment.get("suspicious_patterns", []),
        assessment_summary=assessment.get("reasoning_summary", "No AML assessment available."),
        required_evidence_gaps=assessment.get("required_evidence_gaps", []),
        admitted_evidence=public_registry,
        finding_attributions=findings, regulation_attributions=regulations,
        attribution_status=coverage_status(findings, regulations, patterns, rules),
        critic_action=_string(critic.get("recommended_action")),
        critic_failure_type=_string(critic.get("failure_type")),
        refinement_occurred=count > 1 if valid_count else None,
    )


def sanitize_cached_report(report: dict[str, Any]) -> dict[str, Any]:
    """Sanitize stored explanation only, without rebinding current-request evidence."""
    if report.get("explainability") is None:
        return report
    try:
        payload = dict(report["explainability"])
        payload["admitted_evidence"] = [{key: value for key, value in item.items()
            if key in EvidenceProvenance.model_fields} for item in payload.get("admitted_evidence", [])]
        explanation = ReportExplainability.model_validate(payload)
        patterns = explanation.suspicious_patterns
        rules = report.get("applicable_regulations", [])
        payload["finding_attributions"] = resolve_attributions(explanation.finding_attributions,
            patterns, explanation.admitted_evidence, "transaction_record")
        payload["regulation_attributions"] = resolve_attributions(explanation.regulation_attributions,
            rules, explanation.admitted_evidence, "regulatory_guidance")
        payload["attribution_status"] = coverage_status(payload["finding_attributions"],
            payload["regulation_attributions"], patterns, rules)
        attributed_refs = {ref for item in payload["finding_attributions"] + payload["regulation_attributions"] for ref in item.evidence_references}
        payload["admitted_evidence"] = [{**item.model_dump(), "source_label": item.source_label
            if item.evidence_role == "regulatory_guidance" or item.evidence_reference in attributed_refs else None}
            for item in explanation.admitted_evidence]
        safe = ReportExplainability.model_validate(payload).model_dump()
    except (ValidationError, TypeError, ValueError, AttributeError):
        safe = None
    return {**report, "explainability": safe}
