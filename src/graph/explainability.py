"""Deterministic final-pass provenance and reference membership validation."""

from typing import Any

from pydantic import ValidationError

from src.graph.schemas import (
    AssessmentAttribution, EvidenceAttribution, EvidenceProvenance, ReportExplainability,
)


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def evidence_registry(context: list[dict[str, Any]], pass_index: int) -> list[EvidenceProvenance]:
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
        registry.append(EvidenceProvenance(
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
    """Label the same admitted text without adding metadata or dropping evidence."""
    registry = {item.evidence_reference: item for item in evidence_registry(context, pass_index)}
    blocks = []
    for index, chunk in enumerate(context, 1):
        reference = f"P{pass_index}-E{index}"
        item = registry.get(reference)
        label = f"Evidence {reference} | role: {item.evidence_role}" if item else "Evidence without indexed identity | role: unknown | not citable"
        blocks.append(f"[{label}]\n{chunk.get('content', '')}")
    return "\n\n".join(blocks)


def build_explainability(state: dict[str, Any]) -> ReportExplainability:
    """Resolve links, never infer support from relevance or alter AML decisions."""
    assessment = state.get("aml_assessment") or {}
    critic = state.get("critic_assessment") or {}
    count = state.get("loop_count")
    valid_count = isinstance(count, int) and not isinstance(count, bool) and count >= 1
    registry = evidence_registry(state.get("retrieved_context", []), count) if valid_count else []
    references = {item.evidence_reference for item in registry}
    try:
        attribution = AssessmentAttribution.model_validate(assessment.get("evidence_attribution"))
    except (ValidationError, TypeError, ValueError):
        attribution = None

    def resolve(items: list[EvidenceAttribution], claims: list[str]) -> list[EvidenceAttribution]:
        return [item for item in items if item.claim in claims
                and all(reference in references for reference in item.evidence_references)]

    findings = resolve(attribution.findings, assessment.get("suspicious_patterns", [])) if attribution else []
    regulations = resolve(attribution.regulations, assessment.get("applicable_regulations", [])) if attribution else []
    total = len(attribution.findings) + len(attribution.regulations) if attribution else 0
    accepted = len(findings) + len(regulations)
    return ReportExplainability(
        suspicious_patterns=assessment.get("suspicious_patterns", []),
        assessment_summary=assessment.get("reasoning_summary", "No AML assessment available."),
        required_evidence_gaps=assessment.get("required_evidence_gaps", []),
        admitted_evidence=registry,
        finding_attributions=findings, regulation_attributions=regulations,
        attribution_status="available" if accepted and accepted == total else "partial" if accepted else "unavailable",
        critic_action=_string(critic.get("recommended_action")),
        critic_failure_type=_string(critic.get("failure_type")),
        refinement_occurred=count > 1 if valid_count else None,
    )
