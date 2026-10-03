"""Small HTTP boundary between the Streamlit UI and FinGuard API."""

import json
import os
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_API_BASE_URL = "http://localhost:8000"
AUDIT_ENDPOINT = "/api/v1/audit"
AUDIT_TIMEOUT_SECONDS = 120.0


class AuditApiError(RuntimeError):
    """Safe user-facing error raised when the audit API cannot be used."""


def get_api_base_url() -> str:
    """Return the configured API origin without a trailing slash."""
    return os.getenv("FINGUARD_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def submit_audit(
    query: str,
    *,
    base_url: str | None = None,
    timeout: float = AUDIT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Submit one audit request without retries and validate the response envelope."""
    url = f"{(base_url or get_api_base_url()).rstrip('/')}{AUDIT_ENDPOINT}"
    request = Request(
        url,
        data=json.dumps({"query": query}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            status_code = response.status
            payload_bytes = response.read()
    except HTTPError as exc:
        raise AuditApiError(
            f"FinGuard API returned HTTP {exc.code}. The audit was not completed."
        ) from None
    except (URLError, TimeoutError, OSError):
        raise AuditApiError(
            "Unable to reach the FinGuard API. Confirm the backend is running and try again."
        ) from None

    if not 200 <= status_code < 300:
        raise AuditApiError(
            f"FinGuard API returned HTTP {status_code}. The audit was not completed."
        )

    try:
        payload = json.loads(payload_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise AuditApiError("FinGuard API returned an invalid response.") from None

    if not isinstance(payload, dict):
        raise AuditApiError("FinGuard API returned an invalid response.")
    if payload.get("status") != "SUCCESS":
        raise AuditApiError("FinGuard API did not report a successful audit.")
    if payload.get("cache_status") not in {"CACHE_HIT", "CACHE_MISS"}:
        raise AuditApiError("FinGuard API returned an invalid response.")
    if not isinstance(payload.get("execution_latency_ms"), (int, float)):
        raise AuditApiError("FinGuard API returned an invalid response.")
    if not isinstance(payload.get("report"), dict):
        raise AuditApiError("FinGuard API returned an invalid response.")
    return payload


def _telemetry_display(payload: dict[str, Any]) -> dict[str, str]:
    """Format optional backend observability without estimating cost locally."""
    observability = payload.get("observability")
    usage = observability.get("llm_usage") if isinstance(observability, dict) else None
    if not isinstance(usage, dict):
        return {
            "logical_calls": "Unavailable",
            "total_tokens": "N/A",
            "estimated_cost": "N/A",
            "cost_status": "unavailable",
        }

    logical_calls = usage.get("logical_call_count")
    total_tokens = usage.get("total_tokens")
    cost_status = usage.get("cost_status")
    estimated_cost = usage.get("estimated_cost_usd")

    if cost_status == "estimated" and estimated_cost is not None:
        try:
            cost_display = _format_estimated_cost(Decimal(str(estimated_cost)))
        except (InvalidOperation, ValueError):
            cost_display = "N/A"
    else:
        cost_display = "N/A"

    return {
        "logical_calls": str(logical_calls) if isinstance(logical_calls, int) else "Unavailable",
        "total_tokens": f"{total_tokens:,}" if isinstance(total_tokens, int) else "N/A",
        "estimated_cost": cost_display,
        "cost_status": cost_status if isinstance(cost_status, str) else "unavailable",
    }


def _format_estimated_cost(cost: Decimal) -> str:
    """Render an API-provided cost estimate compactly without recalculating it."""
    if cost == 0:
        return "$0.00"
    if cost >= Decimal("0.01"):
        return f"${cost:.2f}"
    if cost >= Decimal("0.001"):
        return f"${cost:.3f}"
    return f"${cost:.6f}"


def format_latency_ms(latency_ms: float) -> str:
    """Render milliseconds as a compact seconds value for the metric card."""
    return f"{latency_ms / 1000:.1f} s"


def prepare_ui_result(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], str, float, dict[str, str]]:
    """Map the validated API envelope to existing Streamlit display values."""
    cache_status = "HIT" if payload["cache_status"] == "CACHE_HIT" else "MISS"
    return (
        payload["report"],
        cache_status,
        float(payload["execution_latency_ms"]),
        _telemetry_display(payload),
    )


def prepare_diagnostics(payload: dict[str, Any]) -> dict[str, Any]:
    """Select safe diagnostics fields only; retain unavailable versus zero."""
    envelope = payload.get("observability") or {}
    envelope = envelope if isinstance(envelope, dict) else {}
    usage = envelope.get("llm_usage") or {}
    usage = usage if isinstance(usage, dict) else {}
    investigation = envelope.get("investigation") or {}
    investigation = investigation if isinstance(investigation, dict) else {}
    call_fields = ("node", "call_index", "latency_ms", "usage_status", "input_tokens",
                   "output_tokens", "total_tokens", "reported_model")
    calls = [{key: call.get(key) for key in call_fields}
             for call in usage.get("calls", []) if isinstance(call, dict)]
    passes, scores = [], []
    for observation in investigation.get("retrieval_passes", []):
        if not isinstance(observation, dict):
            continue
        passes.append({key: observation.get(key) for key in
                       ("pass_index", "candidate_count", "reranked_count", "shortlist_count", "admitted_count")})
        for candidate in observation.get("candidates", []):
            if isinstance(candidate, dict):
                scores.append({"pass_index": observation.get("pass_index"), **{
                    key: candidate.get(key) for key in
                    ("candidate_index", "vector_distance", "rerank_score", "shortlisted", "admitted")}})
    def display(value):
        return "Unavailable" if value is None else str(value)
    return {
        "configured_model": display(usage.get("model")),
        "provider": display(usage.get("provider")),
        "reported_models": ", ".join(dict.fromkeys(
            call["reported_model"] for call in calls if isinstance(call["reported_model"], str)
        )) or "Unavailable",
        "input_tokens": display(usage.get("input_tokens")),
        "output_tokens": display(usage.get("output_tokens")),
        "llm_latency": (format_latency_ms(usage["total_latency_ms"])
                        if isinstance(usage.get("total_latency_ms"), (int, float)) else "Unavailable"),
        "latency_status": display(usage.get("latency_status")),
        "critic_passes": display(investigation.get("critic_pass_count")),
        "refinements": display(investigation.get("refinement_count")),
        "investigation_status": display(investigation.get("status")),
        "calls": calls, "retrieval_passes": passes, "scores": scores,
    }
