import json
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from src.ui import api_client


ROOT = Path(__file__).resolve().parents[1]


class FakeResponse:
    def __init__(self, payload, status=200):
        self.status = status
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.payload


def successful_payload():
    return {
        "status": "SUCCESS",
        "cache_status": "CACHE_HIT",
        "execution_latency_ms": 12.5,
        "report": {
            "assessment_status": "COMPLETE",
            "risk_rating": "LOW",
            "flagged_wires": [],
            "applicable_regulations": ["Standard ACH Compliance Rules"],
            "audit_summary": "No concerns identified.",
            "source_document_hashes": ["rule_deterministic_v1"],
        },
    }


def test_streamlit_has_no_backend_orchestration_imports():
    source = (ROOT / "src" / "ui" / "app.py").read_text(encoding="utf-8")

    assert "src.graph" not in source
    assert "src.ingestion" not in source
    assert "src.utils.cache" not in source
    assert "build_finguard_graph" not in source
    assert "get_llm" not in source
    assert "FinGuardRetriever" not in source
    assert "load_audit_engine" not in source
    assert ".invoke(" not in source


def test_api_base_url_default_and_configuration(monkeypatch):
    monkeypatch.delenv("FINGUARD_API_BASE_URL", raising=False)
    assert api_client.get_api_base_url() == "http://localhost:8000"

    monkeypatch.setenv("FINGUARD_API_BASE_URL", "http://backend:8000/")
    assert api_client.get_api_base_url() == "http://backend:8000"


def test_submit_audit_uses_endpoint_and_handles_success(monkeypatch):
    calls = []
    payload = successful_payload()

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)

    response = api_client.submit_audit(
        "Audit TXN-1",
        base_url="http://backend:8000/",
    )
    report, cache_status, latency = api_client.prepare_ui_result(response)

    assert len(calls) == 1
    request, timeout = calls[0]
    assert request.full_url == "http://backend:8000/api/v1/audit"
    assert request.method == "POST"
    assert json.loads(request.data) == {"query": "Audit TXN-1"}
    assert timeout == api_client.AUDIT_TIMEOUT_SECONDS
    assert report == payload["report"]
    assert cache_status == "CACHE HIT 🟢"
    assert latency == 12.5


def test_connection_error_is_safe_and_not_retried(monkeypatch):
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        raise URLError("host detail containing secret-value")

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)

    with pytest.raises(api_client.AuditApiError) as exc_info:
        api_client.submit_audit("Audit TXN-1")

    assert calls == 1
    assert "Unable to reach" in str(exc_info.value)
    assert "secret-value" not in str(exc_info.value)


def test_timeout_is_safe_and_not_retried(monkeypatch):
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        raise TimeoutError("backend timeout internals")

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)

    with pytest.raises(api_client.AuditApiError) as exc_info:
        api_client.submit_audit("Audit TXN-1")

    assert calls == 1
    assert "Unable to reach" in str(exc_info.value)
    assert "timeout internals" not in str(exc_info.value)


def test_http_error_is_safe_and_not_retried(monkeypatch):
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        raise HTTPError(
            request.full_url,
            500,
            "backend stack trace secret-value",
            hdrs=None,
            fp=BytesIO(b"internal exception and secret-value"),
        )

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)

    with pytest.raises(api_client.AuditApiError) as exc_info:
        api_client.submit_audit("Audit TXN-1")

    assert calls == 1
    assert str(exc_info.value) == (
        "FinGuard API returned HTTP 500. The audit was not completed."
    )
    assert "secret-value" not in str(exc_info.value)


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json",
        json.dumps([]).encode("utf-8"),
        json.dumps({"status": "SUCCESS"}).encode("utf-8"),
        json.dumps(
            {
                "status": "SUCCESS",
                "cache_status": "CACHE_MISS",
                "execution_latency_ms": 1,
                "report": "not-an-object",
            }
        ).encode("utf-8"),
    ],
)
def test_malformed_response_is_rejected_without_internal_details(monkeypatch, payload):
    monkeypatch.setattr(
        api_client,
        "urlopen",
        lambda request, timeout: FakeResponse(payload),
    )

    with pytest.raises(api_client.AuditApiError) as exc_info:
        api_client.submit_audit("Audit TXN-1")

    assert "invalid response" in str(exc_info.value)
    assert payload.decode("utf-8", errors="ignore") not in str(exc_info.value)
