# src/main.py

import logging
import os
import time
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.graph.workflow import build_finguard_graph
from src.graph.pre_router import route_incoming_audit, run_deterministic_ach_check
from src.graph.schemas import has_valid_assessment_status
from src.ingestion.retriever import RuntimeAssetError, validate_retrieval_assets
from src.observability.llm_usage import (
    AuditObservability,
    LLMUsageCollector,
)
from src.utils.cache import (
    get_semantic_cache,
    set_semantic_cache,
    validate_cache_readiness,
)

load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="FinGuard Orchestrator API",
    version="1.0.0",
    description="Asynchronous Agentic Compliance & AML Audit Microservice",
)

# Initialize compiled LangGraph instance on startup
graph = build_finguard_graph()


class AuditRequest(BaseModel):
    query: str = Field(..., example="Audit wire TXN-984211-X for structuring under FINRA Rule 3310.")
    amount: Optional[float] = Field(None, example=8500.0)
    is_cross_border: bool = Field(False, example=False)
    client_tier: str = Field("Standard_Institutional", example="VIP_Institutional")
    audit_id: Optional[str] = Field(None, example="aud-9988-xx")


class AuditResponse(BaseModel):
    status: str
    cache_status: str
    execution_latency_ms: float
    report: Dict[str, Any]
    observability: AuditObservability | None = None


def _safe_observability(
    collector: LLMUsageCollector,
) -> AuditObservability | None:
    """Keep optional telemetry failures outside the audit result path."""
    try:
        return AuditObservability(llm_usage=collector.snapshot())
    except Exception:
        return None


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "finguard-orchestrator"}


@app.get("/ready")
async def readiness_check():
    """Validate local runtime prerequisites without provider calls or downloads."""
    checks: Dict[str, Any] = {
        "graph": "ready",
        "xai_configuration": (
            "configured" if bool(os.getenv("XAI_API_KEY")) else "missing"
        ),
    }
    ready = checks["xai_configuration"] == "configured"

    try:
        retrieval = validate_retrieval_assets()
        checks["retrieval_assets"] = "ready"
        checks["retrieval_document_count"] = retrieval["document_count"]
    except RuntimeAssetError as exc:
        checks["retrieval_assets"] = "unavailable"
        checks["retrieval_reason"] = str(exc)
        ready = False

    try:
        cache = validate_cache_readiness()
        checks["cache"] = "ready" if cache["ready"] else "unavailable"
        checks["cache_mode"] = cache["mode"]
        if not cache["ready"]:
            checks["cache_reason"] = cache.get("reason", "unavailable")
            ready = False
    except RuntimeError:
        checks["cache"] = "invalid_configuration"
        ready = False

    payload = {
        "status": "ready" if ready else "not_ready",
        "service": "finguard-orchestrator",
        "checks": checks,
    }
    if ready:
        return payload
    return JSONResponse(status_code=503, content=payload)


@app.post("/api/v1/audit", response_model=AuditResponse)
async def execute_audit(request: AuditRequest):
    start_time = time.time()
    usage_collector = LLMUsageCollector(
        provider="xAI",
        model=os.getenv("XAI_MODEL", "grok-4.3"),
    )
    
    # 1. Check Redis / In-Memory Semantic Cache
    cached_report = get_semantic_cache(request.query, threshold=0.80)
    if cached_report and has_valid_assessment_status(cached_report):
        latency = (time.time() - start_time) * 1000
        return AuditResponse(
            status="SUCCESS",
            cache_status="CACHE_HIT",
            execution_latency_ms=round(latency, 2),
            report=cached_report,
            observability=_safe_observability(usage_collector),
        )

    # 2. Evaluate Pre-Router
    route_decision = route_incoming_audit(
        raw_query=request.query,
        amount=request.amount,
        is_cross_border=request.is_cross_border,
    )

    if route_decision == "DETERMINISTIC_PASS":
        report = run_deterministic_ach_check({"query": request.query, "amount": request.amount})
    else:
        initial_state = {
            "raw_query": request.query,
            "doc_type": None,
            "jurisdiction": None,
            "extracted_entities": {},
            "retrieved_context": [],
            "compliance_draft": None,
            "confidence_score": 0.0,
            "loop_count": 0,
            "max_loops": 2,
            "is_audit_complete": False,
            "final_report": None,
        }

        config = {
            "tags": ["AML_AUDIT_RUN", "FASTAPI_SERVICE"],
            "callbacks": [usage_collector],
            "metadata": {
                "client_tier": request.client_tier,
                "audit_id": request.audit_id or f"aud-{int(time.time())}",
                "batch_wire_count": 1,
            },
        }

        try:
            # Asynchronous invocation keeps event loop non-blocking
            result_state = await graph.ainvoke(initial_state, config=config)
            report = result_state.get("final_report", {})
        except Exception as exc:
            logger.error(
                "Graph execution failed error_type=%s",
                type(exc).__name__,
            )
            raise HTTPException(
                status_code=500,
                detail="Audit execution failed.",
            ) from exc

    # 3. Store result in semantic cache
    set_semantic_cache(request.query, report)
    latency = (time.time() - start_time) * 1000

    return AuditResponse(
        status="SUCCESS",
        cache_status="CACHE_MISS",
        execution_latency_ms=round(latency, 2),
        report=report,
        observability=_safe_observability(usage_collector),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
