# src/main.py

import time
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.graph.workflow import build_finguard_graph
from src.graph.pre_router import route_incoming_audit, run_deterministic_ach_check
from src.utils.cache import get_semantic_cache, set_semantic_cache

load_dotenv()

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


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "finguard-orchestrator"}


@app.post("/api/v1/audit", response_model=AuditResponse)
async def execute_audit(request: AuditRequest):
    start_time = time.time()
    
    # 1. Check Redis / In-Memory Semantic Cache
    cached_report = get_semantic_cache(request.query, threshold=0.80)
    if cached_report:
        latency = (time.time() - start_time) * 1000
        return AuditResponse(
            status="SUCCESS",
            cache_status="CACHE_HIT",
            execution_latency_ms=round(latency, 2),
            report=cached_report,
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
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Graph execution failed: {str(e)}")

    # 3. Store result in semantic cache
    set_semantic_cache(request.query, report)
    latency = (time.time() - start_time) * 1000

    return AuditResponse(
        status="SUCCESS",
        cache_status="CACHE_MISS",
        execution_latency_ms=round(latency, 2),
        report=report,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)