# src/graph/nodes.py

import json
from typing import Any, Dict
from src.graph.state import AgentState
from src.graph.schemas import (
    ComplianceReport,
    TransactionExtraction,
    AMLAssessment,
)
from src.ingestion.retriever import FinGuardRetriever
from src.llm.client import get_llm


# Quick patch in src/graph/nodes.py (extraction_node)

def extraction_node(state: AgentState) -> Dict[str, Any]:
    query = state["raw_query"]
    query_lower = query.lower()

    has_swift = any(
        k in query_lower
        for k in ["swift", "field :50k:", "field :59:", "wire", "txn-"]
    )

    has_reg = any(
        k in query_lower
        for k in ["finra", "fincen", "rule", "structuring", "ctr", "sar", "aml"]
    )

    # If query mentions both or is ambiguous, do not constrain doc_type
    # so retrieval can search across transaction and regulatory content.
    if has_swift and has_reg:
        doc_type = None
    elif has_swift:
        doc_type = "swift_log"
    elif has_reg:
        doc_type = "regulatory_pdf"
    else:
        doc_type = None

    jurisdiction = "US_OFAC" if ("ofac" in query_lower or "us" in query_lower) else None

    llm = get_llm()
    structured_llm = llm.with_structured_output(TransactionExtraction)

    extraction = structured_llm.invoke(
        f"""
        Extract AML-relevant information from the audit request below.

        Only extract information explicitly present in the request.
        Do not invent missing transaction IDs, amounts, regulations,
        transaction types, jurisdictions, or suspicious patterns.

        Audit request:
        {query}
        """
    )

    extracted_entities = extraction.model_dump()

    print(
        f"🔹 [Extraction Node] "
        f"doc_type='{doc_type}', "
        f"jurisdiction='{jurisdiction}', "
        f"entities={extracted_entities}"
    )

    return {
        "doc_type": doc_type,
        "jurisdiction": jurisdiction,
        "extracted_entities": extracted_entities,
    }

def aml_audit_node(state: AgentState) -> Dict[str, Any]:
    """Executes vector retrieval against ChromaDB and performs AML reasoning."""

    print(
        f"🔍 [AML Audit Node] Querying vector store "
        f"(Loop {state.get('loop_count', 0)})..."
    )

    retriever = FinGuardRetriever(
        chroma_path="./data/chroma",
        collection_name="finguard_knowledge_base",
        reranker_model="BAAI/bge-reranker-large"
    )

    # If in a refinement loop, append context flags to query
    query = state["raw_query"]

    if state.get("loop_count", 0) > 0:
        query += (
            " FINRA Rule 3310 structuring "
            "Currency Transaction Reporting thresholds"
        )

    chunks = retriever.retrieve(
        query=query,
        doc_type=state.get("doc_type"),
        jurisdiction=state.get("jurisdiction"),
        top_k_vector=10,
        top_n_final=3
    )

    # Filter out chunks below threshold score 0.15
    valid_chunks = [
        c for c in chunks
        if c.get("rerank_score", 0.0) >= 0.15
    ]

    print(
        f"✅ [AML Audit Node] Found {len(valid_chunks)} "
        f"context chunks passing confidence threshold."
    )

    llm = get_llm()
    structured_llm = llm.with_structured_output(AMLAssessment)

    retrieved_text = "\n\n".join(
        c.get("content", "")
        for c in valid_chunks
    )

    assessment = structured_llm.invoke(
        f"""
        You are performing an AML compliance assessment.

        Analyze the transaction information only using:
        1. the user's audit request,
        2. the extracted transaction entities,
        3. the retrieved regulatory context supplied below.

        Do not assume that a relevant regulation means the transaction is suspicious.

        Do not invent transaction behavior, amounts, counterparties, dates,
        countries, or other evidence that is not present.

        If there is not enough transaction evidence to support an AML conclusion,
        set insufficient_evidence to true.

        Audit request:
        {state["raw_query"]}

        Extracted transaction information:
        {json.dumps(state.get("extracted_entities", {}), indent=2)}

        Retrieved regulatory context:
        {retrieved_text}
        """
    )

    print(
        f"🧠 [AML Reasoning] Assessment: "
        f"{assessment.model_dump()}"
    )

    return {
        "retrieved_context": valid_chunks,
        "aml_assessment": assessment.model_dump(),
    }


def auditor_critic_node(state: AgentState) -> Dict[str, Any]:
    """Evaluates whether the AML assessment has sufficient evidence."""

    assessment = state.get("aml_assessment") or {}
    current_loop = state.get("loop_count", 0)
    max_loops = state.get("max_loops", 2)

    insufficient_evidence = assessment.get("insufficient_evidence", True)

    loop_count = current_loop + 1

    if insufficient_evidence and loop_count < max_loops:
        confidence = 0.50
        is_complete = False
    else:
        confidence = 0.90 if not insufficient_evidence else 0.60
        is_complete = True

    print(
        f"⚖️ [Auditor Critic Node] "
        f"Insufficient Evidence: {insufficient_evidence} | "
        f"Confidence: {confidence:.2f} | "
        f"Complete: {is_complete} "
        f"(Loop {loop_count}/{max_loops})"
    )

    return {
        "confidence_score": confidence,
        "loop_count": loop_count,
        "is_audit_complete": is_complete,
    }


def structured_generation_node(state: AgentState) -> Dict[str, Any]:
    """Formats the AML assessment into a Pydantic ComplianceReport."""

    context = state.get("retrieved_context", [])
    assessment = state.get("aml_assessment") or {}

    risk_rating = assessment.get("risk_rating", "Low")

    flagged_wires = assessment.get(
        "flagged_transactions",
        []
    )

    regulations = assessment.get(
        "applicable_regulations",
        []
    )

    summary = assessment.get(
        "reasoning_summary",
        "No AML assessment available."
    )

    sources = list({
        c["metadata"].get("source", "unknown")
        for c in context
    })

    report = ComplianceReport(
        risk_rating=risk_rating.upper(),
        flagged_wires=flagged_wires,
        applicable_regulations=regulations,
        audit_summary=summary,
        source_document_hashes=sources
    )

    print(
        f"📝 [Generation Node] ComplianceReport created successfully "
        f"with Risk Rating: {report.risk_rating}"
    )

    return {
        "final_report": report.model_dump(),
        "compliance_draft": report.audit_summary
    }