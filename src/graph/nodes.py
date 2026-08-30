# src/graph/nodes.py

import json
import re
from typing import Any, Dict
from src.graph.state import AgentState
from src.graph.schemas import (
    ComplianceReport,
    TransactionExtraction,
    AMLAssessment,
    CriticAssessment,
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

    has_us_jurisdiction = bool(
        re.search(
            r"\b(?:ofac|united\s+states)\b|(?<!\w)u\.s\.(?!\w)",
            query_lower,
        )
        or re.search(r"\bUS\b", query)
    )
    jurisdiction = "US_OFAC" if has_us_jurisdiction else None

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
        f"[Extraction Node] "
        f"doc_type='{doc_type}', "
        f"jurisdiction='{jurisdiction}', "
        f"entities={extracted_entities!a}"
    )

    return {
        "doc_type": doc_type,
        "jurisdiction": jurisdiction,
        "extracted_entities": extracted_entities,
    }

def aml_audit_node(state: AgentState) -> Dict[str, Any]:
    """Executes vector retrieval against ChromaDB and performs AML reasoning."""

    print(
        f"[AML Audit Node] Querying vector store "
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
        f"[AML Audit Node] Found {len(valid_chunks)} "
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

        Treat the audit request and extracted entities as evidence about what
        happened in the transaction. Treat retrieved context as evidence about
        regulatory standards and guidance. Regulatory documents do not need to
        repeat transaction facts.

        Do not assume that a relevant regulation means the transaction is suspicious.

        Do not invent transaction behavior, amounts, counterparties, dates,
        countries, or other evidence that is not present.

        Evaluate evidence sufficiency for the specific conclusion being made.
        A missing field is not automatically fatal: require an amount for an
        amount- or threshold-dependent conclusion, timing for a timing-dependent
        conclusion, jurisdiction for a jurisdiction-specific conclusion, and so
        on. A supported LOW/no-indicator conclusion does not require every
        commonly useful field. A positive suspicious-pattern conclusion requires
        the transaction facts that constitute that pattern. Assert an applicable
        regulation only when adequate retrieved regulatory context supports it.

        Set insufficient_evidence to true only when the available evidence does
        not support finalizing the requested AML conclusion reliably. An
        insufficient assessment may still preserve evidence-grounded suspicious
        patterns, a provisional risk rating, and transaction IDs requiring AML
        review; those flags do not confirm illegal activity.

        Audit request:
        {state["raw_query"]}

        Extracted transaction information:
        {json.dumps(state.get("extracted_entities", {}), indent=2)}

        Retrieved regulatory context:
        {retrieved_text}
        """
    )

    print(
        f"[AML Reasoning] Assessment: "
        f"{assessment.model_dump()!a}"
    )

    return {
        "retrieved_context": valid_chunks,
        "aml_assessment": assessment.model_dump(),
    }


def auditor_critic_node(state: AgentState) -> Dict[str, Any]:
    """Critiques the AML assessment and determines the next workflow action."""

    current_loop = state.get("loop_count", 0)
    max_loops = state.get("max_loops", 2)

    llm = get_llm()
    structured_llm = llm.with_structured_output(CriticAssessment)

    assessment = structured_llm.invoke(
        f"""
        You are the independent critic for an AML investigation.

        Review the AML assessment and determine whether the available evidence
        is sufficient to finalize the audit.

        Distinguish carefully between:

        MISSING_TRANSACTION_DATA:
        Transaction facts required for the specific proposed conclusion are
        missing. Depending on that conclusion, these may include amounts, dates,
        counterparties, related transactions, geographic information, or
        transaction history; these fields are not a universal mandatory checklist.
        Searching for additional regulations will NOT solve this problem.

        MISSING_REGULATORY_CONTEXT:
        Transaction facts exist, but relevant regulatory guidance is missing
        or inadequate. Additional retrieval may help.

        INCONSISTENT_ANALYSIS:
        The AML assessment contradicts the supplied evidence or appears unsupported.

        NONE:
        The assessment is adequately supported.

        Apply conclusion-relative evidence sufficiency. A missing field is not
        automatically fatal, and a supported LOW/no-indicator conclusion need
        not contain every commonly useful transaction field. Positive suspicious
        conclusions require the facts that constitute the pattern. Regulatory
        applicability claims require adequate retrieved regulatory context;
        regulatory documents do not need to repeat transaction facts.

        If the AML assessment has insufficient_evidence=true, NONE and GENERATE
        are inconsistent unless a sufficient AML reassessment has replaced it.
        Your critique does not rewrite the AML assessment.

        Choose exactly one recommended action:

        GENERATE:
        Evidence is sufficient and the report can be finalized.

        RETRIEVE_MORE:
        Additional regulatory retrieval could resolve the problem.

        STOP_INSUFFICIENT:
        The required transaction evidence is unavailable, so additional
        regulatory retrieval would not help.

        Current loop:
        {current_loop}

        Maximum loops:
        {max_loops}

        Original audit request:
        {state["raw_query"]}

        Extracted entities:
        {json.dumps(state.get("extracted_entities", {}), indent=2)}

        AML assessment:
        {json.dumps(state.get("aml_assessment", {}), indent=2)}
        """
    )

    critic = assessment.model_dump()

    # Never allow unlimited retrieval loops.
    if critic["recommended_action"] == "RETRIEVE_MORE" and current_loop + 1 >= max_loops:
        critic["recommended_action"] = "STOP_INSUFFICIENT"
        critic["critique"] += " Maximum refinement loops reached."

    loop_count = current_loop + 1

    print(
        f"[Auditor Critic] "
        f"Failure Type: {critic['failure_type']!a} | "
        f"Action: {critic['recommended_action']!a} | "
        f"Missing: {critic['missing_evidence']!a} | "
        f"Loop {loop_count}/{max_loops}"
    )

    return {
        "critic_assessment": critic,
        "loop_count": loop_count,
        "is_audit_complete": critic["recommended_action"] != "RETRIEVE_MORE",
    }


def structured_generation_node(state: AgentState) -> Dict[str, Any]:
    """Formats the AML assessment into a Pydantic ComplianceReport."""

    context = state.get("retrieved_context", [])
    assessment = state.get("aml_assessment") or {}
    critic = state.get("critic_assessment") or {}

    critic_action = critic.get("recommended_action")
    assessment_is_sufficient = not assessment.get("insufficient_evidence", True)
    assessment_status = (
        "COMPLETE"
        if critic_action == "GENERATE" and assessment_is_sufficient
        else "INSUFFICIENT_EVIDENCE"
    )

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
        assessment_status=assessment_status,
        risk_rating=risk_rating.upper(),
        flagged_wires=flagged_wires,
        applicable_regulations=regulations,
        audit_summary=summary,
        source_document_hashes=sources
    )

    print(
        f"[Generation Node] ComplianceReport created successfully "
        f"with Risk Rating: {report.risk_rating!a}"
    )

    return {
        "final_report": report.model_dump(),
        "compliance_draft": report.audit_summary
    }
