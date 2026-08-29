# src/ui/app.py

import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from src.graph.schemas import has_valid_assessment_status

load_dotenv()

st.set_page_config(
    page_title="FinGuard Orchestrator | AML Audit Cockpit",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #30363d;
    }
    .badge-high {
        background-color: #ff4b4b;
        color: white;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-low {
        background-color: #00c853;
        color: white;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR: INGESTION GATE -----------------
with st.sidebar:
    st.title("🛡️ FinGuard Gate")
    st.caption("Enterprise Regulatory & Ledger Ingestion")
    
    st.markdown("### 📥 Document Upload")
    uploaded_files = st.file_uploader(
        "Ingest Batch SWIFT / Compliance PDFs",
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"Loaded {len(uploaded_files)} file(s) into ingestion queue.")
    
    st.markdown("---")
    st.markdown("### 📊 Vector Index Inventory")
    st.info("Collection: `finguard_knowledge_base`\n\n• FINRA Rule 3310 (AML Manual)\n• FinCEN Structuring Advisory\n• SWIFT ISO 20022 Ledgers")


# ----------------- MAIN AUDITOR WORKSPACE -----------------
st.title("🛡️ FinGuard AML Audit Cockpit")
st.markdown("Autonomous multi-agent transaction auditing with verified regulatory grounding.")

query = st.text_input(
    "Auditor Instruction / Transaction Query:",
    value="Audit wire TXN-984211-X for structuring and CTR threshold evasions under FINRA Rule 3310."
)

col_run, _ = st.columns([1, 5])
with col_run:
    run_audit = st.button("🚀 Execute Audit", type="primary", use_container_width=True)

# ----------------- CACHED GRAPH INITIALIZER -----------------
@st.cache_resource(show_spinner="Initializing FinGuard Multi-Agent Engine...")
def load_audit_engine():
    from src.graph.workflow import build_finguard_graph
    return build_finguard_graph()


if run_audit and query:
    start_time = time.time()
    cache_status = "CACHE MISS 🔴"
    
    # Lazy import cache and router utilities
    from src.graph.pre_router import route_incoming_audit, run_deterministic_ach_check
    from src.utils.cache import get_semantic_cache, set_semantic_cache
    
    # 1. Check Semantic Cache
    cached_report = get_semantic_cache(query, threshold=0.80)
    
    if cached_report and has_valid_assessment_status(cached_report):
        cache_status = "CACHE HIT 🟢"
        final_report = cached_report
        latency = (time.time() - start_time) * 1000
    else:
        # 2. Check Pre-Router
        route_decision = route_incoming_audit(query)
        
        if route_decision == "DETERMINISTIC_PASS":
            with st.status("Executing Deterministic ACH Engine...", expanded=True) as status:
                st.write("⚡ Bypassing Agentic Core (Low Risk Local Wire)")
                final_report = run_deterministic_ach_check({})
                status.update(label="Audit Completed via Deterministic Scanner!", state="complete")
        else:
            with st.status("Executing Multi-Agent State Machine...", expanded=True) as status:
                st.write("🔄 Step 1: Normalizing SWIFT transactional data via Extraction Node...")
                st.write("🔍 Step 2: Cross-referencing entities against ChromaDB via AML Audit Node...")
                st.write("⚖️ Step 3: Calculating compliance score via Auditor Critic Node...")
                st.write("📝 Step 4: Compiling finalized compliance draft...")
                
                app = load_audit_engine()
                initial_state = {
                    "raw_query": query,
                    "doc_type": None,
                    "jurisdiction": None,
                    "extracted_entities": {},
                    "retrieved_context": [],
                    "compliance_draft": None,
                    "confidence_score": 0.0,
                    "loop_count": 0,
                    "max_loops": 2,
                    "is_audit_complete": False,
                    "final_report": None
                }
                
                output = app.invoke(initial_state)
                final_report = output.get("final_report", {})
                status.update(label="Multi-Agent Audit Completed!", state="complete")
        
        # Save to semantic cache
        set_semantic_cache(query, final_report)
        latency = (time.time() - start_time) * 1000

    # ----------------- COMPLIANCE SUMMARY REPORT -----------------
    st.markdown("---")
    st.subheader("📋 Compliance Summary Report")
    
    assessment_status = final_report.get("assessment_status")
    risk = final_report.get("risk_rating", "LOW")
    if assessment_status == "INSUFFICIENT_EVIDENCE":
        st.warning(
            "Assessment status: INSUFFICIENT EVIDENCE. FinGuard could not reach "
            "a supported AML conclusion; the risk rating is not a clearance."
        )
    elif assessment_status == "COMPLETE":
        if risk == "HIGH":
            st.markdown('### Risk Assessment: <span class="badge-high">HIGH RISK</span>', unsafe_allow_html=True)
        else:
            st.markdown('### Risk Assessment: <span class="badge-low">LOW RISK</span>', unsafe_allow_html=True)
    else:
        st.error(
            "Assessment status is missing or invalid. Do not interpret this report "
            "as a completed AML assessment."
        )

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Flagged Transactions:**")
        wires = final_report.get("flagged_wires", [])
        if wires:
            df = pd.DataFrame({"Transaction Reference ID": wires, "Status": ["FLAGGED FOR SAR"] * len(wires)})
            st.dataframe(df, use_container_width=True)
        else:
            st.write("None flagged.")
            
        st.markdown("**Applicable Regulations:**")
        for reg in final_report.get("applicable_regulations", []):
            st.markdown(f"- `{reg}`")

    with c2:
        st.markdown("**Executive Audit Summary:**")
        st.info(final_report.get("audit_summary", "No findings reported."))

    # ----------------- AUDITOR CITATIONS DRAWER -----------------
    with st.expander("📚 Auditor's Verified Citations & Cryptographic Hashes", expanded=False):
        hashes = final_report.get("source_document_hashes", [])
        if hashes:
            for h in hashes:
                st.code(f"Source Authority: {h}", language="text")
        else:
            st.write("No external source hashes cited.")

    # ----------------- TELEMETRY & DIAGNOSTICS -----------------
    with st.expander("🛠️ Developer Telemetry & Diagnostics", expanded=True):
        t1, t2, t3 = st.columns(3)
        t1.metric("Semantic Cache Status", cache_status)
        t2.metric("Execution Latency", f"{latency:.1f} ms")
        t3.metric("Est. API Token Cost", "$0.00" if cache_status == "CACHE HIT 🟢" else "$0.02")
