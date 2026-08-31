# src/ui/app.py

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from src.ui.api_client import AuditApiError, prepare_ui_result, submit_audit

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

if run_audit and query:
    with st.status("Submitting audit to FinGuard API...", expanded=True) as status:
        try:
            api_response = submit_audit(query)
            final_report, cache_status, latency, telemetry = prepare_ui_result(
                api_response
            )
        except AuditApiError as exc:
            status.update(label="Audit request failed", state="error")
            st.error(str(exc))
            st.stop()
        status.update(label="Audit completed", state="complete")

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
        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("Semantic Cache Status", cache_status)
        t2.metric("Execution Latency", f"{latency:.1f} ms")
        t3.metric("Logical LLM Calls", telemetry["logical_calls"])
        t4.metric("Total Tokens", telemetry["total_tokens"])
        t5.metric("Est. Provider Cost", telemetry["estimated_cost"])
