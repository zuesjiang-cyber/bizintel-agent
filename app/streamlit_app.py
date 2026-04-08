import streamlit as st
import pandas as pd
import json

from agent.artifacts import (
    build_trace_payload,
    build_verification_rows,
    verification_rows_to_csv,
)
from agent.schemas import AnalysisMode, ConfidenceLevel
from agent.orchestrator import BizIntelAgent

# Configuration and Title
st.set_page_config(page_title="证据驱动可验证的企业财务研究Agent Flow", page_icon="📈", layout="wide")

st.title("📈 证据驱动可验证的企业财务研究Agent Flow")
st.markdown("An automated business intelligence research and citation-aware memo generation system for commercial analysis, strategy, and investment workflows.")

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    
    preset_queries = [
        "Select a preset query...",
        "Analyze Stripe in depth",
        "Assess Stripe's revenue quality, valuation drivers, and key monitorables",
        "Compare Stripe vs PayPal",
        "Global digital payments industry trends and outlook"
    ]
    
    selected_preset = st.selectbox("Presets", preset_queries)
    
    # Decide initial query value
    default_query = ""
    if selected_preset != "Select a preset query...":
        default_query = selected_preset
        
    query = st.text_area("Research Query", value=default_query, height=100, 
                         placeholder="e.g. Analyze Stripe's business model and competitive position")
                         
    mode_mapping = {
        "Auto-Detect": None,
        "Company Deep Dive": AnalysisMode.COMPANY,
        "Competitive Analysis": AnalysisMode.COMPETITIVE,
        "Industry Landscape": AnalysisMode.INDUSTRY
    }
    
    selected_mode = st.selectbox("Force Analysis Mode", list(mode_mapping.keys()))
    demo_mode = st.checkbox("Demo mode (offline / no API key)", value=False)
    
    start_btn = st.button("🚀 Start Analysis", use_container_width=True, type="primary")

def get_confidence_color(level: ConfidenceLevel):
    mapping = {
        ConfidenceLevel.STRONG: "🟢",
        ConfidenceLevel.MODERATE: "🟡",
        ConfidenceLevel.WEAK: "🟠",
        ConfidenceLevel.UNSUPPORTED: "🔴"
    }
    return mapping.get(level, "⚪️")

# Main Execution Logic
if start_btn and query:
    st.divider()
    
    # Progress indication
    progress_text = "Initializing research orchestrator..."
    my_bar = st.progress(0, text=progress_text)
    
    try:
        my_bar.progress(10, text="Agent Planning: Generating Query Graph...")
        agent = BizIntelAgent(load_models=not demo_mode, demo_mode=demo_mode)
        
        my_bar.progress(30, text="Executing Retrieval & LLM Synthesis...")
        mode_enum = mode_mapping[selected_mode]
        with st.spinner('Running multi-step hybrid retrieval and analysis pipeline... This may take up to a minute.'):
            # Run the agent
            result = agent.research(query=query, mode=mode_enum)

        my_bar.progress(80, text="Running NLI Fact Verification...")
        # (This is logically bundled inside agent.research, but we simulate progress for UX)

        my_bar.progress(100, text="Report Generation Complete!")

        memo = result["memo_object"]
        workflow_events = result["workflow_events"]
        verification_rows = build_verification_rows(memo)
        trace_payload = build_trace_payload(result)
        research_task = result.get("research_task")
        subquestion_results = result.get("subquestion_results", [])

        # UI DISPLAY
        st.header(f"Results: {memo.title}")
        confidence_label = "Offline Demo Heuristic Support" if demo_mode else "Verified Support"
        st.caption(
            f"Generated on: {memo.generated_at[:16]} | Base Mode: {memo.mode.value} | "
            f"{confidence_label}: {memo.overall_confidence:.0%}"
        )
        if demo_mode:
            st.info(
                "Offline demo mode uses deterministic stub summarization and a lightweight heuristic verifier. "
                "Treat these support metrics as walkthrough aids, not analyst-grade confidence scores."
            )

        st.subheader("Export Artifacts")
        export_cols = st.columns(3)
        export_cols[0].download_button(
            "Download Memo (.md)",
            data=result["memo_markdown"],
            file_name="bizintel-memo.md",
            mime="text/markdown",
            use_container_width=True,
        )
        export_cols[1].download_button(
            "Download Trace (.json)",
            data=json.dumps(trace_payload, indent=2, ensure_ascii=False),
            file_name="bizintel-trace.json",
            mime="application/json",
            use_container_width=True,
        )
        export_cols[2].download_button(
            "Download Audit (.csv)",
            data=verification_rows_to_csv(verification_rows),
            file_name="bizintel-verification.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.subheader("Executive Summary")
        with st.container(border=True):
            st.markdown(memo.executive_summary)

        st.subheader("Analysis Breakdown")
        for section in memo.sections:
            title_clean = section.title.replace("_", " ").title()
            # Expanders
            with st.expander(f"📖 {title_clean}", expanded=False):
                st.markdown(section.content)

        st.subheader("Agent Plan & Workflow")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Research Tree")
            if subquestion_results:
                st.write(
                    [
                        f"{item.subquestion.question_id} | {item.subquestion.lane.value} | {item.status.value} | {item.subquestion.text}"
                        for item in subquestion_results
                    ]
                )
            elif research_task:
                st.write([research_task.query])
            else:
                st.write([])
        with col2:
            st.markdown("#### Workflow Events")
            st.write([f"{event['node_name']}: {event['event_type']}" for event in workflow_events])
        
        # Verification Summary
        st.subheader("Verification & NLI Auditing")
        all_results = [result for section in memo.sections for result in section.verification_results]
            
        if all_results:
            stats = agent.report_writer.verifier.summary_stats(all_results)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Claims Extracted", stats["total_claims"])
            col2.metric("Heuristic Support" if demo_mode else "Verified Claim Coverage", f"{stats['verified_claim_coverage']:.0%}")
            col3.metric("Avg Heuristic Score" if demo_mode else "Avg NLI Entailment", f"{stats['avg_nli_score']:.2f}")
            col4.metric("Unsupported Claims", stats["unsupported"])
            
            # DataFrame for claims
            st.markdown("#### Claim Validation Details")
            claim_data = []
            for row, verification in zip(verification_rows, all_results):
                icon = get_confidence_color(verification.confidence)
                claim_data.append({
                    "Status": f"{icon} {verification.confidence.value.upper()}",
                    "Section": row["section"].replace("_", " ").title(),
                    "Claim Text": row["claim_text"],
                    "NLI Score": round(verification.nli_score, 2),
                    "Sources": row["cited_sources"],
                })
            
            df = pd.DataFrame(claim_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
        else:
            st.info("No claims verified. (Offline mode or short text)")

        # Sources
        st.subheader("Sources Referenced")
        with st.expander("📚 View Reference Library", expanded=False):
            sources_md = agent.report_writer._format_sources(memo.sources)
            st.markdown(sources_md)

    except Exception as e:
        st.error(f"An error occurred during execution: {str(e)}")
