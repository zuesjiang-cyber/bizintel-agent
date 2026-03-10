import streamlit as st
import pandas as pd
from agent.schemas import AnalysisMode, ConfidenceLevel
from agent.orchestrator import BizIntelAgent

# Configuration and Title
st.set_page_config(page_title="BizIntel Agent", page_icon="📈", layout="wide")

st.title("📈 BizIntel Agent")
st.markdown("An automated business intelligence research and hallucination-free report generation system.")

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    
    preset_queries = [
        "Select a preset query...",
        "Analyze Stripe in depth",
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
    progress_text = "Initializing BizIntel Orchestrator..."
    my_bar = st.progress(0, text=progress_text)
    
    try:
        my_bar.progress(10, text="Agent Planning: Generating Query Graph...")
        agent = BizIntelAgent()
        
        my_bar.progress(30, text="Executing Retrieval & LLM Synthesis...")
        mode_enum = mode_mapping[selected_mode]
        with st.spinner('Running multi-step hybrid retrieval and analysis pipeline... This may take up to a minute.'):
            # Run the agent
            memo = agent.research(query=query, mode=mode_enum)
        
        my_bar.progress(80, text="Running NLI Fact Verification...")
        # (This is logically bundled inside agent.research, but we simulate progress for UX)
        
        my_bar.progress(100, text="Report Generation Complete!")
        
        # UI DISPLAY
        st.header(f"Results: {memo.title}")
        st.caption(f"Generated on: {memo.generated_at[:16]} | Base Mode: {memo.mode.value} | System Confidence: {memo.overall_confidence:.0%}")
        
        st.subheader("Executive Summary")
        with st.container(border=True):
            # Render executive summary correctly by picking the generated text snippet from writer.
            # We will use the report_writer to re-render just the exec summary if we want, or do it inline.
            all_content = "\\n\\n".join(s.content for s in memo.sections)
            exec_summary = agent.report_writer._generate_executive_summary(all_content)
            st.markdown(exec_summary)

        st.subheader("Analysis Breakdown")
        for section in memo.sections:
            title_clean = section.title.replace("_", " ").title()
            # Expanders
            with st.expander(f"📖 {title_clean}", expanded=False):
                st.markdown(section.content)
        
        # Verification Summary
        st.subheader("Verification & NLI Auditing")
        all_results = []
        for s in memo.sections:
            all_results.extend(s.verification_results)
            
        if all_results:
            stats = agent.report_writer.verifier.summary_stats(all_results)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Claims Extracted", stats["total_claims"])
            col2.metric("Citation Coverage", f"{stats['citation_coverage']:.0%}")
            col3.metric("Avg NLI Entailment", f"{stats['avg_nli_score']:.2f}")
            col4.metric("Unsupported Claims", stats["unsupported"])
            
            # DataFrame for claims
            st.markdown("#### Claim Validation Details")
            claim_data = []
            for r in all_results:
                icon = get_confidence_color(r.confidence)
                claim_data.append({
                    "Status": f"{icon} {r.confidence.value.upper()}",
                    "Claim Text": r.claim.text,
                    "NLI Score": round(r.nli_score, 2),
                    "Sources": ", ".join(r.claim.cited_sources)
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
