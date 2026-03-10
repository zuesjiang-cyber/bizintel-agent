"""
BizIntel Agent - Streamlit Frontend
"""

import streamlit as st
import time
from agent.schemas import AnalysisMode
from agent.orchestrator import BizIntelAgent

# Must be called as the first Streamlit command
st.set_page_config(
    page_title="BizIntel Agent",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def get_agent():
    # Cache the agent so models aren't reloaded on every run
    with st.spinner("Initializing AI Models & Retriever..."):
        agent = BizIntelAgent()
    return agent

def main():
    st.title("🤝 BizIntel AI Research Agent")
    st.markdown("Automated Business Intelligence & Research Report Generation")

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        mode_option = st.selectbox(
            "Analysis Mode (Optional):",
            ["Auto-Detect", "Company Profile", "Industry Landscape", "Competitive Comparison"]
        )
        mode = None
        if mode_option == "Company Profile":
            mode = AnalysisMode.COMPANY
        elif mode_option == "Industry Landscape":
            mode = AnalysisMode.INDUSTRY
        elif mode_option == "Competitive Comparison":
            mode = AnalysisMode.COMPETITIVE
            
        st.markdown("---")
        st.markdown("**Supported Companies in Sandbox:**\n- Stripe\n- Notion\n- Databricks")

    # --- Main Interface ---
    st.info("💡 **Example Queries:** 'Analyze Stripe', 'Compare Stripe vs Adyen', 'Fintech industry trends'")
    
    query = st.text_input("Enter your research topic:", placeholder="e.g. Give me a deep dive on Databricks...")

    if st.button("Generate Report", type="primary"):
        if not query.strip():
            st.warning("Please enter a research topic.")
            return

        try:
            agent = get_agent()
            
            with st.status("🧠 Agent is working...", expanded=True) as status:
                st.write(f"Initiating research for: **{query}**")
                
                # Mock progress for UI feedback during the blocking call
                start_time = time.time()
                
                # Run the actual research pipeline
                result = agent.research(query=query, mode=mode)
                
                elapsed = time.time() - start_time
                status.update(label=f"✅ Research completed in {elapsed:.1f}s", state="complete", expanded=False)

            # Display the result
            memo = result["memo_object"]
            markdown_text = result["memo_markdown"]
            events = result["workflow_events"]
            
            # --- Tabs for different views ---
            tab_report, tab_sources, tab_logs = st.tabs(["📝 Final Report", "📚 Sources & Verification", "🛠️ Execution Logs"])
            
            with tab_report:
                # Provide download button
                st.download_button(
                    label="⬇️ Download Markdown",
                    data=markdown_text,
                    file_name=f"BizIntel_{query[:15].replace(' ', '_')}.md",
                    mime="text/markdown"
                )
                st.markdown(markdown_text)
                
            with tab_sources:
                st.subheader("Cited Sources")
                for source in memo.sources:
                    st.markdown(f"- **[{source.source_id}]** {source.title}")
                    
                st.subheader("Confidence Score")
                st.metric(label="Overall Confidence (Citation Coverage)", value=f"{memo.overall_confidence:.0%}")
                
            with tab_logs:
                st.subheader("Workflow Execution Trace")
                for event in events:
                    if event["event_type"] == "started":
                        st.text(f"▶️ [{event['timestamp']}] {event['node_name']} started")
                    elif event["event_type"] == "succeeded":
                        st.text(f"✅ [{event['timestamp']}] {event['node_name']} succeeded ({event.get('detail', '')})")
                    elif event["event_type"] == "failed":
                        st.error(f"❌ [{event['timestamp']}] {event['node_name']} failed: {event.get('detail', '')}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
