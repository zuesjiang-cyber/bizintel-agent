import streamlit as st
import pandas as pd
import json
import csv
import io
from pathlib import Path

from agent.artifacts import (
    build_trace_payload,
    build_verification_rows,
    verification_rows_to_csv,
)
from agent.schemas import AnalysisMode, ConfidenceLevel
from agent.orchestrator import BizIntelAgent
from eval.benchmark_forensics import (
    LEDGER_FIELDS,
    cohort_summary,
    compare_payloads,
    ledger_rows,
    normalize_rows,
    render_forensics_report,
    representative_cases,
)
from eval.trust_standards import finance_hard_gates, metric_mappings

# Configuration and Title
st.set_page_config(page_title="FinTrust RAG 可信金融审计工作台", page_icon="📈", layout="wide")

st.title("📈 FinTrust RAG 可信金融审计工作台")
st.markdown(
    "A local audit workbench for financial RAG: benchmark-aware retrieval, trace replay, "
    "claim-level verification, and finance hard gates."
)


def build_retrieval_lab_rows(trace_payload: dict) -> list[dict]:
    rows = []
    for step in trace_payload.get("step_traces", []):
        step_id = step.get("step", "")
        contracts = step.get("query_contracts") or [{}]
        contract = contracts[0] if contracts else {}
        notes_by_chunk = {
            note.get("chunk_id"): note
            for note in step.get("evidence_notes", [])
            if isinstance(note, dict)
        }
        for index, source in enumerate(step.get("sources_used", []), start=1):
            if not isinstance(source, dict):
                continue
            note = notes_by_chunk.get(source.get("chunk_id"), {})
            rows.append(
                {
                    "step": step_id,
                    "lane": contract.get("lane", ""),
                    "query": step.get("generation_context") or step.get("search_queries", [""])[0],
                    "rank": index,
                    "score": source.get("score"),
                    "chunk_id": source.get("chunk_id", ""),
                    "source_id": source.get("source_id", ""),
                    "period": contract.get("period", ""),
                    "source_types": ", ".join(contract.get("source_types") or []),
                    "fallback_reason": (step.get("gap_reflection") or {}).get("refusal_reason", ""),
                    "snippet": (note.get("evidence_text") or "")[:500],
                }
            )
    return rows


def build_trace_span_rows(trace_payload: dict) -> list[dict]:
    rows = []
    workflow_events = trace_payload.get("workflow_events", [])
    for index, event in enumerate(workflow_events, start=1):
        if not isinstance(event, dict):
            continue
        rows.append(
            {
                "span_id": f"workflow_{index}",
                "stage": event.get("node_name", "workflow"),
                "status": event.get("event_type", ""),
                "input_summary": "",
                "output_summary": "",
                "failure_reason": "",
            }
        )
    for step in trace_payload.get("step_traces", []):
        step_id = step.get("step", "")
        rows.extend(
            [
                {
                    "span_id": f"{step_id}:planning",
                    "stage": "planning",
                    "status": "completed",
                    "input_summary": step.get("generation_context", ""),
                    "output_summary": json.dumps(step.get("query_contracts", []), ensure_ascii=False),
                    "failure_reason": "",
                },
                {
                    "span_id": f"{step_id}:retrieval",
                    "stage": "retrieval",
                    "status": "completed" if step.get("sources_used") else "needs_evidence",
                    "input_summary": json.dumps(step.get("search_queries", []), ensure_ascii=False),
                    "output_summary": f"{len(step.get('sources_used', []))} evidence candidates",
                    "failure_reason": (step.get("gap_reflection") or {}).get("refusal_reason", ""),
                },
                {
                    "span_id": f"{step_id}:verification",
                    "stage": "verification",
                    "status": "completed" if step.get("covered_facts") else "gap_or_refusal",
                    "input_summary": ", ".join(step.get("covered_facts", [])),
                    "output_summary": ", ".join(step.get("missing_facts", [])),
                    "failure_reason": (step.get("gap_reflection") or {}).get("refusal_reason", ""),
                },
                {
                    "span_id": f"{step_id}:publication_gate",
                    "stage": "publication_gate",
                    "status": (step.get("writing_trace") or {}).get("status", ""),
                    "input_summary": (step.get("writing_trace") or {}).get("answer_text", "")[:240],
                    "output_summary": (step.get("writing_trace") or {}).get("supported_content", "")[:240],
                    "failure_reason": (step.get("gap_reflection") or {}).get("refusal_reason", ""),
                },
            ]
        )
    return rows


def benchmark_files() -> list[Path]:
    results_dir = Path("eval/results")
    if not results_dir.exists():
        return []
    return sorted(
        results_dir.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def load_benchmark_picker(label: str, key_prefix: str) -> tuple[dict | None, str]:
    source = st.radio(
        f"{label} source",
        ["Local result", "Upload JSON"],
        horizontal=True,
        key=f"{key_prefix}_source",
    )
    if source == "Upload JSON":
        uploaded = st.file_uploader(f"Upload {label.lower()} benchmark JSON", type=["json"], key=f"{key_prefix}_upload")
        if uploaded is None:
            return None, ""
        try:
            return json.loads(uploaded.getvalue().decode("utf-8")), uploaded.name
        except json.JSONDecodeError as exc:
            st.error(f"Invalid benchmark JSON: {exc}")
            return None, uploaded.name

    files = benchmark_files()
    if not files:
        st.warning("No benchmark JSON files found under eval/results.")
        return None, ""
    options = [str(path) for path in files]
    default_index = 0
    for index, option in enumerate(options):
        if "financebench_open150_all_live_20260410_parallel_merged.json" in option:
            default_index = index
            break
    selected = st.selectbox(f"Select {label.lower()} benchmark", options, index=default_index, key=f"{key_prefix}_local")
    path = Path(selected)
    try:
        return json.loads(path.read_text(encoding="utf-8")), selected
    except json.JSONDecodeError as exc:
        st.error(f"Invalid benchmark JSON: {exc}")
        return None, selected


def ledger_csv_text(payload: dict) -> str:
    rows = ledger_rows(payload)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(LEDGER_FIELDS))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def render_benchmark_forensics_ui() -> None:
    st.subheader("Benchmark Forensics")
    st.caption(
        "Load benchmark artifacts to inspect safety gates, retrieval completeness, failure tags, "
        "standards mapping, and baseline/candidate regressions."
    )
    left, right = st.columns([1, 1])
    with left:
        candidate, candidate_label = load_benchmark_picker("Candidate", "candidate_benchmark")
    with right:
        compare_enabled = st.checkbox("Compare against baseline", value=False)
        baseline = None
        baseline_label = ""
        if compare_enabled:
            baseline, baseline_label = load_benchmark_picker("Baseline", "baseline_benchmark")

    if candidate is None:
        st.info("Select or upload a candidate benchmark JSON to begin.")
        return

    summary = cohort_summary(candidate)
    rows = normalize_rows(candidate)
    cases = representative_cases(rows)

    metric_cols = st.columns(5)
    metric_cols[0].metric("Rows", summary["row_count"])
    metric_cols[1].metric("Statuses", len(summary["status_counts"]))
    metric_cols[2].metric("Failure Tags", sum(summary["failure_tag_counts"].values()))
    metric_cols[3].metric(
        "Unsupported Claims",
        f"{summary['metric_averages'].get('unsupported_claim_rate', 0.0):.2%}",
    )
    metric_cols[4].metric(
        "Required Fact Recall",
        f"{summary['metric_averages'].get('required_fact_recall', 0.0):.2%}",
    )

    metrics_tab, failures_tab, cases_tab, comparison_tab, downloads_tab = st.tabs(
        ["Metrics", "Failures", "Cases", "Comparison", "Downloads"]
    )
    with metrics_tab:
        metric_rows = []
        standards_by_metric = {item["metric_name"]: item for item in metric_mappings()}
        for metric, value in summary["metric_averages"].items():
            standard = standards_by_metric.get(metric, {})
            metric_rows.append(
                {
                    "metric": metric,
                    "value": value,
                    "family": standard.get("metric_family", "uncategorized"),
                    "gate": standard.get("fintrust_gate", "unmapped"),
                    "judge_type": standard.get("judge_type", ""),
                    "failure_action": standard.get("failure_action", ""),
                }
            )
        if metric_rows:
            st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No numeric metrics were found in this benchmark artifact.")

    with failures_tab:
        status_rows = [{"status": key, "count": value} for key, value in summary["status_counts"].items()]
        failure_rows = [{"failure_tag": key, "count": value} for key, value in summary["failure_tag_counts"].items()]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Status Distribution")
            st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)
        with col2:
            st.markdown("#### Failure Distribution")
            if failure_rows:
                st.dataframe(pd.DataFrame(failure_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No failure tags found.")

    with cases_tab:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Representative Successes")
            if cases["successes"]:
                st.dataframe(pd.DataFrame(cases["successes"]), use_container_width=True, hide_index=True)
            else:
                st.info("No representative success cases found.")
        with col2:
            st.markdown("#### Representative Failures")
            if cases["failures"]:
                st.dataframe(pd.DataFrame(cases["failures"]), use_container_width=True, hide_index=True)
            else:
                st.info("No representative failure cases found.")

    with comparison_tab:
        if compare_enabled and baseline is not None:
            comparison = compare_payloads(baseline, candidate)
            st.metric("Matched Items", comparison["matched_items"])
            delta_rows = [
                {"metric": metric, "delta": delta}
                for metric, delta in comparison["metric_deltas"].items()
            ]
            st.markdown("#### Metric Deltas")
            st.dataframe(pd.DataFrame(delta_rows), use_container_width=True, hide_index=True)
            st.markdown("#### Safety Regressions")
            if comparison["safety_regressions"]:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"metric": metric, "delta": delta}
                            for metric, delta in comparison["safety_regressions"].items()
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.success("No average safety regression detected across matched items.")
            st.markdown("#### Largest Item Regressions")
            st.dataframe(pd.DataFrame(comparison["largest_item_regressions"]), use_container_width=True, hide_index=True)
        else:
            st.info("Enable baseline comparison to inspect regression deltas.")

    with downloads_tab:
        report = render_forensics_report(
            candidate,
            candidate_path=Path(candidate_label) if candidate_label else None,
            baseline=baseline if compare_enabled else None,
            baseline_path=Path(baseline_label) if baseline_label else None,
        )
        st.download_button(
            "Download Forensics Report (.md)",
            data=report,
            file_name="benchmark-forensics-report.md",
            mime="text/markdown",
            use_container_width=True,
        )
        st.download_button(
            "Download Ledger (.csv)",
            data=ledger_csv_text(candidate),
            file_name="benchmark-ledger.csv",
            mime="text/csv",
            use_container_width=True,
        )


render_benchmark_forensics_ui()
st.divider()

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
        my_bar.progress(10, text="Planning research contract...")
        agent = BizIntelAgent(load_models=not demo_mode, demo_mode=demo_mode)
        
        my_bar.progress(30, text="Executing retrieval, grounding, and generation...")
        mode_enum = mode_mapping[selected_mode]
        with st.spinner('Running auditable hybrid retrieval and verification pipeline... This may take up to a minute.'):
            # Run the agent
            result = agent.research(query=query, mode=mode_enum)

        my_bar.progress(80, text="Running claim verification and publication gates...")
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
            file_name="fintrust-rag-memo.md",
            mime="text/markdown",
            use_container_width=True,
        )
        export_cols[1].download_button(
            "Download Trace (.json)",
            data=json.dumps(trace_payload, indent=2, ensure_ascii=False),
            file_name="fintrust-rag-trace.json",
            mime="application/json",
            use_container_width=True,
        )
        export_cols[2].download_button(
            "Download Audit (.csv)",
            data=verification_rows_to_csv(verification_rows),
            file_name="fintrust-rag-verification.csv",
            mime="text/csv",
            use_container_width=True,
        )

        lab_tab, trace_tab, standards_tab = st.tabs(["Retrieval Lab", "Trace Viewer", "Trust Standards"])
        with lab_tab:
            retrieval_rows = build_retrieval_lab_rows(trace_payload)
            if retrieval_rows:
                st.dataframe(pd.DataFrame(retrieval_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No retrieval candidates were available in this trace.")
        with trace_tab:
            span_rows = build_trace_span_rows(trace_payload)
            if span_rows:
                st.dataframe(pd.DataFrame(span_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No trace spans were available.")
        with standards_tab:
            st.markdown("#### Trust Metric Mapping")
            st.dataframe(pd.DataFrame(metric_mappings()), use_container_width=True, hide_index=True)
            st.markdown("#### Finance Hard Gates")
            st.dataframe(pd.DataFrame(finance_hard_gates()), use_container_width=True, hide_index=True)

        st.subheader("Executive Summary")
        with st.container(border=True):
            st.markdown(memo.executive_summary)

        st.subheader("Analysis Breakdown")
        for section in memo.sections:
            title_clean = section.title.replace("_", " ").title()
            # Expanders
            with st.expander(f"📖 {title_clean}", expanded=False):
                st.markdown(section.content)

        st.subheader("Research Control Plane")
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
