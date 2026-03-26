import argparse
import sys
from pathlib import Path
import getpass
import json

from agent.artifacts import build_trace_payload, write_artifact_bundle
from agent.schemas import AnalysisMode
from agent.config import settings

def main():
    parser = argparse.ArgumentParser(
        description="证据驱动可验证的企业财务研究Agent Flow: AI-powered business research"
    )
    parser.add_argument("query", type=str, help="The research query (e.g. 'Analyze Stripe')")
    parser.add_argument("--mode", type=str, choices=["company", "industry", "competitive"],
                        help="Force a specific analysis mode")
    parser.add_argument("--output", type=str, help="Output file path (default: prints to stdout)")
    parser.add_argument(
        "--artifact-dir",
        type=str,
        help="Directory for recruiter-friendly exports (memo.md, trace.json, summary.json, verification.csv)",
    )
    parser.add_argument("--demo", action="store_true",
                        help="Run fully offline demo mode with in-memory indexing and stub LLM summaries")
    parser.add_argument("--trace", action="store_true",
                        help="Print plan and workflow trace after the memo")

    args = parser.parse_args()

    # Enum conversion
    mode = None
    if args.mode:
        mode_str = args.mode.lower()
        if mode_str == "company":
            mode = AnalysisMode.COMPANY
        elif mode_str == "industry":
            mode = AnalysisMode.INDUSTRY
        elif mode_str == "competitive":
            mode = AnalysisMode.COMPETITIVE

    if not settings.openai_api_key and not args.demo and settings.llm_mode.lower() != "stub":
        print("未检测到 Minimax API Key。请进行隐式输入：")
        api_key = getpass.getpass("MINIMAX_API_KEY: ")
        if not api_key.strip():
            print("❌ 错误：必须提供 API Key 才能运行", file=sys.stderr)
            sys.exit(1)
        settings.openai_api_key = api_key.strip()
        
    from agent.orchestrator import BizIntelAgent

    try:
        agent = BizIntelAgent(load_models=not args.demo, demo_mode=args.demo)
        print(f"Researching: '{args.query}'...\n")
        
        result = agent.research(query=args.query, mode=mode)

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                f.write(result["memo_markdown"])
            print(f"\n✅ Report saved to {out_path}")
        else:
            print("\n" + "="*80 + "\n")
            print(result["memo_markdown"])
            print("\n" + "="*80 + "\n")

        if args.artifact_dir:
            artifact_paths = write_artifact_bundle(result, Path(args.artifact_dir))
            print("\nArtifact bundle:")
            for name, path in artifact_paths.items():
                print(f"- {name}: {path}")

        if args.trace:
            trace_payload = build_trace_payload(result)
            print(json.dumps(trace_payload, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"\n❌ Error during execution: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
