from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.benchmark_forensics import load_payload, render_forensics_report, write_ledger_csv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a local benchmark forensics report.")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate benchmark JSON.")
    parser.add_argument("--baseline", type=Path, default=None, help="Optional baseline benchmark JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Markdown report output path.")
    parser.add_argument("--ledger-output", type=Path, default=None, help="Optional DuckDB-compatible CSV ledger path.")
    args = parser.parse_args()

    candidate = load_payload(args.candidate)
    baseline = load_payload(args.baseline) if args.baseline else None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_forensics_report(
            candidate,
            candidate_path=args.candidate,
            baseline=baseline,
            baseline_path=args.baseline,
        ),
        encoding="utf-8",
    )
    result = {"report": str(args.output)}

    if args.ledger_output:
        write_ledger_csv(candidate, args.ledger_output)
        result["ledger"] = str(args.ledger_output)

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
