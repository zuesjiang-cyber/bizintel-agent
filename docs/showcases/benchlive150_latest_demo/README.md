# Latest Benchlive 150 Showcase

This folder packages one representative case from the latest FinanceBench Open150 live run
into a demo memo, a trace walkthrough, and a verification preview.

## Run Source

- Cohort: `live_20260410_publishfix_parallel`
- Latest timestamp: `2026-04-10T11:10:13.000757`
- JSON shards: 6
- Total rows in cohort: 150
- Average `strong_support_rate`: 88.24%
- Average `verified_claim_coverage`: 88.51%
- Average `numeric_exact_match_rate`: 92.19%
- Average `primary_source_claim_coverage`: 88.09%
- Average `answer_quality`: 3.43

## Selected Case

- Item ID: `financebench_id_01488`
- Question: Which business segment of JnJ will be treated as a discontinued operation from August 30, 2023 onward?
- Showcase answer: Consumer Health business
- Gold benchmark pass: 1
- Benchmark `strong_support_rate`: 100%
- Raw verification strong rows: 10/12
- Raw verification flagged rows: 2

## Open These Files

- [demo_memo.md](demo_memo.md)
- [trace_overview.md](trace_overview.md)
- [verification_preview.md](verification_preview.md)

## Local Raw Bundle

This Git-tracked showcase keeps the three most readable demo artifacts in-repo.

If you want the full raw bundle (`memo / trace / summary / verification / benchmark_report`),
run:

```bash
make benchlive-demo
```

That regenerates the local bundle under `artifacts/benchlive150_latest_demo/`.
