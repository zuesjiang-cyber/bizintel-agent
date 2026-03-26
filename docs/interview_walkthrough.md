# 证据驱动可验证的企业财务研究Agent Flow Interview Walkthrough

This document is the shortest honest way to present the project in an internship interview.

## One-Sentence Positioning

Built an evidence-aware AI research system that turns curated private-company source packs into first-pass investment memos, combining hybrid retrieval, agent orchestration, and claim verification to reduce unsupported business analysis.

## Multi-View Framing

### Hiring manager view

Why this is worth attention:

- It is not just a prompt wrapper; it shows planning, retrieval, synthesis, verification, and evaluation.
- It has a reproducible workflow: `make check`, `make demo`, CLI, Streamlit, and exported artifacts.
- It is commercially legible. The use case is easier to understand than a generic “AI assistant.”

What would make it weak:

- Overclaiming words like `hallucination-free`, `analyst-grade`, or `production-ready`
- Pretending the static demo corpus is live market research
- Describing it as a broad business copilot instead of a narrow memo workflow

### Finance / strategy practitioner view

What is genuinely useful:

- A first-pass memo structure with business model, financial quality, valuation context, monitorables, and risks
- Traceable evidence instead of a black-box answer
- Claim-level audit artifacts that make weak support visible

What is not yet good enough:

- The static corpus is narrow
- The offline demo is still a prototype, not decision-grade analysis
- It does not replace current, live diligence work

### Technical interviewer view

The three design points to emphasize:

1. BM25 plus dense retrieval plus reranking is more robust than a single retrieval method.
2. Post-generation verification matters because fluent LLM output can still be weakly supported.
3. The project is intentionally inspectable: memo, trace, and verification audit are separate artifacts.

## Resume Bullets

- Built an end-to-end AI research pipeline that converts curated company source packs into structured investment memos using query planning, hybrid retrieval, reranking, LLM synthesis, and post-generation claim verification.
- Implemented a citation-aware retrieval stack with BM25, dense embeddings, Reciprocal Rank Fusion, and cross-encoder reranking; added offline evaluation benchmarks and reproducible demo artifacts for recruiter walkthroughs.
- Shipped a reproducible engineering workflow with CLI and Streamlit interfaces, `make check` validation, offline demo mode, and exported memo/trace/audit artifacts to support technical interviews and project review.

## 3-Minute Demo

### 0:00 to 0:20

Start with:

> I built this as a first-pass investment memo generator for curated private-company source packs. The point is not just to generate text, but to make retrieval, synthesis, and verification inspectable.

### 0:20 to 0:45

Run:

```bash
make demo
```

Say:

> This gives me a reproducible offline walkthrough with no API keys. I use it in interviews because it shows the system shape reliably.

### 0:45 to 1:30

Open `artifacts/demo/memo.md`.

Say:

> The memo is the user-facing output. The important thing is that it follows an analyst-style structure instead of returning one generic paragraph.

Be honest:

> The offline version is still a prototype and uses a static sample corpus, so I present it as an engineering demo rather than a finished research product.

### 1:30 to 2:10

Open `artifacts/demo/trace.json`.

Say:

> This is what makes the project interesting from an AI systems perspective. The query gets planned into steps, those steps run through retrieval and generation, and the workflow is inspectable instead of hidden.

### 2:10 to 2:40

Open `artifacts/demo/verification.csv`.

Say:

> I wanted a second trust layer after generation. This file shows which claims were well-supported and which ones were weak or unsupported.

### 2:40 to 3:00

Close with:

> The strongest part of the project is the architecture and engineering workflow. The next step would be stronger memo substance, fresher data, and a clear baseline comparison against a plain LLM.

## What To Claim

- Strong AI systems project
- Good example of retrieval plus generation plus verification
- Useful first-pass memo drafting workflow on curated data
- Reproducible and easy to evaluate in an interview

## What Not To Claim

- `hallucination-free`
- `analyst-grade`
- `production-ready`
- `live market intelligence`
- `general business copilot`

## Evidence To Show First

Show artifacts in this order:

1. `artifacts/demo/memo.md`
2. `artifacts/demo/trace.json`
3. `artifacts/demo/verification.csv`

That order works because it starts with user value, then shows system design, then shows trust and verification.
