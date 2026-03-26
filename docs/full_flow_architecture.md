# 证据驱动可验证的企业财务研究Agent Flow Full Flow Architecture

This document provides a complete end-to-end architecture diagram for the current verified project flow.

## 1. End-to-End System Flow

```mermaid
flowchart TD
    User["User Query"] --> Entry

    subgraph Entry["Entry Points"]
        CLI["CLI: run.py"]
        UI["Web UI: app/streamlit_app.py"]
        Demo["Offline Demo: make demo"]
    end

    CLI --> Orch
    UI --> Orch
    Demo --> Orch

    subgraph Orchestration["Planning & Orchestration"]
        Orch["BizIntelAgent / agent/orchestrator.py"]
        Planner["Planner / agent/planner.py"]
        Engine["Workflow Engine / agent/workflow_engine.py"]
        Executor["Analysis Executor / agent/executor.py"]
    end

    Orch --> Planner
    Planner --> Engine
    Engine --> Executor

    subgraph Retrieval["Hybrid Retrieval"]
        QueryGen["Step-Specific Search Queries"]
        BM25["BM25 Lexical Search"]
        Dense["Dense Retrieval"]
        RRF["Reciprocal Rank Fusion"]
        Rerank["Cross-Encoder Reranker"]
        Evidence["Retrieved Evidence Chunks"]
    end

    Executor --> QueryGen
    QueryGen --> BM25
    QueryGen --> Dense
    BM25 --> RRF
    Dense --> RRF
    RRF --> Rerank
    Rerank --> Evidence

    subgraph Synthesis["Section Synthesis & Memo Assembly"]
        StepDraft["Per-Step Analysis Drafts"]
        Writer["Report Writer / agent/report_writer.py"]
        Memo["Structured Memo Object"]
        Markdown["Rendered Markdown Memo"]
    end

    Evidence --> StepDraft
    Executor --> StepDraft
    StepDraft --> Writer
    Writer --> Memo
    Memo --> Markdown

    subgraph Verification["Claim Verification"]
        Claims["Claim Extractor / verification/claim_extractor.py"]
        Verifier["Evidence Verifier / verification/evidence_verifier.py"]
        Summary["Support Summary & Confidence Labels"]
    end

    Memo --> Claims
    Evidence --> Verifier
    Claims --> Verifier
    Verifier --> Summary
    Summary --> Memo

    subgraph Outputs["Outputs & Review Artifacts"]
        Terminal["Terminal Output"]
        Streamlit["Interactive Streamlit View"]
        Artifacts["memo.md / trace.json / summary.json / verification.csv"]
    end

    Markdown --> Terminal
    Markdown --> Streamlit
    Memo --> Artifacts
    Summary --> Artifacts
    Orch --> Artifacts
```

## 2. Corpus Preparation Flow

```mermaid
flowchart LR
    subgraph Sources["Source Pack Inputs"]
        Profile["profile.json"]
        Txt[".txt source docs"]
        Json["Supplemental .json source docs"]
        Pdf["PDF docs (optional)"]
    end

    subgraph Parsing["Parsing & Normalization"]
        Parser["tools/doc_parser.py"]
        Meta["DocumentMeta records"]
        Raw["Normalized source text"]
    end

    subgraph Chunking["Chunking & Storage"]
        Chunker["retrieval/chunking.py"]
        ProcChunks["data/processed/<company>/chunks.json"]
        ProcSources["data/processed/<company>/sources.json"]
    end

    subgraph RuntimeUse["Runtime Consumption"]
        DemoLoad["Demo mode: in-memory chunk loading"]
        IndexBuild["Optional live index build"]
        Hybrid["HybridRetriever"]
    end

    Profile --> Parser
    Txt --> Parser
    Json --> Parser
    Pdf --> Parser

    Parser --> Meta
    Parser --> Raw
    Raw --> Chunker
    Meta --> Chunker

    Chunker --> ProcChunks
    Meta --> ProcSources

    ProcChunks --> DemoLoad
    ProcChunks --> IndexBuild
    IndexBuild --> Hybrid
    DemoLoad --> Hybrid
```

## 3. Interview-Friendly Narrative

The shortest accurate way to explain this diagram is:

1. Source packs are normalized into chunked evidence.
2. A user query is classified and decomposed into analysis steps.
3. Each step runs through hybrid retrieval instead of a single vector lookup.
4. Retrieved evidence is turned into section drafts and assembled into a memo.
5. Claims in the memo are checked against evidence and exported with trace artifacts.

## 4. Scope Notes

- This diagram reflects the current verified project flow.
- It includes offline demo mode, reproducible artifacts, and current hybrid retrieval.
- It does not claim live web ingestion or production-grade evaluation beyond what is already implemented.
