# BizIntel Architecture

The system is built on 5 core pillars designed to execute high-fidelity business research:

## 1. Document Processing & Ingestion (`retrieval/`)
- Accepts Raw Texts, JSONs, and Markdown.
- `TextChunker` breaks them into token-limited blocks with overlap, tagging source metadata.
- `ingest.py` standardizes them into `processed/` output JSON sequences.

## 2. Hybrid Search Engine (`retrieval/hybrid_retriever.py`)
- **Stage 1**: BM25 (Rank-BM25) grabs exact match relevance.
- **Stage 2**: Dense Embedding (BAAI/bge-base-en) grabs semantic meaning.
- **Stage 3**: Reciprocal Rank Fusion (RRF) interleaves to prevent single-bias.
- **Stage 4**: Cross-Encoder (MS-MARCO) performs final fine-grained scoring.

## 3. Workflow Automation (`agent/workflow_engine.py`)
- Executes Directed Acyclic Graphs (DAG) consisting of `WorkflowNode` instances.
- Handles retry tracking, timeouts, and state preservation.

## 4. LLM synthesis & Extraction (`agent/planner.py`, `agent/prompts/synthesis.py`)
- Analyzes unstructured queries against known entities/frameworks.
- Synthesizes findings against pre-defined Analyst templates (Company Deep-dive, Industry Landscape).

## 5. Factual Verification (`verification/evidence_verifier.py`)
- The `ClaimExtractor` isolates facts, numbers, and logic clauses.
- The `EvidenceVerifier` maps claims to their quoted sources, scoring textual entailment via DeBERTa NLI. Limits hallucinated output to `UNSUPPORTED`.
