from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime


# ========== 分析模式 ==========

class AnalysisMode(Enum):
    COMPANY = "company_deep_dive"
    INDUSTRY = "industry_landscape"
    COMPETITIVE = "competitive_comparison"


# ========== 文档与 Chunk ==========

@dataclass
class DocumentMeta:
    source_id: str          # 唯一标识
    source_type: str        # "webpage" | "pdf" | "news" | "filing"
    title: str
    url: Optional[str] = None
    date: Optional[str] = None
    company: Optional[str] = None


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    source_id: str          # 关联到 DocumentMeta
    page: Optional[int] = None
    section: Optional[str] = None
    token_count: int = 0


# ========== 检索结果 ==========

@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source_id: str
    page: Optional[int]
    score: float
    bm25_rank: int
    dense_rank: int
    rerank_score: float


# ========== 分析计划 ==========

@dataclass
class AnalysisStep:
    name: str
    description: str
    required: bool
    search_queries: List[str] = field(default_factory=list)
    output_schema: Dict[str, str] = field(default_factory=dict)


@dataclass
class AnalysisPlan:
    mode: AnalysisMode
    user_query: str
    steps: List[AnalysisStep] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ========== 工作流 ==========

class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class NodeResult:
    status: NodeStatus
    data: Any = None
    error: Optional[str] = None
    duration_ms: int = 0
    retry_count: int = 0


@dataclass
class WorkflowEvent:
    timestamp: str
    node_name: str
    event_type: str
    detail: Optional[str] = None


# ========== 验证 ==========

class ConfidenceLevel(Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    UNSUPPORTED = "unsupported"


@dataclass
class Claim:
    claim_id: str
    text: str
    section: str                       # 来自 memo 的哪个 section
    cited_sources: List[str] = field(default_factory=list)
    contains_numbers: bool = False
    extracted_numbers: List[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    claim: Claim
    confidence: ConfidenceLevel
    nli_score: float
    numeric_verified: Optional[bool]
    supporting_evidence: List[str]     # 支撑的原文片段
    explanation: str


# ========== Memo ==========

@dataclass
class MemoSection:
    title: str
    content: str
    claims: List[Claim] = field(default_factory=list)
    verification_results: List[VerificationResult] = field(default_factory=list)


@dataclass
class GeneratedMemo:
    title: str
    mode: AnalysisMode
    query: str
    sections: List[MemoSection] = field(default_factory=list)
    sources: List[DocumentMeta] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    overall_confidence: float = 0.0
