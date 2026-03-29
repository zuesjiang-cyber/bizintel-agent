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
    period: Optional[str] = None
    published_at: Optional[str] = None
    issuer: Optional[str] = None
    is_primary: Optional[bool] = None


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
    company: Optional[str] = None
    doc_id: Optional[str] = None
    source_type: Optional[str] = None
    period: Optional[str] = None
    title: Optional[str] = None
    is_primary: Optional[bool] = None
    trust_level: Optional[int] = None
    content_type: Optional[str] = None
    metric_signals: List[str] = field(default_factory=list)


# ========== 分析计划 ==========

@dataclass
class AnalysisStep:
    name: str
    description: str
    required: bool
    search_queries: List[str] = field(default_factory=list)
    query_contracts: List[Dict[str, Any]] = field(default_factory=list)
    output_schema: Dict[str, str] = field(default_factory=dict)
    evidence_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisPlan:
    mode: AnalysisMode
    user_query: str
    steps: List[AnalysisStep] = field(default_factory=list)
    contract: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ========== 深度研究控制层 ==========

class ResearchLane(Enum):
    HARD_FACT = "hard_fact"
    SEMANTIC = "semantic"


class ResearchQuestionStatus(Enum):
    PENDING = "pending"
    RETRIEVING = "retrieving"
    NEEDS_FOLLOWUP = "needs_followup"
    COMPLETED = "completed"
    CONFLICT = "conflict"
    REFUSED = "refused"


@dataclass
class ResearchTask:
    query: str
    company_id: str
    period: Optional[str] = None
    mode: AnalysisMode = AnalysisMode.COMPANY
    mentioned_company_ids: List[str] = field(default_factory=list)
    max_subquestions: int = 6
    max_followup_rounds: int = 2
    max_evidence_per_question: int = 8
    llm_call_budget: int = 8


@dataclass
class ResearchSubquestion:
    question_id: str
    text: str
    lane: ResearchLane
    priority: int
    required: bool = True
    fact_slot: str = ""
    metric_family: str = ""
    required_period: Optional[str] = None
    allowed_source_types: List[str] = field(default_factory=list)
    needs_numeric_verification: bool = False
    status: ResearchQuestionStatus = ResearchQuestionStatus.PENDING
    rounds_used: int = 0
    max_rounds: int = 2


@dataclass
class ResearchDecisionRecord:
    decision_type: str
    question_id: str
    reason: str
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ResearchReplayRecord:
    task: ResearchTask
    subquestions: List[ResearchSubquestion] = field(default_factory=list)
    decisions: List[ResearchDecisionRecord] = field(default_factory=list)
    llm_calls_used: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class EvidenceAssessment:
    valid_chunk_count: int = 0
    best_entailment: float = 0.0
    mean_entailment: float = 0.0
    numeric_match: bool = False
    metric_match: bool = False
    high_trust_hit: bool = False
    sufficient: bool = False
    insufficient: bool = False
    conflict: bool = False
    reasons: List[str] = field(default_factory=list)
    matched_chunk_ids: List[str] = field(default_factory=list)


@dataclass
class ResearchQuestionResult:
    subquestion: ResearchSubquestion
    status: ResearchQuestionStatus
    evidence: List[RetrievedChunk] = field(default_factory=list)
    assessments: List[EvidenceAssessment] = field(default_factory=list)
    answer_text: str = ""
    verified_claims: List["VerificationResult"] = field(default_factory=list)
    supported_content: str = ""
    refusal_reason: Optional[str] = None
    followup_queries: List[str] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)


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
    cited_chunks: List[str] = field(default_factory=list)
    contains_numbers: bool = False
    extracted_numbers: List[str] = field(default_factory=list)
    specificity_score: float = 0.0


@dataclass
class VerificationResult:
    claim: Claim
    confidence: ConfidenceLevel
    nli_score: float
    numeric_verified: Optional[bool]
    supporting_evidence: List[str]     # 支撑的原文片段
    explanation: str
    failure_reason: Optional[str] = None
    supporting_source_ids: List[str] = field(default_factory=list)
    supporting_chunk_ids: List[str] = field(default_factory=list)
    primary_source_supported: Optional[bool] = None


# ========== Memo ==========

@dataclass
class MemoSection:
    title: str
    content: str
    evidence_notes: List[Dict[str, Any]] = field(default_factory=list)
    writing_trace: Dict[str, Any] = field(default_factory=dict)
    claims: List[Claim] = field(default_factory=list)
    verification_results: List[VerificationResult] = field(default_factory=list)


@dataclass
class GeneratedMemo:
    title: str
    mode: AnalysisMode
    query: str
    executive_summary: str = ""
    contract: Dict[str, Any] = field(default_factory=dict)
    sections: List[MemoSection] = field(default_factory=list)
    sources: List[DocumentMeta] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    overall_confidence: float = 0.0
