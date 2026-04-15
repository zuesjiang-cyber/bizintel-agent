from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import AliasChoices, ConfigDict, Field


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_API_KEY", "MINIMAX_API_KEY", "ANTHROPIC_API_KEY"),
    )
    openai_api_base: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices("OPENAI_BASE_URL", "OPENAI_API_BASE", "MINIMAX_API_BASE", "ANTHROPIC_BASE_URL"),
    )
    openai_model: str = Field(
        default="gpt-5.2",
        validation_alias=AliasChoices("OPENAI_MODEL", "MINIMAX_MODEL", "ANTHROPIC_MODEL"),
    )
    llm_mode: str = "auto"
    strict_live_mode: bool = Field(
        default=False,
        validation_alias=AliasChoices("STRICT_LIVE_MODE", "FORCE_LIVE_STRICT"),
    )
    llm_request_timeout_seconds: float = Field(
        default=120.0,
        validation_alias=AliasChoices("LLM_REQUEST_TIMEOUT_SECONDS", "OPENAI_TIMEOUT_SECONDS"),
    )
    llm_request_max_retries: int = Field(
        default=2,
        validation_alias=AliasChoices("LLM_REQUEST_MAX_RETRIES", "OPENAI_MAX_RETRIES"),
    )
    llm_judge_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("LLM_JUDGE_ENABLED"),
    )
    benchmark_question_max_retries: int = Field(
        default=2,
        validation_alias=AliasChoices("BENCHMARK_QUESTION_MAX_RETRIES", "BENCHMARK_ITEM_MAX_RETRIES"),
    )
    search_api_key: str = ""

    data_dir: Path = Path("data")
    company_packs_dir: Path = Path("data/company_packs")
    eval_cases_dir: Path = Path("data/eval_cases")

    # Retrieval 配置
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    bm25_weight: float = 0.4
    dense_weight: float = 0.6
    retrieval_top_k: int = 10
    retrieval_mode: str = "full_hybrid"

    # Verification 配置
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    entailment_strong_threshold: float = 0.7
    entailment_moderate_threshold: float = 0.4
    nli_batch_size: int = 8
    verification_max_chunks_per_source: int = 4
    inference_device: str = Field(
        default="auto",
        validation_alias=AliasChoices("INFERENCE_DEVICE", "TORCH_DEVICE"),
    )

    # Workflow 配置
    default_max_retries: int = 2
    default_timeout_seconds: int = Field(
        default=300,
        validation_alias=AliasChoices("DEFAULT_TIMEOUT_SECONDS", "WORKFLOW_TIMEOUT_SECONDS"),
    )
    retrieval_gap_max_rounds: int = 2

settings = Settings()
