from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-4o"
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

    # Verification 配置
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    entailment_strong_threshold: float = 0.7
    entailment_moderate_threshold: float = 0.4

    # Workflow 配置
    default_max_retries: int = 2
    default_timeout_seconds: int = 120

    class Config:
        env_file = ".env"


settings = Settings()
