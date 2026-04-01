from dataclasses import dataclass, field
import os

@dataclass
class RiskConfig:
    alpha_total: float = 0.30

    # module-level failure threshold
    tau_1: float = 0.0
    tau_2: float = 0.0
    tau_3: float = 0.0

    # "direct" = 只檢查端到端 P(E) <= alpha_total
    # "weighted" = 先把 alpha_total 分配成 alpha_1, alpha_2, alpha_3
    allocation_mode: str = "weighted"

    # 只有 allocation_mode="weighted" 時才會用到
    w_retrieval: float = 0.3
    w_reranker: float = 0.3
    w_generator: float = 0.4

    # 是否同時要求個別模組風險也要 <= 分配到的 alpha_j
    enforce_module_budgets: bool = False

    # 如果之後想做有限樣本保守修正，可加 safety margin
    safety_margin: float = 0.0

@dataclass
class SearchConfig:
    # -------- stage 1: retriever --------
    # 最大搜尋範圍
    max_top_k: int = 500
    min_top_k: int = 10
    add_top_k_buffer: tuple = (0, 2, 5)

    # -------- stage 2: reranker --------
    top_K_mode: str = "auto_sparse"
    min_top_K: int = 3

    # -------- stage 3: generator --------
    fix_n_rag_to_top_K: bool = True # N_rag是否要等於top-K
    max_lambda_g: int = 1 # 只生成1個答案
    lambda_s_candidates: list = field(default_factory=lambda: [0.8])
    max_stage12_candidates: int = 10
    min_N_rag: int = 1


@dataclass
class ModelConfig:
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    reranker_model: str = "BAAI/bge-reranker-base"
    generator_model: str = "openai/gpt-oss-20b"
    generator_api_base: str = os.getenv("GPTOSS_BASE_URL", "")
    generator_api_key: str = os.getenv("GPTOSS_API_KEY", "dummy")
    temperature: float = 0.0