from dataclasses import dataclass, field

@dataclass
class RiskConfig:
    alpha_total: float = 0.30
    tau_1: float = 0.0
    tau_2: float = 0.0
    tau_3: float = 0.60

    # alpha allocation 的搜尋格點
    # 只搜尋 alpha_1, alpha_2，alpha_3 用總 FWER 公式反推
    alpha_grid_1: list = field(default_factory=lambda: [0.01, 0.03, 0.05, 0.08, 0.10, 0.15])
    alpha_grid_2: list = field(default_factory=lambda: [0.01, 0.03, 0.05, 0.08, 0.10, 0.15])

    # alpha_grid: list = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3])


@dataclass
class SearchGrid:
    top_k_list: list = field(default_factory=lambda: [5, 10, 20, 30])
    top_K_list: list = field(default_factory=lambda: [1, 2, 3, 5, 8])
    N_rag_list: list = field(default_factory=lambda: [1, 2, 3, 5])
    lambda_g_list: list = field(default_factory=lambda: [1, 2])
    lambda_s_list: list = field(default_factory=lambda: [0.80])


@dataclass
class ModelConfig:
    embedding_model: str = "text-embedding-3-large"
    reranker_model: str = "BAAI/bge-reranker-base"
    generator_model: str = "gpt-4.1-mini"
    temperature: float = 0.0