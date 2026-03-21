from dataclasses import dataclass, field

@dataclass
class RiskConfig:
    alpha_total: float = 0.5

    # 三個 tau 還是保留，因為它們是 loss/risk 的失敗門檻
    tau_1: float = 0.0
    tau_2: float = 0.0
    tau_3: float = 0.0

    # alpha allocation 的搜尋格點
    alpha_candidates = [(0.0, 0.2, 0.5)]
    # alpha_grid: list = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3])


@dataclass
class SearchGrid:
    top_k_list: list = field(default_factory=lambda: [3, 5, 10])
    top_K_list: list = field(default_factory=lambda: [1, 2, 3])
    N_rag_list: list = field(default_factory=lambda: [1, 2])
    lambda_g_list: list = field(default_factory=lambda: [1])
    lambda_s_list: list = field(default_factory=lambda: [0.8])


@dataclass
class ModelConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ollama_model: str = "llama3.1:8b"