from dataclasses import dataclass, field
import os

@dataclass
class RiskConfig:
    alpha_total: float = 0.30
    tau_1: float = 0.0
    tau_2: float = 0.0
    tau_3: float = 0.0 # generator_fail 是 0/1 exact-match

    alpha_grid_1: list = field(default_factory=lambda: [0.00, 0.03, 0.05, 0.08, 0.10])
    alpha_grid_2: list = field(default_factory=lambda: [0.10, 0.125, 0.15, 0.175, 0.20])

@dataclass
class SearchGrid:
    # 先用小 grid 找方向
    top_k_list: list = field(default_factory=lambda: [40, 80, 120])
    top_K_list: list = field(default_factory=lambda: [3, 5, 8, 10])
    N_rag_list: list = field(default_factory=lambda: [3, 5])
    lambda_g_list: list = field(default_factory=lambda: [1, 2, 4])
    lambda_s_list: list = field(default_factory=lambda: [0.80, 0.90])

@dataclass
class ModelConfig:
    # Hugging Face embedding model
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2" 
    #"models/all-mpnet-base-v2"
    # 如果不想先手動下載，直接寫：
    # embedding_model: str = "sentence-transformers/all-mpnet-base-v2"

    reranker_model: str = "BAAI/bge-reranker-base"

    # gpt-oss
    generator_model: str = "openai/gpt-oss-20b"
    generator_api_base: str = os.getenv("GPTOSS_BASE_URL", "")
    generator_api_key: str = os.getenv("GPTOSS_API_KEY", "dummy")

    temperature: float = 0.0