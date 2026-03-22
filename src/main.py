from config import RiskConfig, SearchGrid, ModelConfig
from data_utils import load_jsonl
from retriever_module import RetrieverModule
from reranker_module import SimpleReranker
from generator_module import GeneratorModule
from calibrator import grid_search
from pipeline import RiskControlledRAG

# 整個程式的入口 : 
# 1. 設定風險、
# 2. 載入資料、
# 3. 建 retriever / reranker / generator、
# 4. 做 calibration 找最佳參數、
# 5. 用最佳參數回答問題

def main():
    # 使用者只輸入整體 FWER
    risk_cfg = RiskConfig(
    alpha_total=0.30,   # 先用 0.30 或 0.40 測試
    tau_1=0.0,          # retriever 先保留 binary fail
    tau_2=0.0,          # reranker 先保留 binary fail
    tau_3=0.60          # 生成風險容忍門檻，不能設 0
    )

    # alpha_grid = AlphaGrid()
    grid_cfg = SearchGrid()
    model_cfg = ModelConfig()

    corpus = load_jsonl("data/corpus.jsonl")
    calib = load_jsonl("data/calib.jsonl")

    retriever = RetrieverModule(model_cfg.embedding_model)
    retriever.build_index(corpus)

    reranker = SimpleReranker(model_cfg.reranker_model)
    generator = GeneratorModule(model_cfg.ollama_model)

    best_params, all_results, feasible_results = grid_search(
    calib_data=calib,
    retriever=retriever,
    reranker=reranker,
    generator=generator,
    risk_cfg=risk_cfg,
    grid_cfg=grid_cfg
)

    if best_params is None:
        print("找不到滿足整體 FWER 門檻的參數組合。")
        print("\n各組模型參數的原始結果如下：")
        for r in all_results:
            print(r)
        return

    print("最佳參數：")
    print(best_params)

    rag = RiskControlledRAG(retriever, reranker, generator, best_params)

    test_rows = load_jsonl("data/test.jsonl")
    sample_q = test_rows[0]["question"]
    result = rag.answer(sample_q)

    print("Sample test question:", sample_q)
    print(result)

if __name__ == "__main__":
    main()