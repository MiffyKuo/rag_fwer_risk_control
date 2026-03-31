from config import RiskConfig, SearchConfig, ModelConfig
from data_utils import load_jsonl
from retriever_module import RetrieverModule
from reranker_module import SimpleReranker
from generator_module import GeneratorModule
from calibrator import grid_search
from pipeline import RiskControlledRAG


def main():
    # 1. 載入資料
    corpus_data = load_jsonl("data/corpus.jsonl")
    calib_data = load_jsonl("data/calib.jsonl")
    test_rows = load_jsonl("data/test.jsonl")

    # 2. 建立 config
    risk_cfg = RiskConfig(
        alpha_total=0.30,
        allocation_mode="direct",   # 或 "weighted"
        w_retrieval=0.1,
        w_reranker=0.2,
        w_generator=0.7,
        tau_1=0.0,
        tau_2=0.0,
        tau_3=0.0,
    )
    grid_cfg = SearchConfig()
    model_cfg = ModelConfig()

    # 3. 建立模組
    retriever = RetrieverModule(model_cfg.embedding_model)
    retriever.build_index(corpus_data)

    reranker = SimpleReranker(model_cfg.reranker_model)
    generator = GeneratorModule(
        model_name=model_cfg.generator_model,
        api_base=model_cfg.generator_api_base,
        api_key=model_cfg.generator_api_key,
        temperature=model_cfg.temperature,
    )

    # 4. grid search
    best_params, all_results, feasible_results = grid_search(
        calib_data=calib_data,
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        risk_cfg=risk_cfg,
        grid_cfg=grid_cfg,
    )

    # 5. 搜尋結束後，再一次性存 cache
    generator.save_cache()

    # 6.1 後續印結果(沒成功的話)
    if best_params is None:
        print("找不到滿足整體 FWER 門檻的參數組合。")
        ranked = sorted(
            all_results,
            key=lambda x: (x["FWER_3"], x["FWER_2"], x["FWER_1"], -x["top_k"], -x["lambda_g"])
        )
        print("\n最接近可行的前 10 組：")
        for r in ranked[:10]:
            print(r)
        return

    print("最佳參數：", best_params)

    # 6.2 後續印結果(成功的話)
    rag = RiskControlledRAG(retriever, reranker, generator, best_params)
    sample_q = test_rows[0]["question"]
    result = rag.answer(sample_q)

    print("Sample test question:", sample_q)
    print(result)
if __name__ == "__main__":
    main()