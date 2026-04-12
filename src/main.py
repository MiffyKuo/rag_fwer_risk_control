from config import RiskConfig, SearchConfig, ModelConfig
from data_utils import load_jsonl
from retriever_module import RetrieverModule
from reranker_module import SimpleReranker
from generator_module import GeneratorModule
from calibrator import grid_search, end_to_end_fwer, evaluate_fixed_params_on_dataset
from pipeline import RiskControlledRAG


def main():
    # 1. 載入資料
    corpus_data = load_jsonl("data/corpus.jsonl")
    calib_data = load_jsonl("data/calib.jsonl")
    test_rows = load_jsonl("data/test.jsonl")
    print(f"[1] done. corpus={len(corpus_data)}, calib={len(calib_data)}, test={len(test_rows)}")

    # 2. 建立 config
    risk_cfg = RiskConfig(
        alpha_total=0.50,
        allocation_mode="weighted",   # 或 "direct"
        w_retrieval=0.3,
        w_reranker=0.3,
        w_generator=0.4,
        tau_1=0.2,
        tau_2=0.4,
        tau_3=0.4,

        # 建議開啟
        enforce_module_budgets=True,
        use_data_split=True,
        stage12_ratio=0.5,
        use_stage12_tcrcs=False, # True
        stage12_i1_ratio=0.5,
        use_stage3_certified_bound=True,

        # 可自行調整
        delta_total=0.10,
        delta_1=0.03,
        delta_2=0.03,
        delta_3=0.04,
        random_seed=42,
    )
    search_cfg = SearchConfig()
    model_cfg = ModelConfig()

    # 3. 建立模組
    print("[2] building retriever...")
    retriever = RetrieverModule(model_cfg.embedding_model)
    retriever.build_index(corpus_data)
    print("[2] retriever ready.")

    print("[3] building reranker...")
    reranker = SimpleReranker(model_cfg.reranker_model)
    print("[3] reranker ready.")

    print("[4] building generator...")
    generator = GeneratorModule(
        model_name=model_cfg.generator_model,
        api_base=model_cfg.generator_api_base,
        api_key=model_cfg.generator_api_key,
        temperature=model_cfg.temperature,
        max_concurrent=model_cfg.generator_max_concurrent,
        request_timeout=model_cfg.generator_request_timeout,
        max_tokens=model_cfg.generator_max_tokens,
    )
    print("[4] generator ready.")

    # 4. grid search
    print("[5] start grid_search...")
    best_params, all_results, feasible_results = grid_search(
        calib_data=calib_data,
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        risk_cfg=risk_cfg,
        search_cfg=search_cfg,
    )
    print("[5] grid_search done.")

    # 5. 搜尋結束後，再一次性存 cache
    generator.save_cache()

    # 6.1 後續印結果(沒成功的話)
    if best_params is None:
        print("找不到滿足整體 FWER 門檻的參數組合。")

        ranked = []
        for r in all_results:
            pe_hat = end_to_end_fwer(r["FWER_1"], r["FWER_2"], r["FWER_3"])
            gap = r.get("P(E)_cert", pe_hat) - risk_cfg.alpha_total
            ranked.append({
                **r,
                "P(E)_hat": round(pe_hat, 4),
                "P(E)_cert": round(r.get("P(E)_cert", pe_hat), 4),
                "gap_to_alpha_total": round(gap, 4),
            })

        ranked.sort(
            key=lambda x: (
                x["gap_to_alpha_total"],
                x["P(E)_cert"],
                -x["top_K"],
                -x["N_rag"],
                x["lambda_g"],
            )
        )

        print("\n最接近可行的前 10 組：")
        for r in ranked[:10]:
            print(r)
        return

    print("最佳參數：", best_params)

    # 6.2 獨立 test set 檢查
    test_summary = evaluate_fixed_params_on_dataset(
        rows=test_rows,
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        params=best_params,
        tau_1=risk_cfg.tau_1,
        tau_2=risk_cfg.tau_2,
        tau_3=risk_cfg.tau_3,
    )
    print("Independent test summary:", test_summary)

    # 6.3 後續印結果(成功的話)
    rag = RiskControlledRAG(
        retriever,
        reranker,
        generator,
        best_params,
        fix_n_rag_to_top_K=search_cfg.fix_n_rag_to_top_K
    )
    sample_q = test_rows[0]["question"]
    result = rag.answer(sample_q)

    print("Sample test question:", sample_q)
    print(result)


if __name__ == "__main__":
    main()