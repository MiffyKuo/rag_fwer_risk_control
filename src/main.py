from config import RiskConfig, SearchGrid, ModelConfig
from data_utils import load_jsonl
from retriever_module import RetrieverModule
from reranker_module import SimpleReranker
from generator_module import GeneratorModule
from calibrator import grid_search
from pipeline import RiskControlledRAG


def main():
    risk_cfg = RiskConfig(
        alpha_total=0.30,
        tau_1=0.0,
        tau_2=0.0,
        tau_3=0.60
    )

    grid_cfg = SearchGrid()
    model_cfg = ModelConfig()

    corpus = load_jsonl("data/corpus.jsonl")
    calib = load_jsonl("data/calib.jsonl")

    retriever = RetrieverModule(model_cfg.embedding_model)
    retriever.build_index(corpus)

    reranker = SimpleReranker(model_cfg.reranker_model)

    generator = GeneratorModule(
        model_name=model_cfg.generator_model,
        api_base=model_cfg.generator_api_base,
        api_key=model_cfg.generator_api_key,
        temperature=model_cfg.temperature
    )

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

        ranked = sorted(
            all_results,
            key=lambda x: (x["FWER_3"], x["FWER_2"], x["FWER_1"], -x["top_k"], -x["lambda_g"])
        )

        print("\n最接近可行的前 10 組：")
        for r in ranked[:10]:
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