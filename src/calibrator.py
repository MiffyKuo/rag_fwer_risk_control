from itertools import product
from metrics import retriever_fail, reranker_fail, generator_fail

def allocation_total(alpha_1, alpha_2, alpha_3):
    return alpha_1 + (1 - alpha_1) * alpha_2 + (1 - alpha_1) * (1 - alpha_2) * alpha_3

def solve_alpha_3(alpha_total, alpha_1, alpha_2):
    denom = (1 - alpha_1) * (1 - alpha_2)
    if denom <= 0:
        return None
    alpha_3 = (alpha_total - alpha_1 - (1 - alpha_1) * alpha_2) / denom
    if 0 <= alpha_3 <= 1:
        return alpha_3
    return None

def time_proxy(top_k, top_K, N_rag, lambda_g, avg_doc_tokens=180, L_query=30, L_out=64):
    # 依照你 PPT 的時間近似概念，給 reranker 和 generator 較高權重
    retrieval_cost = 1.0 * top_k
    rerank_cost = 4.0 * top_k
    gen_cost = lambda_g * (0.03 * (L_query + N_rag * avg_doc_tokens) + 1.0 * L_out)
    return retrieval_cost + rerank_cost + gen_cost

def evaluate_one_setting(
    calib_data,
    retriever,
    reranker,
    generator,
    top_k,
    top_K,
    N_rag,
    lambda_g,
    lambda_s,
    tau_1,
    tau_2,
    tau_3,
    retrieve_cache,
    rerank_cache,
    gen_cache
):
    A_list = []
    B_list = []
    C_list = []

    for row in calib_data:
        q = row["question"]
        gold_doc_id = row["gold_doc_id"]
        gold_answer = row["gold_answer"]

        # 1) Retriever cache
        ret_key = (q, top_k)
        if ret_key not in retrieve_cache:
            retrieve_cache[ret_key] = retriever.retrieve(q, top_k=top_k)
        retrieved = retrieve_cache[ret_key]

        _, A_i = retriever_fail(retrieved, gold_doc_id, tau_1)
        A_list.append(A_i)
        if A_i == 1:
            continue

        # 2) Reranker cache
        rerank_key = (q, top_k, top_K)
        if rerank_key not in rerank_cache:
            rerank_cache[rerank_key] = reranker.rerank(q, retrieved, top_K=top_K)
        reranked = rerank_cache[rerank_key]

        _, B_i = reranker_fail(reranked, gold_doc_id, tau_2)
        B_list.append(B_i)
        if B_i == 1:
            continue

        # 3) Generator cache
        contexts = reranked[:N_rag]
        doc_ids = tuple(d.metadata["doc_id"] for d in contexts)
        gen_key = (q, doc_ids, lambda_g, lambda_s)

        if gen_key not in gen_cache:
            gen_cache[gen_key] = generator.generate_answers(
                q, contexts, lambda_g=lambda_g, lambda_s=lambda_s
            )
        generation_set = gen_cache[gen_key]

        gen_risk, C_i = generator_fail(
            generation_set=generation_set,
            gold_answer=gold_answer,
            tau_3=tau_3
        )
        C_list.append(C_i)

    fwer_1 = sum(A_list) / max(len(A_list), 1)
    fwer_2 = sum(B_list) / max(len(B_list), 1)
    fwer_3 = sum(C_list) / max(len(C_list), 1)

    return {
        "top_k": top_k,
        "top_K": top_K,
        "N_rag": N_rag,
        "lambda_g": lambda_g,
        "lambda_s": lambda_s,
        "FWER_1": fwer_1,
        "FWER_2": fwer_2,
        "FWER_3": fwer_3,
    }

def grid_search(calib_data, retriever, reranker, generator, risk_cfg, grid_cfg):
    raw_results = []
    feasible_results = []

    retrieve_cache = {}
    rerank_cache = {}
    gen_cache = {}

    model_param_list = list(product(
        grid_cfg.top_k_list,
        grid_cfg.top_K_list,
        grid_cfg.N_rag_list,
        grid_cfg.lambda_g_list,
        grid_cfg.lambda_s_list
    ))

    alpha_pairs = list(product(risk_cfg.alpha_grid_1, risk_cfg.alpha_grid_2))

    for top_k, top_K, N_rag, lambda_g, lambda_s in model_param_list:
        if top_K > top_k:
            continue
        if N_rag > top_K:
            continue

        res = evaluate_one_setting(
            calib_data=calib_data,
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            top_k=top_k,
            top_K=top_K,
            N_rag=N_rag,
            lambda_g=lambda_g,
            lambda_s=lambda_s,
            tau_1=risk_cfg.tau_1,
            tau_2=risk_cfg.tau_2,
            tau_3=risk_cfg.tau_3,
            retrieve_cache=retrieve_cache,
            rerank_cache=rerank_cache,
            gen_cache=gen_cache
        )

        raw_results.append(res)

        for alpha_1, alpha_2 in alpha_pairs:
            alpha_3 = solve_alpha_3(risk_cfg.alpha_total, alpha_1, alpha_2)
            if alpha_3 is None:
                continue

            pe = allocation_total(alpha_1, alpha_2, alpha_3)

            if (
                res["FWER_1"] <= alpha_1
                and res["FWER_2"] <= alpha_2
                and res["FWER_3"] <= alpha_3
            ):
                candidate = {
                    **res,
                    "alpha_1": alpha_1,
                    "alpha_2": alpha_2,
                    "alpha_3": alpha_3,
                    "P(E)_budget": pe
                }
                feasible_results.append(candidate)

    if not feasible_results:
        return None, raw_results, feasible_results

    feasible_results.sort(
        key=lambda x: (
            time_proxy(x["top_k"], x["top_K"], x["N_rag"], x["lambda_g"]),
            x["P(E)_budget"]
        )
    )
    return feasible_results[0], raw_results, feasible_results