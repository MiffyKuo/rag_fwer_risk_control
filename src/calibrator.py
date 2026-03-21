from itertools import product
from metrics import retriever_fail, reranker_fail, generator_fail

def compute_total_fwer(alpha_1, alpha_2, alpha_3):
    return alpha_1 + (1 - alpha_1) * alpha_2 + (1 - alpha_1) * (1 - alpha_2) * alpha_3



def time_proxy(top_k, top_K, N_rag, lambda_g):
    return top_k + top_K + N_rag + 2 * lambda_g

from itertools import product
from metrics import retriever_fail, reranker_fail, generator_fail


def evaluate_one_setting(calib_data, retriever, reranker, generator,
                         top_k, top_K, N_rag, lambda_g, lambda_s,
                         tau_1, tau_2, tau_3):
    A_list = []
    B_list = []
    C_list = []

    for row in calib_data:
        q = row["question"]
        gold_doc_id = row["gold_doc_id"]
        gold_answer = row["gold_answer"]

        # 1) Retriever
        retrieved = retriever.retrieve(q, top_k=top_k)
        _, A_i = retriever_fail(retrieved, gold_doc_id, tau_1)
        A_list.append(A_i)

        if A_i == 1:
            continue

        # 2) Reranker
        reranked = reranker.rerank(q, retrieved, top_K=top_K)
        _, B_i = reranker_fail(reranked, gold_doc_id, tau_2)
        B_list.append(B_i)

        if B_i == 1:
            continue

        # 3) Generator
        contexts = reranked[:N_rag]

        generation_set = generator.generate_answers(
            q,
            contexts,
            lambda_g=lambda_g,
            lambda_s=lambda_s
        )

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


def allocation_total(alpha_1, alpha_2, alpha_3):
    return alpha_1 + (1 - alpha_1) * alpha_2 + (1 - alpha_1) * (1 - alpha_2) * alpha_3


def time_proxy(top_k, top_K, N_rag, lambda_g):
    return top_k + top_K + N_rag + 2 * lambda_g


def grid_search(calib_data, retriever, reranker, generator, risk_cfg, grid_cfg):
    raw_results = []
    feasible_results = []

    # 先枚舉所有模型參數
    model_param_list = list(product(
        grid_cfg.top_k_list,
        grid_cfg.top_K_list,
        grid_cfg.N_rag_list,
        grid_cfg.lambda_g_list,
        grid_cfg.lambda_s_list
    ))


    alpha_triplets = risk_cfg.alpha_candidates
    # # 再枚舉所有 alpha allocation
    # alpha_triplets = list(product(
    #     risk_cfg.alpha_grid,
    #     risk_cfg.alpha_grid,
    #     risk_cfg.alpha_grid
    # ))

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
            tau_3=risk_cfg.tau_3
        )

        # 對這一組模型參數，再檢查哪些 alpha allocation 可行
        for alpha_1, alpha_2, alpha_3 in alpha_triplets:
            pe = allocation_total(alpha_1, alpha_2, alpha_3)

            if pe > risk_cfg.alpha_total:
                continue

            if (
                res["FWER_1"] <= alpha_1 and
                res["FWER_2"] <= alpha_2 and
                res["FWER_3"] <= alpha_3
            ):
                candidate = {
                    **res,
                    "alpha_1": alpha_1,
                    "alpha_2": alpha_2,
                    "alpha_3": alpha_3,
                    "P(E)_budget": pe
                }
                feasible_results.append(candidate)

        raw_results.append(res)

    if not feasible_results:
        return None, raw_results, feasible_results

    feasible_results.sort(
        key=lambda x: (
            time_proxy(x["top_k"], x["top_K"], x["N_rag"], x["lambda_g"]),
            x["P(E)_budget"]
        )
    )

    return feasible_results[0], raw_results, feasible_results
