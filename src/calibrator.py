from itertools import product
from metrics import retriever_fail, reranker_fail, generator_fail
from collections import defaultdict

def allocation_total(alpha_1, alpha_2, alpha_3):
    return alpha_1 + (1 - alpha_1) * alpha_2 + (1 - alpha_1) * (1 - alpha_2) * alpha_3

def end_to_end_fwer(fwer_1, fwer_2, fwer_3):
    return (
        fwer_1
        + (1 - fwer_1) * fwer_2
        + (1 - fwer_1) * (1 - fwer_2) * fwer_3
    )

def allocate_budgets(alpha_total, w1, w2, w3):
    s = 1.0 - alpha_total
    alpha_1 = 1.0 - s ** w1
    alpha_2 = 1.0 - s ** w2
    alpha_3 = 1.0 - s ** w3
    return alpha_1, alpha_2, alpha_3

def solve_alpha_3(alpha_total, alpha_1, alpha_2):
    denom = (1 - alpha_1) * (1 - alpha_2)
    if denom <= 0:
        return None
    alpha_3 = (alpha_total - alpha_1 - (1 - alpha_1) * alpha_2) / denom
    if 0 <= alpha_3 <= 1:
        return alpha_3
    return None

# top-k 自動產生函數
def auto_top_k_candidates(calib_data, retriever, search_cfg):
    """
    Data-driven top-k candidates (Method B):
    1. For each calibration query, retrieve up to max_top_k
    2. Find the first rank where gold_doc_id appears
    3. Add that rank plus small buffers
    4. Merge with a few safe anchors
    """
    ranks = set()

    for row in calib_data:
        q = row["question"]
        gold_doc_id = row["gold_doc_id"]

        docs = retriever.retrieve(q, top_k=search_cfg.max_top_k)

        found_rank = None
        for idx, d in enumerate(docs, start=1):
            doc_id = d.metadata.get("doc_id", None)
            if doc_id == gold_doc_id:
                found_rank = idx
                break

        if found_rank is not None:
            for b in search_cfg.add_top_k_buffer:
                cand = found_rank + b
                if search_cfg.min_top_k <= cand <= search_cfg.max_top_k:
                    ranks.add(cand)

    # safe anchors
    anchors = [
        search_cfg.min_top_k,
        10, 20, 30, 50,
        search_cfg.max_top_k
    ]
    for a in anchors:
        if search_cfg.min_top_k <= a <= search_cfg.max_top_k:
            ranks.add(a)

    top_k_candidates = sorted(ranks)
    return top_k_candidates

# top-K 自動產生函數
def auto_top_K_candidates(top_k, mode="auto_sparse"):
    """
    Build reranker top-K candidates under a given top-k.
    Must satisfy top_K <= top_k.
    """
    if mode == "auto_full":
        return list(range(1, top_k + 1))

    elif mode == "auto_sparse":
        vals = {
            1, 2, 3, 5,
            max(1, top_k // 4),
            max(1, top_k // 2),
            top_k
        }
        return sorted(v for v in vals if 1 <= v <= top_k)

    else:
        raise ValueError(f"Unknown top_K_mode: {mode}")


# generator thresholds 自動產生函數
def auto_N_rag_candidates(top_K):
    """
    N_rag must satisfy N_rag <= top_K
    """
    return list(range(1, top_K + 1))


def auto_lambda_g_candidates(search_cfg):
    return list(range(1, search_cfg.max_lambda_g + 1))


def auto_lambda_s_candidates(search_cfg):
    return list(search_cfg.lambda_s_candidates)


def build_threshold_candidates(calib_data, retriever, reranker, search_cfg):
    """
    Build all threshold candidate sets automatically.
    """
    top_k_candidates = auto_top_k_candidates(calib_data, retriever, search_cfg)

    top_K_candidates_map = {}
    N_rag_candidates_map = {}

    for top_k in top_k_candidates:
        top_Ks = auto_top_K_candidates(top_k, mode=search_cfg.top_K_mode)
        top_K_candidates_map[top_k] = top_Ks

        for top_K in top_Ks:
            N_rag_candidates_map[(top_k, top_K)] = auto_N_rag_candidates(top_K)

    lambda_g_candidates = auto_lambda_g_candidates(search_cfg)
    lambda_s_candidates = auto_lambda_s_candidates(search_cfg)

    return {
        "top_k_candidates": top_k_candidates,
        "top_K_candidates_map": top_K_candidates_map,
        "N_rag_candidates_map": N_rag_candidates_map,
        "lambda_g_candidates": lambda_g_candidates,
        "lambda_s_candidates": lambda_s_candidates,
    }

def time_proxy(top_k, top_K, N_rag, lambda_g, avg_doc_tokens=180, L_query=30, L_out=64):
    # 依照時間近似概念，給 reranker 和 generator 較高權重
    retrieval_cost = 1.0 * top_k
    rerank_cost = 4.0 * top_k + 0.5 * top_K
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
    A_list, B_list, C_list = [], [], []
    fail_cases = {"retriever": [], "reranker": [], "generator": []}

    for row in calib_data:
        qid = row["qid"]
        q = row["question"]
        gold_doc_id = row["gold_doc_id"]
        gold_answer = row["gold_answer"]

        ret_key = (q, top_k)
        if ret_key not in retrieve_cache:
            retrieve_cache[ret_key] = retriever.retrieve(q, top_k=top_k)
        retrieved = retrieve_cache[ret_key]

        _, A_i = retriever_fail(retrieved, gold_doc_id, tau_1)
        A_list.append(A_i)
        if A_i == 1:
            fail_cases["retriever"].append(qid)
            continue

        rerank_key = (q, top_k, top_K)
        if rerank_key not in rerank_cache:
            rerank_cache[rerank_key] = reranker.rerank(q, retrieved, top_K=top_K)
        reranked = rerank_cache[rerank_key]

        _, B_i = reranker_fail(reranked, gold_doc_id, tau_2)
        B_list.append(B_i)
        if B_i == 1:
            fail_cases["reranker"].append(qid)
            continue

        contexts = reranked[:N_rag]
        doc_ids = tuple(d.metadata["doc_id"] for d in contexts)
        gen_key = (q, doc_ids, lambda_g, lambda_s)
        if gen_key not in gen_cache:
            gen_cache[gen_key] = generator.generate_answers(
                q, contexts, lambda_g=lambda_g, lambda_s=lambda_s
            )
        generation_set = gen_cache[gen_key]

        _, C_i = generator_fail(generation_set, gold_answer, tau_3=tau_3)
        C_list.append(C_i)
        if C_i == 1:
            fail_cases["generator"].append(qid)

    fwer_1 = sum(A_list) / max(len(A_list), 1)
    fwer_2 = sum(B_list) / max(len(B_list), 1)
    fwer_3 = sum(C_list) / max(len(C_list), 1)

    pe_hat = allocation_total(fwer_1, fwer_2, fwer_3)

    return {
        "top_k": top_k,
        "top_K": top_K,
        "N_rag": N_rag,
        "lambda_g": lambda_g,
        "lambda_s": lambda_s,
        "FWER_1": fwer_1,
        "FWER_2": fwer_2,
        "FWER_3": fwer_3,
        "P(E)_hat": pe_hat,
        "fail_cases": fail_cases,
    }

def grid_search(calib_data, retriever, reranker, generator, risk_cfg, search_cfg):
    # -------------------------
    # 1. alpha allocation
    # -------------------------
    if risk_cfg.allocation_mode == "weighted":
        alpha_1, alpha_2, alpha_3 = allocate_budgets(
            risk_cfg.alpha_total,
            risk_cfg.w_retrieval,
            risk_cfg.w_reranker,
            risk_cfg.w_generator
        )
    else:
        alpha_1 = alpha_2 = alpha_3 = None

    # -------------------------
    # 2. auto threshold candidates
    # -------------------------
    cand = build_threshold_candidates(
        calib_data=calib_data,
        retriever=retriever,
        reranker=reranker,
        search_cfg=search_cfg,
    )

    top_k_candidates = cand["top_k_candidates"]
    top_K_candidates_map = cand["top_K_candidates_map"]
    N_rag_candidates_map = cand["N_rag_candidates_map"]
    lambda_g_candidates = cand["lambda_g_candidates"]
    lambda_s_candidates = cand["lambda_s_candidates"]

    retrieve_cache = {}
    rerank_cache = {}
    gen_cache = {}

    raw_results = []
    feasible_results = []
    stage12_candidates = []

    # -------------------------
    # 3. search stage 1 + 2
    # -------------------------
    for top_k in top_k_candidates:
        for top_K in top_K_candidates_map[top_k]:
            MIN_TOP_K = 3 # reranker 至少保留 3 篇
            MIN_N_RAG = 3 # generator 至少吃 3 篇

            if top_K > top_k:
                continue
            if top_K < MIN_TOP_K:
                continue
            
            s12 = evaluate_stage12(
                calib_data=calib_data,
                retriever=retriever,
                reranker=reranker,
                top_k=top_k,
                top_K=top_K,
                tau_1=risk_cfg.tau_1,
                tau_2=risk_cfg.tau_2,
                retrieve_cache=retrieve_cache,
                rerank_cache=rerank_cache,
            )

            if risk_cfg.allocation_mode == "weighted":
                if s12["FWER_1"] > alpha_1 + risk_cfg.safety_margin:
                    continue
                if s12["FWER_2"] > alpha_2 + risk_cfg.safety_margin:
                    continue

            stage12_candidates.append(s12)

    # prune
    stage12_candidates.sort(
        key=lambda x: (x["FWER_1"] + x["FWER_2"], x["top_k"], x["top_K"])
    )
    stage12_candidates = stage12_candidates[:search_cfg.max_stage12_candidates]

    print(f"stage12 candidates kept: {len(stage12_candidates)}")

    # -------------------------
    # 4. search stage 3
    # -------------------------
    for s12 in stage12_candidates:
        top_k = s12["top_k"]
        top_K = s12["top_K"]

        for N_rag in N_rag_candidates_map[(top_k, top_K)]:
            for lambda_g in lambda_g_candidates:
                for lambda_s in lambda_s_candidates:
                    s3 = evaluate_stage3(
                        passed_rows=s12["passed_rows"],
                        generator=generator,
                        top_k=top_k,
                        top_K=top_K,
                        N_rag=N_rag,
                        lambda_g=lambda_g,
                        lambda_s=lambda_s,
                        tau_3=risk_cfg.tau_3,
                        gen_cache=gen_cache,
                    )

                    fwer_1 = s12["FWER_1"]
                    fwer_2 = s12["FWER_2"]
                    fwer_3 = s3["FWER_3"]

                    pe_hat = end_to_end_fwer(fwer_1, fwer_2, fwer_3)
                    total_time = time_proxy(top_k, top_K, N_rag, lambda_g)

                    result = {
                        "top_k": top_k,
                        "top_K": top_K,
                        "N_rag": N_rag,
                        "lambda_g": lambda_g,
                        "lambda_s": lambda_s,
                        "FWER_1": fwer_1,
                        "FWER_2": fwer_2,
                        "FWER_3": fwer_3,
                        "P(E)_hat": pe_hat,
                        "time_proxy": total_time,
                    }

                    if risk_cfg.allocation_mode == "weighted":
                        result["alpha_1"] = alpha_1
                        result["alpha_2"] = alpha_2
                        result["alpha_3"] = alpha_3

                        feasible = (
                            fwer_1 <= alpha_1 + risk_cfg.safety_margin
                            and fwer_2 <= alpha_2 + risk_cfg.safety_margin
                            and fwer_3 <= alpha_3 + risk_cfg.safety_margin
                            and pe_hat <= risk_cfg.alpha_total + risk_cfg.safety_margin
                        )
                    else:
                        feasible = pe_hat <= risk_cfg.alpha_total + risk_cfg.safety_margin

                    raw_results.append(result)

                    if feasible:
                        feasible_results.append(result)

    if not feasible_results:
        raw_results.sort(key=lambda x: (x["P(E)_hat"], x["time_proxy"]))
        return None, raw_results, []

    feasible_results.sort(
        key=lambda x: (
            time_proxy(x["top_k"], x["top_K"], x["N_rag"], x["lambda_g"]),
            -x["top_K"],        # 同樣成本時，優先保留更多篇
            -x["N_rag"],        # 同樣成本時，優先更多上下文
            x["P(E)_budget"],
        )
    )
    best = feasible_results[0]
    return best, raw_results, feasible_results

def evaluate_stage12(
    calib_data, retriever, reranker, top_k, top_K, tau_1, tau_2,
    retrieve_cache, rerank_cache
):
    A_list, B_list = [], []
    passed_rows = []

    for row in calib_data:
        qid = row["qid"]
        q = row["question"]
        gold_doc_id = row["gold_doc_id"]
        gold_answer = row["gold_answer"]

        ret_key = (q, top_k)
        if ret_key not in retrieve_cache:
            retrieve_cache[ret_key] = retriever.retrieve(q, top_k=top_k)
        retrieved = retrieve_cache[ret_key]

        _, A_i = retriever_fail(retrieved, gold_doc_id, tau_1)
        A_list.append(A_i)
        if A_i == 1:
            continue

        rerank_key = (q, top_k, top_K)
        if rerank_key not in rerank_cache:
            rerank_cache[rerank_key] = reranker.rerank(q, retrieved, top_K=top_K)
        reranked = rerank_cache[rerank_key]

        _, B_i = reranker_fail(reranked, gold_doc_id, tau_2)
        B_list.append(B_i)
        if B_i == 1:
            continue

        passed_rows.append({
            "qid": qid,
            "question": q,
            "gold_answer": gold_answer,
            "reranked_docs": reranked,
        })

    fwer_1 = sum(A_list) / max(len(A_list), 1)
    fwer_2 = sum(B_list) / max(len(B_list), 1)

    return {
        "top_k": top_k,
        "top_K": top_K,
        "FWER_1": fwer_1,
        "FWER_2": fwer_2,
        "passed_rows": passed_rows,
    }

def evaluate_stage3(
    passed_rows, generator, top_k, top_K, N_rag, lambda_g, lambda_s, tau_3, gen_cache
):
    C_list = []

    for row in passed_rows:
        q = row["question"]
        gold_answer = row["gold_answer"]
        reranked = row["reranked_docs"]

        contexts = reranked[:N_rag]
        doc_ids = tuple(d.metadata["doc_id"] for d in contexts)
        gen_key = (q, doc_ids, lambda_g, lambda_s)

        if gen_key not in gen_cache:
            gen_cache[gen_key] = generator.generate_answers(
                q, contexts, lambda_g=lambda_g, lambda_s=lambda_s
            )

        generation_set = gen_cache[gen_key]
        _, C_i = generator_fail(generation_set, gold_answer, tau_3=tau_3)
        C_list.append(C_i)

    fwer_3 = sum(C_list) / max(len(C_list), 1)

    return {
        "top_k": top_k,
        "top_K": top_K,
        "N_rag": N_rag,
        "lambda_g": lambda_g,
        "lambda_s": lambda_s,
        "FWER_3": fwer_3,
    }