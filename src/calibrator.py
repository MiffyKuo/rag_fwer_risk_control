import math
import random
from metrics import retriever_fail, reranker_fail, generator_fail


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


def split_rows(rows, ratio, seed=42):
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    cut = int(len(rows) * ratio)
    return rows[:cut], rows[cut:]

def get_effective_modes(calib_data, risk_cfg):
    """
    小資料時自動放寬設定，避免 split 之後每段只剩 1~2 題，
    造成 finite-sample 條件永遠不可能通過。
    """
    n_total = len(calib_data)

    use_data_split = risk_cfg.use_data_split
    use_stage12_tcrcs = risk_cfg.use_stage12_tcrcs
    use_stage3_certified_bound = risk_cfg.use_stage3_certified_bound

    # calibration 太小，直接不要 split
    if n_total < 12:
        use_data_split = False
        use_stage12_tcrcs = False
        use_stage3_certified_bound = False
        print(f"[auto-relax] calib size={n_total} 太小，關閉 data split / tCRC-s / certified bound")
        return use_data_split, use_stage12_tcrcs, use_stage3_certified_bound

    # stage12 太小時，不做 stage1/stage2 的二次 split
    stage12_n = int(n_total * risk_cfg.stage12_ratio) if use_data_split else n_total
    if stage12_n < 8:
        use_stage12_tcrcs = False
        print(f"[auto-relax] stage12 size={stage12_n} 太小，關閉 stage12 tCRC-s")

    # stage3 太小時，不做 certified upper bound
    stage3_n = n_total - stage12_n if use_data_split else n_total
    if stage3_n < 10:
        use_stage3_certified_bound = False
        print(f"[auto-relax] stage3 size={stage3_n} 太小，關閉 stage3 certified bound")

    return use_data_split, use_stage12_tcrcs, use_stage3_certified_bound

def finite_sample_pass(num_fail, n, alpha):
    if n <= 0:
        return False
    rhs = (n + 1) * alpha - 1
    return num_fail <= rhs


def _binom_cdf(k, n, p):
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    s = 0.0
    for i in range(k + 1):
        s += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return min(max(s, 0.0), 1.0)


def hb_upper_bound(r_hat, n, delta):
    """
    簡化版 Hoeffding-Bentkus / binomial inversion upper bound
    不需要 scipy，直接用二分搜尋。
    """
    if n <= 0:
        return 1.0

    r_hat = min(max(r_hat, 0.0), 1.0)
    delta = min(max(delta, 1e-12), 1.0 - 1e-12)

    k = math.ceil(n * r_hat)
    target = delta / math.e

    lo = r_hat
    hi = 1.0

    for _ in range(60):
        mid = (lo + hi) / 2.0
        cdf = _binom_cdf(k, n, mid)
        if cdf <= target:
            hi = mid
        else:
            lo = mid

    return hi


# top-k 自動產生函數
def auto_top_k_candidates(calib_data, retriever, search_cfg):
    """
    Multi-gold version:
    對每個 query，找第一個 relevant doc 出現的位置，
    再根據這個位置產生 top-k 候選。
    """
    ranks = set()

    for row in calib_data:
        q = row["question"]

        gold_doc_ids = row.get("gold_doc_ids")
        if gold_doc_ids is None:
            gold_doc_ids = [row["gold_doc_id"]]
        gold_set = set(gold_doc_ids)

        docs = retriever.retrieve(q, top_k=search_cfg.max_top_k)

        found_rank = None
        for idx, d in enumerate(docs, start=1):
            doc_id = d.metadata.get("doc_id", None)
            if doc_id in gold_set:
                found_rank = idx
                break

        if found_rank is not None:
            for b in search_cfg.add_top_k_buffer:
                cand = found_rank + b
                if search_cfg.min_top_k <= cand <= search_cfg.max_top_k:
                    ranks.add(cand)

    anchors = [
        search_cfg.min_top_k,
        10, 20, 30, 50,
        search_cfg.max_top_k,
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
            1,
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
            if search_cfg.fix_n_rag_to_top_K:
                N_rag_candidates_map[(top_k, top_K)] = [top_K]
            else:
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

def _get_retrieved_docs(retrieve_cache, retriever, question, top_k, max_top_k_for_cache=None):
    if max_top_k_for_cache is None:
        max_top_k_for_cache = top_k

    ret_key = question

    if ret_key not in retrieve_cache:
        retrieve_cache[ret_key] = retriever.retrieve(question, top_k=max_top_k_for_cache)

    return retrieve_cache[ret_key][:top_k]


def _get_reranked_docs(rerank_cache, reranker, question, retrieved_docs, top_k, top_K):
    """
    只對每個 (question, top_k) 做一次 rerank。
    cache 中存完整 reranked list，不要對每個 top_K 都各存一份。
    """
    rerank_key = (question, top_k)

    if rerank_key not in rerank_cache:
        full_reranked = reranker.rerank(question, retrieved_docs, top_K=top_k)
        rerank_cache[rerank_key] = full_reranked

    return rerank_cache[rerank_key][:top_K]

def _batch_fill_gen_cache(
    rows,
    generator,
    N_rag,
    lambda_g,
    lambda_s,
    gen_cache
):
    """
    將 rows 中尚未存在於 gen_cache 的項目，整批送去 generator。
    rows 的每個元素需包含:
        - question
        - reranked_docs
    """
    requests_data = []

    for row in rows:
        q = row["question"]
        reranked = row["reranked_docs"]

        contexts = reranked[:N_rag]
        doc_ids = tuple(d.metadata["doc_id"] for d in contexts)
        gen_key = (q, doc_ids, lambda_g, lambda_s)

        if gen_key in gen_cache:
            continue

        prompt = generator.build_prompt(q, contexts)
        requests_data.append({
            "user_key": gen_key,
            "prompt": prompt,
        })

    if requests_data:
        chunk_size = max(1, generator.max_concurrent)
        print(f"[GEN] total requests_data = {len(requests_data)}, chunk_size = {chunk_size}")

        for i in range(0, len(requests_data), chunk_size):
            chunk = requests_data[i:i + chunk_size]
            print(f"[GEN] sending chunk {i // chunk_size + 1}, batch_size = {len(chunk)}")
            
            batch_outputs = generator.batch_generate_answers(
                requests_data=chunk,
                lambda_g=lambda_g,
                lambda_s=lambda_s,
                max_retry=6,
            )

            for gen_key, answers in batch_outputs.items():
                gen_cache[gen_key] = answers


def _collect_stage3_rows(
    calib_data,
    retriever,
    reranker,
    top_k,
    top_K,
    tau_1,
    tau_2,
    retrieve_cache,
    rerank_cache
):
    """
    先跑 stage1 + stage2，收集真正要送進 generator 的 rows。
    回傳:
        - stage3_rows
        - A_list
        - B_list
        - fail_cases
        - n_stage1
        - n_stage2
    """
    A_list, B_list = [], []
    fail_cases = {"retriever": [], "reranker": [], "generator": []}
    stage3_rows = []

    n_stage1 = 0
    n_stage2 = 0

    for row in calib_data:
        qid = row["qid"]
        q = row["question"]
        gold_answer = row["gold_answer"]

        gold_doc_ids = row.get("gold_doc_ids")
        if gold_doc_ids is None:
            gold_doc_ids = [row["gold_doc_id"]]

        # Stage 1
        n_stage1 += 1
        retrieved = _get_retrieved_docs(
            retrieve_cache=retrieve_cache,
            retriever=retriever,
            question=q,
            top_k=top_k,
            max_top_k_for_cache=top_k,
        )

        _, A_i = retriever_fail(retrieved, gold_doc_ids, tau_1)
        A_list.append(A_i)

        if A_i == 1:
            fail_cases["retriever"].append(qid)
            continue

        # Stage 2
        n_stage2 += 1
        reranked = _get_reranked_docs(
            rerank_cache=rerank_cache,
            reranker=reranker,
            question=q,
            retrieved_docs=retrieved,
            top_k=top_k,
            top_K=top_K,
        )

        _, B_i = reranker_fail(reranked, gold_doc_ids, tau_2)
        B_list.append(B_i)

        if B_i == 1:
            fail_cases["reranker"].append(qid)
            continue

        stage3_rows.append({
            "qid": qid,
            "question": q,
            "gold_answer": gold_answer,
            "reranked_docs": reranked,
        })

    return {
        "stage3_rows": stage3_rows,
        "A_list": A_list,
        "B_list": B_list,
        "fail_cases": fail_cases,
        "n_stage1": n_stage1,
        "n_stage2": n_stage2,
    }

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
    stage12_out = _collect_stage3_rows(
        calib_data=calib_data,
        retriever=retriever,
        reranker=reranker,
        top_k=top_k,
        top_K=top_K,
        tau_1=tau_1,
        tau_2=tau_2,
        retrieve_cache=retrieve_cache,
        rerank_cache=rerank_cache,
    )

    stage3_rows = stage12_out["stage3_rows"]
    A_list = stage12_out["A_list"]
    B_list = stage12_out["B_list"]
    fail_cases = stage12_out["fail_cases"]
    n_stage1 = stage12_out["n_stage1"]
    n_stage2 = stage12_out["n_stage2"]

    n_stage3 = len(stage3_rows)

    # 先把 cache miss 的 generator request 一次送出去
    _batch_fill_gen_cache(
        rows=stage3_rows,
        generator=generator,
        N_rag=N_rag,
        lambda_g=lambda_g,
        lambda_s=lambda_s,
        gen_cache=gen_cache,
    )

    # 再逐題計算 generator fail
    C_list = []
    for row in stage3_rows:
        qid = row["qid"]
        q = row["question"]
        gold_answer = row["gold_answer"]
        reranked = row["reranked_docs"]

        contexts = reranked[:N_rag]
        doc_ids = tuple(d.metadata["doc_id"] for d in contexts)
        gen_key = (q, doc_ids, lambda_g, lambda_s)

        generation_set = gen_cache[gen_key]
        _, C_i = generator_fail(generation_set, gold_answer, tau_3=tau_3)
        C_list.append(C_i)

        if C_i == 1:
            fail_cases["generator"].append(qid)

    # 條件風險估計
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
        "n_stage1": n_stage1,
        "n_stage2": n_stage2,
        "n_stage3": n_stage3,
    }

def evaluate_stage12_stats_only(
    calib_data, retriever, reranker, top_k, top_K, tau_1, tau_2,
    retrieve_cache, rerank_cache
):
    A_list, B_list = [], []

    for row in calib_data:
        q = row["question"]

        gold_doc_ids = row.get("gold_doc_ids")
        if gold_doc_ids is None:
            gold_doc_ids = [row["gold_doc_id"]]

        retrieved = _get_retrieved_docs(
            retrieve_cache=retrieve_cache,
            retriever=retriever,
            question=q,
            top_k=top_k,
            max_top_k_for_cache=top_k,
        )

        _, A_i = retriever_fail(retrieved, gold_doc_ids, tau_1)
        A_list.append(A_i)
        if A_i == 1:
            continue

        reranked = _get_reranked_docs(
            rerank_cache=rerank_cache,
            reranker=reranker,
            question=q,
            retrieved_docs=retrieved,
            top_k=top_k,
            top_K=top_K,
        )

        _, B_i = reranker_fail(reranked, gold_doc_ids, tau_2)
        B_list.append(B_i)

    fwer_1 = sum(A_list) / max(len(A_list), 1)
    fwer_2 = sum(B_list) / max(len(B_list), 1)

    return {
        "top_k": top_k,
        "top_K": top_K,
        "FWER_1": fwer_1,
        "FWER_2": fwer_2,
        "num_fail_1": sum(A_list),
        "num_fail_2": sum(B_list),
        "n1": len(A_list),
        "n2": len(B_list),
    }


def evaluate_stage12(
    calib_data, retriever, reranker, top_k, top_K, tau_1, tau_2,
    retrieve_cache, rerank_cache
):
    A_list, B_list = [], []
    passed_rows = []

    for row in calib_data:
        qid = row["qid"]
        q = row["question"]
        gold_answer = row["gold_answer"]

        gold_doc_ids = row.get("gold_doc_ids")
        if gold_doc_ids is None:
            gold_doc_ids = [row["gold_doc_id"]]

        retrieved = _get_retrieved_docs(
            retrieve_cache=retrieve_cache,
            retriever=retriever,
            question=q,
            top_k=top_k,
            max_top_k_for_cache=top_k,
        )

        _, A_i = retriever_fail(retrieved, gold_doc_ids, tau_1)
        A_list.append(A_i)
        if A_i == 1:
            continue

        reranked = _get_reranked_docs(
            rerank_cache=rerank_cache,
            reranker=reranker,
            question=q,
            retrieved_docs=retrieved,
            top_k=top_k,
            top_K=top_K,
        )

        _, B_i = reranker_fail(reranked, gold_doc_ids, tau_2)
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
        "num_fail_1": sum(A_list),
        "num_fail_2": sum(B_list),
        "n1": len(A_list),
        "n2": len(B_list),
    }


def evaluate_stage3(
    passed_rows, generator, top_k, top_K, N_rag, lambda_g, lambda_s, tau_3, gen_cache
):
    # 先把 cache miss 的 generator request 一次送出去
    _batch_fill_gen_cache(
        rows=passed_rows,
        generator=generator,
        N_rag=N_rag,
        lambda_g=lambda_g,
        lambda_s=lambda_s,
        gen_cache=gen_cache,
    )

    C_list = []

    for row in passed_rows:
        q = row["question"]
        gold_answer = row["gold_answer"]
        reranked = row["reranked_docs"]

        contexts = reranked[:N_rag]
        doc_ids = tuple(d.metadata["doc_id"] for d in contexts)
        gen_key = (q, doc_ids, lambda_g, lambda_s)

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
        "num_fail_3": sum(C_list),
        "n3": len(C_list),
    }


def evaluate_fixed_params_on_dataset(
    rows,
    retriever,
    reranker,
    generator,
    params,
    tau_1,
    tau_2,
    tau_3,
):
    retrieve_cache = {}
    rerank_cache = {}
    gen_cache = {}

    return evaluate_one_setting(
        calib_data=rows,
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        top_k=params["top_k"],
        top_K=params["top_K"],
        N_rag=params["N_rag"],
        lambda_g=params["lambda_g"],
        lambda_s=params["lambda_s"],
        tau_1=tau_1,
        tau_2=tau_2,
        tau_3=tau_3,
        retrieve_cache=retrieve_cache,
        rerank_cache=rerank_cache,
        gen_cache=gen_cache
    )


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
    # 2. effective modes (auto relax on tiny calibration)
    # -------------------------
    use_data_split, use_stage12_tcrcs, use_stage3_certified_bound = get_effective_modes(
        calib_data, risk_cfg
    )

    # -------------------------
    # 3. data split
    # -------------------------
    if use_data_split:
        calib_stage12, calib_stage3 = split_rows(
            calib_data,
            ratio=risk_cfg.stage12_ratio,
            seed=risk_cfg.random_seed
        )
    else:
        calib_stage12 = list(calib_data)
        calib_stage3 = list(calib_data)

    if use_stage12_tcrcs:
        calib_stage12_I1, calib_stage12_I2 = split_rows(
            calib_stage12,
            ratio=risk_cfg.stage12_i1_ratio,
            seed=risk_cfg.random_seed + 1
        )
    else:
        calib_stage12_I1 = list(calib_stage12)
        calib_stage12_I2 = list(calib_stage12)

    # -------------------------
    # 3. auto threshold candidates
    # -------------------------
    cand = build_threshold_candidates(
        calib_data=calib_stage12,
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
    gen_cache = {}

    raw_results = []
    feasible_results = []
    stage12_candidates = []

    # -------------------------
    # 4. search stage 1 + 2
    # -------------------------
    print("Start stage 1+2 search...")
    for top_k in top_k_candidates:
        rerank_cache = {}
        print(f"[stage12] top_k={top_k}, candidates={top_K_candidates_map[top_k]}")
        for top_K in top_K_candidates_map[top_k]:
            print(f"  evaluating top_k={top_k}, top_K={top_K}")
            MIN_TOP_K = search_cfg.min_top_K

            if top_K > top_k:
                continue
            if top_K < MIN_TOP_K:
                continue

            # 4.1 stage1/2 search on split data (finite-sample style)
            if use_stage12_tcrcs:
                s12_I1 = evaluate_stage12_stats_only(
                    calib_data=calib_stage12_I1,
                    retriever=retriever,
                    reranker=reranker,
                    top_k=top_k,
                    top_K=top_K,
                    tau_1=risk_cfg.tau_1,
                    tau_2=risk_cfg.tau_2,
                    retrieve_cache=retrieve_cache,
                    rerank_cache=rerank_cache,
                )
                s12_I2 = evaluate_stage12_stats_only(
                    calib_data=calib_stage12_I2,
                    retriever=retriever,
                    reranker=reranker,
                    top_k=top_k,
                    top_K=top_K,
                    tau_1=risk_cfg.tau_1,
                    tau_2=risk_cfg.tau_2,
                    retrieve_cache=retrieve_cache,
                    rerank_cache=rerank_cache,
                )
                s12_stage3_stats = evaluate_stage12_stats_only(
                    calib_data=calib_stage3,
                    retriever=retriever,
                    reranker=reranker,
                    top_k=top_k,
                    top_K=top_K,
                    tau_1=risk_cfg.tau_1,
                    tau_2=risk_cfg.tau_2,
                    retrieve_cache=retrieve_cache,
                    rerank_cache=rerank_cache,
                )

                if risk_cfg.allocation_mode == "weighted" and risk_cfg.enforce_module_budgets:
                    if not finite_sample_pass(s12_I1["num_fail_1"], s12_I1["n1"], alpha_1):
                        continue
                    if not finite_sample_pass(s12_I2["num_fail_2"], s12_I2["n2"], alpha_2):
                        continue

                s12 = {
                    "top_k": top_k,
                    "top_K": top_K,
                    "FWER_1": s12_stage3_stats["FWER_1"],
                    "FWER_2": s12_stage3_stats["FWER_2"],
                    "FWER_1_I1": s12_I1["FWER_1"],
                    "FWER_2_I2": s12_I2["FWER_2"],
                    "num_fail_1_I1": s12_I1["num_fail_1"],
                    "num_fail_2_I2": s12_I2["num_fail_2"],
                    "n1_I1": s12_I1["n1"],
                    "n2_I2": s12_I2["n2"],
                }
            else:
                s12 = evaluate_stage12_stats_only(
                    calib_data=calib_stage12,
                    retriever=retriever,
                    reranker=reranker,
                    top_k=top_k,
                    top_K=top_K,
                    tau_1=risk_cfg.tau_1,
                    tau_2=risk_cfg.tau_2,
                    retrieve_cache=retrieve_cache,
                    rerank_cache=rerank_cache,
                )

                if risk_cfg.allocation_mode == "weighted" and risk_cfg.enforce_module_budgets:
                    if s12["FWER_1"] > alpha_1 + risk_cfg.safety_margin:
                        continue
                    if s12["FWER_2"] > alpha_2 + risk_cfg.safety_margin:
                        continue

            stage12_candidates.append(s12)

    # prune
    stage12_candidates.sort(
        key=lambda x: (
            x["FWER_1"] + x["FWER_2"],
            x["top_k"],
            x["top_K"]
        )
    )
    stage12_candidates = stage12_candidates[:search_cfg.max_stage12_candidates]

    # 只對保留下來的少數候選，重建真正要送進 stage3 的 passed_rows
    rebuilt_stage12_candidates = []
    for s12 in stage12_candidates:
        rebuilt = evaluate_stage12(
            calib_data=calib_stage3 if use_stage12_tcrcs else calib_stage12,
            retriever=retriever,
            reranker=reranker,
            top_k=s12["top_k"],
            top_K=s12["top_K"],
            tau_1=risk_cfg.tau_1,
            tau_2=risk_cfg.tau_2,
            retrieve_cache=retrieve_cache,
            rerank_cache=rerank_cache,
        )

        merged = dict(s12)
        merged["passed_rows"] = rebuilt["passed_rows"]
        rebuilt_stage12_candidates.append(merged)

    stage12_candidates = rebuilt_stage12_candidates

    print(f"stage12 candidates kept: {len(stage12_candidates)}")

    # -------------------------
    # 5. search stage 3
    # -------------------------
    print("Start stage 3 search...")
    for s12 in stage12_candidates:
        top_k = s12["top_k"]
        top_K = s12["top_K"]

        print(f"[stage3] top_k={top_k}, top_K={top_K}, passed_rows={len(s12['passed_rows'])}")

        for N_rag in N_rag_candidates_map[(top_k, top_K)]:
            if (not search_cfg.fix_n_rag_to_top_K) and N_rag < search_cfg.min_N_rag:
                continue

            if search_cfg.fix_n_rag_to_top_K:
                print(f"  N_rag fixed to top_K = {N_rag}")
            else:
                print(f"  N_rag={N_rag}")

            for lambda_g in lambda_g_candidates:
                for lambda_s in lambda_s_candidates:
                    print(f"    lambda_g={lambda_g}, lambda_s={lambda_s}")

                    local_gen_cache = {}

                    s3 = evaluate_stage3(
                        passed_rows=s12["passed_rows"],
                        generator=generator,
                        top_k=top_k,
                        top_K=top_K,
                        N_rag=N_rag,
                        lambda_g=lambda_g,
                        lambda_s=lambda_s,
                        tau_3=risk_cfg.tau_3,
                        gen_cache=local_gen_cache,
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

                    # 保留 split finite-sample 的診斷資訊
                    if "FWER_1_I1" in s12:
                        result["FWER_1_I1"] = s12["FWER_1_I1"]
                        result["FWER_2_I2"] = s12["FWER_2_I2"]
                        result["num_fail_1_I1"] = s12["num_fail_1_I1"]
                        result["num_fail_2_I2"] = s12["num_fail_2_I2"]
                        result["n1_I1"] = s12["n1_I1"]
                        result["n2_I2"] = s12["n2_I2"]

                    # certified generator bound
                    if use_stage3_certified_bound and risk_cfg.allocation_mode == "weighted":
                        alpha_3_hat = hb_upper_bound(
                            r_hat=fwer_3,
                            n=s3["n3"],
                            delta=risk_cfg.delta_3
                        )
                        result["alpha_3_hat"] = alpha_3_hat
                        result["P(E)_cert"] = allocation_total(alpha_1, alpha_2, alpha_3_hat)

                        if risk_cfg.enforce_module_budgets:
                            feasible = (
                                alpha_3_hat <= alpha_3 + risk_cfg.safety_margin
                                and result["P(E)_cert"] <= risk_cfg.alpha_total + risk_cfg.safety_margin
                            )
                        else:
                            feasible = result["P(E)_cert"] <= risk_cfg.alpha_total + risk_cfg.safety_margin
                    else:
                        if risk_cfg.allocation_mode == "weighted":
                            if risk_cfg.enforce_module_budgets:
                                feasible = (
                                    fwer_1 <= alpha_1 + risk_cfg.safety_margin
                                    and fwer_2 <= alpha_2 + risk_cfg.safety_margin
                                    and fwer_3 <= alpha_3 + risk_cfg.safety_margin
                                    and pe_hat <= risk_cfg.alpha_total + risk_cfg.safety_margin
                                )
                            else:
                                feasible = pe_hat <= risk_cfg.alpha_total + risk_cfg.safety_margin
                        else:
                            feasible = pe_hat <= risk_cfg.alpha_total + risk_cfg.safety_margin

                    raw_results.append(result)

                    if feasible:
                        feasible_results.append(result)

    if not feasible_results:
        raw_results.sort(
            key=lambda x: (
                x.get("P(E)_cert", x["P(E)_hat"]),
                x["time_proxy"]
            )
        )
        return None, raw_results, []

    feasible_results.sort(
        key=lambda x: (
            x["time_proxy"],
            -x["top_K"],
            -x["N_rag"],
            x.get("P(E)_cert", x["P(E)_hat"]),
        )
    )
    best = feasible_results[0]
    return best, raw_results, feasible_results