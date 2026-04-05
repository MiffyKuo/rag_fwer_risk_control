from rouge_score import rouge_scorer
import math
import re

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def retriever_fail(retrieved_docs, gold_doc_ids, tau_1=0.0):
    """
    Multi-gold version.
    loss_1 = 1 - (retrieved relevant docs / total relevant docs)
    """
    retrieved_ids = {d.metadata["doc_id"] for d in retrieved_docs}
    gold_set = set(gold_doc_ids)

    if len(gold_set) == 0:
        loss_1 = 1.0
    else:
        covered = len(retrieved_ids & gold_set)
        loss_1 = 1.0 - covered / len(gold_set)

    A_i = int(loss_1 > tau_1)
    return loss_1, A_i


def reranker_fail(reranked_docs, gold_doc_ids, tau_2=0.0):
    """
    Multi-gold simplified nDCG-style loss.
    loss_2 = 1 - nDCG_mod
    """
    doc_ids = [d.metadata["doc_id"] for d in reranked_docs]
    gold_set = set(gold_doc_ids)

    if len(gold_set) == 0:
        loss_2 = 1.0
        B_i = int(loss_2 > tau_2)
        return loss_2, B_i

    dcg = 0.0
    for rank, doc_id in enumerate(doc_ids, start=1):
        if doc_id in gold_set:
            dcg += 1.0 / math.log2(rank + 1.0)

    ideal_hits = min(len(gold_set), len(doc_ids))
    idcg = 0.0
    for rank in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(rank + 1.0)

    if idcg == 0:
        loss_2 = 1.0
    else:
        ndcg_mod = dcg / idcg
        loss_2 = 1.0 - ndcg_mod

    B_i = int(loss_2 > tau_2)
    return loss_2, B_i


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def exact_or_contained_match(pred: str, gold: str) -> bool:
    pred_n = normalize_text(pred)
    gold_n = normalize_text(gold)
    return gold_n in pred_n or pred_n in gold_n


def generator_fail(generation_set, gold_answer, tau_3=0.4):
    """
    用連續風險：
        risk = 1 - max_cand ROUGE-L(cand, gold)
    若有 exact/contained match，直接視為 risk=0

    這樣 tau_3 才有意義，不會像原本一樣只要沒精確命中就一律 fail。
    """
    if len(generation_set) == 0:
        return 1.0, 1

    gold = gold_answer.strip()

    # 先給 exact/contained match 一個捷徑
    for ans in generation_set:
        if exact_or_contained_match(ans, gold):
            return 0.0, 0

    best_score = 0.0
    for ans in generation_set:
        rougeL = scorer.score(gold, ans)["rougeL"].fmeasure
        if rougeL > best_score:
            best_score = rougeL

    risk = 1.0 - best_score
    fail = int(risk > tau_3)
    return risk, fail