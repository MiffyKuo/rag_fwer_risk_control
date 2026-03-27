from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# 正確文件沒撈到就失敗
def retriever_fail(retrieved_docs, gold_doc_id, tau_1=0.0):
    doc_ids = [d.metadata["doc_id"] for d in retrieved_docs]
    loss_1 = 0.0 if gold_doc_id in doc_ids else 1.0
    A_i = int(loss_1 > tau_1)
    return loss_1, A_i

# 正確文件沒留在 top-K 就失敗
def reranker_fail(reranked_docs, gold_doc_id, tau_2=0.0):
    doc_ids = [d.metadata["doc_id"] for d in reranked_docs]
    loss_2 = 0.0 if gold_doc_id in doc_ids else 1.0
    B_i = int(loss_2 > tau_2)
    return loss_2, B_i

# 答案跟標準答案差太多就失敗

# 標準答案有沒有被回答出來
import re

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def generator_fail(generation_set, gold_answer, tau_3=0.0):
    gold = normalize_text(gold_answer)

    if len(generation_set) == 0:
        return 1.0, 1

    hit = 0
    for ans in generation_set:
        pred = normalize_text(ans)
        if gold in pred:
            hit = 1
            break

    risk = 0.0 if hit else 1.0
    fail = int(risk > tau_3)
    return risk, fail

# # ROUGE -> 句子像不像
# def generator_fail(pred_answer, gold_answer, tau_3=0.4):
#     rougeL = scorer.score(gold_answer, pred_answer)["rougeL"].fmeasure
#     risk = 1 - rougeL
#     C_i = int(risk > tau_3)
#     return risk, C_i