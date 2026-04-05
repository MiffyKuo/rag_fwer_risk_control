import re
import json
import os
from pathlib import Path
from datasets import load_dataset
from pprint import pprint

OUTPUT_DIR = Path("data")
TRAIN_SLICE = os.getenv("TRAIN_SLICE", "train[:120]") 
VALID_SLICE = os.getenv("VALID_SLICE", "validation[:60]") 
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "1000")) 


def save_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_nonempty(value):
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        return v if v else None
    if isinstance(value, list):
        for item in value:
            ans = first_nonempty(item)
            if ans:
                return ans
    return None


def get_nested_first_text(obj, possible_keys):
    if not isinstance(obj, dict):
        return None
    for key in possible_keys:
        if key in obj:
            ans = first_nonempty(obj[key])
            if ans:
                return ans
    return None


def extract_answer(row):
    for key in ["answer", "value", "normalized_value", "aliases"]:
        if key in row:
            ans = first_nonempty(row[key])
            if ans:
                return ans

    answer_obj = row.get("answer")
    if isinstance(answer_obj, dict):
        for key in ["value", "normalized_value", "aliases"]:
            ans = get_nested_first_text(answer_obj, [key])
            if ans:
                return ans
    return None


def extract_question(row):
    for key in ["question", "query"]:
        if key in row:
            q = first_nonempty(row[key])
            if q:
                return q
    return None


def extract_qid(row, fallback_idx):
    for key in ["question_id", "qid", "id"]:
        if key in row:
            qid = first_nonempty(row[key])
            if qid:
                return str(qid)
    return f"row_{fallback_idx}"


def extract_all_contexts(row):
    """
    回傳 list of dict:
    [
      {"source": "entity_pages", "title": ..., "context": ...},
      {"source": "search_results", "title": ..., "context": ...},
      ...
    ]
    """
    candidates = []

    entity_pages = row.get("entity_pages")
    if isinstance(entity_pages, dict):
        titles = entity_pages.get("title", [])
        contexts = entity_pages.get("wiki_context", [])
        for i, (t, c) in enumerate(zip(titles, contexts)):
            if c and str(c).strip():
                text = str(c).strip()
                if len(text) > MAX_CONTEXT_CHARS:
                    text = text[:MAX_CONTEXT_CHARS]
                candidates.append({
                    "source": "entity_pages",
                    "title": str(t).strip() if t else "",
                    "context": text,
                    "local_idx": i,
                })

    search_results = row.get("search_results")
    if isinstance(search_results, dict):
        titles = search_results.get("title", [])
        contexts = search_results.get("search_context", [])
        descs = search_results.get("description", [])
        for i, (t, c, d) in enumerate(zip(titles, contexts, descs)):
            text = c if c and str(c).strip() else d
            if text and str(text).strip():
                text = str(text).strip()
                if len(text) > MAX_CONTEXT_CHARS:
                    text = text[:MAX_CONTEXT_CHARS]
                candidates.append({
                    "source": "search_results",
                    "title": str(t).strip() if t else "",
                    "context": text,
                    "local_idx": i,
                })

    # 去重：title + context 相同就視為同一篇
    seen = set()
    deduped = []
    for item in candidates:
        key = (item["title"], item["context"])
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    return deduped


def normalize_text_for_match(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def supports_answer(text, answer):
    text_n = normalize_text_for_match(text)
    ans_n = normalize_text_for_match(answer)
    return ans_n in text_n


def row_to_example(row, idx, prefix):
    question = extract_question(row)
    answer = extract_answer(row)
    qid = extract_qid(row, idx)
    contexts = extract_all_contexts(row)

    if not question:
        return None, "missing_question"
    if not answer:
        return None, "missing_answer"
    if not contexts:
        return None, "missing_context"

    corpus_rows = []

    for j, item in enumerate(contexts):
        source_tag = "ep" if item["source"] == "entity_pages" else "sr"
        doc_id = f"{prefix}_{qid}_{source_tag}_{j}"

        text = f"{item['title']}.\n{item['context']}" if item["title"] else item["context"]

        corpus_rows.append({
            "doc_id": doc_id,
            "text": text,
            "title": item["title"],
            "source": item["source"],
            "qid": f"{prefix}_{qid}",
        })

    # 只把真正包含答案的文件視為 gold
    gold_doc_ids = [
        row["doc_id"]
        for row in corpus_rows
        if supports_answer(row["text"], answer)
    ]

    # 如果完全找不到，就退一步只保留第一篇，避免空集合
    if len(gold_doc_ids) == 0:
        gold_doc_ids = [corpus_rows[0]["doc_id"]]

    qa_row = {
    "qid": f"{prefix}_{qid}",
    "question": question,
    "gold_answer": answer,
    "gold_doc_ids": gold_doc_ids,
    "gold_doc_id": gold_doc_ids[0],          # 向下相容
    "primary_gold_doc_id": gold_doc_ids[0],  # 保留
    }

    return (corpus_rows, qa_row), None


def build_rows(dataset, prefix):
    corpus_rows = []
    qa_rows = []
    used_doc_ids = set()

    stats = {
        "ok": 0,
        "missing_question": 0,
        "missing_answer": 0,
        "missing_context": 0,
    }

    for idx, row in enumerate(dataset):
        converted, err = row_to_example(row, idx, prefix)
        if converted is None:
            stats[err] += 1
            continue

        row_corpus_rows, qa_row = converted

        for corpus_row in row_corpus_rows:
            if corpus_row["doc_id"] not in used_doc_ids:
                used_doc_ids.add(corpus_row["doc_id"])
                corpus_rows.append(corpus_row)

        qa_rows.append(qa_row)
        stats["ok"] += 1

    return corpus_rows, qa_rows, stats


def deduplicate_corpus(rows):
    seen = set()
    out = []
    for row in rows:
        key = (row["doc_id"], row["text"])
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def show_sample_row(ds, name):
    print(f"\n===== SAMPLE FROM {name} =====")
    sample = ds[0]
    pprint(sample)
    print("===== END SAMPLE =====\n")


def main():
    print("Loading TriviaQA train slice:", TRAIN_SLICE)
    train_ds = load_dataset("mandarjoshi/trivia_qa", "rc", split=TRAIN_SLICE)

    print("Loading TriviaQA validation slice:", VALID_SLICE)
    valid_ds = load_dataset("mandarjoshi/trivia_qa", "rc", split=VALID_SLICE)

    # 印一筆樣本，幫助除錯
    show_sample_row(train_ds, "TRAIN")

    print("Converting train slice to calibration rows...")
    train_corpus, calib_rows, train_stats = build_rows(train_ds, prefix="train")

    print("Converting validation slice to test rows...")
    valid_corpus, test_rows, valid_stats = build_rows(valid_ds, prefix="valid")

    corpus_rows = deduplicate_corpus(train_corpus + valid_corpus)

    print("\nTrain conversion stats:", train_stats)
    print("Valid conversion stats:", valid_stats)

    print(f"\nFinal corpus size: {len(corpus_rows)}")
    print(f"Calibration size: {len(calib_rows)}")
    print(f"Test size: {len(test_rows)}")

    save_jsonl(corpus_rows, OUTPUT_DIR / "corpus.jsonl")
    save_jsonl(calib_rows, OUTPUT_DIR / "calib.jsonl")
    save_jsonl(test_rows, OUTPUT_DIR / "test.jsonl")

    print("\nSaved files:")
    print(" - data/corpus.jsonl")
    print(" - data/calib.jsonl")
    print(" - data/test.jsonl")


if __name__ == "__main__":
    main()