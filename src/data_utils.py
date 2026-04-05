import json


def normalize_qa_row(row):
    row = dict(row)

    # 統一 gold doc 欄位
    if "gold_doc_ids" not in row:
        if "gold_doc_id" in row:
            row["gold_doc_ids"] = [row["gold_doc_id"]]
        elif "primary_gold_doc_id" in row:
            row["gold_doc_ids"] = [row["primary_gold_doc_id"]]
        else:
            row["gold_doc_ids"] = []

    if "gold_doc_id" not in row and row["gold_doc_ids"]:
        row["gold_doc_id"] = row["gold_doc_ids"][0]

    if "primary_gold_doc_id" not in row and row["gold_doc_ids"]:
        row["primary_gold_doc_id"] = row["gold_doc_ids"][0]

    return row


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows.append(normalize_qa_row(row))
    return rows