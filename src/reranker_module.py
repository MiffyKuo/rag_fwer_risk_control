from sentence_transformers import CrossEncoder
from copy import deepcopy


class SimpleReranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def rerank(self, question: str, docs: list, top_K: int, batch_size: int = 8):
        pairs = [(question, doc.page_content) for doc in docs]

        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False
        )

        scored_docs = []
        for score, doc in zip(scores, docs):
            new_doc = deepcopy(doc)
            new_doc.metadata["rerank_score"] = float(score)
            scored_docs.append(new_doc)

        scored_docs.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)
        return scored_docs[:top_K]
    