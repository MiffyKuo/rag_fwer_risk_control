from sentence_transformers import CrossEncoder


class SimpleReranker:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, question: str, docs: list, top_K: int, batch_size: int = 1): # batch_size: int = 8
        pairs = [(question, doc.page_content) for doc in docs]

        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False
        )

        scored = list(zip(scores, docs))
        scored.sort(key=lambda x: float(x[0]), reverse=True)

        top_docs = []
        for score, doc in scored[:top_K]:
            doc.metadata["rerank_score"] = float(score)
            top_docs.append(doc)

        return top_docs