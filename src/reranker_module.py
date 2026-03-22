from sentence_transformers import CrossEncoder

class SimpleReranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def rerank(self, question: str, docs: list, top_K: int):
        pairs = [(question, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)

        scored = list(zip(scores, docs))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:top_K]]