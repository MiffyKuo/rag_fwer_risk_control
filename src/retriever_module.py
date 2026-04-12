import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document


class RetrieverModule:
    def __init__(self, embedding_model: str, device: str = "cpu", batch_size: int = 8):
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(embedding_model, device=device)
        self.index = None
        self.docs = []

    def build_index(self, corpus):
        self.docs = [
            Document(
                page_content=row["text"],
                metadata={"doc_id": row["doc_id"]}
            )
            for row in corpus
        ]

        all_vecs = []

        for i in range(0, len(self.docs), self.batch_size):
            batch_docs = self.docs[i:i + self.batch_size]
            batch_texts = [d.page_content for d in batch_docs]

            vecs = self.model.encode(
                batch_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            ).astype("float32")

            all_vecs.append(vecs)

        embeddings = np.vstack(all_vecs).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def retrieve(self, question: str, top_k: int):
        qvec = self.model.encode(
            [question],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        scores, indices = self.index.search(qvec, top_k)
        return [self.docs[i] for i in indices[0] if i != -1]