import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


class RetrieverModule:
    def __init__(self, embedding_model: str, device: str = "cpu"):
        self.device = device

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vectorstore = None

    def build_index(self, corpus):
        docs = [
            Document(
                page_content=row["text"],
                metadata={"doc_id": row["doc_id"]}
            )
            for row in corpus
        ]
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)

    def retrieve(self, question: str, top_k: int):
        return self.vectorstore.similarity_search(question, k=top_k)