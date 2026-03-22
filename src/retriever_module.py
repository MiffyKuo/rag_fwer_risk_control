from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

class RetrieverModule:
    def __init__(self, embedding_model: str):
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
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