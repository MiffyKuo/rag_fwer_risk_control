from langchain_ollama import ChatOllama
from difflib import SequenceMatcher
from typing import List


class GeneratorModule:
    def __init__(self, model_name: str = "llama3.1:8b", temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature
        )

    def build_prompt(self, question: str, contexts: List) -> str:
        """
        Build a QA prompt from retrieved/reranked contexts.
        """
        context_texts = []
        for i, doc in enumerate(contexts, 1):
            text = getattr(doc, "page_content", str(doc))
            context_texts.append(f"[Document {i}]\n{text}")

        joined_context = "\n\n".join(context_texts)

        prompt = f"""
You are a helpful question-answering assistant.
Answer the question only based on the provided documents.
If the answer is not supported by the documents, say you do not know.

Documents:
{joined_context}

Question:
{question}

Answer:
""".strip()

        return prompt

    def generate_answer(self, question: str, contexts: List) -> str:
        """
        Generate a single answer.
        This keeps backward compatibility with your original pipeline.
        """
        prompt = self.build_prompt(question, contexts)
        response = self.llm.invoke(prompt)
        return response.content.strip()

    def answer_similarity(self, text1: str, text2: str) -> float:
        """
        Simple string-level similarity.
        Used to avoid collecting near-duplicate answers.
        """
        return SequenceMatcher(None, text1, text2).ratio()

    def generate_answers(
        self,
        question: str,
        contexts: List,
        lambda_g: int = 3,
        lambda_s: float = 0.8,
        max_retry: int = 20
    ) -> List[str]:
        """
        Generate a set of candidate answers.

        lambda_g: target number of kept answers
        lambda_s: similarity threshold; if new answer is too similar
                  to existing ones, discard it
        """
        prompt = self.build_prompt(question, contexts)

        answers = []
        tries = 0

        while len(answers) < lambda_g and tries < max_retry:
            response = self.llm.invoke(prompt)
            candidate = response.content.strip()

            if not candidate:
                tries += 1
                continue

            too_similar = False
            for old_ans in answers:
                sim = self.answer_similarity(candidate, old_ans)
                if sim > lambda_s:
                    too_similar = True
                    break

            if not too_similar:
                answers.append(candidate)

            tries += 1

        # fallback: at least one answer
        if len(answers) == 0:
            response = self.llm.invoke(prompt)
            fallback = response.content.strip()
            answers.append(fallback if fallback else "I do not know.")

        return answers