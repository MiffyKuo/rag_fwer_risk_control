import json
import hashlib
from pathlib import Path
from typing import List
from langchain_openai import ChatOpenAI
from difflib import SequenceMatcher

class GeneratorModule:
    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        cache_path: str = "data/generator_cache.json"
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.cache_path = Path(cache_path)
        self.cache = self._load_cache()

    def _load_cache(self):
        if self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def build_prompt(self, question: str, contexts: List) -> str:
        context_texts = []
        for i, doc in enumerate(contexts, 1):
            text = getattr(doc, "page_content", str(doc))
            context_texts.append(f"[Document {i}]\n{text}")

        joined_context = "\n\n".join(context_texts)

        prompt = f"""
You are a helpful question-answering assistant.
Answer the question only based on the provided documents.
If the answer is not supported by the documents, say "I do not know".

Documents:
{joined_context}

Question:
{question}

Answer:
""".strip()
        return prompt

    def _cache_key(self, prompt: str, lambda_g: int, lambda_s: float):
        raw = f"{self.model_name}|{self.temperature}|{lambda_g}|{lambda_s}|{prompt}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def generate_answer(self, question: str, contexts: List) -> str:
        prompt = self.build_prompt(question, contexts)
        response = self.llm.invoke(prompt)
        return response.content.strip()

    def answer_similarity(self, text1: str, text2: str) -> float:
        return SequenceMatcher(None, text1, text2).ratio()

    def generate_answers(
        self,
        question: str,
        contexts: List,
        lambda_g: int = 2,
        lambda_s: float = 0.8,
        max_retry: int = 6
    ) -> List[str]:
        prompt = self.build_prompt(question, contexts)
        key = self._cache_key(prompt, lambda_g, lambda_s)

        if key in self.cache:
            return self.cache[key]

        answers = []
        tries = 0

        while len(answers) < lambda_g and tries < max_retry:
            response = self.llm.invoke(prompt)
            candidate = response.content.strip()

            if not candidate:
                tries += 1
                continue

            too_similar = any(
                self.answer_similarity(candidate, old_ans) > lambda_s
                for old_ans in answers
            )

            if not too_similar:
                answers.append(candidate)

            tries += 1

        if len(answers) == 0:
            answers = ["I do not know."]

        self.cache[key] = answers
        self._save_cache()
        return answers