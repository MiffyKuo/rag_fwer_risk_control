import json
import hashlib
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from difflib import SequenceMatcher

import aiohttp
from langchain_openai import ChatOpenAI


class GeneratorModule:
    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str,
        temperature: float = 0.0,
        cache_path: str = "data/generator_cache.json",
        max_concurrent: int = 64,
        request_timeout: int = 600,
        max_tokens: int = 256,
    ):
        self.model_name = model_name
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_concurrent = max_concurrent
        self.request_timeout = request_timeout
        self.max_tokens = max_tokens

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=api_base
        )

        self.api_url = self._build_api_url(self.api_base)

        self.cache_path = Path(cache_path)
        self.cache = self._load_cache()

    def _build_api_url(self, api_base: str) -> str:
        if api_base.endswith("/chat/completions"):
            return api_base
        if api_base.endswith("/v1"):
            return f"{api_base}/chat/completions"
        return f"{api_base}/v1/chat/completions"

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

Question: {question}

Answer:
""".strip()
        return prompt

    def _cache_key(self, prompt: str, lambda_g: int, lambda_s: float):
        raw = f"{self.model_name}|{self.temperature}|{lambda_g}|{lambda_s}|{prompt}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def answer_similarity(self, text1: str, text2: str) -> float:
        return SequenceMatcher(None, text1, text2).ratio()

    async def _send_request_async(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        request_id: str,
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with session.post(self.api_url, json=payload, headers=headers) as response:
                if response.status != 200:
                    text = await response.text()
                    return {
                        "request_id": request_id,
                        "success": False,
                        "error": f"HTTP {response.status}: {text}"
                    }

                result = await response.json()
                content = result["choices"][0]["message"]["content"].strip()

                return {
                    "request_id": request_id,
                    "success": True,
                    "content": content
                }

        except Exception as e:
            return {
                "request_id": request_id,
                "success": False,
                "error": str(e)
            }

    async def _batch_send_requests_async(
        self,
        prompts_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def send_with_limit(session, data):
            async with semaphore:
                return await self._send_request_async(
                    session=session,
                    prompt=data["prompt"],
                    request_id=data["id"],
                )

        connector = aiohttp.TCPConnector(limit=self.max_concurrent + 20)
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [send_with_limit(session, data) for data in prompts_data]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        normalized = []
        for r in results:
            if isinstance(r, Exception):
                normalized.append({
                    "request_id": "unknown",
                    "success": False,
                    "error": str(r)
                })
            else:
                normalized.append(r)
        return normalized

    def _run_async(self, coro):
        try:
            loop = asyncio.get_running_loop()
            running = loop.is_running()
        except RuntimeError:
            running = False

        if not running:
            return asyncio.run(coro)

        # 保險寫法：若未來在 notebook / 互動環境使用
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()

    def batch_generate_answers(
        self,
        requests_data: List[Dict[str, Any]],
        lambda_g: int = 1,
        lambda_s: float = 0.8,
        max_retry: int = 6,
    ) -> Dict[Any, List[str]]:
        """
        requests_data 格式:
        [
            {
                "user_key": ...,
                "prompt": ...
            },
            ...
        ]

        回傳:
            { user_key: [ans1, ans2, ...] }
        """
        if not requests_data:
            return {}

        output: Dict[Any, List[str]] = {}
        pending_items = []
        tries_map = {}

        # 先檢查 persistent cache
        for item in requests_data:
            user_key = item["user_key"]
            prompt = item["prompt"]
            cache_key = self._cache_key(prompt, lambda_g, lambda_s)

            if cache_key in self.cache:
                output[user_key] = self.cache[cache_key]
            else:
                pending_items.append({
                    "user_key": user_key,
                    "prompt": prompt,
                    "cache_key": cache_key,
                })
                output[user_key] = []
                tries_map[user_key] = 0

        # 逐輪批次送 request
        active_items = list(pending_items)

        while active_items:
            batch_payloads = []
            for idx, item in enumerate(active_items):
                user_key = item["user_key"]
                request_id = str(idx)

                batch_payloads.append({
                    "id": request_id,
                    "prompt": item["prompt"],
                    "user_key": user_key,
                })

            raw_results = self._run_async(self._batch_send_requests_async(batch_payloads))

            id_to_user_key = {
                str(i): batch_payloads[i]["user_key"]
                for i in range(len(batch_payloads))
            }

            for result in raw_results:
                request_id = result.get("request_id", "unknown")
                user_key = id_to_user_key.get(request_id)

                if user_key is None:
                    continue

                tries_map[user_key] += 1

                if not result.get("success", False):
                    continue

                candidate = result.get("content", "").strip()
                if not candidate:
                    continue

                too_similar = any(
                    self.answer_similarity(candidate, old_ans) > lambda_s
                    for old_ans in output[user_key]
                )

                if not too_similar:
                    output[user_key].append(candidate)

            next_active_items = []
            for item in active_items:
                user_key = item["user_key"]
                if len(output[user_key]) < lambda_g and tries_map[user_key] < max_retry:
                    next_active_items.append(item)
            active_items = next_active_items

        # 若還是空，就補預設答案，並寫入 persistent cache
        for item in pending_items:
            user_key = item["user_key"]
            cache_key = item["cache_key"]

            if len(output[user_key]) == 0:
                output[user_key] = ["I do not know."]

            self.cache[cache_key] = output[user_key]

        return output

    def generate_answer(self, question: str, contexts: List) -> str:
        prompt = self.build_prompt(question, contexts)
        result = self.batch_generate_answers(
            requests_data=[{"user_key": "single", "prompt": prompt}],
            lambda_g=1,
            lambda_s=1.0,
            max_retry=2,
        )
        answers = result["single"]
        return answers[0].strip() if answers else "I do not know."

    def generate_answers(
        self,
        question: str,
        contexts: List,
        lambda_g: int = 2,
        lambda_s: float = 0.8,
        max_retry: int = 6
    ) -> List[str]:
        prompt = self.build_prompt(question, contexts)
        result = self.batch_generate_answers(
            requests_data=[{"user_key": "single", "prompt": prompt}],
            lambda_g=lambda_g,
            lambda_s=lambda_s,
            max_retry=max_retry,
        )
        return result["single"]

    def save_cache(self):
        self._save_cache()