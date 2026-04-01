# 拿 calibration 選好的最佳參數，真的去回答新問題
class RiskControlledRAG:
    def __init__(self, retriever, reranker, generator, best_params, fix_n_rag_to_top_K=False):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.best_params = best_params
        self.fix_n_rag_to_top_K = fix_n_rag_to_top_K

    def answer(self, question: str):
        top_k = self.best_params["top_k"]
        top_K = self.best_params["top_K"]
        lambda_g = self.best_params["lambda_g"]
        lambda_s = self.best_params["lambda_s"]

        retrieved = self.retriever.retrieve(question, top_k=top_k)
        reranked = self.reranker.rerank(question, retrieved, top_K=top_K)

        if self.fix_n_rag_to_top_K:
            N_rag = top_K
            contexts = reranked
        else:
            N_rag = self.best_params["N_rag"]
            contexts = reranked[:N_rag]

        answers = self.generator.generate_answers(
            question,
            contexts,
            lambda_g=lambda_g,
            lambda_s=lambda_s
        )

        final_answer = answers[0] if len(answers) > 0 else "I do not know."

        return {
            "question": question,
            "answer": final_answer,
            "candidate_answers": answers,
            "top_k": top_k,
            "top_K": top_K,
            "N_rag": N_rag,
            "lambda_g": lambda_g,
            "lambda_s": lambda_s,
            "contexts": [d.page_content for d in contexts],
        }