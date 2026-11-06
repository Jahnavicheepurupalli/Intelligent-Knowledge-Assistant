import os
import json
import time
from typing import List, Dict, Any
from database.chroma_manager import ChromaManager
from rag.retriever import retrieve_relevant_chunks
from rag.generator import AnswerGenerator
from config import BASE_DIR

class RAGEvaluator:
    """
    RAG Evaluation module. Executes a batch of test queries, analyzes the outputs,
    and returns metrics regarding refusal, retrieve latency, and source presence.
    """
    def __init__(self, queries_path: str = None):
        if queries_path is None:
            queries_path = os.path.join(BASE_DIR, "evaluation", "test_queries.json")
        self.queries_path = queries_path
        self.test_cases = self._load_test_cases()

    def _load_test_cases(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.queries_path):
            with open(self.queries_path, 'r') as f:
                return json.load(f)
        return []

    def run_evaluation(self, chroma_manager: ChromaManager, top_k: int, threshold: float) -> Dict[str, Any]:
        """
        Executes evaluation on loaded test cases and returns summary metrics.
        """
        if not self.test_cases:
            return {"error": "No test cases found in test_queries.json"}

        generator = AnswerGenerator()
        results = []

        total_questions = len(self.test_cases)
        correctly_refused = 0
        supported_answered = 0
        failed_refusal = 0  # hallucinated on unsupported
        failed_answer = 0   # refused supported

        total_latency = 0.0

        refusal_phrase = "I don't have enough information in the provided documents to answer that."

        for case in self.test_cases:
            q_id = case["id"]
            question = case["question"]
            category = case["expected_category"]

            start_time = time.time()

            # Step 1: Retrieve
            retrieved = retrieve_relevant_chunks(
                query=question,
                chroma_manager=chroma_manager,
                top_k=top_k,
                similarity_threshold=threshold
            )

            # Step 2: Generate
            response = generator.generate_answer(question, retrieved)
            latency = time.time() - start_time
            total_latency += latency

            answer = response["answer"]
            sources = response["sources"]

            # Evaluator Classification Logic
            refused = refusal_phrase.lower() in answer.lower()

            status = "Unknown"
            if category == "unsupported":
                if refused:
                    correctly_refused += 1
                    status = "Correct Refusal"
                else:
                    failed_refusal += 1
                    status = "Hallucinated/Invalid Answer"
            else:  # supported
                if not refused:
                    supported_answered += 1
                    status = "Answered Successfully"
                else:
                    failed_answer += 1
                    status = "Incorrect Refusal (Under-retrieval)"

            results.append({
                "id": q_id,
                "question": question,
                "expected_category": category,
                "actual_answer": answer,
                "sources_retrieved": [s["source"] for s in sources],
                "status": status,
                "latency_seconds": round(latency, 2)
            })

        avg_latency = round(total_latency / total_questions, 2) if total_questions > 0 else 0.0

        summary = {
            "total_questions": total_questions,
            "correct_refusals": correctly_refused,
            "supported_answered": supported_answered,
            "unsupported_failed_refusal_count": failed_refusal,
            "supported_failed_answered_count": failed_answer,
            "average_latency_seconds": avg_latency,
            "results": results
        }

        return summary
