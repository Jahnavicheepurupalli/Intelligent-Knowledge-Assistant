from typing import List, Tuple, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from config import GROQ_API_KEY, DEFAULT_LLM_MODEL
from rag.prompts import RAG_SYSTEM_PROMPT

class AnswerGenerator:
    """
    RAG Answer Generator utilizing Llama 3.1 LLM via Groq API and strict context templates.
    """
    def __init__(self, model_name: str = DEFAULT_LLM_MODEL, api_key: str = GROQ_API_KEY):
        if not api_key:
            raise ValueError("Groq API Key is missing. Please set the GROQ_API_KEY environment variable or in the UI.")

        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=0.0  # Force zero temperature for deterministic, factual responses
        )
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=RAG_SYSTEM_PROMPT
        )

    def generate_answer(self, question: str, retrieved_chunks: List[Tuple[Document, float]]) -> Dict[str, Any]:
        """
        Generates a factual answer based on retrieved documents and returns response along with source information.
        """
        # If no chunks were retrieved (below threshold or empty database)
        if not retrieved_chunks:
            return {
                "answer": "I don't have enough information in the provided documents to answer that.",
                "sources": []
            }

        # Build context block
        context_str = ""
        sources = []
        for i, (doc, score) in enumerate(retrieved_chunks):
            source_name = doc.metadata.get("source", "Unknown Document")
            page_num = doc.metadata.get("page", "N/A")
            context_str += f"[Context {i+1}] (Source: {source_name}, Page: {page_num})\n{doc.page_content}\n\n"

            sources.append({
                "source": source_name,
                "page": page_num,
                "text": doc.page_content,
                "score": round(score, 4)
            })

        # Run formatting prompt and LLM invocation
        prompt = self.prompt_template.format(context=context_str, question=question)

        try:
            response = self.llm.invoke(prompt)
            answer = response.content.strip()
        except Exception as e:
            return {
                "answer": f"Error communicating with Groq LLM API: {str(e)}",
                "sources": sources
            }

        return {
            "answer": answer,
            "sources": sources
        }
