from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from config import GROQ_API_KEY, DEFAULT_LLM_MODEL

QUERY_REWRITER_PROMPT = """You are a Query Reformulator. Given a conversation history and a follow-up question, your task is to rewrite the follow-up question into a standalone question that can be searched in a vector database.
The standalone question must contain all context and pronouns resolved from the chat history.

Chat History:
{chat_history}

Follow-up Question: {follow_up}

Rewrite the follow-up question to be complete and self-contained. Return ONLY the rewritten question text. Do not add explanations, conversational filler, or formatting.
Standalone Question:"""

def rewrite_query(follow_up: str, chat_history: List[Dict[str, Any]], api_key: str = GROQ_API_KEY, model_name: str = DEFAULT_LLM_MODEL) -> str:
    """
    Rewrite follow-up questions to resolve conversational references (such as 'it', 'third one')
    using preceding messages context.
    """
    if not chat_history or not api_key:
        return follow_up

    # Format history snippet
    history_str = ""
    for msg in chat_history[-3:]:  # Focus on the last 3 exchanges
        role_label = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role_label}: {msg['content']}\n"

    try:
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=0.0
        )
        prompt = PromptTemplate(
            input_variables=["chat_history", "follow_up"],
            template=QUERY_REWRITER_PROMPT
        ).format(chat_history=history_str, follow_up=follow_up)

        response = llm.invoke(prompt)
        rewritten = response.content.strip().strip('"').strip("'")
        return rewritten if rewritten else follow_up
    except Exception:
        # Fallback to original question in case of rate limits or errors
        return follow_up
