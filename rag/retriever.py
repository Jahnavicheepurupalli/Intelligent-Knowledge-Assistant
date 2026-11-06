# from typing import List, Tuple
# from langchain_core.documents import Document
# from database.chroma_manager import ChromaManager

# def retrieve_relevant_chunks(
#     query: str,
#     chroma_manager: ChromaManager,
#     top_k: int = 4,
#     similarity_threshold: float = 0.35,
#     document_scope: str = None
# ) -> List[Tuple[Document, float]]:
#     """
#     Retrieve top-k relevant text chunks from the Chroma vector database.
#     Applies a similarity threshold filter to discard irrelevant snippets.
#     Supports scoping query context searches to a single document.
#     """
#     vector_store = chroma_manager.get_vector_store()

#     # Configure filter metadata mapping if scope is specified
#     search_kwargs = {}
#     if document_scope and document_scope != "All Documents":
#         search_kwargs["filter"] = {"source": document_scope}

#     # Retrieve with relevance scores
#     results = vector_store.similarity_search_with_relevance_scores(query, k=top_k, **search_kwargs)

#     # Filter by similarity threshold
#     filtered_results = []
#     for doc, score in results:
#         if score >= similarity_threshold:
#             filtered_results.append((doc, score))

#     return filtered_results

import os
from typing import List, Tuple

from langchain_core.documents import Document
from database.chroma_manager import ChromaManager


def retrieve_relevant_chunks(
    query: str,
    chroma_manager: ChromaManager,
    top_k: int = 8,
    similarity_threshold: float = 0.20,
    document_scope: str = None
) -> List[Tuple[Document, float]]:
    """
    Retrieve relevant document chunks from ChromaDB.

    Args:
        query: User's question.
        chroma_manager: ChromaManager instance.
        top_k: Maximum number of chunks to retrieve.
        similarity_threshold: Minimum relevance score.
        document_scope: Filename to search within, or "All Documents".

    Returns:
        List of (Document, relevance_score) tuples.
    """

    if not query or not query.strip():
        return []

    vector_store = chroma_manager.get_vector_store()

    # ---------------------------------------------------------
    # Normalize document scope
    # ---------------------------------------------------------
    if document_scope:
        document_scope = document_scope.strip()

        if document_scope != "All Documents":
            document_scope = os.path.basename(document_scope)

    # ---------------------------------------------------------
    # Build Chroma filter
    # ---------------------------------------------------------
    search_kwargs = {}

    if document_scope and document_scope != "All Documents":
        search_kwargs["filter"] = {
            "source": document_scope
        }

    # ---------------------------------------------------------
    # Retrieve chunks
    # ---------------------------------------------------------
    try:
        results = vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=top_k,
            **search_kwargs
        )
    except Exception as e:
        print(f"\nâŒ Retrieval error: {e}")
        return []

    # ---------------------------------------------------------
    # Debug information
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("RAG RETRIEVAL DEBUG")
    print("=" * 80)

    print("QUESTION:", query)
    print("TOP K:", top_k)
    print("THRESHOLD:", similarity_threshold)
    print("DOCUMENT SCOPE:", document_scope)

    if not results:
        print("\nâŒ NO RESULTS FOUND")

        if document_scope and document_scope != "All Documents":
            print(
                "Possible reason: source metadata does not exactly match "
                f"'{document_scope}'."
            )

        print("=" * 80)
        return []

    print(f"\nâœ… RAW RESULTS: {len(results)}")

    for i, (doc, score) in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print("Similarity Score:", round(score, 4))
        print("Source:", doc.metadata.get("source"))
        print("File Hash:", doc.metadata.get("file_hash"))
        print("Page:", doc.metadata.get("page"))
        print("Content:")
        print(doc.page_content[:500])

    # ---------------------------------------------------------
    # Apply similarity threshold
    # ---------------------------------------------------------
    filtered_results = [
        (doc, score)
        for doc, score in results
        if score >= similarity_threshold
    ]

    print("\n" + "-" * 80)
    print("FILTERED RESULTS:", len(filtered_results))

    if not filtered_results:
        print(
            "âš ï¸ Results were retrieved, but all were removed "
            "by the similarity threshold."
        )

    print("=" * 80)

    return filtered_results
