import os
import sys
import unittest
sys.path.insert(0, r"c:\Users\jaanu\OneDrive\Desktop\DATA I2I PROJECT")

from database.chroma_manager import ChromaManager
from rag.retriever import retrieve_relevant_chunks
from ingestion.loader import load_single_document, get_file_hash
from ingestion.chunker import split_documents

class TestRAGPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chroma_manager = ChromaManager()
        cls.test_file = os.path.join(r"c:\Users\jaanu\OneDrive\Desktop\DATA I2I PROJECT", "data", "project_docs.txt")

    def test_1_deduplication(self):
        fhash = get_file_hash(self.test_file)
        docs = load_single_document(self.test_file)
        chunks = split_documents(docs, 500, 100)
        # Attempt to insert, might return False if already loaded, but checking method handles correctly
        res = self.chroma_manager.add_documents(chunks, "project_docs.txt", fhash)
        # Re-adding must return False
        dup_res = self.chroma_manager.add_documents(chunks, "project_docs.txt", fhash)
        self.assertFalse(dup_res)

    def test_2_scoping(self):
        # Retrieve with specific document scope filter
        res = retrieve_relevant_chunks(
            "What is the definition of RAG?",
            self.chroma_manager,
            top_k=2,
            similarity_threshold=0.1,
            document_scope="project_docs.txt"
        )
        self.assertTrue(len(res) > 0)
        # Retrieve with irrelevant scope should yield 0 results
        res_empty = retrieve_relevant_chunks(
            "What is the definition of RAG?",
            self.chroma_manager,
            top_k=2,
            similarity_threshold=0.1,
            document_scope="non_existent.pdf"
        )
        self.assertEqual(len(res_empty), 0)

if __name__ == "__main__":
    unittest.main()
