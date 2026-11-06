from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Split standard documents into overlapping text chunks using RecursiveCharacterTextSplitter.
    Preserves existing metadata in the generated chunk Documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)
