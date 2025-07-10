"""
Main orchestration script for running Retrieval-Augmented Generation (RAG)
over synthetic clinical notes using Gemini and FAISS.

Steps:
1. Preprocess and chunk input note
2. Embed chunks using Gemini
3. Build FAISS index
4. Embed user query
5. Retrieve relevant chunks
6. Generate final answer
"""

import argparse
from utils.preprocessing import load_patient_data, clean_text, chunk_text
from rag.embedder import GeminiEmbedder
from rag.retriever import FaissRetriever
from rag.generator import GeminiGenerator


def run_rag_pipeline(note_path: str, query: str, top_k: int = 3) -> str:
    """
    Runs the full RAG pipeline over a clinical note.

    Parameters
    ----------
    note_path : str
        Path to the clinical note file (plain text) or EHR (.json).

    query : str
        User question

    top_k : int, optional
        Number of top context chunks to retrieve (default is 3).

    Returns
    -------
    str
        The generated answer based on retrieved clinical content.
    """
    # Load from .txt or .json
    note_text = load_patient_data(note_path)

    # Clean and Chunk
    clean = clean_text(note_text)
    chunks = chunk_text(clean)

    # Initialize components
    embedder = GeminiEmbedder()
    retriever = FaissRetriever(embedding_dim=768)
    generator = GeminiGenerator()

    # Embed chunks and build FAISS index
    chunk_embeddings = [embedder.get_embedding(c) for c in chunks]
    retriever.build_index(chunk_embeddings, chunks)

    # Embed query and retrieve top chunks
    query_embedding = embedder.get_embedding(query)
    top_chunks = retriever.retrieve(query_embedding, top_k=top_k)

    # Generate and return answer
    return generator.generate_answer(query, top_chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gemini RAG pipeline on a clinical note.")
    parser.add_argument("--note", type=str, required=True, help="Path to clinical note (txt) or EHR (.json)")
    parser.add_argument("--query", type=str, required=True, help="User question")
    parser.add_argument("--top_k", type=int, default=3, help="Number of chunks to retrieve")

    args = parser.parse_args()

    answer = run_rag_pipeline(note_path=args.note,_
