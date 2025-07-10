"""
This script runs a full PubMed RAG pipeline using:
- PubMed search + fetch (search_pubmed.py)
- Embedding via Gemini
- Retrieval via FAISS
- Generation via Gemini
"""
import argparse
from pubmed_rag.search_pubmed import search_pubmed
from rag.embedder import GeminiEmbedder
from rag.retriever import FaissRetriever
from rag.generator import GeminiGenerator


def run_pubmed_rag_pipeline(query: str, top_k: int = 3, max_results: int = 10, email: str = "") -> str:
    """
    Run the PubMed RAG pipeline.

    Parameters
    ----------
    query : str
        User question to search and answer.

    top_k : int, optional
        Number of top chunks to retrieve (default is 3).

    max_results : int, optional
        Max number of PubMed abstracts to fetch (default is 10).

    email : str, optional
        Required email to access the PubMed API (Entrez).

    Returns
    -------
    str
        Gemini-generated answer based on literature context.
    """
    print("🔍 Searching PubMed...")
    abstracts = search_pubmed(query, max_results=max_results, email=email)
    if not abstracts:
        return "No relevant literature found on PubMed."

    # Step 2: Embed abstracts
    print(f"📚 Retrieved {len(abstracts)} abstracts. Embedding...")
    embedder = GeminiEmbedder()
    embeddings = [embedder.get_embedding(text) for text in abstracts]

    # Step 3: Build FAISS index
    retriever = FaissRetriever(embedding_dim=768)
    retriever.build_index(embeddings, abstracts)

    # Step 4: Embed query and retrieve top context
    print("🔎 Retrieving top relevant abstracts...")
    query_embedding = embedder.get_embedding(query)
    top_chunks = retriever.retrieve(query_embedding, top_k=top_k)

    # Step 5: Generate answer
    print("🧠 Generating answer with Gemini...")
    generator = GeminiGenerator()
    answer = generator.generate_answer(query, top_chunks)
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PubMed RAG QA pipeline")
    parser.add_argument("--query", type=str, required=True, help="User question to search PubMed for")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top chunks to retrieve")
    parser.add_argument("--max_results", type=int, default=10, help="Max number of PubMed abstracts to fetch")
    parser.add_argument("--email", type=str, required=True, help="Your email for Entrez API access")

    args = parser.parse_args()

    answer = run_pubmed_rag_pipeline(
        query=args.query,
        top_k=args.top_k,
        max_results=args.max_results,
        email=args.email
    )
    print("\n📘 Generated Answer:\n")
    print(answer)

