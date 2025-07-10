"""
This module provides functions to build and query a FAISS index
for fast semantic similarity search using text embeddings.
"""

import faiss
import numpy as np


class FaissRetriever:
    """
    A simple wrapper around FAISS for similarity-based retrieval.

    Parameters
    ----------
    embedding_dim : int
        Dimension of the embedding vectors.
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.chunk_store = []  # Store original text chunks aligned with embeddings

    def build_index(self, embeddings: list, chunks: list):
        """
        Build a FAISS index from embeddings and store the corresponding text chunks.

        Parameters
        ----------
        embeddings : list of list of float
            List of embedding vectors for each chunk.

        chunks : list of str
            Original text chunks corresponding to each embedding.

        Raises
        ------
        ValueError
            If the number of embeddings and chunks do not match.
        """
        if len(embeddings) != len(chunks):
            raise ValueError("Mismatch between number of embeddings and text chunks.")

        self.chunk_store = chunks
        self.index.add(np.array(embeddings).astype("float32"))

    def retrieve(self, query_embedding: list, top_k: int = 3) -> list:
        """
        Retrieve top-k most similar text chunks for a given query embedding.

        Parameters
        ----------
        query_embedding : list of float
            The embedding vector of the query.

        top_k : int, optional
            Number of most similar chunks to return (default is 3).

        Returns
        -------
        list of str
            List of retrieved text chunks sorted by similarity.
        """
        query_vec = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_vec, top_k)
        return [self.chunk_store[i] for i in indices[0] if i < len(self.chunk_store)]
