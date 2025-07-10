"""
This module provides a wrapper around the Gemini Embedding API
for generating text embeddings using the free Gemini 1.5 model.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv() # lets you keep your key in a .env file and avoid hardcoding.

# Set API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini SDK
genai.configure(api_key=GEMINI_API_KEY)

class GeminiEmbedder:
    """
    Wrapper class for generating embeddings using Gemini's Embedding API.

    Parameters
    ----------
    model_name : str, optional
        The name of the embedding model to use (default is 'models/embedding-001').

    Attributes
    ----------
    model_name : str
        The embedding model used for generating vector representations.
    """

    def __init__(self, model_name: str = "models/embedding-001"):
        self.model_name = model_name

    def get_embedding(self, text: str) -> list:
        """
        Generate a vector embedding for a single text input using Gemini API.

        Parameters
        ----------
        text : str
            The input text string to embed.

        Returns
        -------
        list of float
            The embedding vector representing the input text.

        Raises
        ------
        ValueError
            If the API response does not contain a valid embedding.
        """
        response = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_document",  # Use "retrieval_query" for queries
            title="clinical_note"
        )
        embedding = response.get("embedding", [])
        if not embedding:
            raise ValueError("Embedding failed: empty response from Gemini API.")
        return embedding