"""
This module provides a wrapper for generating answers using the
Gemini 1.5 Pro model from Google's Generative AI API.
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables (e.g., GEMINI_API_KEY)
load_dotenv()

# Configure Gemini with your API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)


class GeminiGenerator:
    """
    Wrapper class for generating natural language responses using Gemini 1.5 Pro.

    Parameters
    ----------
    model_name : str, optional
        The name of the generative model to use (default is "gemini-1.5-pro").

    Attributes
    ----------
    model : generativeai.GenerativeModel
        Instance of the generative model used to produce completions.
    """

    def __init__(self, model_name: str = "gemini-1.5-pro"):
        self.model = genai.GenerativeModel(model_name)

    def generate_answer(self, query: str, context_chunks: list) -> str: # takes a list of context chunks + query and builds a structured prompt
        """
        Generate an answer to a user query based on retrieved context.

        Parameters
        ----------
        query : str
            The user's question (e.g., "Why was the patient prescribed antibiotics?").

        context_chunks : list of str
            Relevant text chunks retrieved from the clinical notes.

        Returns
        -------
        str
            The generated answer based on the provided context and query.

        Raises
        ------
        RuntimeError
            If the Gemini model fails to return a response.
        """
        # Construct a structured prompt
        prompt = (
            "You are a helpful clinical assistant. Based on the following information, "
            "answer the user's question accurately and clearly.\n\n"
            "Context:\n"
            + "\n---\n".join(context_chunks)
            + f"\n\nQuestion: {query}\nAnswer:"
        )

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {str(e)}")
