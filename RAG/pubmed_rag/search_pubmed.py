"""
Module to query PubMed and fetch abstracts using Entrez (NCBI API).
"""

from Bio import Entrez
from typing import List

def search_pubmed(query: str, max_results: int = 10, email: str = "") -> List[str]:
    """
    Search PubMed and fetch abstracts related to a query.

    Parameters
    ----------
    query : str
        The PubMed search query string.

    max_results : int, optional
        Maximum number of abstracts to retrieve (default is 10).

    email : str, optional
        User email for NCBI API compliance.

    Returns
    -------
    List[str]
        A list of abstracts as text blocks (title + abstract).
    """
    if not email:
        raise ValueError("Entrez email must be provided for PubMed API access.")

    Entrez.email = email  # Set per call

    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()

    id_list = record.get("IdList", [])
    if not id_list:
        return []

    handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="abstract", retmode="text")
    abstract_text = handle.read()
    handle.close()

    entries = abstract_text.strip().split("\n\n")
    return [entry.strip() for entry in entries if entry.strip()]
