import re
import json
from pathlib import Path
from typing import Union

def clean_text(text: str) -> str:
    """
    Clean clinical text by removing excessive whitespace and line breaks.

    Parameters
    ----------
    text : str
        Raw clinical text.

    Returns
    -------
    str
        Cleaned clinical text with consistent formatting.
    """
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def chunk_text(text: str, max_tokens: int = 300) -> list:
    """
    Split cleaned text into approximately token-sized chunks.

    Parameters
    ----------
    text : str
        Input text to split.

    max_tokens : int, optional
        Approximate max tokens per chunk (default is 300).

    Returns
    -------
    list of str
        List of text chunks.
    """
    paragraphs = text.split("\n")
    chunks = []
    chunk = ""
    for para in paragraphs:
        if len(chunk) + len(para) < max_tokens * 4:  # ~4 characters per token
            chunk += para + "\n"
        else:
            chunks.append(chunk.strip())
            chunk = para + "\n"
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def load_patient_data(path: Union[str, Path]) -> str:
    """
    Load clinical content from a .txt file or structured .json EHR file.

    Parameters
    ----------
    path : str or Path
        Path to a .txt or .json file.

    Returns
    -------
    str
        Flattened clinical note text for processing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix == ".txt":
        return path.read_text(encoding="utf-8")

    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return extract_text_from_ehr(data)

    else:
        raise ValueError("Unsupported file format. Use .txt or .json")

def extract_text_from_ehr(ehr: dict) -> str:
    """
    Convert structured EHR data into readable free-text format.

    Parameters
    ----------
    ehr : dict
        Synthetic EHR dictionary (e.g., from Synthea).

    Returns
    -------
    str
        Free-text formatted summary of patient data.
    """
    lines = []

    # Patient demographics
    patient = ehr.get("name", "Unknown")
    lines.append(f"Patient: {patient}")
    if "gender" in ehr:
        lines.append(f"Gender: {ehr['gender']}")
    if "birthDate" in ehr:
        lines.append(f"Birth Date: {ehr['birthDate']}")

    # Medical conditions
    if "conditions" in ehr:
        lines.append("\nConditions:")
        for c in ehr["conditions"]:
            lines.append(f"- {c.get('code', {}).get('text', '')}")

    # Medications
    if "medications" in ehr:
        lines.append("\nMedications:")
        for m in ehr["medications"]:
            lines.append(f"- {m.get('medicationCodeableConcept', {}).get('text', '')}")

    # Observations and labs
    if "observations" in ehr:
        lines.append("\nObservations:")
        for o in ehr["observations"]:
            val = o.get("valueQuantity", {}).get("value")
            unit = o.get("valueQuantity", {}).get("unit")
            text = o.get("code", {}).get("text", "")
            if val and unit:
                lines.append(f"- {text}: {val} {unit}")
            elif text:
                lines.append(f"- {text}")

    return "\n".join(lines)
