# Gemini RAG for Healthcare

A modular framework for building Retrieval-Augmented Generation (RAG) pipelines in healthcare using:
- 🧬 Gemini 1.5 Pro API (free tier) for embeddings and answer generation  
- 🔍 FAISS for semantic retrieval
- 📝 Synthetic or public datasets (e.g., clinical notes, PubMed)

---

 ## Table of Contents
* Overview

* Project Structure

* Quick Start

* RAG Pipelines

  * Clinical Notes QA

  * PubMed Literature QA

* Example Input

* Upcoming Extensions

* Author
---

## Project Structure

```
gemini-rag-healthcare/
├── rag/                    # Shared RAG infrastructure
│   ├── embedder.py
│   ├── retriever.py
│   └── generator.py
├── clinical_rag/           # Clinical note QA module
│   └── pipeline.py         # Old rag/pipeline.py moved here
├── pubmed_rag/             # PubMed QA module
│   ├── search_pubmed.py
│   └── pipeline.py
├── data/
│   └── synthetic_note.txt
├── app/                    # optional
│   └── streamlit_app.py
├── utils/
│   └── preprocessing.py
├── .env.example
├── requirements.txt
└── README.md

```

---

## Quick Start

### 1. Install Dependencies
```bash
git clone https://github.com/yourusername/gemini-rag-healthcare.git
cd gemini-rag-healthcare
pip install -r requirements.txt
```
### 2. Add Gemini API Key

Create a .env file with your key:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

---
## RAG Pipelines
### 1. Clinical Note QA
Extract answers from synthetic .txt or EHR .json notes:

```bash
python rag/pipeline.py \
    --note data/sample.txt \
    --query "Why was the patient given antibiotics?" \
    --top_k 3
    
    
python rag/pipeline.py \
    --note data/sample.json \
    --query "Why was the patient given antibiotics?" \
    --top_k 3
```

#### Example 

```text
Patient: John Doe
Visit Date: 2024-06-20

Chief Complaint:
Patient reports shortness of breath and cough.

Assessment:
Findings consistent with pneumonia. Chest X-ray: right lower lobe consolidation.

Plan:
Started on amoxicillin 500mg TID. Advised rest. Follow-up in 1 week.
```
Query: Why was the patient given antibiotics?
Answer: To treat pneumonia based on symptoms and imaging.

### 2. PubMed Literature QA
Retrieve abstracts from PubMed and synthesize a response:

```bash
python pubmed_pipeline.py \
    --query "What are the effects of metformin on Alzheimer's?" \
    --top_k 3 \
    --max_results 10 \
    --email "your.name@example.com"
```

| Argument        | Description                                   |
| --------------- | --------------------------------------------- |
| `--note`        | Path to clinical note `.txt`                  |
| `--query`       | Clinical or biomedical question               |
| `--top_k`       | Number of top documents to retrieve           |
| `--max_results` | (PubMed) Number of abstracts to fetch         |
| `--email`       | (PubMed) Required email for Entrez API access |

---

## Extending the Project

| Module           | Description                       | Status   |
|------------------|-----------------------------------|----------|
| `clinical_rag/`  | QA from patient notes             | Complete |
| `pubmed_rag/`    | Literature QA from PubMed         | Complete |
| `lab_rag/`       | Explain lab abnormalities         | WIP      |
| `trial_rag/`     | Match patients to clinical trials | WIP      |
| `guideline_rag/` | QA from clinical care guidelines  | WIP      |
| `streamlit_ui/`  | UI interface                      | WIP      |

---
## Author
Ojaswita Lokre