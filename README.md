# Invoice Extraction Workspace

A lightweight Streamlit workspace for extracting structured invoice data from PDFs with local Ollama models and a temporary Chroma vector store.

## What changed

- Modernized the Streamlit UI with configurable extraction settings, success/error feedback, summary metrics, and CSV export.
- Refactored the invoice pipeline for clearer typing, safer temporary-file handling, deterministic chunk deduplication, and automatic cleanup of temporary vector stores.
- Kept the workflow local-first: PDFs are processed on the machine running Streamlit, and the app queries Ollama for structured invoice extraction.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally
- The configured Ollama models, such as:
  - `llama3.2`
  - `nomic-embed-text`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Then open the local Streamlit URL, upload one or more PDF invoices, and export the extracted results as CSV.

## Project structure

- `app/streamlit_app.py` – Streamlit interface
- `app/functions.py` – PDF loading, chunking, retrieval, and structured extraction pipeline
- `data/` – example invoice PDFs for local testing
