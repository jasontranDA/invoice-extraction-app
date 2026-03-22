from __future__ import annotations

import streamlit as st

from functions import AppConfig, DEFAULT_QUERY, InvoiceProcessingError, process_multiple_pdfs

st.set_page_config(
    page_title="Invoice Extraction Workspace",
    page_icon="🧾",
    layout="wide",
)

st.title("🧾 Invoice Extraction Workspace")
st.caption(
    "Upload one or more invoice PDFs, run local Ollama-powered extraction, and export the results as CSV."
)

with st.sidebar:
    st.header("Extraction Settings")
    chat_model = st.text_input("Chat model", value="llama3.2")
    embedding_model = st.text_input("Embedding model", value="nomic-embed-text")
    chunk_size = st.slider("Chunk size", min_value=500, max_value=2000, value=1200, step=100)
    chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=400, value=150, step=25)
    retrieval_k = st.slider("Retrieved chunks", min_value=1, max_value=8, value=4)

query = st.text_area("Extraction prompt", value=DEFAULT_QUERY, height=100)
uploaded_files = st.file_uploader(
    "Upload PDF invoices",
    type=["pdf"],
    accept_multiple_files=True,
    help="The app processes each PDF independently and merges the extracted rows into one table.",
)

extract = st.button("Extract invoice information", type="primary", use_container_width=True)

if extract:
    if not uploaded_files:
        st.warning("Upload at least one PDF invoice to continue.")
    else:
        config = AppConfig(
            chat_model=chat_model,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            retrieval_k=retrieval_k,
        )
        with st.spinner("Reading invoices and extracting structured data..."):
            try:
                result_df = process_multiple_pdfs(uploaded_files, query=query, config=config)
            except InvoiceProcessingError as exc:
                st.error(str(exc))
            else:
                st.success(f"Processed {len(result_df)} invoice(s).")
                metrics = st.columns(3)
                metrics[0].metric("Invoices", len(result_df))
                metrics[1].metric("Unique businesses", result_df["Business Name"].nunique())
                metrics[2].metric("Unique dates", result_df["Invoice Date"].nunique())

                st.dataframe(result_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download CSV",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="invoice-extraction-results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

with st.expander("Local setup checklist"):
    st.markdown(
        """
        - Start Ollama locally before running extractions.
        - Pull the configured chat and embedding models if they are not already available.
        - Keep invoices under your control; the app runs fully on your machine.
        """
    )
