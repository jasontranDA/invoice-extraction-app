from __future__ import annotations

import re
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import pysqlite3
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel, Field

# Force Python to use pysqlite3 instead of the system sqlite module for Chroma.
sys.modules["sqlite3"] = pysqlite3

DEFAULT_QUERY = "Extract the key details from this business invoice."
DEFAULT_CHAT_MODEL = "llama3.2"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

PROMPT_TEMPLATE = """
You extract structured data from invoices.
Use only the retrieved invoice context. If a value is missing, return "Not found".
Do not invent details.

Retrieved context:
{context}

Question: {question}
"""


class ExtractedInfo(BaseModel):
    """Structured invoice information returned by the LLM."""

    invoice_items: str = Field(description="Line items or services listed on the invoice")
    invoice_date: str = Field(description="Invoice issue date")
    business_name: str = Field(description="Vendor or business name on the invoice")
    total_amount: str = Field(description="Invoice total amount due")


@dataclass(frozen=True)
class AppConfig:
    chat_model: str = DEFAULT_CHAT_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chunk_size: int = 1200
    chunk_overlap: int = 150
    retrieval_k: int = 4


class InvoiceProcessingError(RuntimeError):
    """Raised when a PDF cannot be processed into invoice data."""



def clean_filename(filename: str) -> str:
    """Normalize a filename into a valid Chroma collection name."""
    stem = Path(filename).stem
    cleaned = re.sub(r"\s*\(\d+\)", "", stem)
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", cleaned).strip("-")
    return (cleaned or "invoice")[:63]



def get_pdf_documents(uploaded_file) -> list[Document]:
    """Load an uploaded PDF into LangChain documents."""
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = Path(temp_file.name)

        loader = PyPDFLoader(str(temp_path))
        return loader.load()
    except Exception as exc:  # noqa: BLE001
        raise InvoiceProcessingError(f"Unable to read PDF '{uploaded_file.name}'.") from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()



def split_documents(documents: Sequence[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split documents into chunks sized for embedding and retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(list(documents))



def get_embedding_function(model_name: str) -> OllamaEmbeddings:
    """Create the embedding function used for Chroma."""
    return OllamaEmbeddings(model=model_name)



def deduplicate_chunks(chunks: Iterable[Document]) -> tuple[list[Document], list[str]]:
    """Deduplicate chunks while preserving order and aligned IDs."""
    unique_chunks: list[Document] = []
    unique_ids: list[str] = []
    seen_ids: set[str] = set()

    for chunk in chunks:
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.page_content))
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        unique_chunks.append(chunk)
        unique_ids.append(chunk_id)

    return unique_chunks, unique_ids



def create_vectorstore(chunks: Sequence[Document], embedding_model: str, file_name: str) -> tuple[Chroma, str]:
    """Build a temporary Chroma vector store for a single invoice."""
    unique_chunks, unique_ids = deduplicate_chunks(chunks)
    persist_directory = tempfile.mkdtemp(prefix="invoice-chroma-")

    vectorstore = Chroma.from_documents(
        documents=unique_chunks,
        ids=unique_ids,
        collection_name=clean_filename(file_name),
        embedding=get_embedding_function(embedding_model),
        persist_directory=persist_directory,
    )
    vectorstore.persist()
    return vectorstore, persist_directory



def format_docs(docs: Sequence[Document]) -> str:
    """Render retrieved documents into prompt context."""
    return "\n\n".join(doc.page_content for doc in docs)



def query_document(vectorstore: Chroma, query: str, config: AppConfig) -> pd.DataFrame:
    """Query the vector store and return a single-row invoice dataframe."""
    llm = ChatOllama(model=config.chat_model, temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.retrieval_k})
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm.with_structured_output(ExtractedInfo)
    )

    try:
        response = rag_chain.invoke(query)
    except Exception as exc:  # noqa: BLE001
        raise InvoiceProcessingError(
            "The Ollama models could not be reached. Confirm Ollama is running and the required models are installed."
        ) from exc

    return pd.DataFrame(
        [
            {
                "Invoice Items": response.invoice_items,
                "Invoice Date": response.invoice_date,
                "Business Name": response.business_name,
                "Total Amount": response.total_amount,
            }
        ]
    )



def process_multiple_pdfs(pdf_files, query: str = DEFAULT_QUERY, config: AppConfig | None = None) -> pd.DataFrame:
    """Extract invoice details from multiple uploaded PDF files."""
    active_config = config or AppConfig()
    results: list[pd.DataFrame] = []

    for pdf_file in pdf_files:
        documents = get_pdf_documents(pdf_file)
        chunks = split_documents(documents, active_config.chunk_size, active_config.chunk_overlap)
        vectorstore, persist_directory = create_vectorstore(chunks, active_config.embedding_model, pdf_file.name)

        try:
            result = query_document(vectorstore, query=query, config=active_config)
        finally:
            shutil.rmtree(persist_directory, ignore_errors=True)

        result.insert(0, "File Name", pdf_file.name)
        results.append(result)

    if not results:
        return pd.DataFrame(columns=["File Name", "Invoice Items", "Invoice Date", "Business Name", "Total Amount"])

    return pd.concat(results, ignore_index=True)
