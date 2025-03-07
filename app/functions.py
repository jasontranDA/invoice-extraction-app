# Import Langchain modules
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel  # ✅ FIXED HERE
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings


# Other modules and packages
import streamlit as st  
import pandas as pd

import uuid
import re
import os
import tempfile


def clean_filename(filename):
    """
    Remove "(number)" pattern from a filename 
    (because this could cause error when used as collection name when creating Chroma database).

    Parameters:
        filename (str): The filename to clean

    Returns:
        str: The cleaned filename
    """
    # Regular expression to find "(number)" pattern
    new_filename = re.sub(r'\s\(\d+\)', '', filename)
    
    return new_filename


def get_pdf_text(uploaded_file): 
    """
    Load a PDF document from an uploaded file and return it as a list of documents

    Parameters:
        uploaded_file (file-like object): The uploaded PDF file to load

    Returns:
        list: A list of documents created from the uploaded PDF file
    """
    try:
        # Read file content
        input_file = uploaded_file.read()

        # Create a temporary file (PyPDFLoader requires a file path to read the PDF,
        # it can't work directly with file-like objects or byte streams that we get from Streamlit's uploaded_file)
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(input_file)
        temp_file.close()

        # load PDF document
        loader = PyPDFLoader(temp_file.name)
        documents = loader.load()

        return documents
    
    finally:
        # Ensure the temporary file is deleted when we're done with it
        os.unlink(temp_file.name)


def split_document(documents, chunk_size, chunk_overlap):    
    """
    Function to split generic text into smaller chunks.
    chunk_size: The desired maximum size of each chunk (default: 400)
    chunk_overlap: The number of characters to overlap between consecutive chunks (default: 20).

    Returns:
        list: A list of smaller text chunks created from the generic text
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                          chunk_overlap=chunk_overlap,
                                          length_function=len,
                                          separators=["\n\n", "\n", " "])
    
    return text_splitter.split_documents(documents)


def get_embedding_function():
    """
    Return an OpenAIEmbeddings object, which is used to create vector embeddings from text.
    The embeddings model used is "text-embedding-ada-002" and the OpenAI API key is provided
    as an argument to the function.

    Parameters:
        api_key (str): The OpenAI API key to use when calling the OpenAI Embeddings API.

    Returns:
        OpenAIEmbeddings: An OpenAIEmbeddings object, which can be used to create vector embeddings from text.
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text",show_progress=False)

    return embeddings


def create_vectorstore(chunks, embedding_function, file_name, vector_store_path="db"):

    """
    Create a vector store from a list of text chunks.

    :param chunks: A list of generic text chunks
    :param embedding_function: A function that takes a string and returns a vector
    :param file_name: The name of the file to associate with the vector store
    :param vector_store_path: The directory to store the vector store

    :return: A Chroma vector store object
    """

    # Create a list of unique ids for each document based on the content
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]
    
    # Ensure that only unique docs with unique ids are kept
    unique_ids = set()
    unique_chunks = []
    
    unique_chunks = [] 
    for chunk, id in zip(chunks, ids):     
        if id not in unique_ids:       
            unique_ids.add(id)
            unique_chunks.append(chunk)        

    # Create a new Chroma database from the documents
    vectorstore = Chroma.from_documents(documents=unique_chunks, 
                                        collection_name=clean_filename(file_name),
                                        embedding=embedding_function, 
                                        ids=list(unique_ids), 
                                        persist_directory = vector_store_path)

    # The database should save automatically after we create it
    # but we can also force it to save using the persist() method
    vectorstore.persist()
    
    return vectorstore


def create_vectorstore_from_texts(documents, file_name):
    """
    Create a vector store from a list of texts.

    :param documents: A list of generic text documents
    :param file_name: The name of the file to associate with the vector store

    :return: A Chroma vector store object
    """

    # Split the documents into chunks
    chunks = split_document(documents, chunk_size=1000, chunk_overlap=200)
    
    # Step 3 define embedding function
    embedding_function = get_embedding_function()

    # Step 4 create a vector store  
    vectorstore = create_vectorstore(chunks, embedding_function, file_name)
    
    return vectorstore


# Prompt template
PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer
the question. If you don't know the answer, say that you
don't know. DON'T MAKE UP ANYTHING.

{context}

---

Answer the question based on the above context: {question}
"""

class ExtractedInfo(BaseModel):
    """Extracted information about the research article"""
    invoice_items: str =  Field(description="Extract invoice items")
    invoice_date: str =  Field(description="Extract invoice date")
    business_name: str =  Field(description="Extract business name")
    total_amount: str =  Field(description="Extract total amount")


def format_docs(docs):
    """
    Format a list of Document objects into a single string.

    :param docs: A list of Document objects
    :return: A string containing the text of all the documents joined by two newlines
    """
    return "\n\n".join(doc.page_content for doc in docs)


def query_document(vectorstore, query="Extract relevant details from this business invoice."):

    """
    Query a vector store with a question and return a structured response.

    :param vectorstore: A Chroma vector store object
    :param query: The question to ask the vector store
    :param api_key: The OpenAI API key to use when calling the OpenAI Embeddings API

    :return: A pandas DataFrame with three rows: 'answer', 'source', and 'reasoning'
    """
    llm = ChatOllama(model='llama3.2', temperature=0)
    retriever = vectorstore.as_retriever(search_type="similarity")

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm.with_structured_output(ExtractedInfo)
        )

    structured_response = rag_chain.invoke(query)

    df = pd.DataFrame([{
        "Invoice Items": structured_response.invoice_items,
        "Invoice Date": structured_response.invoice_date,
        "Business Name": structured_response.business_name,
        "Total Amount": structured_response.total_amount,
    }])
    
    return df


def process_multiple_pdfs(pdf_files, query):
    results = []

    # Loop through all uploaded PDF files
    for pdf_file in pdf_files:
        file_name = pdf_file.name
        documents = get_pdf_text(pdf_file)
        vectorstore = create_vectorstore_from_texts(documents, file_name)
        df = query_document(vectorstore, query=query)
        df.insert(0, 'file_name', file_name)
        results.append(df)

    final_df = pd.concat(results, ignore_index=True)

    return final_df
