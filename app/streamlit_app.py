
import streamlit as st  
from functions import *

# Streamlit app
st.title("AI-Powered Invoice Extraction Tool")
uploaded_files = st.file_uploader("Upload PDF invoices:", type=["pdf"], accept_multiple_files=True)
user_query = "Extract relevant details from this business invoice."

if st.button("Extract invoice information") and uploaded_files:
    final_df = process_multiple_pdfs(uploaded_files, query=user_query)
    st.write(final_df)