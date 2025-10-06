import streamlit as st
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import os

st.set_page_config(page_title="NHAI QA Bot", page_icon="🛣️", layout="centered")

st.title("🛣️ NHAI Question Answering Bot")
st.markdown("""
Ask me anything about **NHAI (National Highways Authority of India)**:
- Projects & Status  
- Toll Info  
- Policies & FAQs  

This bot searches through uploaded or preloaded NHAI documents to find relevant answers.
""")

# Folder containing NHAI PDFs
PDF_FOLDER = "nhai_pdfs"

# Load PDFs
@st.cache_data
def load_documents(folder):
    docs = []
    filenames = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            with open(os.path.join(folder, file), "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                docs.append(text)
                filenames.append(file)
    return docs, filenames

documents, filenames = load_documents(PDF_FOLDER)

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()
doc_embeddings = model.encode(documents, convert_to_tensor=True)

# User query
query = st.text_input("Enter your question:")

if query:
    st.info("Searching documents...")
    
    query_emb = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, doc_embeddings, top_k=3)
    
    if hits[0]:
        with st.expander("Document-based Answers"):
            for i, hit in enumerate(hits[0], 1):
                doc_id = hit['corpus_id']
                score = hit['score']
                st.markdown(f"**Result {i} (Score: {score:.2f}) from `{filenames[doc_id]}`:**")
                st.write(documents[doc_id][:500] + "...")  # Show first 500 characters
    else:
        st.warning("No relevant information found in the documents.")
