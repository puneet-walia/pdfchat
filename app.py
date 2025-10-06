import streamlit as st
import wikipedia
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import os

st.set_page_config(page_title="NHAI QA Bot", page_icon="🛣️", layout="wide")

st.title("🛣️ NHAI Question Answering Bot")
st.markdown("""
Ask me anything about **NHAI (National Highways Authority of India)**:
- Projects & Status  
- Toll Info  
- History & Policies  

This bot searches both Wikipedia and official NHAI documents for accurate answers.
""")

query = st.text_input("Enter your question:")

# Load your local PDFs (example: NHAI FAQs)
pdf_folder = "nhai_pdfs"
documents = []
for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        with open(os.path.join(pdf_folder, file), "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                documents.append(page.extract_text())

# Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = model.encode(documents, convert_to_tensor=True)

def answer_from_documents(query):
    query_emb = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, doc_embeddings, top_k=3)
    results = []
    for hit in hits[0]:
        results.append(documents[hit['corpus_id']])
    return results

if query:
    st.info("Searching for answers...")
    # Step 1: Search in Wikipedia
    try:
        search_results = wikipedia.search(query, results=3)
        if search_results:
            page = wikipedia.page(search_results[0])
            st.subheader(f"Wikipedia: {page.title}")
            st.write(page.summary)
            st.markdown(f"[Read more]({page.url})")
        else:
            st.warning("No Wikipedia results found.")
    except Exception as e:
        st.error(f"Wikipedia error: {e}")
    
    # Step 2: Search in PDFs (FAQ/Projects)
    doc_answers = answer_from_documents(query)
    if doc_answers:
        with st.expander("Document-based Answers"):
            for i, ans in enumerate(doc_answers, 1):
                st.markdown(f"**Result {i}:**\n{ans[:500]}...")  # Show first 500 chars
    else:
        st.info("No relevant document found.")
