import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

# ---- CONFIG ----
os.environ["OPENAI_API_KEY"] = "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"

st.set_page_config(page_title="NHAI Query Assistant", page_icon="🛣️", layout="wide")
st.title("🛣️ NHAI Query Assistant")
st.markdown(
    """
    Ask any question related to NHAI based on the uploaded documents.  
    **Note:** This tool provides answers based on the PDF content and may not always be 100% accurate.
    """
)

# ---- LOAD PDF AND PREPROCESS ----
@st.cache_data(show_spinner=True)
def load_faiss_index(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    raw_text = ""
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            raw_text += content

    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_text(raw_text)

    # Create embeddings and FAISS vector store
    embeddings = OpenAIEmbeddings()
    document_search = FAISS.from_texts(texts, embeddings)
    return document_search

# Load the PDF in backend
document_search = load_faiss_index("A.pdf")  # PDF path in repo/folder
chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

# ---- USER QUERY ----
query = st.text_input("Ask a question about NHAI:")

if query:
    with st.spinner("Fetching answer..."):
        docs = document_search.similarity_search(query)
        answer = chain.run(input_documents=docs, question=query)
    st.markdown(f"**Answer:** {answer}")
