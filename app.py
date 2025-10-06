import streamlit as st
from sentence_transformers import SentenceTransformer, util
import wikipedia

st.set_page_config(page_title="🛣️ NHAI QA Bot", page_icon="🛣️", layout="centered")

st.title("🛣️ NHAI Question Answering Bot")
st.markdown("""
Ask anything about **NHAI (National Highways Authority of India)**:  
- Projects & Status  
- Toll Info  
- Policies & FAQs  

The bot will find the most relevant information for your question.
""")

# -------------------- UTILITIES -------------------- #
def fetch_nhai_text():
    """Fetch NHAI information (from Wikipedia, backend only)."""
    try:
        page = wikipedia.page("National Highways Authority of India")
        text = page.content
        return text
    except wikipedia.exceptions.DisambiguationError as e:
        return " ".join(e.options[:5])
    except wikipedia.exceptions.PageError:
        return "Information not found."

def chunk_text(text, max_words=150):
    """Split text into smaller chunks for semantic search."""
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# -------------------- LOAD MODEL AND DATA -------------------- #
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def prepare_documents():
    text = fetch_nhai_text()
    chunks = chunk_text(text)
    return chunks

model = load_model()
docs = prepare_documents()
doc_embeddings = model.encode(docs, convert_to_tensor=True)

# -------------------- USER INTERFACE -------------------- #
query = st.text_input("Enter your question about NHAI:")

if query:
    st.info("Finding relevant information...")
    
    query_emb = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, doc_embeddings, top_k=5)[0]
    
    st.success("Top Results:")
    for i, hit in enumerate(hits, 1):
        snippet = docs[hit['corpus_id']][:600] + "..."
        score = hit['score']
        st.markdown(f"**Result {i} (Score: {score:.2f}):**")
        st.write(snippet)
