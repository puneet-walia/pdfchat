import streamlit as st
import requests

st.set_page_config(page_title="🛣️ NHAI Chatbot", page_icon="🛣️", layout="centered")

st.title("🛣️ NHAI Chatbot")
st.markdown("""
Ask anything about **NHAI (National Highways Authority of India)**:
- Projects & Status  
- Toll Info  
- Policies & FAQs  

This chatbot will answer questions using AI.
""")

# Your Gemini API Key
API_KEY = "AIzaSyBbMy_RNIGMs1R33aBr0k-rdDCxkNYh-us"

# Gemini endpoint
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta2/models/chat-bison-001:generateMessage?key={API_KEY}"

def query_gemini(prompt):
    """
    Send user query to Gemini API and get AI response.
    Restrict answers to NHAI only via system prompt.
    """
    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "prompt": {
            "messages": [
                {"author": "system", "content": "You are a helpful assistant that ONLY answers questions related to the National Highways Authority of India (NHAI). If the question is unrelated, politely say you cannot answer."},
                {"author": "user", "content": prompt}
            ]
        },
        "temperature": 0.2,
        "candidate_count": 1,
        "top_p": 0.95
    }

    response = requests.post(GEMINI_ENDPOINT, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        try:
            return result['candidates'][0]['content']
        except:
            return "Sorry, I couldn't process the response."
    else:
        return f"Error {response.status_code}: {response.text}"

# -------------------- Streamlit Interface -------------------- #
query = st.text_input("Enter your question about NHAI:")

if query:
    with st.spinner("Getting answer from AI..."):
        answer = query_gemini(query)
        st.success("Answer:")
        st.write(answer)
