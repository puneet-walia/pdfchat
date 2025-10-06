import streamlit as st
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession
import json

st.title("🛣️ NHAI Chatbot with Gemini")

# Load your service account JSON key
SERVICE_ACCOUNT_FILE = "path_to_your_service_account.json"

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

authed_session = AuthorizedSession(credentials)

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta2/models/chat-bison-001:generateMessage"

def query_gemini(prompt):
    data = {
        "prompt": {
            "messages": [
                {"author": "system", "content": "You are a helpful assistant that only answers questions about NHAI."},
                {"author": "user", "content": prompt}
            ]
        },
        "temperature": 0.2,
        "candidate_count": 1,
        "top_p": 0.95
    }

    response = authed_session.post(GEMINI_ENDPOINT, json=data)
    if response.status_code == 200:
        result = response.json()
        try:
            return result['candidates'][0]['content']
        except:
            return "Error reading response."
    else:
        return f"Error {response.status_code}: {response.text}"

# Streamlit interface
query = st.text_input("Enter your NHAI question:")

if query:
    with st.spinner("Getting AI answer..."):
        answer = query_gemini(query)
        st.success("Answer:")
        st.write(answer)
