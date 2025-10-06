# Install dependencies first:
# pip install streamlit google-genai

import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="🛣️ NHAI Chatbot", page_icon="🛣️", layout="centered")

st.title("🛣️ NHAI Chatbot")
st.markdown("""
Ask anything about **NHAI (National Highways Authority of India)**:
- Projects & Status  
- Toll Info  
- Policies & FAQs  

The AI will answer questions strictly related to NHAI.
""")

# -------------------- Hardcoded API key --------------------
API_KEY = "AIzaSyBbMy_RNIGMs1R33aBr0k-rdDCxkNYh-us"

client = genai.Client(api_key=API_KEY)

# -------------------- User input --------------------
query = st.text_input("Enter your question about NHAI:")

if query:
    with st.spinner("Generating answer..."):
        # Create the content with system prompt
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=f"""
You are a helpful assistant that ONLY answers questions related to the National Highways Authority of India (NHAI). 
If the question is unrelated, politely say you cannot answer.
User question: {query}
""")
                ],
            )
        ]

        # Optional: no external tools
        tools = []

        # Generate content config
        generate_content_config = types.GenerateContentConfig(
            tools=tools,
        )

        # Stream response
        answer_placeholder = st.empty()
        full_answer = ""
        try:
            for chunk in client.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents=contents,
                config=generate_content_config,
            ):
                full_answer += chunk.text
                answer_placeholder.markdown(full_answer)
        except Exception as e:
            st.error(f"Error: {e}")
