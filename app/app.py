import streamlit as st
import requests

API_BASE = "http://localhost:8000/api"

st.set_page_config(page_title="AI Model Interface", layout="centered")

st.title("AI Model Interface")

# Sidebar settings
st.sidebar.header("Configuration")
model_name = st.sidebar.text_input("Model Name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
spacy_model = st.sidebar.text_input("SpaCy Model", "en_core_web_sm")
chunk_size = st.sidebar.slider("Chunk Size (KB)", 0.5, 10.0, 4.0)
llm_model = st.sidebar.text_input("LLM Model", "llama3.2")

# Choose task
task = st.selectbox("Choose Task", ["Chat", "RAG", "Search"])

# Input area
st.subheader(f"{task} with Model")
user_input = st.text_area("Enter your question or search query:")

if st.button("Submit") and user_input.strip():
    payload = {
        "question": user_input,
        "query": user_input,
        "model": model_name,
        "spacy_model": spacy_model,
        "chunk_size_in_kb": chunk_size,
        "llm_model": llm_model,
    }

    try:
        if task == "Chat":
            response = requests.post(f"{API_BASE}/prompt", json=payload)
            st.markdown(f"### Answer:\n{response.json()['answer']}")
        elif task == "RAG":
            response = requests.post(f"{API_BASE}/rag", json=payload)
            result = response.json()
            st.markdown(f"### Context:\n{result['context']}")
            st.markdown(f"### Answer:\n{result['answer']}")
        elif task == "Search":
            response = requests.post(f"{API_BASE}/search", json=payload)
            st.markdown("### Search Results:")
            for i, res in enumerate(response.json()["results"], 1):
                st.markdown(f"{i}. {res}")
    except Exception as e:
        st.error(f"Error: {e}")
