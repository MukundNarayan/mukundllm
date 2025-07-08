# streamlit_app.py

import streamlit as st
import os

from langchain_ollama import ChatOllama
from document_loader import load_documents_into_database
from models import check_if_model_is_available
from llm import getChatChain

# ---- Config ----
PERSIST_DIRECTORY = "db"
DOCUMENT_PATH = "Research"
EMBED_MODEL_NAME = "nomic-embed-text"
OLLAMA_MODEL = "mistral"

# ---- SQL Configuration ----
SQL_CONFIG = {
    "server": "localhost",
    "database": "test_llm",
    "trusted_connection": True,
    "queries": {
        "get_all_employees": "SELECT * FROM employees;"
    }
}

# ---- Load or Create Vector DB ----
@st.cache_resource
def load_db():
    try:
        check_if_model_is_available(OLLAMA_MODEL)
        check_if_model_is_available(EMBED_MODEL_NAME)
        return load_documents_into_database(
            model_name=EMBED_MODEL_NAME,
            documents_path=DOCUMENT_PATH,
            reload=True,
            load_excel=True,
            load_sql=True,
            sql_config=SQL_CONFIG
        )
    except FileNotFoundError as e:
        st.error(f"📁 Document path not found: {e}")
    except Exception as e:
        st.error(f"❌ Error while loading DB: {e}")

# ---- Streamlit UI ----
st.set_page_config(page_title="📚 Internal RAG Chat", layout="wide")
st.title("🧠 Internal Document + SQL Chatbot")

question = st.text_input("🔍 Ask anything from documents + SQL database:")
submit = st.button("Ask")

if submit and question:
    with st.spinner("Processing..."):
        db = load_db()
        if db:
            llm = ChatOllama(model=OLLAMA_MODEL)
            chain = getChatChain(llm, db)
            result = chain(question)

            st.markdown("### 💬 Answer")
            st.write(result["result"] if isinstance(result, dict) else result)

            if isinstance(result, dict) and "source_documents" in result:
                st.markdown("### 📄 Sources")
                for doc in result["source_documents"]:
                    st.markdown(f"- {doc.metadata['source']}")
