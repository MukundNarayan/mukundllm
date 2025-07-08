# ui.py

import streamlit as st
import os
from langchain_ollama import ChatOllama

from document_loader import load_documents_into_database
from models import get_list_of_models
from llm import getStreamingChain

# Embedding and default path
EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_DOCS_PATH = "Research"

# 🔐 SQL Server Configuration
SQL_CONFIG = {
    "server": "localhost",
    "database": "EmployeeManagementDB",
    "trusted_connection": True,
    "queries": {
        "get_all_employees": "SELECT * FROM employees;"
    }
}


# Streamlit setup
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("💬 Local LLM with RAG (Ollama + Streamlit)")

# Load available LLM models
if "list_of_models" not in st.session_state:
    st.session_state["list_of_models"] = get_list_of_models()

selected_model = st.sidebar.selectbox("Select a chat model:", st.session_state["list_of_models"])

# Validate selected model
if "embed" in selected_model or "nomic" in selected_model:
    st.error("❌ This is an embedding model, not a chat model. Please choose a valid chat model.")
    st.stop()

# Load or update chat model
if st.session_state.get("ollama_model") != selected_model:
    st.session_state["ollama_model"] = selected_model
    st.session_state["llm"] = ChatOllama(model=selected_model)

# Folder path input
folder_path = st.sidebar.text_input("📁 Enter documents folder path:", value=DEFAULT_DOCS_PATH)

# Initialize memory state
if "db" not in st.session_state:
    st.session_state.db = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Indexing logic
if folder_path:
    if not os.path.isdir(folder_path):
        st.error("🚫 Invalid folder path. Please check and try again.")
        

    if st.sidebar.button("📚 Index Documents + SQL"):
        # reload_index = st.sidebar.checkbox("Force re-index", value=False)
    # elif st.sidebar.button("📚 Index Documents + SQL"):
        with st.spinner("Indexing and embedding documents + SQL data..."):
            print("🔍 Using SQL database:", SQL_CONFIG["database"])
            st.session_state.db = load_documents_into_database(
                model_name=EMBEDDING_MODEL,
                documents_path=folder_path,
                reload=True,
                load_excel=True,
                load_sql=True,
                sql_config=SQL_CONFIG
            )
        st.success("✅ Documents and SQL data indexed successfully!")
else:
    st.warning("⚠️ Please enter a valid folder path to load documents.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat section
if st.session_state.db is None:
    st.warning("📌 Please index documents before chatting.")
    st.chat_input("Waiting for documents...", disabled=True)
else:
    if user_input := st.chat_input("Ask something from the documents + SQL..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            stream = getStreamingChain(
                user_input,
                st.session_state.messages,
                st.session_state.llm,
                st.session_state.db,
            )
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
