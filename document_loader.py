import os
import pandas as pd
import pyodbc
from typing import List, Dict

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # ✅ updated per deprecation warning

# Constants
PERSIST_DIRECTORY = "storage"
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# ------------------------
# Load Documents into Chroma
# ------------------------

def load_documents_into_database(
    model_name: str,
    documents_path: str,
    reload: bool = True,
    load_excel: bool = True,
    load_sql: bool = True,
    sql_config: Dict = None
) -> Chroma:
    if reload:
        print("🔄 Loading and splitting documents...")
        raw_documents = load_documents(documents_path, load_excel, load_sql, sql_config)
        documents = TEXT_SPLITTER.split_documents(raw_documents)

        print("🔌 Creating embeddings and storing in Chroma...")
        return Chroma.from_documents(
            documents=documents,
            embedding=OllamaEmbeddings(model=model_name),
            persist_directory=PERSIST_DIRECTORY,
        )
    else:
        print("🔁 Loading existing Chroma index...")
        return Chroma(
            embedding_function=OllamaEmbeddings(model=model_name),
            persist_directory=PERSIST_DIRECTORY,
        )

# ------------------------
# Load from Folder + Excel + SQL
# ------------------------

def load_documents(
    path: str,
    load_excel: bool = True,
    load_sql: bool = False,
    sql_config: Dict = None
) -> List[Document]:

    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ The specified path does not exist: {path}")

    documents = []

    # 1. PDFs
    pdf_loader = DirectoryLoader(
        path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    documents.extend(pdf_loader.load())

    # 2. Markdown
    md_loader = DirectoryLoader(
        path,
        glob="**/*.md",
        loader_cls=TextLoader,
        show_progress=True,
    )
    documents.extend(md_loader.load())

    # 3. Excel
    if load_excel:
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".xlsx"):
                    full_path = os.path.join(root, file)
                    documents.extend(load_excel_documents(full_path))

    # 4. SQL
    if load_sql and sql_config:
        print("🔎 Loading SQL data...")
        documents.extend(load_sql_documents(sql_config))

    return documents

# ------------------------
# Load Excel Documents
# ------------------------

def load_excel_documents(path: str) -> List[Document]:
    docs = []
    try:
        df = pd.read_excel(path, engine="openpyxl")
        for _, row in df.iterrows():
            content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
            docs.append(Document(page_content=content, metadata={"source": path}))
        print(f"✅ Loaded Excel file: {path}")
    except Exception as e:
        print(f"❌ Error loading Excel {path}: {e}")
    return docs

# ------------------------
# Load SQL Documents
# ------------------------

def load_sql_documents(config: Dict) -> List[Document]:
    docs = []
    try:
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={config['server']};"
            f"DATABASE={config['database']};"
            f"Trusted_Connection=yes;"
        )

        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        for name, query in config["queries"].items():
            cursor.execute(query)
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()

            print(f"✅ Loaded {len(rows)} rows from query: {name}")
            for row in rows:
                content_parts = []
                for col, val in zip(columns, row):
                    if val is not None:
                        col_clean = col.replace('_', ' ').capitalize()
                        content_parts.append(f"{col_clean} is {val}")
                content = ". ".join(content_parts) + "."
                docs.append(Document(page_content=content, metadata={"source": name}))

        conn.close()

    except Exception as e:
        print(f"❌ Error loading SQL: {e}")

    return docs
