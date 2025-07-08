from operator import itemgetter

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import format_document
from langchain.prompts.prompt import PromptTemplate

from langchain_community.llms import Ollama
from langchain_experimental.sql import SQLDatabaseChain

from langchain_community.utilities import SQLDatabase
import re
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import SQLDatabase



condense_question = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question)

answer = """
### Instruction:
You're a helpful research assistant, who answers questions based on provided research in a clear way and easy-to-understand way.
If there is no research, or the research is irrelevant to answering the question, simply reply that you can't answer.
Please reply with just the detailed answer and your sources. If you're unable to answer the question, do not list sources

## Research:
{context}

## Question:
{question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(answer)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
    template="Source Document: {source}, Page {page}:\n{page_content}"
)


def get_sql_chain(model_name: str, sql_config: dict):
    conn_str = (
        f"mssql+pyodbc://{sql_config['server']}/{sql_config['database']}"
        "?driver=ODBC+Driver+17+for+SQL+Server"
        "&trusted_connection=yes"
    )
    db = SQLDatabase.from_uri(conn_str)
    llm = Ollama(model=model_name)

    return SQLDatabaseChain.from_llm(
        llm=llm,
        db=db,
        verbose=True,
        return_intermediate_steps=True
    )

# def _combine_documents(
#     docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
# ):
#     doc_strings = [format_document(doc, document_prompt) for doc in docs]
#     return document_separator.join(doc_strings)

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    # Patch missing metadata
    for doc in docs:
        doc.metadata.setdefault("page", "N/A")
        doc.metadata.setdefault("source", "unknown")

    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)



memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")

def clean_sql(query: str) -> str:
    import re

    query = re.sub(r"```(?:sql)?\s*", "", query, flags=re.IGNORECASE).replace("```", "").strip()

    # Convert LIMIT to TOP
    match = re.search(r"LIMIT\s+(\d+)", query, re.IGNORECASE)
    if match:
        limit_value = match.group(1)
        query = re.sub(r"SELECT", f"SELECT TOP {limit_value}", query, count=1, flags=re.IGNORECASE)
        query = re.sub(r"LIMIT\s+\d+", "", query, flags=re.IGNORECASE)

    return query.strip()


# def clean_sql(query: str) -> str:
#     import re
#     # Remove triple backticks (```sql or ```) and any leading/trailing whitespace
#     query = re.sub(r"```(?:sql)?\s*", "", query, flags=re.IGNORECASE)
#     query = query.replace("```", "").strip()
    
#     # Replace MySQL-style field quotes with SQL Server style
#     query = re.sub(r"`(.*?)`", r"[\1]", query)

#     return query

def getStreamingChain(question: str, memory, llm, db):
    # ✅ Simple SQL intent detection
    sql_keywords = ["list", "employee", "department", "joined", "show", "email", "from table", "salary", "project", "select", "where"]
    is_sql_question = any(word in question.lower() for word in sql_keywords)

    if is_sql_question:
        # ✅ Build SQL connection URI
        sql_config = {
            "server": "localhost",
            "database": "EmployeeManagementDB",
            "trusted_connection": True,
        }
        conn_str = (
            f"mssql+pyodbc://{sql_config['server']}/{sql_config['database']}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
        )
        print("🖥️ SQL Server connection:", conn_str)
        db_sql = SQLDatabase.from_uri(conn_str)

        # ✅ Create SQL Agent with context tracking
        agent = create_sql_agent(
            llm=Ollama(model=llm.model),
            toolkit=SQLDatabaseToolkit(db=db_sql, llm=llm),
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

        cleaned_query = clean_sql(question)
        print("🧾 Final cleaned SQL:\n", cleaned_query)
        # result = agent.invoke({"input": cleaned_query})
        # result = agent.invoke(cleaned_query, handle_parsing_errors=True)
        result = agent.invoke({"input": cleaned_query}, handle_parsing_errors=True)


        # result = agent.invoke(cleaned_query)
        yield result
        return

    # ✅ Fallback to RAG for non-SQL questions
    retriever = db.as_retriever(search_kwargs={"k": 10})

    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(
            lambda x: "\n".join(
                [f"{item['role']}: {item['content']}" for item in x["memory"]]
            )
        ),
    )

    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
        | CONDENSE_QUESTION_PROMPT
        | llm
        | (lambda x: x.content if hasattr(x, "content") else x)
    }

    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }

    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }

    answer = final_inputs | ANSWER_PROMPT | llm
    final_chain = loaded_memory | standalone_question | retrieved_documents | answer

    yield from final_chain.stream({"question": question, "memory": memory})


# def getStreamingChain(question: str, memory, llm, db):
#     # ✅ Simple SQL intent detection (replace with LLM-based if needed)
#     sql_keywords = ["list", "employee", "department", "joined", "show", "email", "from table"]
#     is_sql_question = any(word in question.lower() for word in sql_keywords)

#     if is_sql_question:
#         # ✅ Define your SQL config
#         sql_config = {
#             "server": "localhost",
#             "database": "test_llm",
#             "trusted_connection": True,
#         }
#         # ✅ Create SQLAlchemy-compatible URI
#         conn_str = (
#             f"mssql+pyodbc://{sql_config['server']}/{sql_config['database']}"
#             "?driver=ODBC+Driver+17+for+SQL+Server"
#             "&trusted_connection=yes"
#         )
#         print("Mukund connection string is ", conn_str)
#         db_sql = SQLDatabase.from_uri(conn_str)
#         sql_chain = SQLDatabaseChain.from_llm(
#             llm=Ollama(model=llm.model),  # Use same LLM as chat
#             db=db_sql,
#             verbose=True,
#             return_intermediate_steps=True,
#         )

#         cleaned_question = clean_sql(question)

#         print("🧾 Final cleaned SQL:\n", cleaned_question)

#         output = sql_chain.invoke({"query": cleaned_question})

#         # # Clean any LLM-wrapped markdown SQL code
#         # cleaned_question = re.sub(r"```(?:sql)?\s*", "", question, flags=re.IGNORECASE).replace("```", "").strip()

#         # # Now use the cleaned query
#         # output = sql_chain.invoke({"query": cleaned_question})

#         # output = sql_chain.invoke({"query": question})
#         result = output["result"]

#         print("🧠 SQL used:", output["intermediate_steps"])

#         yield result
#         return  # Important: stop here

#     # ✅ Otherwise: run your default RAG chain
#     retriever = db.as_retriever(search_kwargs={"k": 10})

#     loaded_memory = RunnablePassthrough.assign(
#         chat_history=RunnableLambda(
#             lambda x: "\n".join(
#                 [f"{item['role']}: {item['content']}" for item in x["memory"]]
#             )
#         ),
#     )

#     standalone_question = {
#         "standalone_question": {
#             "question": lambda x: x["question"],
#             "chat_history": lambda x: x["chat_history"],
#         }
#         | CONDENSE_QUESTION_PROMPT
#         | llm
#         | (lambda x: x.content if hasattr(x, "content") else x)
#     }

#     retrieved_documents = {
#         "docs": itemgetter("standalone_question") | retriever,
#         "question": lambda x: x["standalone_question"],
#     }

#     final_inputs = {
#         "context": lambda x: _combine_documents(x["docs"]),
#         "question": itemgetter("question"),
#     }

#     answer = final_inputs | ANSWER_PROMPT | llm
#     final_chain = loaded_memory | standalone_question | retrieved_documents | answer

#     yield from final_chain.stream({"question": question, "memory": memory})

def getChatChain(llm, db):
    retriever = db.as_retriever(search_kwargs={"k": 10})

    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables)
        | itemgetter("history"),
    )

    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | CONDENSE_QUESTION_PROMPT
        | llm
        | (lambda x: x.content if hasattr(x, "content") else x)
    }

    # Now we retrieve the documents
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }

    # Now we construct the inputs for the final prompt
    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }

    # And finally, we do the part that returns the answers
    answer = {
        "answer": final_inputs
        | ANSWER_PROMPT
        | llm.with_config(callbacks=[StreamingStdOutCallbackHandler()]),
        "docs": itemgetter("docs"),
    }

    final_chain = loaded_memory | standalone_question | retrieved_documents | answer

    def chat(question: str):
        inputs = {"question": question}
        result = final_chain.invoke(inputs)
        memory.save_context(inputs, {"answer": result["answer"].content if hasattr(result["answer"], "content") else result["answer"]})

    return chat
