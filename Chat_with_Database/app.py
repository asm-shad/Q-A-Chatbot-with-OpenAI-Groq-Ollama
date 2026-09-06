import os
import sqlite3
from pathlib import Path

import streamlit as st
from langchain_community.agent_toolkits import (
    SQLDatabaseToolkit,
    create_sql_agent,
)
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from sqlalchemy import create_engine

# ============================================================
# Streamlit configuration
# ============================================================

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")

st.title("🦜 LangChain: Chat with SQL DB")


# ============================================================
# Database options
# ============================================================

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = ["Use SQLite 3 Database - Student.db", "Connect to your MySQL Database"]

selected_opt = st.sidebar.radio(label="Choose the database", options=radio_opt)


# ============================================================
# Database configuration
# ============================================================

if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL

    mysql_host = st.sidebar.text_input("Provide MySQL Host")

    mysql_user = st.sidebar.text_input("MySQL User")

    mysql_password = st.sidebar.text_input("MySQL Password", type="password")

    mysql_db = st.sidebar.text_input("MySQL Database")

else:
    db_uri = LOCALDB


# ============================================================
# Groq API key
# ============================================================

api_key = st.sidebar.text_input("Groq API Key", type="password")


if not api_key:
    st.info("Please enter your Groq API key.")

    st.stop()


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    api_key=api_key,
    model="llama-3.1-8b-instant",
    temperature=0,
)


# ============================================================
# Configure database
# ============================================================


@st.cache_resource(ttl="2h")
def configure_db(
    db_uri,
    mysql_host=None,
    mysql_user=None,
    mysql_password=None,
    mysql_db=None,
):

    # --------------------------------------------------------
    # SQLite
    # --------------------------------------------------------

    if db_uri == LOCALDB:
        dbfilepath = (Path(__file__).parent / "student.db").absolute()

        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)

        engine = create_engine("sqlite:///", creator=creator)

        return SQLDatabase(engine)

    # --------------------------------------------------------
    # MySQL
    # --------------------------------------------------------

    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")

            st.stop()

        mysql_uri = (
            f"mysql+mysqlconnector://"
            f"{mysql_user}:{mysql_password}"
            f"@{mysql_host}/{mysql_db}"
        )

        engine = create_engine(mysql_uri)

        return SQLDatabase(engine)


# ============================================================
# Create database connection
# ============================================================

if db_uri == MYSQL:
    db = configure_db(
        db_uri,
        mysql_host,
        mysql_user,
        mysql_password,
        mysql_db,
    )

else:
    db = configure_db(db_uri)


# ============================================================
# SQL Toolkit
# ============================================================

toolkit = SQLDatabaseToolkit(db=db, llm=llm)


# ============================================================
# Create SQL Agent
# ============================================================

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    agent_type="tool-calling",
    verbose=True,
    agent_executor_kwargs={"handle_parsing_errors": True},
)


# ============================================================
# Chat history
# ============================================================

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you?"}
    ]


# ============================================================
# Display chat history
# ============================================================

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# ============================================================
# User question
# ============================================================

user_query = st.chat_input("Ask anything from the database")


if user_query:
    # --------------------------------------------------------
    # Save user question
    # --------------------------------------------------------

    st.session_state.messages.append({"role": "user", "content": user_query})

    st.chat_message("user").write(user_query)

    # --------------------------------------------------------
    # Run SQL agent
    # --------------------------------------------------------

    with st.chat_message("assistant"):
        try:
            result = agent.invoke({"input": user_query})

            response = result["output"]

            st.write(response)

            # ------------------------------------------------
            # Save assistant response
            # ------------------------------------------------

            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"Error: {e}")
