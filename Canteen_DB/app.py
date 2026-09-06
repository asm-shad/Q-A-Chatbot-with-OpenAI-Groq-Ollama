import os
from urllib.parse import quote_plus

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from sqlalchemy import create_engine

# ============================================================
# Load environment variables
# ============================================================

load_dotenv()


# ============================================================
# Get credentials from .env
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "db_canteen")


# ============================================================
# Check Groq API key
# ============================================================

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing from your .env file.")

    st.stop()


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="Canteen Database Q&A", page_icon="🍽️")

st.title("🍽️ Canteen Database Q&A")

st.write("Ask questions about the canteen_application table.")


# ============================================================
# Create MySQL connection URL
# ============================================================

encoded_password = quote_plus(MYSQL_PASSWORD)

DATABASE_URL = (
    f"mysql+mysqlconnector://"
    f"{MYSQL_USER}:{encoded_password}"
    f"@{MYSQL_HOST}:{MYSQL_PORT}/"
    f"{MYSQL_DATABASE}"
)


# ============================================================
# Connect to MySQL
# ============================================================


@st.cache_resource
def get_database():

    engine = create_engine(DATABASE_URL)

    db = SQLDatabase(engine)

    return db


# ============================================================
# Get database
# ============================================================

try:
    db = get_database()

except Exception as e:
    st.error("Could not connect to MySQL database.")

    st.error(str(e))

    st.stop()


# ============================================================
# Show database information
# ============================================================

st.success(f"Connected to database: {MYSQL_DATABASE}")


# ============================================================
# Create Groq LLM
# ============================================================

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="qwen/qwen3.8-27b",
    temperature=0,
)


# ============================================================
# Create SQL toolkit
# ============================================================

toolkit = SQLDatabaseToolkit(db=db, llm=llm)


# ============================================================
# Get SQL tools
# ============================================================

tools = toolkit.get_tools()


# ============================================================
# Create modern LangChain agent
# ============================================================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
    You are a helpful SQL database assistant.

    You are connected to a MySQL database.

    The main table the user wants to work with is:
    canteen_application

    Answer questions using the database.

    Before writing SQL, inspect the available tables and
    the schema when necessary.

    Only perform read-only SQL operations.

    Do not INSERT, UPDATE, DELETE, DROP, ALTER, or TRUNCATE
    any data.

    When the user asks for records, return useful information
    clearly.

    When the user asks for a count, calculate the count
    from the database.

    If the requested information does not exist in the database,
    clearly say that you could not find it.

    Do not invent database information.
    """,
)


# ============================================================
# Chat history
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I can answer questions about the canteen_application database."
            ),
        }
    ]


# ============================================================
# Display previous messages
# ============================================================

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])


# ============================================================
# User question
# ============================================================

user_question = st.chat_input("Ask something about canteen_application...")


if user_question:
    # --------------------------------------------------------
    # Display user question
    # --------------------------------------------------------

    st.session_state.messages.append({"role": "user", "content": user_question})

    st.chat_message("user").write(user_question)

    # --------------------------------------------------------
    # Run agent
    # --------------------------------------------------------

    with st.chat_message("assistant"):
        with st.spinner("Checking the database..."):
            try:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": user_question}]}
                )

                # --------------------------------------------
                # Get final message
                # --------------------------------------------

                final_message = result["messages"][-1]

                response = final_message.content

                # --------------------------------------------
                # Display answer
                # --------------------------------------------

                st.write(response)

                # --------------------------------------------
                # Save response
                # --------------------------------------------

                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

            except Exception as e:
                st.error(f"Error while querying database: {e}")
