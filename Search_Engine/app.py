import os

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.tools import (
    ArxivQueryRun,
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
)
from langchain_community.utilities import (
    ArxivAPIWrapper,
    WikipediaAPIWrapper,
)
from langchain_groq import ChatGroq

# ============================================================
# Load environment variables
# ============================================================

load_dotenv()


# ============================================================
# Streamlit UI
# ============================================================

st.title("🔎 LangChain - Chat with Search")

st.write("Ask questions and the agent can search the web, Arxiv, or Wikipedia.")


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("Settings")

api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")


# ============================================================
# Chat history
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi, I'm a chatbot who can search the web. How can I help you?"
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

prompt = st.chat_input("What is machine learning?")


if prompt:
    # ========================================================
    # Check API key
    # ========================================================

    if not api_key:
        st.warning("Please enter your Groq API key.")

        st.stop()

    # ========================================================
    # Add user message to history
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    st.chat_message("user").write(prompt)

    # ========================================================
    # LLM
    # ========================================================

    llm = ChatGroq(
        api_key=api_key,
        model="llama-3.1-8b-instant",
        temperature=0,
    )

    # ========================================================
    # Arxiv tool
    # ========================================================

    arxiv_wrapper = ArxivAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=200,
    )

    arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

    # ========================================================
    # Wikipedia tool
    # ========================================================

    wikipedia_wrapper = WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=200,
    )

    wiki = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

    # ========================================================
    # DuckDuckGo tool
    # ========================================================

    search = DuckDuckGoSearchRun(name="Search")

    # ========================================================
    # Tools
    # ========================================================

    tools = [
        search,
        arxiv,
        wiki,
    ]

    # ========================================================
    # Modern LangChain Agent
    # ========================================================

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful research assistant. "
            "You have access to web search, Arxiv, and Wikipedia. "
            "Use the tools when they are useful. "
            "Give a clear and concise final answer."
        ),
    )

    # ========================================================
    # Run Agent
    # ========================================================

    with st.chat_message("assistant"):
        try:
            result = agent.invoke({"messages": st.session_state.messages})

            # The modern agent returns messages.
            final_message = result["messages"][-1]

            response = final_message.content

            st.write(response)

            # =================================================
            # Save assistant response
            # =================================================

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )

        except Exception as e:
            st.error(f"Error: {e}")
