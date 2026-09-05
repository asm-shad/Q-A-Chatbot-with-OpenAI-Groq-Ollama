import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

# ============================================================
# Load environment variables
# ============================================================

load_dotenv()

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")


# ============================================================
# LangSmith Configuration
# ============================================================

if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with Ollama"


# ============================================================
# Prompt Template
# ============================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant.
Answer the user's question directly and concisely.
Do not show your thinking process, reasoning, analysis, or <think> tags.
Only provide the final answer to the user.""",
        ),
        ("user", "Question: {question}"),
    ]
)


# ============================================================
# Generate Response
# ============================================================


def generate_response(question, model, temperature, max_tokens):

    llm = OllamaLLM(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    response = chain.invoke({"question": question})

    return response


# ============================================================
# Title of the app
# ============================================================

st.title("Q&A Chatbot With Groq")


# ============================================================
# Sidebar for settings
# ============================================================

st.sidebar.title("Settings")


# ============================================================
# Select Groq Model
# ============================================================

engine = st.sidebar.selectbox(
    "Select Groq Model",
    [
        "llama3",
        "gemma2",
        # "openai/gpt-oss-120b",
        # "openai/gpt-oss-20b",
    ],
)


# ============================================================
# Adjust response parameters
# ============================================================

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=1000, value=150)


# ============================================================
# Main interface for user input
# ============================================================

st.write("Go ahead and ask any question")

user_input = st.text_input("You:")


# ============================================================
# Generate Response
# ============================================================

if user_input:
    response = generate_response(user_input, engine, temperature, max_tokens)
    st.write(response)

else:
    st.write("Please provide the user input")
