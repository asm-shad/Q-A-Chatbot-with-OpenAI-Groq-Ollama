import os
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# Load environment variables
# ============================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


# ============================================================
# Check API key
# ============================================================

if not groq_api_key:
    st.error("GROQ_API_KEY is not set in your .env file.")
    st.stop()


# ============================================================
# Initialize Groq LLM
# ============================================================

llm = ChatGroq(
    api_key=groq_api_key,
    model="qwen/qwen3.6-27b",
)


# ============================================================
# Prompt
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the provided context.

    If the answer is not available in the context,
    say that you don't know based on the provided documents.

    Be accurate and concise.

    <context>
    {context}
    </context>

    Question:
    {question}
    """
)


# ============================================================
# Clean LLM Response
# ============================================================


def clean_response(response):
    """
    Remove Qwen's <think>...</think> section
    and return only the final answer.
    """

    answer = response.content

    if "<think>" in answer and "</think>" in answer:
        answer = answer.split("</think>", 1)[1].strip()

    return answer


# ============================================================
# Create Vector Database
# ============================================================


def create_vector_embedding():
    """
    Load PDF files, split them into chunks,
    create embeddings and store them in FAISS.
    """

    # Only create the vector database once
    if "vectors" not in st.session_state:
        # ----------------------------------------------------
        # 1. Create Ollama embedding model
        # ----------------------------------------------------

        st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # ----------------------------------------------------
        # 2. Load PDF documents
        # ----------------------------------------------------

        st.session_state.loader = PyPDFDirectoryLoader("research_papers")

        st.session_state.docs = st.session_state.loader.load()

        # ----------------------------------------------------
        # Check whether PDFs were found
        # ----------------------------------------------------

        if not st.session_state.docs:
            st.error("No PDF documents were found in the 'research_papers' folder.")
            return False

        # ----------------------------------------------------
        # 3. Split documents into chunks
        # ----------------------------------------------------

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        st.session_state.final_documents = (
            st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        )

        # ----------------------------------------------------
        # 4. Create FAISS vector database
        # ----------------------------------------------------

        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings,
        )

        return True

    return True


# ============================================================
# Streamlit UI
# ============================================================

st.title("📚 Research Paper Q&A")

st.write(
    "Ask questions about the research papers stored in the `research_papers` folder."
)


# ============================================================
# Create Vector Database Button
# ============================================================

if st.button("Document Embedding"):
    with st.spinner("Loading documents and creating vector database..."):
        success = create_vector_embedding()

    if success:
        st.success("Vector Database is ready.")


# ============================================================
# User Question
# ============================================================

user_prompt = st.text_input("Enter your query from the research paper:")


# ============================================================
# RAG Question Answering
# ============================================================

if user_prompt:
    # --------------------------------------------------------
    # Check whether vector database exists
    # --------------------------------------------------------

    if "vectors" not in st.session_state:
        st.warning("Please click 'Document Embedding' first.")

        st.stop()

    # --------------------------------------------------------
    # Create retriever
    # --------------------------------------------------------

    retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 4})

    # --------------------------------------------------------
    # Retrieve relevant documents
    # --------------------------------------------------------

    start = time.perf_counter()

    retrieved_documents = retriever.invoke(user_prompt)

    retrieval_time = time.perf_counter() - start

    # --------------------------------------------------------
    # Combine retrieved documents into context
    # --------------------------------------------------------

    context = "\n\n".join(document.page_content for document in retrieved_documents)

    # --------------------------------------------------------
    # Create prompt
    # --------------------------------------------------------

    formatted_prompt = prompt.invoke(
        {
            "context": context,
            "question": user_prompt,
        }
    )

    # --------------------------------------------------------
    # Send prompt to Groq
    # --------------------------------------------------------

    start = time.perf_counter()

    response = llm.invoke(formatted_prompt)

    response_time = time.perf_counter() - start

    # --------------------------------------------------------
    # Clean response
    # --------------------------------------------------------

    answer = clean_response(response)

    # --------------------------------------------------------
    # Display answer
    # --------------------------------------------------------

    st.subheader("Answer")

    st.write(answer)

    # --------------------------------------------------------
    # Display timing
    # --------------------------------------------------------

    st.caption(f"Retrieval time: {retrieval_time:.2f} seconds")

    st.caption(f"LLM response time: {response_time:.2f} seconds")

    # --------------------------------------------------------
    # Display retrieved documents
    # --------------------------------------------------------

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(retrieved_documents):
            st.write(f"### Document {i + 1}")

            st.write(doc.page_content)

            st.write("-----------------------------")
