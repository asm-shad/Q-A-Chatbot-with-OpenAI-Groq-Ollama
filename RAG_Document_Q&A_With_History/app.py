import os

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# Load environment variables
# ============================================================

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


# ============================================================
# Embeddings
# ============================================================

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ============================================================
# Streamlit UI
# ============================================================

st.title("Conversational RAG With PDF Uploads and Chat History")

st.write("Upload PDFs and chat with their content")


# ============================================================
# Groq API Key
# ============================================================

api_key = st.text_input("Enter your Groq API key:", type="password")


if api_key:
    # ========================================================
    # LLM
    # ========================================================

    llm = ChatGroq(api_key=api_key, model="gemma2-9b-it")

    # ========================================================
    # Session ID
    # ========================================================

    session_id = st.text_input("Session ID", value="default_session")

    # ========================================================
    # Chat history storage
    # ========================================================

    if "store" not in st.session_state:
        st.session_state.store = {}

    # ========================================================
    # PDF Upload
    # ========================================================

    uploaded_files = st.file_uploader(
        "Choose PDF files", type="pdf", accept_multiple_files=True
    )

    # ========================================================
    # Process PDFs
    # ========================================================

    if uploaded_files:
        documents = []

        for uploaded_file in uploaded_files:
            temp_pdf = "./temp.pdf"

            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_pdf)

            docs = loader.load()

            documents.extend(docs)

        # ====================================================
        # Split documents
        # ====================================================

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000, chunk_overlap=500
        )

        splits = text_splitter.split_documents(documents)

        # ====================================================
        # Create vector database
        # ====================================================

        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

        # ====================================================
        # Retriever
        # ====================================================

        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # ====================================================
        # STEP 1
        # Understand the user's question using history
        # ====================================================

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Given the chat history and the latest user question,
                    rewrite the latest question so that it can be
                    understood without the chat history.

                    Do NOT answer the question.
                    Only rewrite it when necessary.
                    If it is already clear, return it as it is.
                    """,
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # ====================================================
        # Function to create standalone question
        # ====================================================

        def create_standalone_question(input_question, chat_history):

            prompt_value = contextualize_q_prompt.invoke(
                {"input": input_question, "chat_history": chat_history}
            )

            response = llm.invoke(prompt_value)

            return response.content

        # ====================================================
        # STEP 2
        # Retrieve documents
        # ====================================================

        def retrieve_documents(input_question, chat_history):

            standalone_question = create_standalone_question(
                input_question, chat_history
            )

            documents = retriever.invoke(standalone_question)

            return documents

        # ====================================================
        # STEP 3
        # Format documents
        # ====================================================

        def format_documents(documents):

            return "\n\n".join(document.page_content for document in documents)

        # ====================================================
        # STEP 4
        # Answer the question
        # ====================================================

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are an assistant for question-answering tasks.

                    Use the following retrieved context to answer
                    the question.

                    If you don't know the answer from the context,
                    say that you don't know.

                    Keep the answer concise and use a maximum
                    of three sentences.

                    Retrieved context:
                    {context}
                    """,
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # ====================================================
        # Modern RAG function
        # ====================================================

        def answer_question(input_question, chat_history):

            # -----------------------------------------------
            # Retrieve relevant PDF chunks
            # -----------------------------------------------

            documents = retrieve_documents(input_question, chat_history)

            # -----------------------------------------------
            # Convert documents into text
            # -----------------------------------------------

            context = format_documents(documents)

            # -----------------------------------------------
            # Create final prompt
            # -----------------------------------------------

            prompt_value = qa_prompt.invoke(
                {
                    "context": context,
                    "chat_history": chat_history,
                    "input": input_question,
                }
            )

            # -----------------------------------------------
            # Ask LLM
            # -----------------------------------------------

            response = llm.invoke(prompt_value)

            return {"answer": response.content, "documents": documents}

        # ====================================================
        # Chat history function
        # ====================================================

        def get_session_history(session: str) -> BaseChatMessageHistory:

            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()

            return st.session_state.store[session]

        # ====================================================
        # User question
        # ====================================================

        user_input = st.text_input("Your question:")

        if user_input:
            # -----------------------------------------------
            # Get current conversation history
            # -----------------------------------------------

            session_history = get_session_history(session_id)

            # -----------------------------------------------
            # Generate answer
            # -----------------------------------------------

            result = answer_question(user_input, session_history.messages)

            # -----------------------------------------------
            # Save user message
            # -----------------------------------------------

            session_history.add_user_message(user_input)

            # -----------------------------------------------
            # Save AI message
            # -----------------------------------------------

            session_history.add_ai_message(result["answer"])

            # =================================================
            # Display answer
            # =================================================

            st.subheader("Assistant")

            st.write(result["answer"])

            # =================================================
            # Display chat history
            # =================================================

            with st.expander("Chat History"):
                for message in session_history.messages:
                    st.write(f"**{message.type}:** {message.content}")

            # =================================================
            # Display retrieved documents
            # =================================================

            with st.expander("Retrieved Documents"):
                for i, document in enumerate(result["documents"]):
                    st.write(f"### Document {i + 1}")

                    st.write(document.page_content)

else:
    st.warning("Please enter the Groq API key.")
