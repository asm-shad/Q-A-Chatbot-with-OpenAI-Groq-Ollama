import streamlit as st

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv


# ============================================================
# Load Environment Variables
# ============================================================

load_dotenv()


# ============================================================
# Streamlit App Configuration
# ============================================================

st.set_page_config(
    page_title="Text to Math Problem Solver and Data Search Assistant",
    page_icon="🧮"
)

st.title("🧮 Text to Math Problem Solver Using Groq")


# ============================================================
# Groq API Key
# ============================================================

groq_api_key = st.sidebar.text_input(
    label="Groq API Key",
    type="password"
)

if not groq_api_key:
    st.info("Please add your Groq API key to continue.")
    st.stop()


# ============================================================
# Initialize Groq LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=groq_api_key,
    temperature=0
)


# ============================================================
# Wikipedia Tool
# ============================================================

wikipedia_wrapper = WikipediaAPIWrapper()


@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for information about a topic.

    Use this tool when the user asks for factual information
    that can be found on Wikipedia.
    """

    return wikipedia_wrapper.run(query)


# ============================================================
# Calculator Tool
# ============================================================

@tool
def calculator(expression: str) -> str:
    """
    Calculate mathematical expressions.

    Examples:
    5 + 7
    12 * 25
    (5 - 2) + (7 - 3)
    12 + (2 * 25)
    """

    try:
        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except Exception as e:
        return f"Could not calculate the expression: {e}"


# ============================================================
# Reasoning Tool
# ============================================================

@tool
def reasoning_tool(question: str) -> str:
    """
    Solve logic-based and reasoning questions.

    Provide a clear step-by-step explanation.
    """

    reasoning_prompt = f"""
    Solve the following reasoning problem.

    Think logically and provide a clear,
    step-by-step explanation.

    Question:
    {question}
    """

    response = llm.invoke(reasoning_prompt)

    return response.content


# ============================================================
# Create Modern LangChain Agent
# ============================================================

agent = create_agent(
    model=llm,
    tools=[
        wikipedia_search,
        calculator,
        reasoning_tool
    ],
    system_prompt="""
    You are a helpful math and information assistant.

    You have access to three tools:

    1. Calculator
       Use it for mathematical calculations.

    2. Wikipedia
       Use it when factual information from Wikipedia
       is required.

    3. Reasoning Tool
       Use it for logic-based and reasoning questions.

    For mathematical problems:

    - Understand the problem carefully.
    - Use the calculator when calculations are required.
    - Explain the solution clearly.
    - Break the answer into logical steps.

    Always provide a helpful and understandable final answer.
    """
)


# ============================================================
# Initialize Chat History
# ============================================================

if "messages" not in st.session_state:

    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "Hi! 👋 I'm a math chatbot. "
                "I can solve mathematical problems, "
                "reasoning questions, and search Wikipedia."
            )
        }
    ]


# ============================================================
# Display Chat History
# ============================================================

for msg in st.session_state.messages:

    st.chat_message(
        msg["role"]
    ).write(
        msg["content"]
    )


# ============================================================
# User Question
# ============================================================

question = st.text_area(
    "Enter your question:",
    value=(
        "I have 5 bananas and 7 grapes. "
        "I eat 2 bananas and give away 3 grapes. "
        "Then I buy a dozen apples and 2 packs of blueberries. "
        "Each pack of blueberries contains 25 berries. "
        "How many total pieces of fruit do I have at the end?"
    )
)


# ============================================================
# Find Answer
# ============================================================

if st.button("🔍 Find My Answer"):

    if question.strip():

        with st.spinner("Generating response..."):

            # ------------------------------------------------
            # Add user message to chat history
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            st.chat_message(
                "user"
            ).write(
                question
            )


            # ------------------------------------------------
            # Invoke LangChain Agent
            # ------------------------------------------------

            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": question
                        }
                    ]
                }
            )


            # ------------------------------------------------
            # Get Final AI Response
            # ------------------------------------------------

            response = result["messages"][-1].content


            # ------------------------------------------------
            # Save Assistant Response
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response
                }
            )


            # ------------------------------------------------
            # Display Response
            # ------------------------------------------------

            st.write("### Response:")

            st.success(response)

    else:

        st.warning(
            "Please enter a question."
        )