import validators
import streamlit as st

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader

from youtube_transcript_api import YouTubeTranscriptApi


# ============================================================
# Streamlit App
# ============================================================

st.set_page_config(
    page_title="LangChain: Summarize Text From YT or Website",
    page_icon="🦜"
)

st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")


# ============================================================
# Get Groq API Key
# ============================================================

with st.sidebar:

    groq_api_key = st.text_input(
        "Groq API Key",
        value="",
        type="password"
    )


# ============================================================
# Get URL
# ============================================================

generic_url = st.text_input(
    "URL",
    label_visibility="collapsed"
)


# ============================================================
# Prompt
# ============================================================

prompt_template = """
Provide a concise summary of the following content in 300 words.

Content:
{text}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)


# ============================================================
# Button
# ============================================================

if st.button("Summarize the Content from YT or Website"):

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    if not groq_api_key.strip() or not generic_url.strip():

        st.error(
            "Please provide the Groq API Key and URL."
        )

        st.stop()


    if not validators.url(generic_url):

        st.error(
            "Please enter a valid URL."
        )

        st.stop()


    try:

        with st.spinner("Loading and summarizing..."):

            # =================================================
            # 1. Create Groq model
            # =================================================

            llm = ChatGroq(
                model="qwen/qwen3.8-27b",
                groq_api_key=groq_api_key,
                temperature=0,
                max_tokens=700
            )


            # =================================================
            # 2. Get content
            # =================================================

            if "youtube.com" in generic_url or "youtu.be" in generic_url:

                # ------------------------------------------------
                # Extract YouTube video ID
                # ------------------------------------------------

                if "youtube.com" in generic_url:

                    video_id = generic_url.split("v=")[1].split("&")[0]

                else:

                    video_id = generic_url.split("youtu.be/")[1].split("?")[0]


                # ------------------------------------------------
                # Get YouTube transcript
                # ------------------------------------------------

                ytt_api = YouTubeTranscriptApi()

                transcript = ytt_api.fetch(
                    video_id,
                    languages=["en"]
                )


                # ------------------------------------------------
                # Convert transcript to normal text
                # ------------------------------------------------

                text = "\n".join(
                    snippet.text
                    for snippet in transcript
                )


            else:

                # =================================================
                # Website
                # =================================================

                loader = UnstructuredURLLoader(
                    urls=[generic_url],
                    ssl_verify=False,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 "
                            "(Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 "
                            "(KHTML, like Gecko) "
                            "Chrome/58.0.3029.110 "
                            "Safari/537.3"
                        )
                    }
                )

                docs = loader.load()


                # ------------------------------------------------
                # Convert website documents to text
                # ------------------------------------------------

                text = "\n\n".join(
                    document.page_content
                    for document in docs
                )


            # =================================================
            # 3. Create prompt
            # =================================================

            formatted_prompt = prompt.invoke(
                {
                    "text": text
                }
            )


            # =================================================
            # 4. Send to Groq
            # =================================================

            response = llm.invoke(
                formatted_prompt
            )


            # =================================================
            # 5. Display summary
            # =================================================

            st.success(response.content)


    except Exception as e:

        st.error("Something went wrong.")

        st.exception(e)