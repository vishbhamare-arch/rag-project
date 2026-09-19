# Adapted from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming

import os

import base64
import gc
import random
import tempfile
import time
import uuid

from IPython.display import Markdown, display

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader

import streamlit as st

from token_counter import TokenCounter
from token_tracker import TokenTracker

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.token_tracker = TokenTracker()
    st.session_state.token_counter = TokenCounter()

session_id = st.session_state.id
client = None



@st.cache_resource
def load_llm():
    llm = Ollama(model="llama3.2:1b", request_timeout=120.0)
    return llm

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    st.session_state.token_tracker.reset()
    gc.collect()


def display_pdf(file):
    # Opening file from file path

    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


with st.sidebar:
    st.header("📄 Add your documents!")

    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")

                if file_key not in st.session_state.get('file_cache', {}):

                    if os.path.exists(temp_dir):
                            loader = SimpleDirectoryReader(
                                input_dir = temp_dir,
                                required_exts=[".pdf"],
                                recursive=True
                            )
                    else:    
                        st.error('Could not find the file you uploaded, please check again...')
                        st.stop()
                    
                    docs = loader.load_data()

                    # Count embedding tokens
                    doc_texts = [doc.get_content() for doc in docs]
                    embedding_tokens = st.session_state.token_counter.estimate_embedding_tokens(
                        len(docs),
                        st.session_state.token_counter.get_avg_tokens_per_doc(doc_texts)
                    )
                    st.session_state.token_tracker.add_embedding_tokens(
                        int(embedding_tokens),
                        uploaded_file.name
                    )

                    # setup llm & embedding model
                    llm=load_llm()
                    embed_model = HuggingFaceEmbedding( model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
                    # Creating an index over loaded data
                    Settings.embed_model = embed_model
                    index = VectorStoreIndex.from_documents(docs, show_progress=True)

                    # Create the query engine, where we use a cohere reranker on the fetched nodes
                    Settings.llm = llm
                    query_engine = index.as_query_engine(streaming=True)

                    # ====== Customise prompt template ======
                    qa_prompt_tmpl_str = (
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                    )
                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

                    query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                    )
                    
                    st.session_state.file_cache[file_key] = query_engine
                else:
                    query_engine = st.session_state.file_cache[file_key]

                # Inform the user that the file is processed and Display the PDF uploaded
                st.success("Ready to Chat!")
                display_pdf(uploaded_file)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

    # Display token usage metrics in sidebar
    st.divider()
    st.subheader("📊 Token Usage")
    summary = st.session_state.token_tracker.get_summary()

    col1_metric, col2_metric = st.columns(2)
    with col1_metric:
        st.metric("Total Tokens", summary['total_tokens'])
        st.metric("Embedding Tokens", summary['embedding_tokens'])
    with col2_metric:
        st.metric("Prompt Tokens", summary['prompt_tokens'])
        st.metric("Response Tokens", summary['response_tokens'])

    st.metric("Messages", summary['message_count'])

    if st.session_state.messages:
        with st.expander("📈 Token Breakdown"):
            breakdown = st.session_state.token_tracker.get_per_message_summary()
            for i, item in enumerate(breakdown):
                st.write(f"{item['role'].capitalize()}: {item['tokens']} tokens")

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Chat with Docs using Llama-3.2")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("What's up?"):
    if not st.session_state.file_cache:
        st.error("Please upload a PDF first!")
    else:
        # Count prompt tokens
        prompt_tokens = st.session_state.token_counter.count_tokens(prompt)
        st.session_state.token_tracker.add_prompt_tokens(prompt_tokens, prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Get the cached query engine from the first uploaded file
            query_engine = next(iter(st.session_state.file_cache.values()))
            # Simulate stream of response with milliseconds delay
            streaming_response = query_engine.query(prompt)

            for chunk in streaming_response.response_gen:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            # Count response tokens
            response_tokens = st.session_state.token_counter.count_tokens(full_response)
            st.session_state.token_tracker.add_response_tokens(response_tokens, full_response)
            st.session_state.token_tracker.add_message_tokens(
                message_id=str(len(st.session_state.messages)),
                role="assistant",
                content=full_response,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens
            )

            # full_response = query_engine.query(prompt)

            message_placeholder.markdown(full_response)
            # st.session_state.context = ctx

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})