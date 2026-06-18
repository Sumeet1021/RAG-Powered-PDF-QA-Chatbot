import streamlit as st
import os

from utils import (
    load_pdf,
    split_text,
    create_vector_store
)

from rag import generate_answer

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

# -----------------------------------
# SESSION STATE
# -----------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------------
# SIDEBAR
# -----------------------------------

with st.sidebar:

    st.title("🤖 RAG PDF Chatbot")

    st.markdown("""
### Tech Stack

- LangChain
- FAISS
- Hugging Face Embeddings
- Groq API
- Llama 3.1
- Streamlit

### Features

✅ Multi-PDF Upload

✅ Semantic Search

✅ Source Citations

✅ Chat History

✅ LLM Question Answering
""")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# -----------------------------------
# MAIN TITLE
# -----------------------------------

st.title("📄 RAG-Powered PDF Chatbot")

# -----------------------------------
# FILE UPLOAD
# -----------------------------------

uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# -----------------------------------
# PDF PROCESSING
# -----------------------------------

if uploaded_files:

    all_docs = []

    for uploaded_file in uploaded_files:

        temp_path = f"temp_{uploaded_file.name}"

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        docs = load_pdf(temp_path)

        all_docs.extend(docs)

    chunks = split_text(all_docs)

    db = create_vector_store(chunks)

    st.success("PDFs processed successfully!")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Pages Loaded",
            len(all_docs)
        )

    with col2:
        st.metric(
            "Chunks Created",
            len(chunks)
        )

    # -----------------------------------
    # DISPLAY CHAT HISTORY
    # -----------------------------------

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # -----------------------------------
    # CHAT INPUT
    # -----------------------------------

    query = st.chat_input(
        "Ask a question about your PDFs..."
    )

    if query:

        # USER MESSAGE

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        with st.chat_message("user"):
            st.markdown(query)

        # RETRIEVE + GENERATE

        with st.spinner("Generating Answer..."):

            retrieved_docs = db.similarity_search(
                query,
                k=3
            )

            answer = generate_answer(
                query,
                retrieved_docs
            )

        # ASSISTANT RESPONSE

        with st.chat_message("assistant"):

            st.markdown(answer)

            st.markdown("---")

            st.subheader("📚 Sources")

            for i, doc in enumerate(retrieved_docs):

                page_num = doc.metadata.get(
                    "page",
                    "Unknown"
                )

                source_name = doc.metadata.get(
                    "source",
                    "PDF"
                )

                filename = os.path.basename(
                    source_name
                )

                filename = filename.replace(
                    "temp_",
                    ""
                )

                with st.expander(
                    f"Source {i+1} | Page {page_num + 1}"
                ):

                    st.write(
                        f"📄 File: {filename}"
                    )

                    st.write(
                        doc.page_content[:1000]
                    )

        # SAVE CHAT HISTORY

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )