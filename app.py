"""
app.py
------
Streamlit chat interface for the Salesforce AI Assistant.

Run with:
    streamlit run app.py
"""

import streamlit as st
from src.rag import SalesforceRAG

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Salesforce AI Assistant",
    page_icon="☁️",
    layout="centered",
)

st.title("☁️ Salesforce AI Assistant")
st.caption("Ask me anything about Salesforce — powered by RAG + Claude")


# ---------------------------------------------------------------------------
# Initialize the RAG pipeline (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing AI pipeline...")
def get_rag() -> SalesforceRAG:
    """
    Create and return the RAG pipeline.
    Using @st.cache_resource means this runs once and is reused across
    all Streamlit reruns (i.e., every time the user interacts with the page).
    """
    return SalesforceRAG()


rag = get_rag()


# ---------------------------------------------------------------------------
# Sidebar — document management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Document Management")

    doc_count = len(rag.vector_store._docs)
    if doc_count > 0:
        st.success(f"{doc_count} chunks loaded in vector store")
    else:
        st.warning("No documents loaded yet. Click below to load docs.")

    if st.button("Load / Reload Salesforce Docs", use_container_width=True):
        with st.spinner("Fetching and indexing Salesforce documentation..."):
            try:
                count = rag.load_documents(force_reload=True)
                st.success(f"Loaded {count} chunks successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading documents: {e}")

    st.divider()
    st.subheader("Add a custom URL")
    custom_url = st.text_input("Paste a Salesforce doc URL:")
    if st.button("Ingest URL", use_container_width=True) and custom_url:
        with st.spinner(f"Loading {custom_url}..."):
            try:
                from src.ingestion import ingest_urls, chunk_documents
                docs = ingest_urls([custom_url])
                chunks = chunk_documents(docs)
                rag.vector_store.add_documents(chunks)
                st.success(f"Added {len(chunks)} chunks from URL!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

    st.divider()
    st.markdown(
        "**Tips:**\n"
        "- Load docs once; they persist between sessions.\n"
        "- Add your own Salesforce docs via URL.\n"
        "- Put `.txt` files in `data/` for local content."
    )


# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------

# Initialize chat history in session state (persists across reruns)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input box — Streamlit pauses here until the user submits
if prompt := st.chat_input("Ask a Salesforce question..."):

    # Check that documents are loaded before answering
    if rag.vector_store.is_empty():
        st.warning(
            "Please load Salesforce documents first using the sidebar button."
        )
        st.stop()

    # Show the user's message in the chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # Append to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Stream Claude's response
    with st.chat_message("assistant"):
        # st.write_stream consumes our generator and displays tokens as they arrive
        full_response = st.write_stream(rag.query(prompt))

    # Save the completed response to history
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )

    # Optionally show the retrieved sources in an expander
    with st.expander("View sources used for this answer"):
        sources = rag.get_sources(prompt)
        for i, chunk in enumerate(sources, 1):
            st.markdown(f"**Source {i}** — `{chunk['source']}` (score: {chunk['score']:.2f})")
            st.text(chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"])
            st.divider()
