"""Streamlit app for Ask My PDF Bot."""

import streamlit as st

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    GEMINI_MODEL_NAME,
    TOP_K_RESULTS,
    create_project_directories,
    is_api_key_configured,
)
from embeddings_utils import load_vector_store, vector_store_exists
from pdf_utils import list_uploaded_pdf_names, process_uploaded_files
from rag_pipeline import answer_question, build_vector_store_from_pdfs


st.set_page_config(
    page_title="Ask My PDF Bot",
    page_icon="PDF",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state() -> None:
    """Create Streamlit session state variables used by the app."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

    if "last_stats" not in st.session_state:
        st.session_state.last_stats = {"pages": 0, "chunks": 0}

    if "vector_store_load_error" not in st.session_state:
        st.session_state.vector_store_load_error = ""


def load_existing_vector_store() -> None:
    """Load a saved FAISS vector store if one exists."""
    if st.session_state.vector_store is None and vector_store_exists():
        try:
            vector_store = load_vector_store()
            st.session_state.vector_store = vector_store

            documents = list(vector_store.docstore._dict.values())
            sources = sorted(
                {
                    document.metadata.get("source", "Unknown PDF")
                    for document in documents
                }
            )
            pages = {
                (document.metadata.get("source"), document.metadata.get("page"))
                for document in documents
            }

            st.session_state.processed_files = sources
            st.session_state.last_stats = {
                "pages": len(pages),
                "chunks": len(documents),
            }
            st.session_state.vector_store_load_error = ""
        except Exception as exc:
            st.session_state.vector_store_load_error = (
                "Saved vector store exists, but Streamlit could not load it. "
                f"Restart the app after installing/loading the embedding model. Details: {exc}"
            )
            st.session_state.vector_store = None


def render_sidebar() -> None:
    """Render sidebar with status, uploaded files, and project info."""
    with st.sidebar:
        st.header("Project Info")
        st.write("**Ask My PDF Bot**")
        st.caption("AI-powered RAG chatbot using Streamlit, LangChain, FAISS, and the Google AI Studio Gemini API.")

        st.divider()
        st.subheader("System Status")

        api_status = "Ready" if is_api_key_configured() else "Missing API key"
        vector_status = "Ready" if st.session_state.vector_store is not None else "Not created"

        st.metric("Gemini API", api_status)
        st.metric("Vector Store", vector_status)
        st.metric("Chunk Size", CHUNK_SIZE)
        st.metric("Top-K Retrieval", TOP_K_RESULTS)

        if st.session_state.vector_store_load_error:
            st.warning(st.session_state.vector_store_load_error)

        if st.session_state.vector_store is None and vector_store_exists():
            if st.button("Load Saved Vector Store", use_container_width=True):
                load_existing_vector_store()
                st.rerun()

        st.divider()
        st.subheader("Uploaded Files")
        uploaded_names = list_uploaded_pdf_names()

        if uploaded_names:
            for file_name in uploaded_names:
                st.write(f"- {file_name}")
        else:
            st.caption("No PDFs uploaded yet.")

        st.divider()
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        with st.expander("About this project"):
            st.write(
                "This internship-level project demonstrates a complete RAG workflow: "
                "PDF upload, text extraction, chunking, embeddings, FAISS retrieval, "
                "Gemini answer generation, and source citation display."
            )


def render_header() -> None:
    """Render main title and dashboard metrics."""
    st.title("Ask My PDF Bot")
    st.markdown(
        "Upload one or more PDF documents and ask natural language questions. "
        "The bot retrieves relevant chunks and answers using the Google AI Studio Gemini API."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Processed PDFs", len(st.session_state.processed_files))
    col2.metric("Readable Pages", st.session_state.last_stats["pages"])
    col3.metric("Text Chunks", st.session_state.last_stats["chunks"])


def render_pdf_upload_section() -> None:
    """Render PDF uploader and processing controls."""
    st.subheader("1. Upload PDF Documents")

    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload academic notes, reports, manuals, resumes, or any text-based PDF.",
    )

    process_button = st.button(
        "Process PDFs and Build Vector Store",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    if process_button:
        with st.spinner("Extracting text, creating chunks, generating embeddings, and saving FAISS index..."):
            saved_paths, upload_errors = process_uploaded_files(uploaded_files)

            for error in upload_errors:
                st.error(error)

            if not saved_paths:
                st.warning("No valid PDF files were available for processing.")
                return

            try:
                vector_store, total_pages, total_chunks = build_vector_store_from_pdfs(saved_paths)
                st.session_state.vector_store = vector_store
                st.session_state.processed_files = [path.name for path in saved_paths]
                st.session_state.last_stats = {
                    "pages": total_pages,
                    "chunks": total_chunks,
                }
                st.success("PDF processing completed successfully. You can now ask questions.")
            except Exception as exc:
                st.error(f"Processing failed: {exc}")


def render_chat_history() -> None:
    """Display chat messages stored in Streamlit session state."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                render_sources(message["sources"])


def render_sources(sources) -> None:
    """Display retrieved source chunks below an answer."""
    with st.expander("View source chunks used for this answer"):
        for index, source in enumerate(sources, start=1):
            metadata = source.metadata
            source_name = metadata.get("source", "Unknown source")
            page_number = metadata.get("page", "Unknown page")
            chunk_number = metadata.get("chunk", index)
            preview = source.page_content[:700].strip()

            st.markdown(f"**Source {index}: {source_name} | Page {page_number} | Chunk {chunk_number}**")
            st.write(preview + ("..." if len(source.page_content) > 700 else ""))
            st.divider()


def render_chat_interface() -> None:
    """Render the chatbot interface and answer questions."""
    st.subheader("2. Chat With Your PDF")

    if st.session_state.vector_store is None:
        st.info("Upload and process at least one PDF before asking questions.")

    render_chat_history()

    question = st.chat_input("Ask a question about your uploaded PDF...")

    if question is None:
        return

    if not question.strip():
        st.warning("Please enter a valid question.")
        return

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    if st.session_state.vector_store is None:
        message = "Please upload and process a PDF before asking questions."
        st.session_state.messages.append({"role": "assistant", "content": message, "sources": []})
        with st.chat_message("assistant"):
            st.warning(message)
        return

    with st.chat_message("assistant"):
        with st.spinner("Searching the PDF and generating an answer..."):
            try:
                answer, sources = answer_question(st.session_state.vector_store, question)
                st.markdown(answer)
                render_sources(sources)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )
            except Exception as exc:
                error_message = f"Sorry, I could not answer that question. {exc}"
                st.error(error_message)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )


def render_help_section() -> None:
    """Render small help section for internship demo clarity."""
    with st.expander("How this RAG demo works"):
        st.write(
            "1. PDFs are saved locally.\n"
            "2. PyPDF2 extracts page text.\n"
            "3. LangChain splits the text into overlapping chunks.\n"
            "4. Sentence-transformers creates embeddings.\n"
            "5. FAISS stores and searches the vector index.\n"
            "6. Gemini receives the retrieved context and user question.\n"
            "7. The answer and source chunks are displayed together."
        )


def main() -> None:
    """Main Streamlit app entry point."""
    create_project_directories()
    initialize_session_state()
    load_existing_vector_store()

    render_sidebar()
    render_header()

    upload_column, chat_column = st.columns([1, 1.4], gap="large")

    with upload_column:
        render_pdf_upload_section()
        render_help_section()

    with chat_column:
        render_chat_interface()

    st.caption(
        f"Using {GEMINI_MODEL_NAME}, FAISS, chunk size {CHUNK_SIZE}, "
        f"overlap {CHUNK_OVERLAP}, and top-{TOP_K_RESULTS} retrieval."
    )


if __name__ == "__main__":
    main()
