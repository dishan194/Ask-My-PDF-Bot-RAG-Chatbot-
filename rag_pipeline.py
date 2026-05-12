"""Core RAG pipeline for Ask My PDF Bot."""

import ssl
from typing import Dict, List, Tuple

from google import genai
from google.genai import types
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    GEMINI_MODEL_NAME,
    TOP_K_RESULTS,
    get_api_key,
    is_api_key_configured,
)
from embeddings_utils import create_faiss_vector_store, save_vector_store
from pdf_utils import extract_text_from_pdf


def create_documents_from_pages(page_metadata: List[Dict]) -> List[Document]:
    """Convert extracted page metadata into LangChain Document objects."""
    documents: List[Document] = []

    for page in page_metadata:
        documents.append(
            Document(
                page_content=page["text"],
                metadata={
                    "source": page["source"],
                    "page": page["page"],
                },
            )
        )

    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Split PDF page documents into smaller overlapping chunks."""
    if not documents:
        raise ValueError("No document text is available for chunking.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)

    # Add a simple chunk number to metadata for source display.
    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk"] = index

    return chunks


def build_vector_store_from_pdfs(pdf_paths: List) -> Tuple[object, int, int]:
    """Run extraction, chunking, embedding, and FAISS storage for PDFs.

    Returns:
        vector_store: FAISS vector store object.
        total_pages: Number of extracted PDF pages with readable text.
        total_chunks: Number of chunks embedded.
    """
    all_page_metadata: List[Dict] = []

    for pdf_path in pdf_paths:
        _text, page_metadata = extract_text_from_pdf(pdf_path)
        all_page_metadata.extend(page_metadata)

    documents = create_documents_from_pages(all_page_metadata)
    chunks = split_documents(documents)
    vector_store = create_faiss_vector_store(chunks)
    save_vector_store(vector_store)

    return vector_store, len(all_page_metadata), len(chunks)


def retrieve_relevant_chunks(vector_store, question: str, top_k: int = TOP_K_RESULTS) -> List[Document]:
    """Retrieve top relevant chunks for a user question."""
    if not question.strip():
        raise ValueError("Please enter a question before searching.")

    try:
        return vector_store.similarity_search(question, k=top_k)
    except Exception as exc:
        raise RuntimeError("Could not retrieve relevant chunks from FAISS.") from exc


def format_context(chunks: List[Document]) -> str:
    """Format retrieved chunks into readable context for Gemini."""
    context_parts = []

    for index, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "Unknown source")
        page = chunk.metadata.get("page", "Unknown page")
        context_parts.append(
            f"[Source {index}: {source}, page {page}]\n{chunk.page_content}"
        )

    return "\n\n".join(context_parts)


def create_ssl_context() -> ssl.SSLContext:
    """Create an SSL context compatible with Windows certificate bundles.

    Some Windows/Python 3.13 setups include a local certificate whose Basic
    Constraints extension is not marked critical. OpenSSL 3 rejects that in
    strict mode before Gemini can receive the request. This keeps certificate
    verification enabled while relaxing only that strict OpenSSL flag.
    """
    context = ssl.create_default_context()

    strict_flag = getattr(ssl, "VERIFY_X509_STRICT", None)
    if strict_flag is not None:
        context.verify_flags &= ~strict_flag

    return context


def generate_answer_with_gemini(question: str, context: str) -> str:
    """Generate an answer using Google Gemini and retrieved PDF context."""
    if not is_api_key_configured():
        raise ValueError(
            "Google AI Studio API key is missing. Add GEMINI_API_KEY to your .env file."
        )

    try:
        prompt = f"""
You are Ask My PDF Bot, a helpful AI assistant for answering questions from PDF documents.

Use ONLY the context below to answer the user's question.
If the answer is not available in the context, say:
"I could not find this information in the uploaded PDF."

Keep the answer clear, beginner-friendly, and professional.

Context:
{context}

Question:
{question}

Answer:
"""

        client = genai.Client(
            api_key=get_api_key(),
            http_options=types.HttpOptions(
                client_args={"verify": create_ssl_context()},
            ),
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
        )

        if not response or not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        return response.text.strip()

    except Exception as exc:
        error_text = str(exc)
        if "API key" in error_text or "API_KEY" in error_text or "400" in error_text:
            raise RuntimeError(
                "Gemini rejected the API request. Please paste a real Google AI Studio "
                "key into `.env` as GEMINI_API_KEY, save the file, and restart Streamlit."
            ) from exc

        raise RuntimeError(
            f"Gemini API request failed: {error_text}"
        ) from exc


def answer_question(vector_store, question: str) -> Tuple[str, List[Document]]:
    """Complete query-time RAG flow: retrieve chunks and generate answer."""
    chunks = retrieve_relevant_chunks(vector_store, question)

    if not chunks:
        return "I could not find relevant information in the uploaded PDF.", []

    context = format_context(chunks)
    answer = generate_answer_with_gemini(question, context)
    return answer, chunks
