"""Embedding and FAISS vector store helper functions."""

import os

from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, VECTOR_STORE_DIR


# Avoid a noisy Windows-only warning from Hugging Face cache symlink behavior.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


class MiniLMEmbeddings(Embeddings):
    """LangChain-compatible wrapper around sentence-transformers MiniLM."""

    def __init__(self, model_name: str, local_files_only: bool = False) -> None:
        self.model = SentenceTransformer(
            model_name,
            device="cpu",
            local_files_only=local_files_only,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for document chunks."""
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Create an embedding for a user question."""
        vector = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()


def get_embedding_model() -> MiniLMEmbeddings:
    """Load the sentence-transformers embedding model.

    The model downloads on the first run and is reused locally afterwards.
    """
    try:
        return MiniLMEmbeddings(EMBEDDING_MODEL_NAME, local_files_only=True)
    except Exception as local_exc:
        try:
            return MiniLMEmbeddings(EMBEDDING_MODEL_NAME, local_files_only=False)
        except Exception as online_exc:
            raise RuntimeError(
                "Could not load the embedding model. The app first tried the local "
                "Hugging Face cache, then tried downloading it online. If this is a "
                "Windows SSL certificate issue, run `pip install python-certifi-win32` "
                "in the same Python environment used by Streamlit, then restart the app. "
                f"Local error: {local_exc}. Online error: {online_exc}"
            ) from online_exc


def create_faiss_vector_store(documents: List[Document]) -> FAISS:
    """Create a FAISS vector store from LangChain Document chunks."""
    if not documents:
        raise ValueError("No text chunks were provided for embedding.")

    try:
        embeddings = get_embedding_model()
        return FAISS.from_documents(documents, embeddings)
    except Exception as exc:
        raise RuntimeError(
            "Could not create the FAISS vector store from PDF chunks. "
            "The embedding model may still be downloading or your network "
            "certificate settings may need the Windows certificate helper."
        ) from exc


def save_vector_store(vector_store: FAISS, save_path: Path = VECTOR_STORE_DIR) -> None:
    """Persist the FAISS vector store locally."""
    save_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(save_path))


def load_vector_store(load_path: Path = VECTOR_STORE_DIR) -> FAISS:
    """Load an existing FAISS vector store from disk."""
    index_file = load_path / "index.faiss"
    metadata_file = load_path / "index.pkl"

    if not index_file.exists() or not metadata_file.exists():
        raise FileNotFoundError("No saved FAISS vector store was found.")

    embeddings = get_embedding_model()
    return FAISS.load_local(
        str(load_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def vector_store_exists(load_path: Path = VECTOR_STORE_DIR) -> bool:
    """Check whether a saved FAISS vector store exists."""
    return (load_path / "index.faiss").exists() and (load_path / "index.pkl").exists()
