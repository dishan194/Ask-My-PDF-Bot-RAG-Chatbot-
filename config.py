"""Configuration settings for Ask My PDF Bot.

This file keeps project paths, model names, and RAG settings in one place.
The values are intentionally simple so beginners can understand and modify
the project during an internship demo.
"""

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv


# Base project directory: ask_my_pdf_bot/
BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

# Always load the .env file that sits beside this config file. override=True is
# important because Streamlit/Windows may already have an old or empty variable.
load_dotenv(dotenv_path=ENV_FILE, override=True)

# Local storage folders.
UPLOAD_DIR = BASE_DIR / "uploaded_pdfs"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
ASSETS_DIR = BASE_DIR / "assets"
CHAT_HISTORY_DIR = BASE_DIR / "chat_history"

# Model settings.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Chunking settings used by LangChain.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings.
TOP_K_RESULTS = 4

def _clean_env_value(value: str | None) -> str:
    """Return an environment value without surrounding whitespace or quotes."""
    if value is None:
        return ""
    return value.strip().strip('"').strip("'")


def get_api_key() -> str:
    """Return the Google AI Studio key, preferring this project's .env file."""
    file_values = dotenv_values(ENV_FILE) if ENV_FILE.exists() else {}
    candidates = [
        file_values.get("GEMINI_API_KEY"),
        file_values.get("GOOGLE_API_KEY"),
        os.getenv("GEMINI_API_KEY"),
        os.getenv("GOOGLE_API_KEY"),
    ]

    for candidate in candidates:
        api_key = _clean_env_value(candidate)
        if api_key:
            return api_key

    return ""


# Google AI Studio / Gemini API key. Both names are supported, with
# GEMINI_API_KEY preferred for Google AI Studio examples.
GEMINI_API_KEY = _clean_env_value(dotenv_values(ENV_FILE).get("GEMINI_API_KEY"))
GOOGLE_API_KEY = _clean_env_value(dotenv_values(ENV_FILE).get("GOOGLE_API_KEY"))
API_KEY = get_api_key()


def is_placeholder_api_key(value: str) -> bool:
    """Detect common placeholder values shipped in example .env files."""
    normalized_value = value.strip().lower()
    placeholder_words = ["your_", "your-", "actual", "placeholder", "example", "here"]
    return any(word in normalized_value for word in placeholder_words)


def create_project_directories() -> None:
    """Create local folders required by the app if they do not exist."""
    for folder in [UPLOAD_DIR, VECTOR_STORE_DIR, ASSETS_DIR, CHAT_HISTORY_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def is_api_key_configured() -> bool:
    """Return True when the Gemini API key is available."""
    api_key = get_api_key()
    return bool(api_key and not is_placeholder_api_key(api_key))
