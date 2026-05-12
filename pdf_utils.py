"""PDF upload, validation, saving, and text extraction utilities."""

import re
from pathlib import Path
from typing import BinaryIO, Dict, List, Tuple

import streamlit as st
from PyPDF2 import PdfReader

from config import UPLOAD_DIR


def is_valid_pdf(uploaded_file) -> Tuple[bool, str]:
    """Validate a Streamlit uploaded file.

    Returns:
        A tuple of (is_valid, message).
    """
    if uploaded_file is None:
        return False, "No file was uploaded."

    if not uploaded_file.name.lower().endswith(".pdf"):
        return False, f"{uploaded_file.name} is not a PDF file."

    if uploaded_file.size == 0:
        return False, f"{uploaded_file.name} is empty."

    return True, "Valid PDF file."


def clean_text(text: str) -> str:
    """Clean extracted PDF text for better chunking and retrieval."""
    if not text:
        return ""

    # Normalize line endings and remove excessive whitespace.
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +\n", "\n", text)
    return text.strip()


def save_uploaded_pdf(uploaded_file) -> Path:
    """Save a Streamlit uploaded PDF to uploaded_pdfs/."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = Path(uploaded_file.name).name
    file_path = UPLOAD_DIR / safe_name

    with file_path.open("wb") as output_file:
        output_file.write(uploaded_file.getbuffer())

    return file_path


def extract_text_from_pdf(file_path: Path) -> Tuple[str, List[Dict]]:
    """Extract text page by page from a PDF using PyPDF2.

    Returns:
        combined_text: Full cleaned text from the PDF.
        page_metadata: List containing page number, source, and page text.
    """
    page_metadata: List[Dict] = []

    try:
        reader = PdfReader(str(file_path))
    except Exception as exc:
        raise ValueError(f"Could not read PDF file: {file_path.name}") from exc

    if len(reader.pages) == 0:
        raise ValueError(f"{file_path.name} does not contain any pages.")

    all_text_parts = []

    for page_index, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""

        page_text = clean_text(raw_text)

        if page_text:
            all_text_parts.append(page_text)
            page_metadata.append(
                {
                    "source": file_path.name,
                    "page": page_index,
                    "text": page_text,
                }
            )

    combined_text = clean_text("\n\n".join(all_text_parts))

    if not combined_text:
        raise ValueError(
            f"No readable text was found in {file_path.name}. "
            "Scanned/image-only PDFs may not work with PyPDF2."
        )

    return combined_text, page_metadata


def process_uploaded_files(uploaded_files: List[BinaryIO]) -> Tuple[List[Path], List[str]]:
    """Validate and save uploaded PDFs.

    Returns:
        saved_paths: Successfully saved PDF paths.
        errors: User-friendly error messages for failed uploads.
    """
    saved_paths: List[Path] = []
    errors: List[str] = []

    for uploaded_file in uploaded_files:
        is_valid, message = is_valid_pdf(uploaded_file)

        if not is_valid:
            errors.append(message)
            continue

        try:
            saved_paths.append(save_uploaded_pdf(uploaded_file))
        except Exception as exc:
            errors.append(f"Could not save {uploaded_file.name}: {exc}")

    return saved_paths, errors


def list_uploaded_pdf_names() -> List[str]:
    """Return names of PDFs already saved in uploaded_pdfs/."""
    if not UPLOAD_DIR.exists():
        return []

    return sorted(path.name for path in UPLOAD_DIR.glob("*.pdf"))
