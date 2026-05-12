# Ask My PDF Bot - AI-Powered RAG Chatbot

Ask My PDF Bot is a professional internship-level AI project that lets users upload PDF files and chat with their content. It uses a simple Retrieval-Augmented Generation (RAG) workflow with Streamlit, LangChain, FAISS, sentence-transformers embeddings, PyPDF2, and the Google AI Studio Gemini API.

This project is designed to be easy to run locally in VS Code and easy to explain during an internship evaluation.

## Project Overview

The application allows users to:

- Upload one or more PDF documents
- Extract readable text from PDFs
- Split text into chunks
- Convert chunks into embeddings
- Store embeddings in a FAISS vector database
- Ask questions in natural language
- Retrieve relevant PDF chunks
- Generate answers using the Google AI Studio Gemini API
- Display source chunks used for every answer
- Maintain simple chat history during the session

## Features

- Professional Streamlit dashboard
- Single and multiple PDF upload support
- PDF validation and local saving
- Page-by-page text extraction using PyPDF2
- Text cleaning and whitespace normalization
- LangChain recursive text chunking
- Embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- FAISS vector store creation and persistence
- Gemini API answer generation
- Source chunk citation display
- Streamlit chat interface
- Session-based chat history
- Beginner-friendly error handling
- Clean modular Python structure

## Technologies Used

| Area | Technology |
| --- | --- |
| UI | Streamlit |
| RAG Framework | LangChain |
| LLM | Google AI Studio Gemini API |
| Vector Database | FAISS |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| PDF Processing | PyPDF2 |
| Environment Variables | python-dotenv |
| Utilities | pandas, os, pathlib |

## Folder Structure

```text
ask_my_pdf_bot/
|
├── app.py
├── rag_pipeline.py
├── pdf_utils.py
├── embeddings_utils.py
├── config.py
├── requirements.txt
├── README.md
├── .env
|
├── uploaded_pdfs/
├── vector_store/
├── assets/
└── chat_history/
```

## RAG Architecture Explanation

The project follows a simple RAG pipeline:

1. User uploads PDF files in the Streamlit app.
2. PyPDF2 extracts text from each PDF page.
3. LangChain splits extracted text into chunks of 1000 characters with 200 characters overlap.
4. Sentence-transformers generates embeddings for every chunk.
5. FAISS stores the chunk embeddings locally.
6. User asks a question.
7. FAISS retrieves the most relevant chunks.
8. The retrieved chunks and question are sent to Gemini through the Google AI Studio API key.
9. Gemini generates an answer based only on the retrieved context.
10. The app displays the answer and the source chunks used.

## Installation Steps

### 1. Open the project folder

```bash
cd ask_my_pdf_bot
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The first run may take a few minutes because the embedding model and machine learning dependencies need to install.
On Windows, the requirements include `python-certifi-win32` so Python can use the Windows certificate store when downloading the sentence-transformers model.

## API Key Setup

This project uses the Google AI Studio Gemini API.

1. Create or open the `.env` file.
2. Add your Google AI Studio Gemini API key:

```env
GEMINI_API_KEY=your_google_ai_studio_api_key_here
```

3. Save the file.
4. Restart Streamlit after editing `.env`.

You can get this key from Google AI Studio. The app also supports `GOOGLE_API_KEY`, but `GEMINI_API_KEY` is recommended for this project.

## How to Run

From inside the `ask_my_pdf_bot` folder, run:

```bash
streamlit run app.py
```

Streamlit will open the app in your browser, usually at:

```text
http://localhost:8501
```

## How to Use

1. Open the Streamlit app.
2. Upload one or more PDF files.
3. Click **Process PDFs and Build Vector Store**.
4. Wait for the success message.
5. Ask a question in the chat box.
6. Review the AI answer.
7. Expand the source section to see retrieved PDF chunks.

## Screenshots

Add screenshots here after running the app locally:

- `assets/home_screen.png`
- `assets/pdf_upload.png`
- `assets/chat_answer_sources.png`

## Business Use Cases

- Chat with company policy documents
- Search academic research papers
- Summarize legal or compliance documents
- Query technical manuals
- Analyze training materials
- Support HR onboarding document Q&A
- Extract answers from reports and proposals

## Future Improvements

- Add OCR support for scanned PDFs
- Add PDF deletion and vector store reset controls
- Export chat history to CSV or PDF
- Add answer confidence indicators
- Support more document formats such as DOCX and TXT
- Add user authentication for private documents
- Deploy the app to Streamlit Community Cloud

## Troubleshooting

### Missing Gemini API key

Make sure `.env` contains:

```env
GEMINI_API_KEY=your_actual_google_ai_studio_key_here
```

Then restart Streamlit.

### PDF has no readable text

PyPDF2 works best with text-based PDFs. Scanned image PDFs may return no text. Use a text-based PDF for this demo.

### FAISS installation error

Make sure you are using a supported Python version. Python 3.10, 3.11, 3.12, or 3.13 should work with the version ranges in `requirements.txt`.

### Embedding model takes time

The first run downloads `sentence-transformers/all-MiniLM-L6-v2`. Later runs are faster because the model is cached.

### Hugging Face certificate or closed-client error

If PDF processing shows a message such as `Cannot send a request, as the client has been closed`, reinstall the requirements and restart Streamlit:

```bash
pip install -r requirements.txt
streamlit run app.py
```

This usually happens on Windows when Python cannot verify certificates while downloading the embedding model.

### Gemini API failure

Check:

- API key is correct
- Internet connection is working
- Google API quota has not been exceeded
- The model name in `config.py` is available for your account. The default is `gemini-2.5-flash`.

## Academic Notes

This project is intentionally simple and modular. It avoids enterprise tools such as Docker, Kubernetes, Celery, Redis, Kafka, microservices, and advanced DevOps. The goal is a stable, understandable, presentation-ready RAG chatbot.
