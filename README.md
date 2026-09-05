# Ground Truth — Local RAG PDF Chatbot

A local RAG chatbot for asking questions about PDFs with grounded, page-level citations.

## Screenshots
<img width="1882" height="847" alt="image" src="https://github.com/user-attachments/assets/0464e677-4a85-43c9-9aa3-ee828e3ae34e" />

## Features

- Multiple PDF upload
- FAISS + BM25 hybrid retrieval
- Sentence Transformer embeddings
- OCR support for scanned PDFs
- Page-level source citations
- Grounding threshold to reduce hallucinations
- Ollama / OpenAI-compatible / extractive answering

## Tech Stack

Python · Streamlit · FAISS · BM25 · Sentence Transformers · Ollama · pdfplumber · pypdf · Tesseract

## Architecture

```text
PDF → Text Extraction → Chunking
          ↓
    FAISS + BM25
          ↓
   Grounding Check
          ↓
    Answer + Citations
```

## Setup

```bash
git clone https://github.com/diyaaa19/Ground-Truth.git
cd pdfchat

python -m venv .venv
```

### Windows

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### macOS / Linux

```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501`.

## Usage

1. Upload PDF(s)
2. Enable OCR if needed
3. Build the index
4. Ask questions
5. View grounded answers and source passages

## Project Structure

```text
pdfchat/
├── app.py
├── requirements.txt
├── README.md
└── rag/
    ├── extract.py
    ├── chunk.py
    ├── store.py
    └── llm.py
```

## License

MIT
