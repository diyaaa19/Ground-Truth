# Ground Truth — Local RAG PDF Chatbot

> **Ask questions about your PDFs and get grounded, page-level cited answers — without letting the model invent information.**

Ground Truth is a local Retrieval-Augmented Generation (RAG) chatbot built with Python and Streamlit. Upload one or more PDFs, build an index, and ask questions about their contents.

The application combines **semantic retrieval** with **BM25 keyword retrieval**, then sends the retrieved passages to a configurable answering backend. If the relevant information cannot be found with sufficient confidence, the application refuses instead of guessing.

---

## ✨ Features

-  **Multiple PDF upload** — upload one or more PDFs at once.
-  **Hybrid retrieval** — FAISS semantic search + BM25 keyword search.
-  **Local embeddings** — `all-MiniLM-L6-v2`, running locally.
-  **Page-level provenance** — retrieved passages retain their source PDF and page information.
-  **Grounded answers** — the answering layer is instructed to stay within retrieved document context.
-  **Hallucination guard** — a configurable grounding threshold allows the bot to refuse low-confidence questions.
-  **Visible citations** — answers expose the passages used to produce them.
-  **Multiple answering backends**:
  - Ollama (local/offline)
  - OpenAI-compatible API
  - Extractive mode (no LLM; quote retrieved passages)
-  **OCR fallback** for scanned/image-only PDFs.
-  **Retrieval controls** for passage count, grounding threshold, and semantic-vs-keyword weighting.
-  **Clean Streamlit interface** with document statistics and source panels.

The core application implements these retrieval, embedding, PDF-processing, and answering choices in the project code. 

---

## 🏗️ Architecture

```text
                    ┌─────────────────┐
                    │   PDF Uploads   │
                    └────────┬────────┘
                             │
                             ▼
                 ┌──────────────────────┐
                 │   PDF Text Extraction │
                 │ pdfplumber / pypdf   │
                 │      + OCR fallback  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Page-aware Chunks  │
                 └──────────┬───────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
    ┌──────────────────┐        ┌──────────────────┐
    │ Semantic Search  │        │   BM25 Search    │
    │ FAISS + MiniLM   │        │    Keywords      │
    └────────┬─────────┘        └────────┬─────────┘
             │                           │
             └─────────────┬─────────────┘
                           ▼
                  ┌─────────────────┐
                  │ Hybrid Ranking  │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Grounding Check │
                  └────────┬────────┘
                           │
                    ┌──────┴───────┐
                    │              │
                 Relevant      Not enough
                  context       evidence
                    │              │
                    ▼              ▼
             ┌─────────────┐   Refuse to
             │ Answering   │   guess
             │ Backend     │
             └──────┬──────┘
                    │
                    ▼
             Answer + Sources
```

The project uses FAISS with normalized embeddings and BM25 scores for retrieval, `all-MiniLM-L6-v2` for embeddings, and supports Ollama, OpenAI-compatible APIs, or passage-only answering.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| UI | Streamlit |
| Vector retrieval | FAISS |
| Keyword retrieval | BM25 |
| Embeddings | Sentence Transformers — `all-MiniLM-L6-v2` |
| PDF extraction | pdfplumber, pypdf |
| OCR | Tesseract + Poppler |
| Local LLM | Ollama |
| Hosted/API LLM | OpenAI-compatible API |
| Numerical processing | NumPy |

The required Python packages are listed in `requirements.txt`. 

---

# 🚀 Installation

## 1. Clone or open the project

```bash
cd pdfchat
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
source .venv/bin/activate
```

You should see:

```text
(.venv)
```

at the beginning of your terminal prompt.

## 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project requires Streamlit, FAISS, Sentence Transformers, pypdf, pdfplumber, BM25, NumPy, requests, and optional OCR packages.

---

# 🪟 Windows: Important Environment Check

Make sure your terminal is using the project's virtual environment.

Run:

```powershell
python -c "import sys; print(sys.executable)"
```

It should point to something similar to:

```text
C:\...\pdfchat\.venv\Scripts\python.exe
```

Then verify PyTorch:

```powershell
python -c "import torch; print(torch.__version__)"
```

And verify the embedding model:

```powershell
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('EMBEDDINGS OK')"
```

If Streamlit is installed in the virtual environment, start the application with:

```powershell
python -m streamlit run app.py
```

Using `python -m streamlit` is useful on Windows because it guarantees that Streamlit is launched by the same Python interpreter as the active `.venv`.

---

# ▶️ Run the Application

With the virtual environment activated:

```bash
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

The application currently provides document upload, OCR selection, index building, model-backend selection, retrieval controls, chat, and source displays directly in the Streamlit UI. 

---

# 📚 Using the App

### 1. Upload PDFs

Drag and drop your PDFs into the **Documents** section.

Multiple PDFs can be indexed together.

### 2. Enable OCR when needed

Turn on:

**OCR scanned pages**

if the PDF contains image-only/scanned pages.

OCR requires Tesseract and Poppler to be installed on the system. 

### 3. Build the index

Click:

**Build index**

The application:

1. Reads the PDF pages.
2. Extracts text.
3. Creates page-aware chunks.
4. Generates embeddings.
5. Builds the FAISS/BM25 retrieval index.

The UI reports document count, pages read, indexed passages, and index-build time. 

### 4. Ask questions

After indexing, ask a question in the chat box.

The retriever searches the indexed document and the selected answering backend generates a grounded response.

### 5. Inspect the sources

Each answer can display the passages used to support it, including their citation and retrieval score. 

---

# 🤖 Answering Backends

## Ollama — Local / Offline

Ollama is the recommended option if you want the answering model to run locally.

Install Ollama, then download a model:

```bash
ollama pull llama3.1
```

You can use another model available in your Ollama installation as well.

Then select:

```text
Ollama (local)
```

in the application.

The UI detects locally available Ollama models and lets you select one. 

---

## OpenAI-Compatible API

Select:

```text
OpenAI-compatible API
```

Set your API key before starting a query.

### Windows PowerShell

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

### macOS / Linux

```bash
export OPENAI_API_KEY="your-api-key"
```

The application also supports an optional OpenAI-compatible base URL through `OPENAI_BASE_URL`. 

> **Security:** Never commit API keys to GitHub or place them directly in source code.

---

## Extractive Mode

You can also answer without an LLM.

Select:

```text
No model — quote passages
```

This mode is useful when you want retrieved document evidence without generative answering.

The application exposes this as one of its supported answering backends. 

---

# 🔍 Hybrid Retrieval

Ground Truth does not rely on only one retrieval strategy.

It combines:

### Semantic retrieval

FAISS searches using embeddings generated by:

```text
all-MiniLM-L6-v2
```

This helps with:

- paraphrased questions
- concept-level matching
- natural-language queries

### Keyword retrieval

BM25 helps with:

- exact terms
- names
- identifiers
- technical vocabulary
- phrases that semantic retrieval may underweight

The **Meaning ↔ keyword balance** slider controls the weighting between the two approaches:

```text
0.0 → keyword-focused
0.65 → balanced default
1.0 → semantic-focused
```

The current UI exposes this control alongside the number of passages and grounding threshold. 

---

# 🛡️ Hallucination / Grounding Controls

Ground Truth is designed to prioritize **document-grounded answers**.

## Grounding threshold

The **Grounding threshold** determines how strong a retrieved match must be before the system is willing to answer.

Higher threshold:

```text
More conservative
↓
More refusals
↓
Less risk of unsupported answers
```

Lower threshold:

```text
More permissive
↓
Fewer refusals
↓
Potentially weaker evidence
```

The application explicitly exposes this threshold and passes it to the answering layer. 

## Refusal behavior

If the required information cannot be sufficiently grounded in the uploaded documents, the intended response is:

```text
I couldn't find that in the document.
```

The project's original README describes the same grounding strategy: a score threshold, strict prompting, temperature 0, and citations mapped to visible source passages. 

---

# 🖨️ OCR for Scanned PDFs

For normal text PDFs, OCR may not be necessary.

For scanned/image-only PDFs, install:

### Windows

Install:

- Tesseract OCR
- Poppler

and make sure both are available through your system `PATH`.

### macOS

```bash
brew install tesseract poppler
```

### Ubuntu / Debian

```bash
sudo apt install tesseract-ocr poppler-utils
```

The Python OCR dependencies are already included in `requirements.txt`; the system-level Tesseract and Poppler executables are separate requirements. fileciteturn0file1L42-L47

---

# ⚙️ Retrieval Tuning

The sidebar exposes three retrieval controls:

| Setting | What it does |
|---|---|
| **Passages per answer** | Controls how many retrieved passages are supplied as context |
| **Grounding threshold** | Controls how strong the retrieval evidence must be before answering |
| **Meaning ↔ keyword balance** | Controls semantic vs. keyword retrieval weighting |

The application's defaults are currently:

```text
Passages per answer: 6
Grounding threshold: 0.25
Meaning ↔ keyword balance: 0.65
```

These controls are defined directly in the Streamlit UI. fileciteturn0file0L113-L118

---

# 📁 Project Structure

```text
pdfchat/
│
├── app.py
├── requirements.txt
├── README.md
│
└── rag/
    ├── __init__.py
    ├── extract.py
    ├── chunk.py
    ├── store.py
    └── llm.py
```

### `app.py`

Streamlit interface, document upload, index building, retrieval controls, chat, and source display. fileciteturn0file0L124-L155

### `rag/extract.py`

PDF-to-page text extraction with OCR fallback.

### `rag/chunk.py`

Creates chunks while preserving page provenance.

### `rag/store.py`

Handles the retrieval store, including FAISS and BM25.

### `rag/llm.py`

Handles grounded prompting and answering backends.

The project structure and responsibilities are also reflected in the existing project documentation. 

---

# 🧪 Troubleshooting

## `ModuleNotFoundError: No module named 'streamlit'`

Your virtual environment probably does not contain the project dependencies.

Activate `.venv` and run:

```powershell
python -m pip install -r requirements.txt
```

Then:

```powershell
python -m streamlit run app.py
```

---

## `ModuleNotFoundError: No module named 'torch'`

Install PyTorch into the **same `.venv`** that runs the application.

For a CPU-only Windows setup:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Then verify:

```powershell
python -c "import torch; print(torch.__version__)"
```

---

## `ModuleNotFoundError: No module named 'sentence_transformers'`

Install the project's dependencies:

```powershell
python -m pip install -r requirements.txt
```

or specifically:

```powershell
python -m pip install sentence-transformers
```

Then test:

```powershell
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('EMBEDDINGS OK')"
```

---

## App says "compatibility search mode"

This means the neural embedding model could not start and the application has fallen back to keyword/character search.

First verify that your app and terminal use the same `.venv`:

```powershell
python -c "import sys; print(sys.executable)"
```

Then test:

```powershell
python -c "import torch; print(torch.__version__)"
```

and:

```powershell
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('EMBEDDINGS OK')"
```

Restart the application with:

```powershell
python -m streamlit run app.py
```

If the embedding model loads successfully, rebuild the PDF index.

---

## Ollama is not detected

Check that Ollama is installed and running, then pull a model:

```bash
ollama pull llama3.1
```

Restart the Streamlit application afterwards.

---

## Scanned PDF returns little or no text

Enable:

```text
OCR scanned pages
```

and make sure Tesseract and Poppler are installed and available in `PATH`.

---

# 🔐 Privacy

When using the **Ollama (local)** backend, the answering model can run locally.

However, if you choose an OpenAI-compatible API backend, retrieved document content may be sent to that external API as part of the answering request.

Do not upload confidential documents to an external model provider unless your organization's policies allow it.

---

# 📌 Current Limitations

- OCR depends on external Tesseract and Poppler installations.
- Local model performance depends on your machine's available CPU/RAM.
- Retrieval quality depends on PDF text extraction and chunking quality.
- A low grounding score can intentionally result in a refusal even when the document contains related information.
- Extractive mode provides passages rather than a generated natural-language answer.

---

# 🧭 Typical Workflow

```text
1. Start the application
        ↓
2. Upload PDF(s)
        ↓
3. Enable OCR if required
        ↓
4. Build index
        ↓
5. Choose answering backend
        ↓
6. Ask a question
        ↓
7. Hybrid retrieval
        ↓
8. Grounding check
        ↓
9. Generate / quote answer
        ↓
10. Inspect cited source passages
```

---

# 📄 License

Add your preferred license here before publishing the project publicly.

For example:

```text
MIT License
```

if you intend to release the project under MIT.

---

## 👩‍💻 Project

**Ground Truth — Local RAG PDF Chatbot**

Built with:

```text
Python + Streamlit + FAISS + BM25
+ Sentence Transformers + PDF extraction
+ Ollama / OpenAI-compatible APIs
```
