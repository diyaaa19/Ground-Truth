"""Ground Truth — a local RAG chatbot for any PDF.

Run:  streamlit run app.py
"""

from __future__ import annotations

import os
import tempfile
import time

import streamlit as st

from rag import NO_ANSWER, VectorStore, answer, chunk_pages, extract_pages, ollama_models
from rag import store as rag_store


st.set_page_config(
    page_title="Ground Truth · PDF chat",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root{
  --ink:#101418; --muted:#6b7280; --line:#e6e3dc;
  --paper:#faf8f4; --card:#ffffff; --accent:#1f6f5c; --accent-soft:#e7f1ed;
}
html, body, [class*="css"]{ font-family:"Inter","Segoe UI",system-ui,sans-serif; }
.stApp{ background:
  radial-gradient(1100px 480px at 12% -10%, #f1efe7 0%, transparent 60%),
  var(--paper); color:var(--ink); }
#MainMenu, footer, header{ visibility:hidden; }
.block-container{ padding-top:2.2rem; max-width:1180px; }

.gt-hero{ display:flex; align-items:baseline; gap:.8rem; margin-bottom:.2rem; }
.gt-hero h1{ font-size:2.05rem; font-weight:700; letter-spacing:-.03em; margin:0; }
.gt-hero span{ color:var(--accent); font-weight:600; letter-spacing:.14em;
  font-size:.7rem; text-transform:uppercase; border:1px solid var(--accent);
  border-radius:999px; padding:.18rem .6rem; }
.gt-sub{ color:var(--muted); font-size:.95rem; margin:.1rem 0 1.4rem; }

.gt-stat{ background:var(--card); border:1px solid var(--line); border-radius:14px;
  padding:.85rem 1rem; }
.gt-stat b{ display:block; font-size:1.35rem; letter-spacing:-.02em; }
.gt-stat small{ color:var(--muted); text-transform:uppercase; letter-spacing:.1em;
  font-size:.64rem; }

.gt-cite{ background:var(--card); border:1px solid var(--line); border-left:3px solid var(--accent);
  border-radius:10px; padding:.7rem .9rem; margin:.45rem 0; font-size:.87rem; line-height:1.5; }
.gt-cite .tag{ color:var(--accent); font-weight:600; font-size:.72rem;
  letter-spacing:.08em; text-transform:uppercase; }
.gt-empty{ border:1px dashed var(--line); border-radius:16px; padding:2.6rem;
  text-align:center; color:var(--muted); background:rgba(255,255,255,.6); }

[data-testid="stChatMessage"]{ background:var(--card); border:1px solid var(--line);
  border-radius:16px; padding:.9rem 1.1rem; margin-bottom:.7rem; }
[data-testid="stSidebar"]{ background:#f4f1ea; border-right:1px solid var(--line); }
.stButton>button{ border-radius:10px; border:1px solid var(--line); font-weight:600; }
.stButton>button[kind="primary"]{ background:var(--accent); border-color:var(--accent); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------- state
if "store" not in st.session_state:
    st.session_state.store = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed" not in st.session_state:
    st.session_state.indexed = []
if "stats" not in st.session_state:
    st.session_state.stats = {"pages": 0, "chunks": 0, "seconds": 0.0}

# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Documents")
    uploads = st.file_uploader(
        "Drop PDFs here", type=["pdf"], accept_multiple_files=True,
        label_visibility="collapsed",
    )
    use_ocr = st.toggle("OCR scanned pages", value=True,
                        help="Needs tesseract + poppler installed. Slower, but reads image-only PDFs.")
    build = st.button("Build index", type="primary", use_container_width=True,
                      disabled=not uploads)

    st.markdown("---")
    st.markdown("### Answering")
    local_models = ollama_models()
    backend_labels = {
        "ollama": "Ollama (local)",
        "openai": "OpenAI-compatible API",
        "extractive": "No model — quote passages",
    }
    backend = st.selectbox(
        "Model backend", list(backend_labels),
        format_func=lambda k: backend_labels[k],
        index=0 if local_models else 2,
    )
    if backend == "ollama":
        model = st.selectbox("Model", local_models or ["llama3.1"])
        if not local_models:
            st.caption("Ollama not detected on :11434 — install it, then `ollama pull llama3.1`.")
    elif backend == "openai":
        model = st.text_input("Model name", "gpt-4o-mini")
        if not os.environ.get("OPENAI_API_KEY"):
            st.caption("Set OPENAI_API_KEY in your environment before asking.")
    else:
        model = ""

    st.markdown("---")
    st.markdown("### Retrieval")
    top_k = st.slider("Passages per answer", 3, 12, 6)
    min_score = st.slider("Grounding threshold", 0.0, 0.6, 0.25, 0.05,
                          help="Below this match score the bot refuses instead of guessing.")
    alpha = st.slider("Meaning ↔ keyword balance", 0.0, 1.0, 0.65, 0.05)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------------------------- index
if build and uploads:
    started = time.time()
    store = VectorStore()
    all_chunks, pages_total, names = [], 0, []
    progress = st.progress(0.0, text="Reading PDFs…")

    for i, upload in enumerate(uploads):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(upload.getbuffer())
            tmp_path = tmp.name
        try:
            pages = extract_pages(tmp_path, use_ocr=use_ocr)
        finally:
            os.unlink(tmp_path)

        pages_total += len(pages)
        names.append(upload.name)
        all_chunks += chunk_pages(pages, upload.name, start_id=len(all_chunks))
        progress.progress((i + 1) / (len(uploads) + 1), text=f"Read {upload.name}")

    progress.progress(0.95, text="Embedding + building FAISS index…")
    store.add(all_chunks)
    progress.empty()

    st.session_state.store = store
    st.session_state.indexed = names
    st.session_state.messages = []
    st.session_state.stats = {
        "pages": pages_total, "chunks": len(all_chunks),
        "seconds": round(time.time() - started, 1),
    }
    if not all_chunks:
        st.error("No readable text found. If these are scans, enable OCR and install tesseract.")

if rag_store.FALLBACK_REASON:
    st.warning(
        "Running in **compatibility search mode** — the neural embedding model could not "
        "start on this machine, so a built-in keyword/character search is used instead. "
        "Answers stay grounded and cited, but matching is a little less flexible.\n\n"
        f"Reason: `{rag_store.FALLBACK_REASON}`\n\n"
        "To restore the full model (usually a broken PyTorch install on Windows): install the "
        "[Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe), "
        "then run `pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu`."
    )


# ---------------------------------------------------------------------- hero
st.markdown(
    '<div class="gt-hero"><h1>Ground Truth</h1><span>local rag</span></div>'
    '<p class="gt-sub">Ask anything about your PDFs. Every answer is drawn from the pages '
    "themselves and cited — if it isn't in the document, you'll be told so.</p>",
    unsafe_allow_html=True,
)

store: VectorStore | None = st.session_state.store

if store and store.size:
    stats = st.session_state.stats
    cols = st.columns(4)
    figures = [
        (len(st.session_state.indexed), "documents"),
        (stats["pages"], "pages read"),
        (stats["chunks"], "indexed passages"),
        (f"{stats['seconds']}s", "index time"),
    ]
    for col, (value, label) in zip(cols, figures):
        col.markdown(f'<div class="gt-stat"><b>{value}</b><small>{label}</small></div>',
                     unsafe_allow_html=True)
    st.caption("Indexed: " + " · ".join(st.session_state.indexed))
else:
    st.markdown(
        '<div class="gt-empty">Upload one or more PDFs in the sidebar, then '
        "<b>Build index</b> to start asking questions.</div>",
        unsafe_allow_html=True,
    )

st.write("")

# ---------------------------------------------------------------------- chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"Sources · {len(message['sources'])} passages"):
                for n, src in enumerate(message["sources"], start=1):
                    st.markdown(
                        f'<div class="gt-cite"><span class="tag">[{n}] {src["citation"]}'
                        f' · match {src["score"]:.2f}</span><br>{src["text"]}</div>',
                        unsafe_allow_html=True,
                    )

question = st.chat_input(
    "Ask a question about your PDFs…" if store and store.size else "Build an index first…",
    disabled=not (store and store.size),
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document…"):
            hits = store.search(question, top_k=top_k, alpha=alpha)
            try:
                reply = answer(
                    question, hits, backend=backend, model=model, min_score=min_score
                )
            except Exception as exc:  # surface backend problems plainly
                reply = f"**The answering model failed.** {exc}"
                hits = hits if reply.startswith("**The") else []

        st.markdown(reply)
        sources = []
        if reply != NO_ANSWER and not reply.startswith("**The answering model failed"):
            for hit in hits:
                text = hit.chunk.text
                sources.append({
                    "citation": hit.chunk.citation,
                    "score": hit.score,
                    "text": (text[:600] + "…") if len(text) > 600 else text,
                })
            if sources:
                with st.expander(f"Sources · {len(sources)} passages"):
                    for n, src in enumerate(sources, start=1):
                        st.markdown(
                            f'<div class="gt-cite"><span class="tag">[{n}] {src["citation"]}'
                            f' · match {src["score"]:.2f}</span><br>{src["text"]}</div>',
                            unsafe_allow_html=True,
                        )

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": sources}
    )
