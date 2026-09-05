"""Answer generation, grounded strictly in retrieved PDF context.

Backends, in order of preference:
  * ollama  - fully local, no API key (default; e.g. `ollama run llama3.1`)
  * openai  - any OpenAI-compatible endpoint via OPENAI_API_KEY / OPENAI_BASE_URL
  * extractive - no LLM at all: returns the best-matching passages verbatim
"""

from __future__ import annotations

import os
from typing import List

import requests

from .store import Hit

SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages from the user's PDF.

Hard rules:
1. Never use outside knowledge, never guess, never fill gaps.
2. Every factual sentence ends with its source marker, e.g. [1] or [2][3].
3. If the passages do not contain the answer, reply exactly:
   "I couldn't find that in the document."
   You may then name the closest topic the document does cover.
4. Quote numbers, names, dates and figures exactly as written.
5. Be concise and use short paragraphs or bullets."""

NO_ANSWER = "I couldn't find that in the document."


def build_context(hits: List[Hit], max_chars: int = 7000) -> str:
    blocks, used = [], 0
    for n, hit in enumerate(hits, start=1):
        block = f"[{n}] ({hit.chunk.citation})\n{hit.chunk.text}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def _user_prompt(question: str, context: str) -> str:
    return f"Context passages:\n\n{context}\n\nQuestion: {question}\n\nGrounded answer with [n] citations:"


# --------------------------------------------------------------------- ollama
def ollama_models(host: str = "http://localhost:11434") -> List[str]:
    try:
        r = requests.get(f"{host}/api/tags", timeout=2)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def _ollama_answer(question, context, model, host, temperature) -> str:
    r = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "stream": False,
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(question, context)},
            ],
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


# --------------------------------------------------------------------- openai
def _openai_answer(question, context, model, temperature) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    r = requests.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(question, context)},
            ],
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ----------------------------------------------------------------- extractive
def _extractive_answer(hits: List[Hit]) -> str:
    lines = ["Closest passages found in the document (no language model configured):", ""]
    for n, hit in enumerate(hits[:3], start=1):
        snippet = hit.chunk.text.strip()
        if len(snippet) > 700:
            snippet = snippet[:700].rsplit(" ", 1)[0] + "…"
        lines.append(f"> {snippet}\n\n— [{n}] {hit.chunk.citation}")
        lines.append("")
    return "\n".join(lines).strip()


def answer(
    question: str,
    hits: List[Hit],
    backend: str = "ollama",
    model: str = "llama3.1",
    host: str = "http://localhost:11434",
    temperature: float = 0.0,
    min_score: float = 0.25,
) -> str:
    """Return a grounded answer, or the refusal string when nothing matches."""
    if not hits or hits[0].score < min_score:
        return NO_ANSWER

    if backend == "extractive":
        return _extractive_answer(hits)

    context = build_context(hits)
    if backend == "ollama":
        return _ollama_answer(question, context, model, host, temperature)
    if backend == "openai":
        return _openai_answer(question, context, model, temperature)
    raise ValueError(f"Unknown backend: {backend}")
