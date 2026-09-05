"""Hybrid retriever: FAISS dense vectors + BM25 keyword scores.

Dense-only retrieval misses exact identifiers (part numbers, section names);
BM25-only misses paraphrases. Fusing both gives fast, grounded retrieval.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .chunk import Chunk

_TOKEN = re.compile(r"[A-Za-z0-9_]+")
_EMBEDDER_CACHE: dict = {}

# Set by get_embedder() when the neural model cannot be loaded (e.g. a broken
# PyTorch install on Windows: "DLL initialization routine failed / c10.dll").
FALLBACK_REASON: str | None = None


class HashingEmbedder:
    """Dependency-free embedding fallback (hashed character n-grams + words).

    Pure NumPy, no PyTorch. Retrieval quality is lower than MiniLM but the app
    keeps working on machines where torch cannot load.
    """

    name = "hashing-fallback"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _features(self, text: str):
        words = tokenize(text)
        feats = list(words)
        feats += [f"{a}_{b}" for a, b in zip(words, words[1:])]
        for word in words:
            padded = f"^{word}$"
            feats += [padded[i:i + 3] for i in range(max(len(padded) - 2, 1))]
        return feats

    def encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype="float32")
        for feat in self._features(text):
            h = int.from_bytes(hashlib.md5(feat.encode("utf-8")).digest()[:8], "little")
            vec[h % self.dim] += 1.0
        vec = np.sqrt(vec)  # dampen frequent terms
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def encode(self, texts, **_) -> np.ndarray:
        return np.vstack([self.encode_one(t) for t in texts]).astype("float32")


def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Small, fast, fully local embedding model (~90MB, CPU friendly).

    Falls back to a pure-NumPy hashing embedder if sentence-transformers/torch
    is unavailable or broken on this machine.
    """
    global FALLBACK_REASON
    if model_name not in _EMBEDDER_CACHE:
        try:
            from sentence_transformers import SentenceTransformer

            _EMBEDDER_CACHE[model_name] = SentenceTransformer(model_name)
            FALLBACK_REASON = None
        except BaseException as exc:  # OSError/ImportError from broken torch DLLs
            FALLBACK_REASON = str(exc).split("\n")[0][:300]
            _EMBEDDER_CACHE[model_name] = HashingEmbedder()
    return _EMBEDDER_CACHE[model_name]


def embed(texts: Sequence[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    model = get_embedder(model_name)
    if isinstance(model, HashingEmbedder):
        return model.encode(list(texts))
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype="float32")



def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense: float


class VectorStore:
    """FAISS inner-product index over L2-normalised embeddings (= cosine)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.chunks: List[Chunk] = []
        self.index = None
        self._bm25 = None

    # ----------------------------------------------------------------- build
    def add(self, chunks: List[Chunk]) -> None:
        import faiss
        from rank_bm25 import BM25Okapi

        if not chunks:
            return
        self.chunks.extend(chunks)
        vectors = embed([c.text for c in chunks], self.model_name)

        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self._bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])

    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def documents(self) -> List[str]:
        seen: List[str] = []
        for c in self.chunks:
            if c.doc not in seen:
                seen.append(c.doc)
        return seen

    # -------------------------------------------------------------- retrieve
    def search(
        self,
        query: str,
        top_k: int = 6,
        candidates: int = 30,
        alpha: float = 0.65,
    ) -> List[Hit]:
        if self.index is None or not self.chunks:
            return []

        q = embed([query], self.model_name)
        k = min(candidates, len(self.chunks))
        scores, ids = self.index.search(q, k)
        dense = {int(i): float(s) for i, s in zip(ids[0], scores[0]) if i >= 0}

        lexical: dict = {}
        if self._bm25 is not None:
            raw = self._bm25.get_scores(tokenize(query))
            if len(raw):
                top = np.argsort(raw)[::-1][:candidates]
                peak = float(raw.max()) or 1.0
                for i in top:
                    lexical[int(i)] = float(raw[i]) / peak

        fused: List[Tuple[int, float, float]] = []
        for idx in set(dense) | set(lexical):
            d = dense.get(idx, 0.0)
            l = lexical.get(idx, 0.0)
            fused.append((idx, alpha * d + (1 - alpha) * l, d))

        fused.sort(key=lambda x: x[1], reverse=True)
        return [Hit(self.chunks[i], s, d) for i, s, d in fused[:top_k]]

    # ------------------------------------------------------------- persist
    def save(self, directory: str) -> None:
        import faiss

        os.makedirs(directory, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "chunks.pkl"), "wb") as fh:
            pickle.dump({"chunks": self.chunks, "model": self.model_name}, fh)

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        import faiss
        from rank_bm25 import BM25Okapi

        with open(os.path.join(directory, "chunks.pkl"), "rb") as fh:
            payload = pickle.load(fh)
        store = cls(payload["model"])
        store.chunks = payload["chunks"]
        path = os.path.join(directory, "index.faiss")
        if os.path.exists(path):
            store.index = faiss.read_index(path)
        if store.chunks:
            store._bm25 = BM25Okapi([tokenize(c.text) for c in store.chunks])
        return store
