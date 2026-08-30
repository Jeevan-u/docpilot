"""Tests for the vector store and retriever using a deterministic embedder."""

import hashlib

import numpy as np

from docpilot.embeddings import Embedder
from docpilot.retriever import Retriever
from docpilot.store import VectorStore


class NGramHashEmbedder(Embedder):
    """Deterministic embedding from character n-grams; no network needed."""

    def __init__(self, dimension: int = 64):
        self._dimension = dimension
        self._cache: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]
        vector = np.zeros(self._dimension)
        for window in (2, 3, 4):
            for start in range(len(text) - window + 1):
                gram = text[start : start + window]
                digest = hashlib.md5(gram.encode()).digest()
                index = int.from_bytes(digest[:2], "big") % self._dimension
                vector[index] += 1.0
        vector = vector / (np.linalg.norm(vector) or 1.0)
        result = list(vector)
        self._cache[text] = result
        return result

    def embed_many(self, texts, batch_size=64):
        return [self.embed(text) for text in texts]


def _build_store():
    embedder = NGramHashEmbedder()
    store = VectorStore()
    store.add(
        embedder.embed("Dense vectors capture meaning rather than exact wording."),
        "Dense vectors capture meaning rather than exact wording.",
        {"source": "a.txt"},
    )
    store.add(
        embedder.embed("The bakery is famous for its sourdough loaves."),
        "The bakery is famous for its sourdough loaves.",
        {"source": "b.txt"},
    )
    store.add(
        embedder.embed("Semantic search finds related passages for a question."),
        "Semantic search finds related passages for a question.",
        {"source": "c.txt"},
    )
    store.add(
        embedder.embed("Vector dimensions must be consistent across the store."),
        "Vector dimensions must be consistent across the store.",
        {"source": "d.txt"},
    )
    return embedder, store


def test_search_returns_the_most_similar_chunk_first():
    embedder, store = _build_store()
    hits = store.search(embedder.embed("How does semantic search work?"), top_k=2)
    assert hits[0].text == "Semantic search finds related passages for a question."
    assert hits[0].score > hits[1].score


def test_exact_match_has_near_perfect_score():
    embedder, store = _build_store()
    target = "Semantic search finds related passages for a question."
    hits = store.search(embedder.embed(target), top_k=1)
    assert hits[0].score > 0.99


def test_retriever_returns_stored_metadata():
    embedder, store = _build_store()
    retriever = Retriever(store, embedder, top_k=3)
    hits = retriever.retrieve("sourdough bakery")
    assert hits[0].metadata["source"] == "b.txt"


def test_inconsistent_dimensions_are_rejected():
    store = VectorStore()
    store.add([0.1, 0.2, 0.3], "first")
    try:
        store.add([0.1, 0.2, 0.3, 0.4], "second")
    except ValueError:
        return
    raise AssertionError("Expected a ValueError for mismatched dimensions")


def test_save_and_load_round_trip(tmp_path):
    embedder, store = _build_store()
    store.save(tmp_path)

    restored = VectorStore().load(tmp_path)
    assert len(restored) == len(store)
    hits = restored.search(embedder.embed("semantic search and vectors"), top_k=1)
    assert hits[0].text == "Semantic search finds related passages for a question."


def test_empty_store_search_returns_empty():
    hits = VectorStore().search([1.0, 2.0], top_k=3)
    assert hits == []
