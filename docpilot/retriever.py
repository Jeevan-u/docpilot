"""Turn a question into the most relevant stored chunks.

Retrieval is the "R" in RAG: rather than asking the language model to
reason over everything at once, we first narrow the document set down to
the handful of chunks that actually resemble the question.
"""

from __future__ import annotations

from .embeddings import Embedder
from .store import SearchHit, VectorStore


class Retriever:
    """Scores stored chunks against a query and returns the best matches."""

    def __init__(self, store: VectorStore, embedder: Embedder, top_k: int = 4):
        self.store = store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        k = top_k or self.top_k
        if not self.store:
            return []
        query_vector = self.embedder.embed(query)
        return self.store.search(query_vector, top_k=k)
