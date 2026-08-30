"""Orchestrate the stages of a retrieval-augmented generation pipeline.

A pipeline owns the store, the embedder, the chunker and the generator,
so callers only need two operations: ``index`` to load knowledge and
``answer`` to query it.
"""

from __future__ import annotations

from pathlib import Path

from .chunker import TokenChunker
from .config import Config
from .embeddings import Embedder, OpenAIBackendEmbedder
from .generator import Answer, ChatGenerator, Generator
from .loader import Document
from .retriever import Retriever
from .store import VectorStore


class RAGPipeline:
    """The glue that combines retrieval with generation."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        api_key = self.config.require_api_key()

        self.embedder: Embedder = OpenAIBackendEmbedder(
            model=self.config.embedding_model, api_key=api_key
        )
        self.store = VectorStore()
        self.chunker = TokenChunker(
            chunk_size=self.config.chunk_size_tokens,
            overlap=self.config.chunk_overlap_tokens,
        )
        self.retriever = Retriever(
            store=self.store, embedder=self.embedder, top_k=self.config.top_k
        )
        self.generator: Generator = ChatGenerator(
            model=self.config.chat_model, api_key=api_key
        )

    def index(self, documents: list[Document]) -> int:
        """Chunk, embed and store the given documents. Returns chunk count."""
        if not documents:
            return 0

        chunks = [
            chunk
            for document in documents
            for chunk in self.chunker.chunk_document(document)
        ]
        vectors = self.embedder.embed_many([chunk.text for chunk in chunks])
        for vector, chunk in zip(vectors, chunks):
            self.store.add(vector, chunk.text, chunk.metadata)
        return len(chunks)

    def answer(self, question: str, top_k: int | None = None) -> Answer:
        """Retrieve the best matches for a question and answer from them."""
        hits = self.retriever.retrieve(question, top_k=top_k)
        if not hits:
            raise RuntimeError(
                "Index is empty. Run `docpilot index` on your documents first."
            )
        return self.generator.generate(question, hits)

    def persist(self, directory: str | Path) -> Path:
        return self.store.save(directory)

    def load(self, directory: str | Path) -> RAGPipeline:
        self.store.load(directory)
        return self
