"""Text-to-vector providers.

All producers implement the small :class:`Embedder` interface so the rest
of the pipeline never cares which model actually turns text into numbers.
The stock provider calls OpenAI's embeddings API; a local sentence encoder
can be dropped in later behind the same interface.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from openai import OpenAI


class Embedder(ABC):
    """Converts text into dense vectors."""

    @abstractmethod
    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class OpenAIBackendEmbedder(Embedder):
    """Embeddings served by OpenAI's embeddings API."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        self.model = model
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self.model, input=text)
        return list(response.data[0].embedding)

    def embed_many(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self._client.embeddings.create(model=self.model, input=batch)
            embeddings.extend(list(item.embedding) for item in response.data)
        return embeddings