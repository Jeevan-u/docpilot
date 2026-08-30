"""Local vector store with cosine similarity search.

Vectors live in memory and can be persisted to disk as a pair of files:
a NumPy matrix of embeddings plus a JSON sidecar holding the matching
text and metadata. The implementation is intentionally plain so the math
behind similarity search is easy to follow:

* embeddings are L2-normalised when stored, so their dot product is the
  cosine similarity (range -1..1, higher is more similar),
* the query is normalised the same way and scored against every stored
  vector in one matrix multiplication.

For personal-scale corpora this brute-force approach is fast enough and
removes a heavyweight dependency; the interface is small enough to swap
for a dedicated vector database later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SearchHit:
    """A single retrieval result: content plus its similarity score."""

    text: str
    metadata: dict
    score: float


def _normalise(vector: list[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(array)
    if norm == 0:
        return array
    return array / norm


class VectorStore:
    """In-memory cosine similarity store with optional persistence."""

    def __init__(self) -> None:
        self._vectors: list[np.ndarray] = []
        self._texts: list[str] = []
        self._metadata: list[dict] = []
        self._dimension: int | None = None

    def __len__(self) -> int:
        return len(self._texts)

    def add(self, vector: list[float], text: str, metadata: dict | None = None) -> None:
        normalised = _normalise(vector)
        if self._dimension is None:
            self._dimension = int(normalised.shape[0])
        if int(normalised.shape[0]) != self._dimension:
            raise ValueError(
                f"Vector dimension {normalised.shape[0]} does not match "
                f"store dimension {self._dimension}"
            )
        self._vectors.append(normalised)
        self._texts.append(text)
        self._metadata.append(metadata or {})

    def search(self, vector: list[float], top_k: int = 4) -> list[SearchHit]:
        if not self._vectors:
            return []
        query = _normalise(vector)
        if self._dimension is not None and query.shape[0] != self._dimension:
            raise ValueError(
                f"Query dimension {query.shape[0]} does not match store dimension "
                f"{self._dimension}"
            )
        matrix = np.stack(self._vectors)
        scores = matrix @ query

        k = min(top_k, len(self._vectors))
        order = np.argsort(scores)[::-1][:k]
        return [
            SearchHit(
                text=self._texts[int(i)],
                metadata=self._metadata[int(i)],
                score=float(scores[int(i)]),
            )
            for i in order
        ]

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self._vectors:
            np.save(directory / "vectors.npy", np.stack(self._vectors))
        else:
            np.save(directory / "vectors.npy", np.zeros((0, 1), dtype=np.float32))
        sidecar = {
            "texts": self._texts,
            "metadata": self._metadata,
            "dimension": self._dimension,
        }
        (directory / "index.json").write_text(
            json.dumps(sidecar, indent=2), encoding="utf-8"
        )
        return directory

    def load(self, directory: str | Path) -> "VectorStore":
        directory = Path(directory)
        vectors = np.load(directory / "vectors.npy")
        sidecar = json.loads((directory / "index.json").read_text(encoding="utf-8"))

        self._vectors = [vectors[i] for i in range(vectors.shape[0])]
        self._texts = sidecar["texts"]
        self._metadata = sidecar["metadata"]
        self._dimension = sidecar.get("dimension")
        return self