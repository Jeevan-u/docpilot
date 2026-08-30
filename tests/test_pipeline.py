"""End-to-end pipeline test with in-memory stand-ins for the API calls."""

from dataclasses import dataclass

from test_retriever import NGramHashEmbedder

from docpilot.generator import Answer, Generator
from docpilot.loader import Document
from docpilot.pipeline import RAGPipeline
from docpilot.retriever import Retriever
from docpilot.store import SearchHit, VectorStore


@dataclass
class _FakeChoice:
    message: object


@dataclass
class _FakeMessage:
    content: str


class _StandInGenerator(Generator):
    def __init__(self):
        self.last_question = None
        self.last_hits = None

    def generate(self, question: str, hits: list[SearchHit]) -> Answer:
        self.last_question = question
        self.last_hits = hits
        return Answer(text=f"Answer to: {question}", sources=[h.metadata for h in hits])


def _hook_pipeline(pipeline: RAGPipeline, generator: Generator) -> Retriever:
    pipeline.generator = generator
    return pipeline.retriever


def test_pipeline_indexes_and_answers_without_network():
    store = VectorStore()
    embedder = NGramHashEmbedder()
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.config = None
    pipeline.store = store
    pipeline.embedder = embedder
    pipeline.chunker = None

    from docpilot.chunker import TokenChunker

    pipeline.chunker = TokenChunker(chunk_size=200, overlap=20)
    pipeline.retriever = Retriever(store, embedder, top_k=2)
    generator = _StandInGenerator()
    pipeline.generator = generator

    doc = Document(
        text="Robust retrieval keeps only the top few passages for the model to read.",
        source="notes.txt",
    )
    chunk_count = pipeline.index([doc])
    assert chunk_count >= 1

    answer = pipeline.answer("How many passages are kept?")
    assert answer.text.startswith("Answer to:")
    assert generator.last_question == "How many passages are kept?"
    assert generator.last_hits
    assert generator.last_hits[0].metadata["source"] == "notes.txt"
