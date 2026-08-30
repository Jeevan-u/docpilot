# DocPilot

Ask questions in plain English and get answers drawn from **your own documents** — with the exact source passages cited as evidence.

DocPilot is a small, readable implementation of **retrieval-augmented generation (RAG)**. It loads your PDFs, notes and Markdown files, builds a searchable semantic index, then answers questions by retrieving the most relevant passages and grounding a language model on exactly those passages. The whole codebase is a few hundred lines, deliberately dependency-light, and easy to follow module by module.

---

## Why this project exists

RAG is one of the most requested skills in applied LLM work, and most tutorials hand you a framework that hides every interesting detail behind a magic import. This project is the opposite: every stage is a small, named module you can read in a minute, and the math behind semantic search is written out instead of imported.

What you can point at and say "I built this":

- **Token-aware chunking** — splits documents on paragraph and sentence boundaries, keeps chunks inside an LLM context budget, and preserves an overlap so no idea gets cut in half.
- **Semantic search from scratch** — embeddings are L2-normalised and scored with a single NumPy matrix multiplication (cosine similarity). No vector database required, and the `VectorStore` interface is small enough to swap for a real one later.
- **Grounded generation** — the model only ever sees the top-k retrieved excerpts, is forbidden from using outside knowledge, and is asked to cite a numbered source for every claim.
- **Measurable quality** — a small offline benchmark reports retrieval hit-rate and answer coverage, so changes can be proven to help or hurt.

## How it works

```
 your documents
      │
      ▼
 ┌────────────┐   ┌──────────────┐   ┌──────────────┐
 │  loader.py │──▶│  chunker.py  │──▶│ embeddings.py│
 │ (PDF, md,  │   │ (token-aware │   │  (OpenAI     │
 │  txt)      │   │  + overlap)  │   │   API)       │
 └────────────┘   └──────────────┘   └──────────────┘
                                            │
                                            ▼
                                    ┌──────────────┐    persisted to disk?
                                    │   store.py   │◀──▶ data/index/*
                                    │ (cosine sim) │
                                    └──────────────┘
                                            ▲
              your question ───────────────┘
                    │
                    ▼
             ┌────────────┐   top-k hits   ┌──────────────┐
             │ retriever  │───────────────▶│  generator   │──▶ answer + sources
             │ .py        │                │  .py (LLM)   │
             └────────────┘                └──────────────┘
```

The pipeline (`pipeline.py`) wires these together; the CLI (`main.py`) exposes everything as three commands.

## Quickstart

Requires Python 3.10+ and an OpenAI API key.

```bash
# 1. install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. set your key
cp .env.example .env          # then add your OPENAI_API_KEY

# 3. index some documents (a sample file is included)
python main.py index sample_docs

# 4. ask a question
python main.py ask "What is retrieval-augmented generation?"
```

### Example output

```
$ python main.py index sample_docs
Loaded 1 document(s) from sample_docs
Indexed 11 chunks into data/index

$ python main.py ask "Why should a question be matched against the top few passages?"
RAG narrows the corpus down to a small set of relevant passages before
generation so the model never reasons over the whole collection at once [1].
Because the model is only shown those excerpts, every claim is tied to a
numbered source [1].

Sources:
  - sample_docs/intro.txt
```

### Commands

| Command | What it does |
| --- | --- |
| `python main.py index <file-or-dir>` | Load, chunk, embed and store your documents |
| `python main.py ask "<question>"` | Retrieve best matches and answer with citations |
| `python main.py eval --questions qs.json` | Run the offline benchmark (hit rate + coverage) |

Evaluation questions live in a small JSON file:

```json
[
  {"question": "What is the advantage of semantic search?", "key_phrase": "meaning"}
]
```

```bash
python main.py eval --questions eval_questions.json --generate-answers
```

## Project layout

```
docpilot/
├── main.py                  CLI: index / ask / eval
├── docpilot/
│   ├── config.py            one place for every setting
│   ├── loader.py            PDF + text + Markdown loading
│   ├── chunker.py           token-aware splitting with overlap
│   ├── embeddings.py        text → vector providers
│   ├── store.py             cosine-similarity vector store + persistence
│   ├── retriever.py         rank chunks against a question
│   ├── generator.py         answer generation grounded on sources
│   ├── pipeline.py          wires the stages together
│   └── evaluate.py          offline benchmark metrics
├── sample_docs/             sample document to try it on
└── tests/                   pytest suite (no network required)
```

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests use a deterministic hash-based embedder and an in-memory stand-in for the
language model, so the whole suite runs offline and fast.

## Ideas to extend it

- Swap the vector store for a real vector database via the small `VectorStore` interface.
- Add hybrid retrieval: combine keyword (BM25) and semantic scores.
- Add a reranking step over the top-k candidates.
- Add a local embedding provider (e.g. sentence-transformers) behind the `Embedder` interface.
- Turn the CLI into a small FastAPI service with a `/ask` endpoint and a chat UI.

## License

[MIT](LICENSE)