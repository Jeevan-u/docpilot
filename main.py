#!/usr/bin/env python3
"""Command-line interface for docpilot.

Examples:

    python main.py index sample_docs
    python main.py ask "How does retrieval work?"
    python main.py eval --questions eval_questions.json --generate-answers
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from docpilot.config import Config
from docpilot.evaluate import evaluate, load_questions
from docpilot.loader import DocumentLoader
from docpilot.pipeline import RAGPipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docpilot",
        description="Query your documents in plain language using RAG.",
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, metavar="{index,ask,eval}"
    )

    index = subparsers.add_parser(
        "index", help="Load documents and build the search index."
    )
    index.add_argument("source", help="A document file or directory to index.")
    index.add_argument("--out", default=None, help="Directory to persist the index in.")

    ask = subparsers.add_parser("ask", help="Answer a question from the index.")
    ask.add_argument("question", help="The question to answer.")
    ask.add_argument(
        "--index", default=None, help="Directory that holds a built index."
    )
    ask.add_argument(
        "--top-k", type=int, default=None, help="Override the number of contexts."
    )

    eval_ = subparsers.add_parser(
        "eval", help="Run a question benchmark against the index."
    )
    eval_.add_argument("--questions", required=True, help="JSON file of questions.")
    eval_.add_argument(
        "--index", default=None, help="Directory that holds a built index."
    )
    eval_.add_argument(
        "--top-k", type=int, default=None, help="Retrieval depth for the metric."
    )
    eval_.add_argument(
        "--generate-answers",
        action="store_true",
        help="Also generate answers and measure coverage (uses the LLM).",
    )
    return parser


def _command_index(args: argparse.Namespace, config: Config) -> int:
    pipeline = RAGPipeline(config)
    documents = DocumentLoader().load(args.source)
    print(f"Loaded {len(documents)} document(s) from {args.source}")

    chunk_count = pipeline.index(documents)
    out = Path(args.out) if args.out else config.index_dir
    pipeline.persist(out)
    print(f"Indexed {chunk_count} chunks into {out}")
    return 0


def _command_ask(args: argparse.Namespace, config: Config) -> int:
    index_dir = Path(args.index) if args.index else config.index_dir
    if not _index_exists(index_dir):
        print(
            f"Index not found at {index_dir}. Run `python main.py index` first.",
            file=sys.stderr,
        )
        return 1

    pipeline = RAGPipeline(config).load(index_dir)
    answer = pipeline.answer(args.question, top_k=args.top_k)

    print(answer.text)
    print()
    print("Sources:")
    for source in sorted({s.get("source") for s in answer.sources}):
        print(f"  - {source}")
    return 0


def _command_eval(args: argparse.Namespace, config: Config) -> int:
    index_dir = Path(args.index) if args.index else config.index_dir
    if not _index_exists(index_dir):
        print(
            f"Index not found at {index_dir}. Run `python main.py index` first.",
            file=sys.stderr,
        )
        return 1

    pipeline = RAGPipeline(config).load(index_dir)
    rows = load_questions(args.questions)

    answer_fn = None
    if args.generate_answers:
        answer_fn = lambda q: pipeline.answer(q, top_k=args.top_k).text

    result = evaluate(pipeline.retriever, rows, answer_fn=answer_fn, top_k=args.top_k)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def _index_exists(path: Path) -> bool:
    return (path / "vectors.npy").is_file() and (path / "index.json").is_file()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = Config()

    try:
        config.require_api_key()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.command == "index":
        return _command_index(args, config)
    if args.command == "ask":
        return _command_ask(args, config)
    if args.command == "eval":
        return _command_eval(args, config)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
