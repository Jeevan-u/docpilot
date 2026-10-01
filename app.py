from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from docpilot.config import Config
from docpilot.loader import DocumentLoader
from docpilot.pipeline import RAGPipeline

st.set_page_config(page_title="DocPilot", page_icon="📚", layout="wide")
st.title("📚 DocPilot")
st.caption("Ask questions about your PDFs, Markdown, and text files with cited retrieval.")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = tempfile.TemporaryDirectory()

uploaded = st.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "md", "mdx"],
    accept_multiple_files=True,
)

if uploaded:
    upload_dir = Path(st.session_state.temp_dir.name) / "documents"
    upload_dir.mkdir(parents=True, exist_ok=True)

    for file in uploaded:
        (upload_dir / file.name).write_bytes(file.getbuffer())

    if st.button("Build document index", type="primary"):
        try:
            config = Config()
            pipeline = RAGPipeline(config)
            documents = DocumentLoader().load(upload_dir)
            chunk_count = pipeline.index(documents)
            st.session_state.pipeline = pipeline
            st.success(
                f"Indexed {len(documents)} document sections into {chunk_count} chunks."
            )
        except Exception as exc:
            st.error(f"Could not build the index: {exc}")

if st.session_state.pipeline is not None:
    question = st.text_input(
        "Ask a question",
        placeholder="What does the document say about ...?",
    )

    if st.button("Ask DocPilot") and question.strip():
        try:
            with st.spinner("Retrieving relevant passages and generating an answer..."):
                answer = st.session_state.pipeline.answer(question.strip())

            st.subheader("Answer")
            st.write(answer.text)

            st.subheader("Sources")
            seen = set()
            for source in answer.sources:
                label = source.get("source", "unknown")
                page = source.get("page")
                if page is not None:
                    label = f"{label} (page {page})"
                if label not in seen:
                    st.write(f"- {label}")
                    seen.add(label)
        except Exception as exc:
            st.error(f"Could not answer the question: {exc}")
else:
    st.info("Upload one or more documents and build the index to start.")
