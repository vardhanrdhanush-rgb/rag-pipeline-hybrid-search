"""
Hybrid Search RAG Pipeline
--------------------------
Production-style Retrieval-Augmented Generation combining BM25 keyword
search and dense vector search for grounded, accurate answers over large
internal document corpora.

Stack: Anthropic Claude · RAGLite · Cohere reranker · Streamlit
"""

import os
import logging
import warnings
from pathlib import Path
from typing import List

import streamlit as st
from raglite import RAGLiteConfig, insert_document, hybrid_search, retrieve_chunks, rerank_chunks, rag
from rerankers import Reranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*torch.classes.*")

RAG_SYSTEM_PROMPT = """
You are a knowledgeable assistant that provides accurate, grounded answers.
Answer only using the context provided. Do not reference the context directly.
If the answer is not in the context, say so clearly.
""".strip()


def initialize_config(openai_key: str, anthropic_key: str, cohere_key: str, db_url: str) -> RAGLiteConfig:
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    os.environ["CO_API_KEY"] = cohere_key
    return RAGLiteConfig(
        db_url=db_url,
        llm="claude-3-5-sonnet-20241022",
        embedder="text-embedding-3-small",
        reranker=Reranker("cohere", lang="en", api_key=cohere_key),
    )


def ingest_document(file_path: str, config: RAGLiteConfig) -> None:
    logger.info(f"Ingesting: {file_path}")
    insert_document(Path(file_path), config=config)
    logger.info("Done.")


def query_pipeline(question: str, config: RAGLiteConfig, top_k: int = 5) -> str:
    chunk_ids, _ = hybrid_search(question, num_results=top_k, config=config)
    chunks = retrieve_chunks(chunk_ids, config=config)
    chunks = rerank_chunks(question, chunks, config=config)
    return rag(question, chunks, config=config, system_prompt=RAG_SYSTEM_PROMPT)


st.set_page_config(page_title="Hybrid Search RAG", page_icon="🔍", layout="wide")
st.title("🔍 Hybrid Search RAG Pipeline")
st.caption("Keyword + semantic retrieval grounded by Anthropic Claude.")

with st.sidebar:
    st.header("API Keys")
    openai_key    = st.text_input("OpenAI API Key",    type="password")
    anthropic_key = st.text_input("Anthropic API Key", type="password")
    cohere_key    = st.text_input("Cohere API Key",    type="password")
    db_url        = st.text_input("Database URL", value="sqlite:///rag.db")
    uploaded      = st.file_uploader("Upload document (PDF/TXT)", type=["pdf", "txt"])

    if st.button("Ingest") and uploaded and all([openai_key, anthropic_key, cohere_key]):
        config = initialize_config(openai_key, anthropic_key, cohere_key, db_url)
        tmp = f"/tmp/{uploaded.name}"
        with open(tmp, "wb") as f:
            f.write(uploaded.read())
        with st.spinner("Indexing..."):
            ingest_document(tmp, config)
        st.success("Indexed.")

question = st.text_input("Ask a question:")
if st.button("Answer") and question:
    if not all([openai_key, anthropic_key, cohere_key]):
        st.warning("Enter all API keys.")
    else:
        config = initialize_config(openai_key, anthropic_key, cohere_key, db_url)
        with st.spinner("Retrieving..."):
            answer = query_pipeline(question, config)
        st.markdown("### Answer")
        st.write(answer)
