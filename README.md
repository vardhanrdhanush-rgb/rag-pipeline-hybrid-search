# Hybrid Search RAG Pipeline

A production-style **Retrieval-Augmented Generation (RAG)** pipeline combining BM25 keyword search and dense vector search for grounded, accurate answers over internal document corpora.

## What it does

- Ingests PDF/text documents and indexes them into a vector store
- Runs **hybrid retrieval** (BM25 + semantic embeddings) at query time
- Reranks results with Cohere for higher precision
- Generates answers using **Anthropic Claude** — grounded only in retrieved context, no hallucination

## Tech stack

| Layer | Tool |
|---|---|
| Generation | Anthropic Claude (claude-3-5-sonnet) |
| Retrieval | RAGLite — BM25 + text-embedding-3-small |
| Reranking | Cohere Reranker |
| UI | Streamlit |

## Why hybrid search

Pure semantic search misses exact keyword matches (product codes, IDs, regulatory terms). Pure BM25 misses paraphrased queries. Combining both consistently outperforms either alone on enterprise corpora.

## Setup

```bash
pip install -r requirements.txt
streamlit run main.py
```

## Skills demonstrated

- RAG pipeline architecture and implementation
- Hybrid retrieval (BM25 + dense vectors)
- Anthropic Claude API integration
- Vector database indexing and semantic search
- Streamlit application development
