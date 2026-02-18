[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000)](https://github.com/astral-sh/ruff)
[![Last commit](https://img.shields.io/github/last-commit/harindermashiana/sec-10q-rag)](https://github.com/harindermashiana/sec-10q-rag/commits/main)

# SEC-Insight

## RAG-Powered Financial Filing Intelligence

SEC-Insight is an end-to-end Retrieval-Augmented Generation (RAG) system that ingests SEC 10-Q filings, converts them into structured knowledge, and enables semantic search and LLM-assisted analysis over financial disclosures.

This project demonstrates practical AI engineering by combining real-world data ingestion, NLP pipelines, vector search, and modern LLM workflows into a production-style architecture.

---

## Overview

Financial filings are dense, lengthy documents that require significant manual effort to analyze. SEC-Insight automates this process by:

* Programmatically retrieving filings from the SEC EDGAR API
* Parsing HTML/XBRL documents into structured text
* Generating semantic embeddings for document understanding
* Performing high-speed similarity search using FAISS
* Constructing citation-ready context for LLM reasoning

The result is a system that allows users to query financial filings using natural language.

---

## Architecture

```
        SEC EDGAR API
              |
              v
     Filing Downloader (CIK Lookup)
              |
              v
     HTML/XBRL Parsing and Cleaning
      (BeautifulSoup + Structuring)
              |
              v
          Text Chunking
              |
              v
   SentenceTransformer Embeddings
              |
              v
        FAISS Vector Index
              |
              v
      Semantic Retrieval (Top-K)
              |
              v
        RAG Prompt Builder
              |
              v
        LLM Reasoning Layer
```

### Design Principles

* Separation of ingestion, retrieval, and generation stages
* Persistent indexing to prevent redundant downloads
* Provider-agnostic LLM integration
* Lightweight local execution
* Modular, testable Python package structure

---

## Tech Stack

| Layer           | Technology                                 |
| --------------- | ------------------------------------------ |
| Language        | Python 3.9+                                |
| Embeddings      | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector Database | FAISS (Facebook AI Similarity Search)      |
| Data Source     | SEC EDGAR REST API                         |
| Parsing         | BeautifulSoup4, lxml                       |
| Processing      | NumPy, Requests, Regex                     |
| LLM Compatible  | GPT-4o, Claude, or local models            |

---

## Project Structure

```
sec-10q-rag/
│
├── src/
│   └── sec10q_rag/
│       ├── config.py
│       ├── sec_client.py
│       ├── parsing.py
│       ├── rag.py
│       ├── storage.py
│       └── cli.py
│
├── examples/
│   └── demo.py
│
├── tests/
│   └── test_chunking.py
│
├── pyproject.toml
├── README.md

```

The `src/` layout follows modern Python packaging practices and ensures reliable imports and deployment behavior.

---

## Setup and Usage

### Clone the repository

```bash
git clone https://github.com/yourusername/sec-10q-rag.git
cd sec-10q-rag
```

### Install dependencies

```bash
pip install -e .
```

### Configure SEC User Agent

The SEC requires identification for automated requests:

```python
user_agent="your_email@example.com"
```

### Run example

```bash
python examples/demo.py
```

Or via CLI:

```bash
python -m sec10q_rag.cli \
  --user-agent "your_email@example.com" \
  --ticker AAPL \
  --year 2024 \
  --quarter Q1 \
  --question "What risk factors were mentioned?"
```

---

## Example Queries

* What risk factors were highlighted this quarter?
* What operational challenges were disclosed?
* Where are supply chain risks discussed?
* What financial risks increased year over year?

---


## Roadmap

* Multi-document comparison (10-Q vs 10-K)
* Interactive Streamlit dashboard
* Advanced financial table extraction using pandas
* Hybrid retrieval (BM25 + embeddings)
* Cross-encoder reranking for improved precision

---

## Motivation

Financial disclosures contain critical insights but remain difficult to explore efficiently. SEC-Insight shows how modern AI systems can transform unstructured regulatory filings into searchable financial intelligence.

Built at the intersection of AI Engineering, NLP, and FinTech.

