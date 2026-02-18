from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import Settings, ensure_data_dir
from .parsing import chunk_text, filing_html_to_text
from .sec_client import SecClient
from .storage import (
    Registry,
    append_jsonl,
    as_float32_matrix,
    load_faiss_index,
    read_jsonl,
    save_faiss_index,
)


@dataclass
class RetrievedChunk:
    text: str
    meta: Dict


class Sec10QRAG:
    """
    Minimal Retrieval-Augmented Generation pipeline for SEC 10-Q filings:
      - fetch SEC filing
      - parse to text
      - chunk & embed
      - FAISS search at query time
    """

    def __init__(self, settings: Settings) -> None:
        ensure_data_dir(settings)
        self.settings = settings
        self.client = SecClient(headers={"User-Agent": settings.user_agent})

        self.model = SentenceTransformer(settings.embedding_model_name)
        self.index = load_faiss_index(settings.faiss_index_path, settings.dim)

        # stores are persisted as JSONL and loaded into memory
        self.text_store: List[str] = [x["text"] for x in read_jsonl(settings.text_store_path)]
        self.meta_store: List[Dict] = read_jsonl(settings.meta_store_path)
        self.registry = Registry.load(settings.registry_path)

    def _key(self, ticker: str, year: int, quarter: str) -> str:
        return f"{ticker.upper()}_{year}_{quarter.upper()}"

    def ensure_indexed(self, ticker: str, year: int, quarter: str) -> None:
        key = self._key(ticker, year, quarter)

        if key in self.registry.data:
            return

        cik = self.client.ticker_to_cik(ticker)
        filing_url = self.client.get_10q_filing_url(cik, year, quarter)

        html = self.client.fetch_html(filing_url)
        text = filing_html_to_text(html)
        chunks = chunk_text(text)

        embeddings = self.model.encode(chunks)
        vecs = as_float32_matrix(embeddings)

        # add to FAISS
        self.index.add(vecs)

        # append to stores
        text_items = [{"text": c} for c in chunks]
        meta_items = []
        base_idx = len(self.meta_store)

        for i in range(len(chunks)):
            meta_items.append(
                {
                    "key": key,
                    "ticker": ticker.upper(),
                    "year": year,
                    "quarter": quarter.upper(),
                    "chunk_id": i,
                    "global_id": base_idx + i,
                    "source_url": filing_url,
                }
            )

        append_jsonl(self.settings.text_store_path, text_items)
        append_jsonl(self.settings.meta_store_path, meta_items)

        self.text_store.extend(chunks)
        self.meta_store.extend(meta_items)

        # save registry and index
        self.registry.data[key] = {
            "ticker": ticker.upper(),
            "year": year,
            "quarter": quarter.upper(),
            "chunks": len(chunks),
            "source_url": filing_url,
        }
        self.registry.save()
        save_faiss_index(self.settings.faiss_index_path, self.index)

    def retrieve(self, question: str, k: int = 5) -> List[RetrievedChunk]:
        q_emb = as_float32_matrix(self.model.encode([question]))
        _, I = self.index.search(q_emb, k)

        results: List[RetrievedChunk] = []
        for idx in I[0]:
            # faiss returns -1 if no result
            if idx < 0 or idx >= len(self.text_store):
                continue
            results.append(RetrievedChunk(text=self.text_store[idx], meta=self.meta_store[idx]))
        return results

    @staticmethod
    def build_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
        context_parts: List[str] = []
        for i, item in enumerate(chunks, 1):
            m = item.meta
            context_parts.append(
                f"[Source {i} | {m['ticker']} {m['year']} {m['quarter']} | chunk {m['chunk_id']}]\n"
                f"{item.text}\n"
            )

        context = "\n".join(context_parts)
        return (
            "You are answering questions about a company's 10-Q filing.\n"
            "Use ONLY the context below. Cite sources like [Source 1].\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )

    def answer(self, ticker: str, year: int, quarter: str, question: str, k: int = 5) -> Tuple[str, List[RetrievedChunk]]:
        """
        Returns:
          - prompt-only answer placeholder (LLM call optional)
          - retrieved chunks used as sources
        """
        self.ensure_indexed(ticker, year, quarter)
        retrieved = self.retrieve(question, k=k)
        prompt = self.build_prompt(question, retrieved)

        # Plug in your preferred LLM provider here (OpenAI, Groq, etc.)
        # For showcasing, recruiters love seeing that you kept the LLM boundary clean.
        answer = "(LLM call disabled in this repo example)\n\n" + prompt
        return answer, retrieved
