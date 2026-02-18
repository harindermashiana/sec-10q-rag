from __future__ import annotations

import argparse

from .config import Settings
from .rag import Sec10QRAG


def main() -> None:
    p = argparse.ArgumentParser(description="SEC 10-Q RAG CLI")
    p.add_argument("--user-agent", required=True, help="SEC-compliant User-Agent (email or app contact).")
    p.add_argument("--ticker", required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--quarter", required=True, choices=["Q1", "Q2", "Q3", "Q4"])
    p.add_argument("--question", required=True)
    p.add_argument("--topk", type=int, default=5)

    args = p.parse_args()

    rag = Sec10QRAG(Settings(user_agent=args.user_agent))
    answer, sources = rag.answer(args.ticker, args.year, args.quarter, args.question, k=args.topk)

    print("\nANSWER\n-----")
    print(answer)

    print("\nSOURCES\n-------")
    for i, s in enumerate(sources, 1):
        m = s.meta
        print(f"[{i}] {m['ticker']} {m['year']} {m['quarter']} chunk={m['chunk_id']}")
        print(f"    url: {m.get('source_url')}")
        print()


if __name__ == "__main__":
    main()
