from __future__ import annotations

import re
from bs4 import BeautifulSoup


REMOVE_TAGS = {"script", "style", "noscript", "footer", "nav"}


def filing_html_to_text(html: str) -> str:
    """
    Convert SEC filing HTML to a structured plain-text representation.

    Includes:
      - [SECTION] markers for headings
      - [TABLE] blocks with row lines
    """
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(list(REMOVE_TAGS)):
        tag.decompose()

    blocks: list[str] = []
    for tag in soup.find_all(["h1", "h2", "h3", "p", "table", "span"]):
        name = tag.name.lower()

        if name in {"h1", "h2", "h3"}:
            blocks.append(f"\n[SECTION] {tag.get_text(' ', strip=True)}")
            continue

        if name in {"p", "span"}:
            txt = tag.get_text(" ", strip=True)
            if len(txt) > 40:
                blocks.append(txt)
            continue

        if name == "table":
            rows: list[str] = []
            for tr in tag.find_all("tr"):
                cells = [
                    td.get_text(" ", strip=True)
                    for td in tr.find_all(["td", "th"])
                ]
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                blocks.append("[TABLE]\n" + "\n".join(rows))

    text = "\n".join(blocks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> list[str]:
    """
    Simple character-based chunker with overlap.

    max_chars: size of each chunk
    overlap: number of chars repeated between consecutive chunks
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= max_chars:
        raise ValueError("overlap must be < max_chars")

    chunks: list[str] = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + max_chars])
        i += max_chars - overlap
    return chunks
