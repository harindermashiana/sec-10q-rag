import pytest
from sec10q_rag.parsing import chunk_text


def test_chunk_text_basic():
    text = "a" * 2000
    chunks = chunk_text(text, max_chars=800, overlap=100)
    assert len(chunks) >= 3
    assert chunks[0] == "a" * 800


def test_chunk_text_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello", max_chars=10, overlap=10)
