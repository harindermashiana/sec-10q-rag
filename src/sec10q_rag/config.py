from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """
    Project configuration.

    Notes:
      - SEC requests should include a descriptive User-Agent per SEC guidance.
      - Storage paths are local, so you can run repeat queries without re-indexing.
    """
    user_agent: str
    embedding_model_name: str = "all-MiniLM-L6-v2"
    dim: int = 384

    data_dir: Path = Path("data")
    registry_path: Path = Path("data/ingested_index.json")
    faiss_index_path: Path = Path("data/faiss.index")
    text_store_path: Path = Path("data/text_store.jsonl")
    meta_store_path: Path = Path("data/meta_store.jsonl")


def ensure_data_dir(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
