from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import faiss
import numpy as np


@dataclass
class Registry:
    path: Path
    data: Dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "Registry":
        if path.exists():
            return cls(path=path, data=json.loads(path.read_text()))
        return cls(path=path, data={})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))


def append_jsonl(path: Path, items: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_faiss_index(path: Path, index: faiss.Index) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_faiss_index(path: Path, dim: int) -> faiss.Index:
    if path.exists():
        return faiss.read_index(str(path))
    return faiss.IndexFlatL2(dim)


def as_float32_matrix(vectors: Any) -> np.ndarray:
    arr = np.array(vectors, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr
