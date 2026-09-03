"""Filesystem helpers: hashing for cache keys, safe JSON I/O."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_fingerprint(path: Path) -> str:
    """Cheap, stable fingerprint for cache keys: name + size + mtime, hashed.

    Avoids hashing full video contents (could be GBs) while still detecting
    if the source file changed.
    """
    stat = path.stat()
    raw = f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def stable_hash(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def list_input_videos(input_dir: Path, extensions: set[str]) -> list[Path]:
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )


def safe_stem(path: Path) -> str:
    """Filesystem-safe stem for building output/cache filenames."""
    keep = "-_.() "
    stem = "".join(c for c in path.stem if c.isalnum() or c in keep)
    return stem.strip() or "video"
