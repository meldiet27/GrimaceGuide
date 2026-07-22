"""File/blob storage adapter — local now, swap for S3 later."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol


class BlobStorage(Protocol):
    def put(self, key: str, data: bytes) -> str: ...

    def get(self, key: str) -> bytes: ...


class LocalBlobStorage:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, data: bytes) -> str:
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return str(target)

    def get(self, key: str) -> bytes:
        return (self.root / key).read_bytes()
