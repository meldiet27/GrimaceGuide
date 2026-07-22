"""Persistence layer — hides SQLite/Postgres behind a Protocol."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Protocol

from grimaceguide.core.models import (
    ActionUnitBreakdown,
    ActionUnitScore,
    GrimaceResult,
)


class ResultRepository(Protocol):
    def save(self, result: GrimaceResult) -> int: ...

    def list_recent(self, limit: int = 20) -> list[GrimaceResult]: ...


class SQLiteResultRepository:
    """SQLite-backed implementation of ResultRepository."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses
                (
                    id
                    INTEGER
                    PRIMARY
                    KEY
                    AUTOINCREMENT,
                    total
                    INTEGER
                    NOT
                    NULL,
                    normalized
                    REAL
                    NOT
                    NULL,
                    pain_likely
                    INTEGER
                    NOT
                    NULL,
                    breakdown_json
                    TEXT
                    NOT
                    NULL,
                    processing_ms
                    REAL
                    NOT
                    NULL,
                    created_at
                    TEXT
                    NOT
                    NULL
                )
                """
            )

    def save(self, result: GrimaceResult) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO analyses
                (total, normalized, pain_likely, breakdown_json,
                 processing_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    result.breakdown.total,
                    result.breakdown.normalized,
                    int(result.pain_likely),
                    json.dumps(result.breakdown.as_dict()),
                    result.processing_ms,
                    result.created_at.isoformat(),
                ),
            )
            return int(cur.lastrowid)

    def list_recent(self, limit: int = 20) -> list[GrimaceResult]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()

        results: list[GrimaceResult] = []
        for row in rows:
            data = json.loads(row["breakdown_json"])
            breakdown = ActionUnitBreakdown(
                ears=ActionUnitScore(data["ears"]),
                eyes=ActionUnitScore(data["eyes"]),
                muzzle=ActionUnitScore(data["muzzle"]),
                whiskers=ActionUnitScore(data["whiskers"]),
                head=ActionUnitScore(data["head"]),
            )
            results.append(
                GrimaceResult(
                    breakdown=breakdown,
                    pain_likely=bool(row["pain_likely"]),
                    processing_ms=float(row["processing_ms"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
            )
        return results
