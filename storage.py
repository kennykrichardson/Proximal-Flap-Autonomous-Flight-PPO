"""Tiny JSON-backed best-score store."""

from __future__ import annotations

import json
from pathlib import Path

from config import DATA_DIR


class ScoreStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DATA_DIR / "best_score.json"
        self.best_score = 0
        self.load()

    def load(self) -> int:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.best_score = max(0, int(payload.get("best_score", 0)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            self.best_score = 0
        return self.best_score

    def save_if_better(self, score: int) -> bool:
        if score <= self.best_score:
            return False
        self.best_score = int(score)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps({"best_score": self.best_score}, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.path)
        except OSError:
            # Gameplay should never fail just because the score file is read-only.
            pass
        return True
