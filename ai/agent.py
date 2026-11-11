"""Small inference wrapper for trained PROXIMAL FLAP PPO policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO


class PPOBirdAgent:
    """Load a PPO checkpoint and choose coast/flap actions from observations."""

    def __init__(self, model_path: str | Path, *, device: str = "auto") -> None:
        path = Path(model_path)
        if path.suffix != ".zip" and path.with_suffix(".zip").exists():
            path = path.with_suffix(".zip")
        if not path.exists():
            raise FileNotFoundError(f"PPO checkpoint not found: {path}")
        self.model = PPO.load(path, device=device)

    def act(self, observation: np.ndarray, *, deterministic: bool = True) -> int:
        action, _state = self.model.predict(
            np.asarray(observation, dtype=np.float32),
            deterministic=deterministic,
        )
        return int(np.asarray(action).item())

    def predict(
        self, observation: np.ndarray, state: Any = None, *,
        episode_start: bool = False, deterministic: bool = True
    ) -> tuple[int, Any]:
        action, next_state = self.model.predict(
            np.asarray(observation, dtype=np.float32),
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        return int(np.asarray(action).item()), next_state
