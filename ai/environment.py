"""Gymnasium environment wrapper around the existing PROXIMAL FLAP simulation."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ai.observations import OBSERVATION_HIGH, OBSERVATION_LOW, observe
from ai.rewards import RewardConfig, calculate_reward
from game.world import GameWorld


class ProximalFlapEnv(gym.Env[np.ndarray, int]):
    """One bird, one episode, and the real deterministic PROXIMAL FLAP physics.

    Actions:
        0: coast for one simulation frame.
        1: flap, then advance one simulation frame.

    The environment is intentionally headless. It does not initialize Pygame or
    render frames, so training can run much faster than real time.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        max_episode_steps: int = 20_000,
        reward_config: RewardConfig | None = None,
    ) -> None:
        super().__init__()
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be at least 1")

        self.max_episode_steps = int(max_episode_steps)
        self.reward_config = reward_config or RewardConfig()
        self.action_space = spaces.Discrete(2)

        # Bounds are shared with the observation encoder to prevent drift.
        self.observation_space = spaces.Box(
            low=OBSERVATION_LOW, high=OBSERVATION_HIGH, dtype=np.float32
        )
        self.render_mode = None

        self.world: GameWorld | None = None
        self.episode_seed: int | None = None
        self.episode_steps = 0
        self._episode_over = True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a fresh course; passing a seed reproduces the same course."""
        super().reset(seed=seed)
        del options  # Reserved for future course variants; no options yet.

        self.episode_seed = int(self.np_random.integers(0, 2**32 - 1))
        if self.world is None:
            self.world = GameWorld(seed=self.episode_seed)
        else:
            self.world.reset(seed=self.episode_seed)

        self.episode_steps = 0
        self._episode_over = False
        observation = observe(self.world)
        info = self._build_info(scored=False, terminated=False, truncated=False)
        return observation, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one discrete action and advance exactly one game frame."""
        if self.world is None:
            raise RuntimeError("Call reset() before step().")
        if self._episode_over:
            raise RuntimeError("Episode ended; call reset() before stepping again.")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}; expected 0 (coast) or 1 (flap).")

        if int(action) == 1:
            self.world.flap()

        result = self.world.step(autopilot=False)
        self.episode_steps += 1
        terminated = bool(result.died)
        truncated = bool(
            not terminated and self.episode_steps >= self.max_episode_steps
        )
        reward = calculate_reward(
            scored=result.scored,
            terminated=terminated,
            config=self.reward_config,
        )
        self._episode_over = terminated or truncated

        observation = observe(self.world)
        info = self._build_info(
            scored=result.scored,
            terminated=terminated,
            truncated=truncated,
        )
        return observation, reward, terminated, truncated, info

    def _build_info(
        self,
        *,
        scored: bool,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        assert self.world is not None
        bird = self.world.bird
        next_pipe = next(
            (
                pipe
                for pipe in self.world.pipes
                if pipe.right >= bird.x - bird.radius_x
            ),
            None,
        )
        return {
            "episode_seed": self.episode_seed,
            "episode_steps": self.episode_steps,
            "score": self.world.score,
            "frames": self.world.frames,
            "difficulty": float(self.world.difficulty),
            "bird_y": float(bird.y),
            "bird_velocity": float(bird.velocity),
            "next_pipe_dx": (
                float(next_pipe.x - bird.x) if next_pipe is not None else None
            ),
            "next_gap_center": (
                float(next_pipe.gap_center) if next_pipe is not None else None
            ),
            "next_gap_size": (
                int(next_pipe.gap_size) if next_pipe is not None else None
            ),
            "scored": bool(scored),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "termination_reason": (
                "collision" if terminated else "time_limit" if truncated else None
            ),
        }

    def render(self) -> None:
        """Rendering is intentionally omitted from this headless training env."""
        return None

    def close(self) -> None:
        """No graphical or external resources are allocated by this environment."""
        return None
