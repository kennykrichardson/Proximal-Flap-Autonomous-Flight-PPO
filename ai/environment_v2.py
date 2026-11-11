"""V2 Gymnasium environment. Does not alter the original champion's contract."""
from __future__ import annotations
from typing import Any
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ai.observations_v2 import OBSERVATION_HIGH_V2, OBSERVATION_LOW_V2, OBSERVATION_SIZE_V2, observe_v2
from ai.rewards_v2 import RewardConfigV2, alignment_potential, calculate_reward_v2
from game.world import GameWorld

class ProximalFlapEnvV2(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}
    def __init__(self, *, max_episode_steps: int = 20_000,
                 reward_config: RewardConfigV2 | None = None):
        super().__init__()
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be at least 1")
        self.max_episode_steps = int(max_episode_steps)
        self.reward_config = reward_config or RewardConfigV2()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(OBSERVATION_LOW_V2, OBSERVATION_HIGH_V2, dtype=np.float32)
        self.world = None
        self.episode_seed = None
        self.episode_steps = 0
        self._episode_over = True
        self._potential = 0.0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        del options
        self.episode_seed = int(self.np_random.integers(0, 2**32 - 1))
        if self.world is None:
            self.world = GameWorld(seed=self.episode_seed)
        else:
            self.world.reset(seed=self.episode_seed)
        self.episode_steps = 0
        self._episode_over = False
        self._potential = alignment_potential(self.world)
        return observe_v2(self.world), self._info(False, False, False)

    def step(self, action: int):
        if self.world is None or self._episode_over:
            raise RuntimeError("Call reset() before stepping or after an episode ends.")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}; expected 0 or 1.")
        if int(action) == 1:
            self.world.flap()
        result = self.world.step(autopilot=False)
        self.episode_steps += 1
        terminated = bool(result.died)
        truncated = bool(not terminated and self.episode_steps >= self.max_episode_steps)
        next_potential = alignment_potential(self.world)
        reward = calculate_reward_v2(
            scored=result.scored, terminated=terminated,
            previous_potential=self._potential, next_potential=next_potential,
            config=self.reward_config,
        )
        self._potential = next_potential
        self._episode_over = terminated or truncated
        return observe_v2(self.world), reward, terminated, truncated, self._info(result.scored, terminated, truncated)

    def _info(self, scored: bool, terminated: bool, truncated: bool):
        assert self.world is not None
        return {"episode_seed": self.episode_seed, "episode_steps": self.episode_steps,
                "score": self.world.score, "frames": self.world.frames,
                "difficulty": float(self.world.difficulty), "scored": bool(scored),
                "terminated": bool(terminated), "truncated": bool(truncated),
                "termination_reason": "collision" if terminated else "time_limit" if truncated else None}
    def render(self):
        return None
    def close(self):
        return None
