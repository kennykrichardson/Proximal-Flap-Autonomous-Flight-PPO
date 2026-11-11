"""Initial, deliberately simple reward function for PPO experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Reward weights; keep these versioned with experiment metadata."""

    survival: float = 0.01
    pipe_passed: float = 10.0
    collision: float = -20.0


def calculate_reward(
    *,
    scored: bool,
    terminated: bool,
    config: RewardConfig = RewardConfig(),
) -> float:
    """Reward one simulation frame.

    Survival is a small per-frame signal, passing a pipe is the main positive
    event, and collision ends the episode with a penalty. Time-limit truncation
    is not treated as a collision.
    """
    reward = config.survival
    if scored:
        reward += config.pipe_passed
    if terminated:
        reward += config.collision
    return float(reward)
