"""Version-2 PPO observations with time-to-contact and ballistic context.

This module is deliberately separate from observations.py: the original 12-value
observation contract belongs to the protected champion checkpoint.
"""
from __future__ import annotations

import numpy as np

from config import BIRD_GRAVITY, WINDOW_HEIGHT, WINDOW_WIDTH
from game.world import GameWorld

OBSERVATION_FEATURES_V2 = (
    "bird_y", "bird_velocity",
    "pipe_1_distance", "pipe_1_time_to_contact", "pipe_1_ballistic_y",
    "pipe_1_gap_center", "pipe_1_relative_center",
    "pipe_1_top_clearance", "pipe_1_bottom_clearance",
    "pipe_1_gap_size", "pipe_1_speed",
    "pipe_2_distance", "pipe_2_time_to_contact", "pipe_2_ballistic_y",
    "pipe_2_gap_center", "pipe_2_relative_center",
    "pipe_2_top_clearance", "pipe_2_bottom_clearance",
    "pipe_2_gap_size", "pipe_2_speed",
)
OBSERVATION_SIZE_V2 = len(OBSERVATION_FEATURES_V2)
OBSERVATION_LOW_V2 = np.asarray(
    [-1, -1, -1, 0, -1, -1, -1, -1, -1, 0, 0,
     -1, 0, -1, -1, -1, -1, -1, 0, 0], dtype=np.float32
)
OBSERVATION_HIGH_V2 = np.asarray(
    [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,
     2, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32
)

def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def _upcoming(world: GameWorld):
    threshold = world.bird.x - world.bird.radius_x
    pipes = sorted((p for p in world.pipes if p.right >= threshold), key=lambda p: p.x)
    return (pipes + [None, None])[:2]

def _pipe_features(pipe, world: GameWorld) -> tuple[float, ...]:
    if pipe is None:
        return (2.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0)
    bird = world.bird
    dx = pipe.x - bird.x
    speed = max(float(pipe.speed), 0.1)
    frames = _clip(dx / speed, 0.0, 90.0)
    # One-step ballistic estimate. It is context, not a claim that the policy
    # must coast all the way to the pipe.
    predicted_y = bird.y + bird.velocity * frames + 0.5 * BIRD_GRAVITY * frames * (frames + 1.0)
    center = pipe.gap_center
    return (
        _clip(dx / WINDOW_WIDTH, -1.0, 2.0),
        _clip(frames / 90.0, 0.0, 1.0),
        _clip(2.0 * predicted_y / WINDOW_HEIGHT - 1.0, -1.0, 1.0),
        _clip(2.0 * center / WINDOW_HEIGHT - 1.0, -1.0, 1.0),
        _clip((center - bird.y) / WINDOW_HEIGHT, -1.0, 1.0),
        _clip((bird.y - bird.radius_y - pipe.gap_top) / WINDOW_HEIGHT, -1.0, 1.0),
        _clip((pipe.gap_bottom - bird.y - bird.radius_y) / WINDOW_HEIGHT, -1.0, 1.0),
        _clip(pipe.gap_size / WINDOW_HEIGHT, 0.0, 1.0),
        _clip(pipe.speed / 4.6, 0.0, 1.0),
    )

def observe_v2(world: GameWorld) -> np.ndarray:
    """Return a bounded 20-value observation for new V2-only experiments."""
    bird = world.bird
    values = (
        _clip(2.0 * bird.y / WINDOW_HEIGHT - 1.0, -1.0, 1.0),
        _clip(bird.velocity / 10.0, -1.0, 1.0),
        *_pipe_features(_upcoming(world)[0], world),
        *_pipe_features(_upcoming(world)[1], world),
    )
    return np.asarray(values, dtype=np.float32)
