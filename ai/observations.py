"""Stable, normalized observations exposed to reinforcement-learning agents."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from config import PIPE_SPEED_MAX, WINDOW_HEIGHT, WINDOW_WIDTH
from game.entities import Pipe
from game.world import GameWorld

# Observation order is part of the model contract. Change it only with a model
# version bump and a fresh training run.
OBSERVATION_FEATURES: tuple[str, ...] = (
    "bird_y_normalized",
    "bird_velocity_normalized",
    "pipe_1_horizontal_distance",
    "pipe_1_gap_center_normalized",
    "pipe_1_gap_center_relative_to_bird",
    "pipe_1_gap_size_normalized",
    "pipe_1_speed_normalized",
    "pipe_2_horizontal_distance",
    "pipe_2_gap_center_normalized",
    "pipe_2_gap_center_relative_to_bird",
    "pipe_2_gap_size_normalized",
    "pipe_2_speed_normalized",
)

OBSERVATION_SIZE = len(OBSERVATION_FEATURES)
OBSERVATION_LOW = np.asarray(
    [-1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0,
     -1.0, -1.0, -1.0, 0.0, 0.0],
    dtype=np.float32,
)
OBSERVATION_HIGH = np.asarray(
    [1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0,
     2.0, 1.0, 1.0, 1.0, 1.0],
    dtype=np.float32,
)
_BIRD_VELOCITY_SCALE = 10.0


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _upcoming_pipes(world: GameWorld) -> Sequence[Pipe | None]:
    """Return the next two not-yet-passed pipes in horizontal order."""
    threshold = world.bird.x - world.bird.radius_x
    upcoming = [pipe for pipe in world.pipes if pipe.right >= threshold]
    upcoming.sort(key=lambda pipe: pipe.x)
    selected: list[Pipe | None] = list(upcoming[:2])
    selected.extend([None] * (2 - len(selected)))
    return selected


def _pipe_features(pipe: Pipe | None, bird_y: float, bird_x: float) -> tuple[float, ...]:
    """Encode one pipe as distance, gap position, gap size, and speed."""
    if pipe is None:
        # A neutral, distant sentinel. Normal gameplay almost always has two
        # upcoming pipes, but keeping the observation finite makes it robust.
        return (2.0, 0.0, 0.0, 0.0, 0.0)

    horizontal_distance = _clip((pipe.x - bird_x) / WINDOW_WIDTH, -1.0, 2.0)
    gap_center = _clip((2.0 * pipe.gap_center / WINDOW_HEIGHT) - 1.0, -1.0, 1.0)
    relative_center = _clip((pipe.gap_center - bird_y) / WINDOW_HEIGHT, -1.0, 1.0)
    gap_size = _clip(pipe.gap_size / WINDOW_HEIGHT, 0.0, 1.0)
    speed = _clip(pipe.speed / PIPE_SPEED_MAX, 0.0, 1.0)
    return (
        horizontal_distance,
        gap_center,
        relative_center,
        gap_size,
        speed,
    )


def observe(world: GameWorld) -> np.ndarray:
    """Build the fixed 12-value observation vector for the current game state.

    All values are float32 and bounded by the matching Gymnasium Box space.
    The vector contains bird position/velocity and information about the next
    two pipes, allowing the policy to react to the current and upcoming gaps.
    """
    bird = world.bird
    bird_y = _clip((2.0 * bird.y / WINDOW_HEIGHT) - 1.0, -1.0, 1.0)
    bird_velocity = _clip(bird.velocity / _BIRD_VELOCITY_SCALE, -1.0, 1.0)

    first_pipe, second_pipe = _upcoming_pipes(world)
    values = (
        bird_y,
        bird_velocity,
        *_pipe_features(first_pipe, bird.y, bird.x),
        *_pipe_features(second_pipe, bird.y, bird.x),
    )
    return np.asarray(values, dtype=np.float32)
