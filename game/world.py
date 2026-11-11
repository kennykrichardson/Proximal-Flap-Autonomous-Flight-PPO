"""Deterministic, renderer-independent single-bird simulation."""

from __future__ import annotations

import random
from dataclasses import dataclass

from config import (
    BIRD_START_Y,
    BIRD_X,
    DIFFICULTY_RAMP_SCORE,
    GAP_CENTER_STEP_MAX,
    GAP_CENTER_STEP_START,
    GROUND_Y,
    PIPE_FIRST_X,
    PIPE_GAP_MAX,
    PIPE_GAP_MIN,
    PIPE_GAP_TOP_MIN,
    PIPE_GAP_CENTER_MAX,
    PIPE_SPEED,
    PIPE_SPEED_MAX,
    PIPE_SPAWN_THRESHOLD,
    PIPE_SPACING_MAX,
    PIPE_SPACING_MIN,
    PIPE_WIDTH,
    WINDOW_HEIGHT,
)
from game.controller import SoloPilot
from game.entities import Bird, Pipe


@dataclass(slots=True)
class StepResult:
    scored: bool = False
    died: bool = False


class GameWorld:
    """Owns one bird, an endless adaptive pipe stream, and the current run's score."""

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self.rng = random.Random(seed)
        self.pilot = SoloPilot()
        self.reset(seed=seed)

    @property
    def difficulty(self) -> float:
        """Difficulty rises smoothly over the first 36 cleared pipes."""
        return min(self.score / DIFFICULTY_RAMP_SCORE, 1.0)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
            self.rng = random.Random(seed)
        elif self._seed is not None:
            # Deliberately continue the existing RNG stream. A fresh world with
            # the same seed reproduces the same course; automatic restarts do not.
            pass

        self.bird = Bird(BIRD_X, BIRD_START_Y)
        self.score = 0
        self.frames = 0
        self.pilot.reset()
        self.last_result = StepResult()
        self._last_gap_center: float | None = None
        self._last_vertical_direction = self.rng.choice((-1, 1))
        self.pipes = [self._make_pipe(PIPE_FIRST_X, first=True)]

    def _difficulty_for_next_pipe(self) -> float:
        # Use current score, so the next obstacle gets harder immediately after
        # a successful pass rather than one full pipe later.
        return self.difficulty

    def _pipe_profile(self) -> tuple[int, int, float, int, int, int]:
        difficulty = self._difficulty_for_next_pipe()
        gap_max = round(PIPE_GAP_MAX - 22 * difficulty)
        gap_min = round(PIPE_GAP_MIN + 4 * (1.0 - difficulty))
        gap_min = min(gap_min, gap_max)
        speed = PIPE_SPEED + (PIPE_SPEED_MAX - PIPE_SPEED) * difficulty
        spacing_min = round(PIPE_SPACING_MIN + 12 * (1.0 - difficulty))
        spacing_max = round(PIPE_SPACING_MAX - 18 * difficulty)
        max_center_step = round(
            GAP_CENTER_STEP_START
            + (GAP_CENTER_STEP_MAX - GAP_CENTER_STEP_START) * difficulty
        )
        return gap_min, gap_max, speed, spacing_min, spacing_max, max_center_step

    def _choose_gap_center(self, gap_size: int, max_step: int, first: bool) -> float:
        min_center = PIPE_GAP_TOP_MIN + gap_size / 2
        max_center = min(PIPE_GAP_CENTER_MAX, GROUND_Y - 48 - gap_size / 2)

        if first or self._last_gap_center is None:
            center = self.rng.uniform(225, 330)
        else:
            # A weighted random walk creates small corrections, abrupt direction
            # changes, and occasional steep climbs or drops without teleporting
            # the safe lane beyond what the bird can physically reach.
            bands = (
                (0.22, 0.38, 0.22, 0.18)
                if self.difficulty < 0.45
                else (0.12, 0.30, 0.32, 0.26)
            )
            band = self.rng.choices((0, 1, 2, 3), weights=bands, k=1)[0]
            ranges = (
                (0.18, 0.40),
                (0.40, 0.66),
                (0.66, 0.86),
                (0.86, 1.00),
            )
            low, high = ranges[band]
            magnitude = self.rng.uniform(low * max_step, high * max_step)
            if self.rng.random() < 0.42:
                self._last_vertical_direction *= -1
            elif self.rng.random() < 0.22:
                self._last_vertical_direction = self.rng.choice((-1, 1))
            center = self._last_gap_center + self._last_vertical_direction * magnitude

        center = max(min_center, min(max_center, center))
        self._last_gap_center = center
        return center

    def _make_pipe(self, x: float, *, first: bool = False) -> Pipe:
        gap_min, gap_max, speed, _, _, max_step = self._pipe_profile()
        gap_size = self.rng.randint(gap_min, gap_max)
        center = self._choose_gap_center(gap_size, max_step, first)
        gap_top = round(center - gap_size / 2)
        return Pipe(x=x, gap_top=gap_top, gap_size=gap_size, speed=speed)

    def _spawn_pipe(self) -> None:
        if not self.pipes or self.pipes[-1].x < PIPE_SPAWN_THRESHOLD:
            if self.pipes:
                _, _, _, spacing_min, spacing_max, _ = self._pipe_profile()
                next_x = self.pipes[-1].x + self.rng.randint(spacing_min, spacing_max)
            else:
                next_x = PIPE_FIRST_X
            self.pipes.append(self._make_pipe(next_x))

    def _collides_with_pipe(self, pipe: Pipe) -> bool:
        bird = self.bird
        horizontal_overlap = (
            pipe.x < bird.x + bird.radius_x
            and pipe.right > bird.x - bird.radius_x
        )
        if not horizontal_overlap:
            return False
        return (
            bird.y - bird.radius_y < pipe.gap_top
            or bird.y + bird.radius_y > pipe.gap_bottom
        )

    def step(self, autopilot: bool = True) -> StepResult:
        """Advance one fixed simulation frame."""
        result = StepResult()
        if not self.bird.alive:
            self.last_result = result
            return result

        if autopilot:
            self.pilot.update(self.bird, self.pipes)

        self.bird.update()
        self.frames += 1

        if (
            self.bird.y - self.bird.radius_y < 0
            or self.bird.y + self.bird.radius_y >= GROUND_Y
            or self.bird.y > WINDOW_HEIGHT
            or self.bird.y < -40
        ):
            self.bird.alive = False
            result.died = True
            self.last_result = result
            return result

        for pipe in self.pipes:
            if self._collides_with_pipe(pipe):
                self.bird.alive = False
                result.died = True
                self.last_result = result
                return result

        for pipe in self.pipes:
            pipe.update()
            if not pipe.scored and pipe.right < self.bird.x - self.bird.radius_x:
                pipe.scored = True
                self.score += 1
                result.scored = True

        self.pipes = [pipe for pipe in self.pipes if pipe.right >= 0]
        self._spawn_pipe()
        self.last_result = result
        return result

    def flap(self) -> None:
        self.bird.flap()

    @property
    def alive(self) -> bool:
        return self.bird.alive
