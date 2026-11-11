"""Small, renderer-independent game entities."""

from __future__ import annotations

from dataclasses import dataclass

from config import (
    BIRD_FLAP_VELOCITY,
    BIRD_GRAVITY,
    BIRD_RADIUS_X,
    BIRD_RADIUS_Y,
    PIPE_GAP_MAX,
    PIPE_SPEED,
    PIPE_WIDTH,
)


@dataclass(slots=True)
class Bird:
    """The single bird currently being controlled."""

    x: float
    y: float
    velocity: float = 0.0
    alive: bool = True
    radius_x: int = BIRD_RADIUS_X
    radius_y: int = BIRD_RADIUS_Y

    def flap(self) -> None:
        if self.alive:
            self.velocity = BIRD_FLAP_VELOCITY

    def update(self) -> None:
        if not self.alive:
            return
        self.velocity += BIRD_GRAVITY
        self.y += self.velocity


@dataclass(slots=True)
class Pipe:
    """A pipe pair described by the top edge of its safe gap."""

    x: float
    gap_top: int
    gap_size: int = PIPE_GAP_MAX
    speed: float = PIPE_SPEED
    scored: bool = False

    @property
    def gap_bottom(self) -> int:
        return self.gap_top + self.gap_size

    @property
    def gap_center(self) -> float:
        return self.gap_top + self.gap_size / 2

    @property
    def right(self) -> float:
        return self.x + PIPE_WIDTH

    def update(self) -> None:
        self.x -= self.speed
