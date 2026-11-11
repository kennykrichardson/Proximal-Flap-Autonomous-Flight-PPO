"""Single-agent feedback pilot for the solo game."""

from __future__ import annotations

from collections.abc import Sequence

from game.entities import Bird, Pipe


class SoloPilot:
    """Deterministic position-and-momentum baseline; deliberately not learned."""

    def __init__(self, vertical_margin: float = 26.0) -> None:
        self.vertical_margin = vertical_margin
        self.last_target_y: float | None = None
        self.flap_count = 0
        self.frames_since_flap = 999

    def reset(self) -> None:
        self.last_target_y = None
        self.flap_count = 0
        self.frames_since_flap = 999

    def update(self, bird: Bird, pipes: Sequence[Pipe]) -> bool:
        """Issue measured flaps based on the next gap's centre and bird momentum."""
        if not bird.alive:
            return False

        self.frames_since_flap += 1
        target_pipe = next(
            (pipe for pipe in pipes if pipe.right >= bird.x - bird.radius_x),
            pipes[-1] if pipes else None,
        )
        if target_pipe is None:
            return False

        target_y = target_pipe.gap_center
        self.last_target_y = target_y
        margin = min(self.vertical_margin, max(20.0, target_pipe.gap_size * 0.20))

        # This controller intentionally reacts to actual position and momentum
        # rather than extrapolating gravity for dozens of frames. A long ballistic
        # prediction would repeatedly flap while the bird is still safely above
        # a lower gap, creating needless climbs and ceiling collisions.
        below_lane = bird.y > target_y + margin
        drifting_down = bird.velocity > 0.4
        hard_recovery = bird.y > target_y + margin * 0.55 and bird.velocity > 2.4
        urgent = bird.y > target_y + margin * 1.8 and bird.velocity > -2.0

        if (below_lane and drifting_down) or hard_recovery or urgent:
            if self.frames_since_flap >= 7:
                bird.flap()
                self.flap_count += 1
                self.frames_since_flap = 0
                return True
        return False
