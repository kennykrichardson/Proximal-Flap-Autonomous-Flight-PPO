"""Versioned reward configuration for V2 training experiments."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class RewardConfigV2:
    survival: float = 0.005
    pipe_passed: float = 15.0
    collision: float = -25.0
    alignment_weight: float = 0.35

def alignment_potential(world) -> float:
    """Bounded potential based on the next gap's center and bird position."""
    bird = world.bird
    pipe = next((p for p in world.pipes if p.right >= bird.x - bird.radius_x), None)
    if pipe is None:
        return 0.0
    return -min(abs(pipe.gap_center - bird.y) / 300.0, 1.0)

def calculate_reward_v2(*, scored: bool, terminated: bool, previous_potential: float,
                         next_potential: float, config: RewardConfigV2,
                         gamma: float = 0.99) -> float:
    # Potential-based shaping: gamma*Phi(s') - Phi(s). The terminal state
    # receives zero potential so shaping cannot reward dying in a favorable pose.
    next_value = 0.0 if terminated else next_potential
    reward = config.survival + config.alignment_weight * (
        gamma * next_value - previous_potential
    )
    if scored:
        reward += config.pipe_passed
    if terminated:
        reward += config.collision
    return float(reward)
