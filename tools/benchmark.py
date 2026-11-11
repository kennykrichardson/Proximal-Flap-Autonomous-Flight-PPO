"""Repeatable deterministic-pilot benchmark across fixed course seeds.

Run: python -m tools.benchmark --seeds 100 --frames 30000
The output is JSON so later AI controllers can be compared on identical seeds.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from game.world import GameWorld


def run_seed(seed: int, frame_limit: int) -> dict[str, int | bool]:
    world = GameWorld(seed=seed)
    while world.alive and world.frames < frame_limit:
        world.step(autopilot=True)
    return {
        "seed": seed,
        "score": world.score,
        "frames": world.frames,
        "survived_to_limit": world.alive,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--frames", type=int, default=30_000)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("data/deterministic-baseline.json"))
    args = parser.parse_args()
    if args.seeds < 1 or args.frames < 1:
        parser.error("--seeds and --frames must be positive")

    results = [run_seed(seed, args.frames) for seed in range(args.start_seed, args.start_seed + args.seeds)]
    scores = [int(row["score"]) for row in results]
    ordered = sorted(scores)
    p90 = ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))]
    summary = {
        "controller": "SoloPilot deterministic baseline",
        "seeds": args.seeds,
        "frame_limit": args.frames,
        "mean_score": round(statistics.mean(scores), 3),
        "median_score": statistics.median(scores),
        "p90_score": p90,
        "best_score": max(scores),
        "worst_score": min(scores),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    print(f"Saved full results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
