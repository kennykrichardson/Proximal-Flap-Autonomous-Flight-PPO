"""Evaluate PPO and SoloPilot on identical deterministic course seeds.

Example:
    python -m ai.evaluate --model ai/checkpoints/ppo_proximal_flap.zip --seeds 100
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from ai.environment import ProximalFlapEnv
from game.world import GameWorld


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_solo_pilot(seed: int, max_frames: int) -> dict[str, int]:
    world = GameWorld(seed=seed)
    while world.alive and world.frames < max_frames:
        world.step(autopilot=True)
    return {"score": int(world.score), "frames": int(world.frames)}


def run_ppo(model: PPO, seed: int, max_frames: int) -> dict[str, int]:
    env = ProximalFlapEnv(max_episode_steps=max_frames)
    observation, _ = env.reset(seed=seed)
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        action, _state = model.predict(observation, deterministic=True)
        observation, _reward, terminated, truncated, info = env.step(int(np.asarray(action).item()))
    env.close()
    return {"score": int(info.get("score", 0)), "frames": int(info.get("frames", 0)), "course_seed": int(info.get("episode_seed", seed))}


def summarize(results: list[dict[str, int]]) -> dict[str, float | int]:
    scores = [item["score"] for item in results]
    return {
        "episodes": len(scores),
        "mean": round(statistics.fmean(scores), 3),
        "median": round(statistics.median(scores), 3),
        "p90": float(np.percentile(scores, 90)),
        "best": max(scores),
        "worst": min(scores),
        "mean_frames": round(statistics.fmean(item["frames"] for item in results), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PPO with SoloPilot on identical seeds.")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "ai" / "checkpoints" / "ppo_proximal_flap.zip")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=10_000)
    parser.add_argument("--max-frames", type=int, default=30_000)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "ai" / "runs" / "evaluation.json")
    args = parser.parse_args()

    model_path = args.model
    if not model_path.exists() and model_path.suffix != ".zip":
        model_path = model_path.with_suffix(".zip")
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}. Train one first with python -m ai.train")

    model = PPO.load(model_path, device="auto")
    seeds = list(range(args.start_seed, args.start_seed + args.seeds))
    ppo_results = [run_ppo(model, seed, args.max_frames) for seed in seeds]
    pilot_results = [run_solo_pilot(seed, args.max_frames) for seed in seeds]

    report = {
        "model": str(model_path),
        "seed_range": [seeds[0], seeds[-1]] if seeds else [],
        "max_frames": args.max_frames,
        "ppo": summarize(ppo_results),
        "solo_pilot": summarize(pilot_results),
        "per_seed": [
            {"seed": seed, "ppo": ppo, "solo_pilot": pilot}
            for seed, ppo, pilot in zip(seeds, ppo_results, pilot_results)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("ppo", "solo_pilot")}, indent=2))
    print(f"Full report saved to: {args.output}")


if __name__ == "__main__":
    main()
