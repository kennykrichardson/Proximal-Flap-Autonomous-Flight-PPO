"""Paired V2-vs-champion evaluation on identical generated course seeds."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from ai.environment import ProximalFlapEnv
from ai.environment_v2 import ProximalFlapEnvV2

ROOT = Path(__file__).resolve().parents[1]

def run(model, env_type, seed, max_frames):
    env = env_type(max_episode_steps=max_frames)
    obs, _ = env.reset(seed=seed)
    done = False
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(np.asarray(action).item()))
        done = terminated or truncated
    world = env.world
    bird = world.bird if world is not None else None
    collision_cause = None
    if terminated and bird is not None:
        from config import GROUND_Y
        if bird.y - bird.radius_y < 0 or bird.y + bird.radius_y >= GROUND_Y:
            collision_cause = "boundary"
        elif any(world._collides_with_pipe(pipe) for pipe in world.pipes):
            collision_cause = "pipe"
        else:
            collision_cause = "other"
    env.close()
    return {"score": int(info["score"]), "frames": int(info["frames"]),
            "course_seed": int(info["episode_seed"]),
            "termination_reason": info.get("termination_reason"),
            "collision_cause": collision_cause,
            "final_bird_y": round(float(bird.y), 3) if bird is not None else None}

def summarize(rows):
    scores = [r["score"] for r in rows]
    return {"episodes": len(scores), "mean": round(statistics.fmean(scores), 3),
            "median": round(statistics.median(scores), 3),
            "p90": round(float(np.percentile(scores, 90)), 3),
            "best": max(scores), "worst": min(scores),
            "mean_frames": round(statistics.fmean(r["frames"] for r in rows), 2)}

def main():
    p = argparse.ArgumentParser(description="Evaluate original champion and V2 on paired courses.")
    p.add_argument("--champion", type=Path, default=ROOT/"ai/checkpoints/ppo_proximal_flap.zip")
    p.add_argument("--candidate", type=Path, default=ROOT/"ai/checkpoints/experiments/ppo_proximal_flap_v2.zip")
    p.add_argument("--seeds", type=int, default=500)
    p.add_argument("--start-seed", type=int, default=50_000)
    p.add_argument("--max-frames", type=int, default=30_000)
    p.add_argument("--progress-every", type=int, default=10,
                   help="Print progress after each batch of paired courses (default: 10).")
    p.add_argument("--output", type=Path, default=ROOT/"ai/runs/v2_comparison.json")
    a = p.parse_args()
    if a.seeds < 1: p.error("--seeds must be positive")
    if a.progress_every < 1: p.error("--progress-every must be positive")
    for path in (a.champion, a.candidate):
        if not path.exists(): raise SystemExit(f"Model not found: {path}")
    champion, candidate = PPO.load(a.champion, device="auto"), PPO.load(a.candidate, device="auto")
    seeds = range(a.start_seed, a.start_seed+a.seeds)
    old, new = [], []
    for index, seed in enumerate(seeds, start=1):
        old.append(run(champion, ProximalFlapEnv, seed, a.max_frames))
        new.append(run(candidate, ProximalFlapEnvV2, seed, a.max_frames))
        if index % a.progress_every == 0 or index == a.seeds:
            print(f"Evaluated {index}/{a.seeds} paired courses", flush=True)
    paired = [n["score"]-o["score"] for o,n in zip(old,new)]
    report = {"champion": summarize(old), "candidate_v2": summarize(new),
              "paired_score_delta_mean": round(statistics.fmean(paired), 3),
              "candidate_wins": sum(d>0 for d in paired),
              "ties": sum(d==0 for d in paired), "candidate_losses": sum(d<0 for d in paired),
              "seed_range": [a.start_seed, a.start_seed+a.seeds-1],
              "per_seed": [{"seed": a.start_seed+i, "champion": o, "candidate_v2": n,
                            "score_delta": n["score"]-o["score"]}
                           for i,(o,n) in enumerate(zip(old,new))]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="per_seed"}, indent=2))
    print(f"Detailed paired report: {a.output}")

if __name__ == "__main__": main()
