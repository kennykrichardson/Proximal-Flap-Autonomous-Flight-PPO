"""Train a PPO policy against the real PROXIMAL FLAP simulation.

Run from the project root:
    python -m ai.train
    python -m ai.train --timesteps 1000000 --seed 123
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from ai.callbacks import TrainingProgressCallback
from ai.environment import ProximalFlapEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PROXIMAL FLAP with PPO.")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "ai" / "checkpoints" / "ppo_proximal_flap")
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "ai" / "runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timesteps < 1 or args.n_envs < 1:
        raise SystemExit("--timesteps and --n-envs must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(
        ProximalFlapEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        monitor_dir=str(args.log_dir / "monitor"),
        wrapper_class=Monitor,
    )

    callback = TrainingProgressCallback(
        checkpoint_dir=args.output.parent / "checkpoints",
        checkpoint_interval=args.checkpoint_every,
        verbose=1,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(args.log_dir),
        device="auto",
    )

    print(f"Starting PPO training: {args.timesteps:,} timesteps using {args.n_envs} vectorized environments.")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            tb_log_name="ppo_proximal_flap",
            progress_bar=False,
        )
        model.save(args.output)
        print(f"Final model saved to: {args.output}.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
