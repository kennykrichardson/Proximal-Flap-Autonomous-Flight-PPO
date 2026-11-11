"""Train an isolated V2 PPO experiment; never writes the champion checkpoint."""
from __future__ import annotations
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from ai.callbacks import TrainingProgressCallback
from ai.environment_v2 import ProximalFlapEnvV2

ROOT = Path(__file__).resolve().parents[1]

def parse_args():
    parser = argparse.ArgumentParser(description="Train an experimental PROXIMAL-FLAP V2 PPO agent.")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=ROOT / "ai/checkpoints/experiments/ppo_proximal_flap_v2")
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="Resume V2 training from an existing V2 checkpoint.")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "ai/runs/v2")
    return parser.parse_args()

def main():
    args = parse_args()
    if min(args.timesteps, args.n_envs, args.checkpoint_every) < 1:
        raise SystemExit("timesteps, n-envs and checkpoint-every must be positive")
    champion_path = (ROOT / "ai/checkpoints/ppo_proximal_flap.zip").resolve()
    if args.output.resolve() == champion_path:
        raise SystemExit("Refusing to overwrite the protected champion checkpoint.")
    if args.resume_from is not None and not args.resume_from.is_file():
        raise SystemExit(f"Resume checkpoint not found: {args.resume_from}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(ProximalFlapEnvV2, n_envs=args.n_envs, seed=args.seed,
                       monitor_dir=str(args.log_dir / "monitor"), wrapper_class=Monitor)
    try:
        if args.resume_from is not None:
            print(f"Resuming V2 from: {args.resume_from}")
            model = PPO.load(args.resume_from, env=env, device="auto")
        else:
            model = PPO("MlpPolicy", env, learning_rate=args.learning_rate,
                        n_steps=2048, batch_size=256, n_epochs=10, gamma=0.99,
                        gae_lambda=0.95, clip_range=0.2, ent_coef=0.005,
                        vf_coef=0.5, max_grad_norm=0.5,
                        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
                        seed=args.seed, verbose=1, tensorboard_log=str(args.log_dir),
                        device="auto")
        callback = TrainingProgressCallback(
            checkpoint_dir=args.output.parent / "v2_periodic",
            checkpoint_interval=args.checkpoint_every, verbose=1)
        print(f"Training V2 for {args.timesteps:,} timesteps; output={args.output}")
        model.learn(total_timesteps=args.timesteps, callback=callback,
                    tb_log_name="ppo_proximal_flap_v2",
                    reset_num_timesteps=args.resume_from is None)
        model.save(args.output)
        print(f"V2 experiment saved: {args.output}.zip")
    finally:
        env.close()

if __name__ == "__main__":
    main()
