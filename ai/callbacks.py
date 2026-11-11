"""Training callbacks for checkpointing PPO models and recording progress."""

from __future__ import annotations

from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback


class TrainingProgressCallback(BaseCallback):
    """Save periodic checkpoints and report episode reward/length metrics."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        checkpoint_interval: int = 100_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        if checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be positive")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = int(checkpoint_interval)
        self._next_checkpoint = self.checkpoint_interval
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []

    def _init_callback(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode")
            if episode:
                self._episode_rewards.append(float(episode["r"]))
                self._episode_lengths.append(int(episode["l"]))

        if self.num_timesteps >= self._next_checkpoint:
            path = self.checkpoint_dir / f"ppo_proximal_flap_{self.num_timesteps}_steps"
            self.model.save(path)
            if self.verbose:
                print(f"Saved checkpoint: {path}.zip")
            self._next_checkpoint += self.checkpoint_interval

        if self._episode_rewards:
            window = self._episode_rewards[-100:]
            self.logger.record("custom/episode_reward_mean_100", sum(window) / len(window))
            self.logger.record("custom/episodes_completed", len(self._episode_rewards))
        if self._episode_lengths:
            window = self._episode_lengths[-100:]
            self.logger.record("custom/episode_length_mean_100", sum(window) / len(window))
        return True
