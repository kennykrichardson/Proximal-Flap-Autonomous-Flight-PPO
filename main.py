"""Launch PROXIMAL FLAP: SOLO."""

from __future__ import annotations

import pygame
from datetime import datetime
from pathlib import Path

from config import (
    ASSET_DIR,
    AUTO_RESTART_SECONDS,
    BACKGROUND_PARALLAX,
    FPS,
    PIPE_SPEED,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from game.renderer import Renderer
from game.world import GameWorld
from storage import ScoreStore

# AI dependencies are optional for ordinary human/SoloPilot gameplay.
try:
    from ai.agent import PPOBirdAgent
    from ai.observations_v2 import observe_v2
except ImportError:
    PPOBirdAgent = None
    observe_v2 = None


class Audio:
    """Optional sound effects. The game remains playable without audio."""

    def __init__(self) -> None:
        self.score = None
        self.crash = None
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            self.score = pygame.mixer.Sound(str(ASSET_DIR / "sounds" / "score.wav"))
            self.crash = pygame.mixer.Sound(str(ASSET_DIR / "sounds" / "slap.wav"))
            self.score.set_volume(0.28)
            self.crash.set_volume(0.35)
        except (pygame.error, OSError):
            self.score = None
            self.crash = None

    def play_score(self) -> None:
        if self.score:
            self.score.play()

    def play_crash(self) -> None:
        if self.crash:
            self.crash.play()



class RunLogger:
    """Append one compact record whenever a game run ends."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.write_text(
                "timestamp | run | mode | score\\n"
                "-----------------------------------------------\\n",
                encoding="utf-8",
            )

    def record(self, run_number: int, mode: str, score: int) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(
                f"{timestamp} | {run_number:04d} | {mode.upper():<6} | {score}\\n"
            )


def main() -> int:
    pygame.init()
    pygame.display.set_caption("PROXIMAL FLAP: SOLO")
    # Keep the original compact game window. The simulation and presentation
    # both use the same 400x600 surface, with no desktop-sized scaling layer.
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    renderer = Renderer(screen)
    audio = Audio()
    score_store = ScoreStore()
    run_logger = RunLogger(Path(__file__).resolve().parent / "proximal_flap_runs.log")
    world = GameWorld()

    model_path = (ASSET_DIR.parent / "ai" / "checkpoints" / "experiments" / "ppo_proximal_flap_v2.zip")
    ppo_agent = None
    if PPOBirdAgent is not None and model_path.exists():
        try:
            ppo_agent = PPOBirdAgent(model_path)
            print(f"PPO pilot loaded: {model_path}")
        except Exception as exc:
            print(f"Could not load PPO pilot: {exc}")
    else:
        print("PPO pilot unavailable. Install requirements-ai.txt and ensure the champion checkpoint exists.")

    # V2 is the default playable AI; use key 1 for manual or 2 for SoloPilot.
    control_mode = "ppo" if ppo_agent is not None else "solo"
    paused = False
    help_visible = False
    resetting = False
    reset_timer = 0.0
    last_score = 0
    run_number = 1
    ground_offset = 0.0
    background_offset = 0.0
    running = True

    while running:
        delta_seconds = min(clock.tick(FPS) / 1000.0, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if help_visible:
                        help_visible = False
                    else:
                        running = False
                elif event.key == pygame.K_h:
                    help_visible = not help_visible
                elif event.key == pygame.K_a and not help_visible:
                    control_mode = "manual" if control_mode != "manual" else "solo"
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_m) and not help_visible:
                    if event.key == pygame.K_1:
                        control_mode = "manual"
                    elif event.key == pygame.K_2:
                        control_mode = "solo"
                    elif ppo_agent is not None:
                        control_mode = "ppo"
                    else:
                        print("PPO mode unavailable: model or AI dependencies could not be loaded.")
                elif event.key == pygame.K_p and not help_visible:
                    paused = not paused
                elif event.key == pygame.K_r and not help_visible:
                    run_logger.record(run_number, control_mode, world.score)
                    score_store.save_if_better(world.score)
                    world.reset()
                    run_number += 1
                    resetting = False
                    reset_timer = 0.0
                elif event.key in (pygame.K_SPACE, pygame.K_UP) and not paused and not help_visible and control_mode == "manual":
                    world.flap()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if not paused and not help_visible and control_mode == "manual":
                    world.flap()

        if not paused and not help_visible and not resetting:
            if control_mode == "ppo" and ppo_agent is not None and observe_v2 is not None:
                action = ppo_agent.act(observe_v2(world), deterministic=True)
                if action == 1:
                    world.flap()
                result = world.step(autopilot=False)
            else:
                result = world.step(autopilot=(control_mode == "solo"))
            ground_speed = world.pipes[0].speed if world.pipes else PIPE_SPEED
            ground_offset = (ground_offset + ground_speed) % WINDOW_WIDTH
            background_offset = (
                background_offset + ground_speed * BACKGROUND_PARALLAX
            ) % (WINDOW_WIDTH * 2)
            if result.scored:
                audio.play_score()
                score_store.save_if_better(world.score)
            if result.died:
                last_score = world.score
                run_logger.record(run_number, control_mode, last_score)
                score_store.save_if_better(last_score)
                audio.play_crash()
                resetting = True
                reset_timer = 0.0

        elif resetting and not paused and not help_visible:
            reset_timer += delta_seconds
            ground_speed = world.pipes[0].speed if world.pipes else PIPE_SPEED
            ground_offset = (ground_offset + ground_speed * 0.45) % WINDOW_WIDTH
            background_offset = (
                background_offset + ground_speed * 0.45 * BACKGROUND_PARALLAX
            ) % (WINDOW_WIDTH * 2)
            if reset_timer >= AUTO_RESTART_SECONDS:
                world.reset()
                run_number += 1
                resetting = False
                reset_timer = 0.0

        renderer.draw(
            world,
            best_score=score_store.best_score,
            run_number=run_number,
            pilot_enabled=(control_mode != "manual"),
            control_mode=control_mode,
            ground_offset=ground_offset,
            background_offset=background_offset,
            paused=paused,
            help_visible=help_visible,
            resetting=resetting,
            last_score=last_score,
        )
        pygame.display.flip()

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
