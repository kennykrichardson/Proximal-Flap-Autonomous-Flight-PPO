"""Pygame presentation layer for PROXIMAL FLAP: SOLO."""

from __future__ import annotations

import pygame

from config import (
    ASSET_DIR,
    BACKGROUND_ASSET,
    BIRD_ASSET,
    GROUND_ASSET,
    PIPE_DOWN_ASSET,
    PIPE_UP_ASSET,
    CHARCOAL,
    CREAM,
    CRIMSON,
    GROUND_Y,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from game.world import GameWorld


class Renderer:
    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        font_path = ASSET_DIR / "fonts" / "BaiJamjuree-Bold.ttf"

        # Art is loaded independently so each generated PNG can be replaced
        # without touching game logic. Images are fitted to the logical canvas.
        self.background = pygame.image.load(str(BACKGROUND_ASSET)).convert()
        self.background = pygame.transform.smoothscale(
            self.background, (WINDOW_WIDTH, WINDOW_HEIGHT)
        )
        # Mirror the artwork horizontally. The original right edge meets its
        # reflected right edge, and the reflected left edge meets the next
        # original left edge. This creates an exactly edge-matched 2x tile.
        mirrored_background = pygame.transform.flip(
            self.background, True, False
        )
        self.background_tile = pygame.Surface(
            (WINDOW_WIDTH * 2, WINDOW_HEIGHT)
        ).convert()
        self.background_tile.blit(self.background, (0, 0))
        self.background_tile.blit(mirrored_background, (WINDOW_WIDTH, 0))
        self.background_tile_width = self.background_tile.get_width()
        self.ground = pygame.image.load(str(GROUND_ASSET)).convert_alpha()
        self.pipe_up = pygame.image.load(str(PIPE_UP_ASSET)).convert_alpha()
        self.pipe_down = pygame.image.load(str(PIPE_DOWN_ASSET)).convert_alpha()
        self._bird_asset = None
        if BIRD_ASSET.exists():
            self._bird_asset = pygame.image.load(str(BIRD_ASSET)).convert_alpha()
            self._bird_asset = pygame.transform.smoothscale(self._bird_asset, (68, 54))

        if font_path.exists():
            self.font_large = pygame.font.Font(str(font_path), 42)
            self.font_medium = pygame.font.Font(str(font_path), 19)
            self.font_small = pygame.font.Font(str(font_path), 12)
            self.font_tiny = pygame.font.Font(str(font_path), 10)
        else:
            self.font_large = pygame.font.SysFont("Arial", 42, bold=True)
            self.font_medium = pygame.font.SysFont("Arial", 19, bold=True)
            self.font_small = pygame.font.SysFont("Arial", 12, bold=True)
            self.font_tiny = pygame.font.SysFont("Arial", 10, bold=True)

        self._bird_up = self._make_bird_surface(wing_up=True)
        self._bird_down = self._make_bird_surface(wing_up=False)
        self._overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)

    @staticmethod
    def _make_bird_surface(*, wing_up: bool) -> pygame.Surface:
        """Draw the crimson retro-tech bird in a compact, expressive sprite."""
        surface = pygame.Surface((68, 54), pygame.SRCALPHA)
        red = CRIMSON
        dark = CHARCOAL
        cream = CREAM

        # Tail feathers and rear silhouette.
        pygame.draw.polygon(surface, dark, [(21, 25), (4, 13), (5, 6), (28, 19)])
        pygame.draw.polygon(surface, red, [(20, 24), (7, 12), (8, 9), (27, 21)])
        pygame.draw.polygon(surface, dark, [(19, 31), (3, 33), (2, 41), (28, 37)])
        pygame.draw.polygon(surface, cream, [(18, 32), (7, 35), (6, 38), (25, 35)])

        # Animated wing, behind the body.
        if wing_up:
            wing_points = [(27, 31), (15, 7), (23, 4), (39, 24), (42, 34)]
            wing_inner = [(27, 29), (18, 11), (23, 9), (37, 26)]
        else:
            wing_points = [(27, 28), (14, 43), (23, 49), (41, 35), (43, 27)]
            wing_inner = [(28, 31), (18, 41), (23, 44), (38, 34)]
        pygame.draw.polygon(surface, dark, wing_points)
        pygame.draw.polygon(surface, red, wing_inner)
        pygame.draw.line(surface, cream, wing_inner[1], wing_inner[2], 2)

        # Main body and cream breast.
        pygame.draw.ellipse(surface, dark, (10, 15, 48, 34))
        pygame.draw.ellipse(surface, red, (12, 16, 44, 30))
        pygame.draw.ellipse(surface, cream, (19, 30, 34, 15))
        pygame.draw.ellipse(surface, dark, (18, 29, 35, 16), 2)

        # Crest feathers give the character a fast, forward-swept silhouette.
        pygame.draw.polygon(surface, dark, [(24, 19), (26, 5), (34, 13), (39, 8), (42, 20)])
        pygame.draw.polygon(surface, red, [(26, 17), (28, 8), (34, 15), (39, 11), (41, 20)])

        # Headset housing on the visible side.
        pygame.draw.ellipse(surface, dark, (23, 20, 24, 25))
        pygame.draw.ellipse(surface, cream, (25, 22, 20, 21))
        pygame.draw.ellipse(surface, red, (28, 25, 14, 15))
        pygame.draw.ellipse(surface, dark, (32, 28, 7, 9))
        pygame.draw.circle(surface, cream, (35, 32), 2)

        # Visor and luminous crimson HUD strip.
        pygame.draw.rect(surface, dark, (33, 9, 30, 18), border_radius=7)
        pygame.draw.rect(surface, cream, (35, 11, 26, 14), width=2, border_radius=5)
        pygame.draw.rect(surface, red, (38, 14, 21, 8), border_radius=3)
        pygame.draw.lines(surface, cream, False, [(42, 19), (47, 15), (51, 20), (55, 16)], 2)

        # Large determined eye and angled brow.
        pygame.draw.ellipse(surface, cream, (45, 24, 14, 15))
        pygame.draw.ellipse(surface, dark, (52, 27, 6, 8))
        pygame.draw.circle(surface, red, (55, 28), 2)
        pygame.draw.polygon(surface, dark, [(44, 24), (58, 22), (60, 25), (48, 27)])

        # Beak, split into upper and lower plates.
        pygame.draw.polygon(surface, dark, [(56, 34), (67, 38), (56, 43)])
        pygame.draw.polygon(surface, red, [(57, 35), (64, 38), (57, 40)])
        pygame.draw.line(surface, cream, (58, 37), (63, 38), 1)
        pygame.draw.polygon(surface, dark, [(56, 41), (63, 42), (58, 46)])
        pygame.draw.line(surface, cream, (57, 42), (61, 42), 1)

        # Small collar/flight harness detail.
        pygame.draw.arc(surface, dark, (21, 37, 23, 14), 0.2, 2.7, 3)
        pygame.draw.line(surface, cream, (27, 44), (39, 47), 2)
        return surface

    def draw(
        self,
        world: GameWorld,
        *,
        best_score: int,
        run_number: int,
        pilot_enabled: bool,
        ground_offset: float,
        background_offset: float,
        control_mode: str = "solo",
        paused: bool = False,
        help_visible: bool = False,
        resetting: bool = False,
        last_score: int = 0,
    ) -> None:
        self._draw_background(background_offset)
        self._draw_pipes(world)
        self._draw_bird(world)
        self._draw_ground(ground_offset)
        self._draw_hud(
            world,
            best_score=best_score,
            run_number=run_number,
            pilot_enabled=pilot_enabled,
            control_mode=control_mode,
        )

        if paused:
            self._draw_message("PAUSED", "Press P to continue")
        if resetting:
            self._draw_message("RUN COMPLETE", f"Score {last_score}  •  Next run starting...")
        if help_visible:
            self._draw_help(pilot_enabled)

    def _draw_background(self, offset: float) -> None:
        # The offset is independent of ground_offset and wraps only after a
        # complete mirrored tile, so there is no periodic backward jump.
        x = -round(offset) % self.background_tile_width
        self.screen.blit(
            self.background_tile, (x - self.background_tile_width, 0)
        )
        self.screen.blit(self.background_tile, (x, 0))

    def _draw_ground(self, offset: float) -> None:
        x = -int(offset) % WINDOW_WIDTH
        self.screen.blit(self.ground, (x - WINDOW_WIDTH, GROUND_Y))
        self.screen.blit(self.ground, (x, GROUND_Y))

    def _draw_pipes(self, world: GameWorld) -> None:
        for pipe in world.pipes:
            top_y = pipe.gap_top - self.pipe_down.get_height()
            self.screen.blit(self.pipe_down, (round(pipe.x), top_y))
            self.screen.blit(self.pipe_up, (round(pipe.x), pipe.gap_bottom))

    def _draw_bird(self, world: GameWorld) -> None:
        bird = world.bird
        if self._bird_asset is not None:
            wing_surface = self._bird_asset
        else:
            wing_surface = self._bird_up if (world.frames // 5) % 2 == 0 else self._bird_down
        angle = max(-24, min(58, -bird.velocity * 2.7))
        sprite = pygame.transform.rotate(wing_surface, angle)
        rect = sprite.get_rect(center=(round(bird.x), round(bird.y)))
        self.screen.blit(sprite, rect)

    def _draw_hud(
        self,
        world: GameWorld,
        *,
        best_score: int,
        run_number: int,
        pilot_enabled: bool,
        control_mode: str = "solo",
    ) -> None:
        panel = pygame.Surface((WINDOW_WIDTH - 20, 82), pygame.SRCALPHA)
        pygame.draw.rect(
            panel, (*CHARCOAL, 218), (0, 0, panel.get_width(), panel.get_height()), border_radius=15
        )
        pygame.draw.rect(
            panel, (*CREAM, 90), (0, 0, panel.get_width(), panel.get_height()),
            width=1, border_radius=15
        )
        self.screen.blit(panel, (10, 10))

        title = self.font_medium.render("PROXIMAL FLAP", True, CREAM)
        subtitle = self.font_tiny.render("RETRO SOLO FLIGHT", True, (220, 20, 60))
        self.screen.blit(title, (22, 18))
        self.screen.blit(subtitle, (23, 43))

        status_color = CRIMSON if pilot_enabled else CREAM
        status_text_color = CREAM if pilot_enabled else CHARCOAL
        status_label = {"ppo": "PPO AI", "solo": "SOLO PILOT", "manual": "MANUAL"}.get(control_mode, "MANUAL")
        status = self.font_tiny.render(status_label, True, status_text_color)
        pill_width = max(61, status.get_width() + 14)
        pill = pygame.Rect(22, 61, pill_width, 17)
        pygame.draw.rect(self.screen, status_color, pill, border_radius=8)
        self.screen.blit(status, status.get_rect(center=pill.center))

        score_text = self.font_large.render(str(world.score), True, CREAM)
        score_rect = score_text.get_rect(center=(216, 50))
        self.screen.blit(score_text, score_rect)
        score_label = self.font_tiny.render("SCORE", True, (220, 20, 60))
        self.screen.blit(score_label, score_label.get_rect(center=(216, 76)))

        best_label = self.font_tiny.render("BEST", True, CREAM)
        best_value = self.font_medium.render(str(best_score), True, (255, 243, 224))
        self.screen.blit(best_label, best_label.get_rect(topright=(378, 25)))
        self.screen.blit(best_value, best_value.get_rect(topright=(378, 51)))

        run_text = self.font_tiny.render(f"RUN {run_number:03d}", True, CHARCOAL)
        self.screen.blit(run_text, (15, 99))

        pressure = int(world.difficulty * 100)
        pressure_text = self.font_tiny.render(
            f"COURSE PRESSURE {pressure:02d}%", True, CHARCOAL
        )
        self.screen.blit(pressure_text, (15, 115))
        bar_bg = pygame.Rect(15, 130, 112, 5)
        pygame.draw.rect(self.screen, CREAM, bar_bg, border_radius=2)
        if pressure:
            bar_fill = pygame.Rect(15, 130, max(2, int(112 * pressure / 100)), 5)
            pygame.draw.rect(self.screen, CRIMSON, bar_fill, border_radius=2)

        controls = "1 MANUAL   2 SOLO   3 PPO AI   P PAUSE   H HELP"
        control_text = self.font_tiny.render(controls, True, CHARCOAL)
        self.screen.blit(control_text, control_text.get_rect(center=(WINDOW_WIDTH // 2, 521)))

    def _draw_message(self, heading: str, detail: str) -> None:
        self._overlay.fill((23, 23, 23, 150))
        self.screen.blit(self._overlay, (0, 0))
        card = pygame.Rect(35, 238, WINDOW_WIDTH - 70, 112)
        pygame.draw.rect(self.screen, CHARCOAL, card, border_radius=14)
        pygame.draw.rect(self.screen, CRIMSON, card, width=2, border_radius=14)
        title = self.font_medium.render(heading, True, CREAM)
        sub = self.font_small.render(detail, True, CREAM)
        self.screen.blit(title, title.get_rect(center=(WINDOW_WIDTH // 2, 276)))
        self.screen.blit(sub, sub.get_rect(center=(WINDOW_WIDTH // 2, 311)))

    def _draw_help(self, pilot_enabled: bool) -> None:
        shade = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        shade.fill((23, 23, 23, 218))
        self.screen.blit(shade, (0, 0))
        card = pygame.Rect(28, 150, WINDOW_WIDTH - 56, 300)
        pygame.draw.rect(self.screen, CHARCOAL, card, border_radius=15)
        pygame.draw.rect(self.screen, CRIMSON, card, width=2, border_radius=15)

        heading = self.font_medium.render("FLIGHT CONTROLS", True, CREAM)
        self.screen.blit(heading, heading.get_rect(center=(WINDOW_WIDTH // 2, 185)))
        rows = [
            ("SPACE / CLICK", "Flap in manual mode"),
            ("1 / 2 / 3", "Manual / SoloPilot / PPO AI"),
            ("A", "Toggle manual and SoloPilot"),
            ("P", "Pause or resume"),
            ("R", "Start a fresh run"),
            ("H / ESC", "Close this panel"),
        ]
        y = 220
        for key, description in rows:
            key_surface = self.font_small.render(key, True, CRIMSON)
            value_surface = self.font_small.render(description, True, CREAM)
            self.screen.blit(key_surface, (54, y))
            self.screen.blit(value_surface, (170, y))
            y += 29

        mode_text = "Autopilot is active" if pilot_enabled else "Manual flight is active"
        mode_surface = self.font_tiny.render(mode_text, True, CREAM)
        self.screen.blit(mode_surface, mode_surface.get_rect(center=(WINDOW_WIDTH // 2, 410)))
        close_text = self.font_tiny.render("Press H or ESC to return", True, CREAM)
        self.screen.blit(close_text, close_text.get_rect(center=(WINDOW_WIDTH // 2, 428)))
