"""Central configuration for PROXIMAL FLAP: SOLO."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
ASSET_DIR = ROOT_DIR / "assets"
DATA_DIR = ROOT_DIR / "data"

# Compact game window and logical simulation dimensions.
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 600
FPS = 60
GROUND_Y = 536


BIRD_X = 92.0
BIRD_START_Y = 280.0
BIRD_RADIUS_X = 14
BIRD_RADIUS_Y = 12
BIRD_GRAVITY = 0.46
BIRD_FLAP_VELOCITY = -7.0

PIPE_WIDTH = 79
PIPE_GAP_MIN = 106
PIPE_GAP_MAX = 140
PIPE_SPEED = 3.45
PIPE_SPEED_MAX = 4.6
PIPE_FIRST_X = 420.0
PIPE_SPAWN_THRESHOLD = 220.0
PIPE_SPACING_MIN = 208
PIPE_SPACING_MAX = 252
PIPE_GAP_TOP_MIN = 78
PIPE_GAP_CENTER_MAX = 360

DIFFICULTY_RAMP_SCORE = 36
GAP_CENTER_STEP_START = 58
GAP_CENTER_STEP_MAX = 102

AUTO_RESTART_SECONDS = 0.85

# Visual palette: Crimson / Charcoal / Cream.
CRIMSON = (220, 20, 60)  # #DC143C
CHARCOAL = (23, 23, 23)  # #171717
CREAM = (255, 243, 224)  # #FFF3E0

# Stable asset contract for the upcoming individually generated artwork.
SPRITE_DIR = ASSET_DIR / "sprites"
BACKGROUND_ASSET = SPRITE_DIR / "background.png"
GROUND_ASSET = SPRITE_DIR / "ground.png"
PIPE_UP_ASSET = SPRITE_DIR / "pipe_up.png"
PIPE_DOWN_ASSET = SPRITE_DIR / "pipe_down.png"
BIRD_ASSET = SPRITE_DIR / "bird.png"  # Optional; no bird.png exists yet.


# Background moves more slowly than the foreground for a calm parallax effect.
BACKGROUND_PARALLAX = 0.20
