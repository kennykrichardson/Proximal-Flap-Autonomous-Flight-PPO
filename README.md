# PROXIMAL FLAP: SOLO

**One bird. One course. No two runs quite alike.**

PROXIMAL FLAP: SOLO is the single-bird playable foundation for the PROXIMAL FLAP rebuild. One autonomous bird plays at a time, the course is generated continuously, and a collision automatically begins the next run. Manual play is available at any time.

## What's new in v1.1

- **A tougher course:** gap sizes begin around 110–140 px and tighten toward 106–118 px as the run progresses.
- **Increasing pipe speed:** obstacles accelerate from approximately 3.45 to 4.6 pixels per frame over the first 36 cleared pipes.
- **Unpredictable but bounded gaps:** gap centres use a weighted random walk, with sharper climbs and drops later in a run. The maximum centre shift grows from about 58 to 102 px, so the course is less repetitive without making obstacles teleport to impossible positions.
- **A more deliberate solo pilot:** the feedback controller uses a measured flap cadence and a gap-size-aware target margin. It is not a trained neural network.
- **Retro-futuristic visual pass:** a crimson / charcoal / cream bird, animated wing poses, recoloured pipe sprites, and a matching HUD. The original blue-sky-and-cloud background is retained.
- **Continuous solo play:** the current bird automatically restarts after a collision while the best score persists.

Difficulty ramps with the score and reaches its maximum after 36 cleared pipes. At maximum difficulty, gaps are narrow, vertical changes are steeper, and pipes move faster. Randomness is constrained by the course generator's vertical-step limit to keep the challenge physically plausible.

## Run it

Python 3.10 or newer is recommended.

```bash
# From the PROXIMAL-FLAP directory
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
# source .venv/bin/activate

python -m pip install -r requirements.txt
python main.py
```

If Windows blocks script activation, run the interpreter directly instead:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py
```

## Controls

| Key / input | Action |
|---|---|
| `1` | Manual mode |
| `2` | SoloPilot deterministic controller |
| `3` or `M` | Trained PPO AI |
| `A` | Toggle Manual / SoloPilot |
| `SPACE`, `UP`, or left click | Flap in Manual mode |
| `P` | Pause / resume |
| `R` | Start a fresh run immediately |
| `H` | Open / close the controls panel |
| `ESC` | Close the panel, or quit from the game |

SoloPilot is enabled at launch. Choose Manual, SoloPilot, or the trained PPO controller with the mode keys above. In manual mode, the bird waits for your input. A collision ends only the current run; the next run starts automatically after a short reset pause. The best score is saved in `data/best_score.json`.

## Difficulty model

The next pipe's profile is based on the current score:

- Gap width decreases progressively.
- Pipe speed increases progressively.
- Pipe spacing contracts slightly as speed rises.
- Gap centres follow a weighted random walk, alternating between modest adjustments and occasional steep changes.
- The vertical change between consecutive gap centres is explicitly bounded. It becomes more demanding as the score rises, but it does not jump arbitrarily from ceiling to floor.

The first 36 pipes are a ramp rather than a flat warm-up. The challenge continues at the maximum profile after that point.

## Solo pilot

SoloPilot is a transparent feedback policy, not a trained neural network. It tracks the next safe gap and issues flaps according to the bird's vertical position, momentum, and available gap size. It remains available alongside the separately trained PPO controller, so both approaches can be compared without rewriting the game loop.

The pilot is not expected to beat the maximum-difficulty course indefinitely. Its job at this milestone is to demonstrate one autonomous bird playing continuously through repeated attempts, while leaving the game genuinely challenging for a human.

## Project layout

```text
PROXIMAL-FLAP/
├── assets/
│   ├── concept/
│   ├── fonts/
│   ├── sounds/
│   └── sprites/
├── data/
├── game/
│   ├── controller.py
│   ├── entities.py
│   ├── renderer.py
│   └── world.py
├── tests/
│   └── test_simulation.py
├── config.py
├── main.py
├── storage.py
├── requirements.txt
└── pyproject.toml
```

## Tests

The simulation logic is renderer-independent and can be tested without opening a Pygame window:

```bash
python -m unittest discover -s tests -v
```

The v1.1 test suite covers physics, increasing difficulty, bounded unpredictable gap movement, seeded autopilot performance, continuous play across automatic restarts, manual controls, and persistent high scores.

## Current scope

- One visible bird and one controller at a time.
- Continuous, procedurally generated pipe stream.
- Score-based escalating difficulty.
- Autopilot enabled by default; manual play is available.
- Automatic restart after a collision.
- Persistent best score.
- Original sky background with a refreshed retro-futuristic character and obstacle palette.

Not included yet: population training, genetic evolution, species tracking, or a training dashboard. PPO training and evaluation are available through the optional AI tools described below.


## Resolution and generated-artwork workflow

The game uses a compact 400x600 simulation and presentation window. Keeping the
logical physics dimensions and rendered surface aligned makes seeded simulations
consistent with the visible game. Replace these files individually when the
new artwork is generated:

- `assets/sprites/background.png` - full cream/crimson/charcoal cyberpunk sky
- `assets/sprites/ground.png` - matching foreground/ground strip
- `assets/sprites/pipe_up.png` and `pipe_down.png` - flat matte obstacle pieces
- `assets/sprites/bird.png` - optional transparent bird sprite; if absent, the
  built-in procedural bird remains active

## Deterministic baseline benchmark

Run a fixed-seed benchmark before changing the controller or introducing AI:

```bash
python -m tools.benchmark --seeds 100 --frames 30000
```

The full per-seed results are written to `data/deterministic-baseline.json`.
There is intentionally no upper score assertion: a high score is not a test
failure. Compare future controllers on the same seed set and frame limit.

### Background scrolling

The background has its own parallax offset; it is not derived from the ground offset. The artwork is mirrored into a 2x-width edge-matched tile, and the offset wraps over that complete tile. This avoids the periodic backward jump caused by scaling a ground offset that already wraps at the viewport width.


## PPO AI training

Install the optional reinforcement-learning dependencies:

```powershell
python -m pip install -r requirements-ai.txt
```

Run a short smoke training session:

```powershell
python -m ai.train --timesteps 20000 --n-envs 2 --checkpoint-every 10000
```

Run the full default training session:

```powershell
python -m ai.train
```

Evaluate the trained policy against SoloPilot on identical generated courses:

```powershell
python -m ai.evaluate --model ai/checkpoints/ppo_proximal_flap.zip --seeds 100
```

See [AI_PHASE2.md](AI_PHASE2.md) for training configuration, checkpoints, TensorBoard,
and inference usage.


## PPO improvement experiments

See [AI_EXPERIMENTS_V2.md](AI_EXPERIMENTS_V2.md) for the isolated V2 observation/reward experiment, training commands, and paired evaluation. The existing `ai/checkpoints/ppo_proximal_flap.zip` champion remains unchanged and is still used by live gameplay.
