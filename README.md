# JetsonCV — Tello SDK-style Demo

This repository contains a modular SDK-style layout for Tello EDU drone autonomy and computer
vision demos. It provides lightweight core components and optional perception/control modules
so you can build and run autonomy pipelines incrementally.

Important: this README documents the refactored SDK and the example apps under `apps/`.
It intentionally excludes the original top-level application file to avoid duplication.

## What you get
- `jetsoncv/` — SDK package (core, perception, control, highlevel stubs)
- `apps/main.py` — Primary demo that uses the SDK components and attempts to connect to a Tello
- `apps/example_basic_cv.py` — Minimal example that runs BasicCV on the Tello stream (safe fallback)
- `requirements.txt` — List of external dependencies (optional modules are stubs when missing)

The SDK is designed so advanced modules (YOLO, ORB-SLAM2, DroneFormer, RL) are optional and
will not crash the app if their dependencies are not installed.

## Quick start (macOS / zsh)

1. Create and activate a virtualenv (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies (the list includes optional packages; the SDK will run without them):

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Run the minimal BasicCV example (does not require YOLO/ORB/DroneFormer):

```bash
python apps/example_basic_cv.py
```

4. Run the primary demo (attempts to connect to Tello and stream):

```bash
python apps/main.py
```

Notes:
- Make sure your computer is connected to the Tello Wi‑Fi network before running the Tello-based apps.
- If the Tello stream is unavailable, the demo falls back to a waiting loop and will not crash.
- To stop any demo, press `ESC` or `q` in the display window; the demo will attempt to land the drone.

## File layout (high level)

- `jetsoncv/sdk/core/` — core runtime utilities: `video_input.py`, `frame_hub.py`, `types.py`, `config.py`, `logging_utils.py`
- `jetsoncv/sdk/perception/` — perception modules: `basic_cv.py` (always available), `yolo_detector.py` (stub), `orb_slam2.py` (stub)
- `jetsoncv/sdk/control/` — controllers and planners: `follow_me.py`, `safety.py`, `behavior.py`, `low_level.py`
- `jetsoncv/sdk/highlevel/` — future integrations: `droneformer.py`, `rl_policy.py` (stubs)
- `apps/` — runnable example apps using the SDK

## Extending the SDK

- To add YOLO: implement `jetsoncv/sdk/perception/yolo_detector.py` real model loading and inference. The stub will warn but not crash when dependencies are missing.
- To plug advanced planners (DroneFormer / RL): implement the high-level interfaces in `jetsoncv/sdk/highlevel/` and integrate with `BehaviorPlanner`.

## Safety

- The demo will not auto-trigger `takeoff()`; you must issue takeoff/land commands manually or implement a controlled takeoff in your app.
- Always test with props removed for initial algorithm validation.

## Development tips

- Import the SDK in a Python REPL to verify availability:

```bash
source .venv/bin/activate
python -c "import jetsoncv; print('OK')"
```

- Use the `apps/main.py` file as a reference for assembling components (video input, frame hub, BasicCV, BehaviorPlanner).

If you want, I can generate a more detailed developer guide, add CI tests, or wire your existing `Follow-me-DJI-Tello` detectors into the SDK.
