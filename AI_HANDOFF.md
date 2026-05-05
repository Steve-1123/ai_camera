# AI Handoff

This file is for future AI coding agents working on this repository. Keep changes small, explicit, and easy to review.

## Project Snapshot

`ai_camera` is a small FastAPI backend. It started as a rule-based photography pose recommendation MVP, but the current active work is image analysis and pose visualization:

- Upload an image and analyze it through `POST /analyze_image`.
- Detect human pose landmarks with MediaPipe Pose.
- Classify scene context with CLIP zero-shot prompts, with a heuristic fallback.
- Read stored pose data and generate a visual pose card through `/poses/{pose_id}/card`.

Do not rebuild the project from scratch. Preserve the existing structure and add incrementally.

## Current Structure

- `app/main.py`: FastAPI entry point. Defines `/health`, `/analyze_image`, `/poses`, `/poses/{pose_id}`, and `/poses/{pose_id}/card`.
- `app/schemas.py`: Pydantic response models for image analysis. Avoid changing response schemas unless explicitly required.
- `app/vision/`: image loading, RGB normalization, MediaPipe pose estimation, and scene classification.
- `app/pose_library/`: active pose library loader. It currently reads `app/legacy/pose_library/poses.json`.
- `app/visualization/`: pose skeleton and pose card rendering with Pillow.
- `app/legacy/`: old recommendation and pose library code. Do not delete it unless the user explicitly asks.
- `tests/`: pytest coverage for image analysis and pose visualization.
- `scripts/test_scene_classifier.py`: local utility to inspect scene classification on one image.

## Important Behaviors

### Image Analysis

`POST /analyze_image` accepts a multipart image upload and returns:

- `image_info`
- `pose`
- `scene`

Keep this response schema stable unless the user explicitly approves a change.

Image loading should guarantee:

- RGB `numpy.ndarray`
- `uint8` dtype
- `H x W x 3` shape
- alpha images converted to RGB

Pose detection should use MediaPipe Pose with:

- `static_image_mode=True`
- `model_complexity=2`
- `min_detection_confidence=0.3`
- longest side resized to 1280 before detection

Debug logging should include image shape, dtype, min/max pixel value, and whether MediaPipe detected landmarks.

### Scene Classification

If CLIP is available, scene classification uses descriptive prompts rather than one-word labels. Return the short label in API results, not the full prompt.

Current labels include:

- `cafe`
- `street`
- `beach`
- `park`
- `stairs`
- `wall`
- `indoor`
- `landmark`
- `restaurant`
- `hotel`
- `museum`

If CLIP cannot load, the heuristic fallback should keep the API working.

### Pose Visualization

The active pose visualization feature includes:

- `PoseRenderer.render_skeleton(landmarks) -> PIL.Image`
- `PoseCardRenderer.render_pose_card(pose) -> PIL.Image`
- `GET /poses`
- `GET /poses/{pose_id}`
- `GET /poses/{pose_id}/card`

Stored poses may or may not have `landmarks`. The loader must preserve a `landmarks` field when present. The card renderer currently falls back to a generic skeleton if landmarks are missing, so old pose data can still render.

Do not implement recommendation logic or LLM suggestions unless the user asks.

## Development Rules

Follow the repository instructions in `AGENTS.md`.

- Use the existing `venv` for Python dependency installs and updates.
- Whenever dependencies change, update `requirement.txt` promptly.
- If `pyproject.toml` dependencies change, keep `requirement.txt` in sync.
- For new files or folders, decide whether they belong in `.gitignore`.
- Do not commit secrets, tokens, local caches, downloaded model files, uploaded images, or local environment files.
- Avoid broad refactors unless the user explicitly asks for them.
- Preserve old code unless deletion is requested.
- Prefer focused tests for changed behavior.

## Common Commands

Install dependencies in the virtual environment:

```bash
venv/bin/pip install -r requirement.txt
```

Run the app locally:

```bash
venv/bin/uvicorn app.main:app --reload
```

Run all tests:

```bash
venv/bin/python -m pytest
```

Run focused tests:

```bash
venv/bin/python -m pytest tests/test_analyze_image.py
venv/bin/python -m pytest tests/test_pose_visualization.py
```

Inspect scene classification for one image:

```bash
venv/bin/python scripts/test_scene_classifier.py images/full/full_01.jpg
```

Example pose card URL after starting the service:

```text
http://127.0.0.1:8000/poses/street_walk_lookback_001/card
```

## Review Expectations

The user sends every commit to Claude for review. Make changes that are easy to defend:

- Keep diffs scoped.
- Explain why each behavior changed.
- Add or update tests when changing observable behavior.
- Avoid hidden interface changes.
- Avoid large style-only churn.

Before handing work back, report:

- Files changed.
- Tests run and whether they passed.
- Any dependency or `.gitignore` changes.
- Any known limitations or follow-up risks.
