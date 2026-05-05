# AI Handoff

This file is for future AI coding agents working on this repository. Keep changes small, explicit, and easy to review.

## Project Snapshot

`ai_camera` is a small FastAPI backend. It started as a rule-based photography pose recommendation MVP, but the current MVP is now limited to image analysis and stored pose ingestion:

- Upload an image and analyze it through `POST /analyze_image`.
- Detect human pose landmarks with MediaPipe Pose.
- Classify scene context with CLIP zero-shot prompts, with a heuristic fallback.
- Given a `pose_id` and reference image, write a `landmark_template`, `scene_analysis`, derived skeleton features, and `embedding_document` into `app/pose_library/poses.json`.

Recommendation APIs, old recommender modules, LLM photo advice, realtime camera features, and pose card rendering are intentionally out of scope.

## Current Structure

- `app/main.py`: FastAPI entry point. Defines only `/health` and `/analyze_image`.
- `app/schemas.py`: Pydantic response models for image analysis. Avoid changing response schemas unless explicitly required.
- `app/vision/`: image loading, RGB normalization, MediaPipe pose estimation, and scene classification.
- `app/features/`: skeleton feature derivation from MediaPipe normalized landmarks.
- `app/pose_library/`: active `poses.json`, loader, and reference image ingestion updater.
- `tests/`: pytest coverage for image analysis, skeleton features, and pose library updater.
- `scripts/ingest_pose_image.py`: CLI to ingest one reference image into a stored pose.
- `scripts/test_scene_classifier.py`: local utility to inspect scene classification on one image.

Current active file tree:

```text
app/
  main.py
  schemas.py
  features/
    __init__.py
    skeleton_features.py
  pose_library/
    __init__.py
    loader.py
    poses.json
    updater.py
  vision/
    __init__.py
    image_loader.py
    pose_estimator.py
    scene_classifier.py

scripts/
  ingest_pose_image.py
  test_scene_classifier.py

tests/
  test_analyze_image.py
  test_pose_library_updater.py
  test_skeleton_features.py
```

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

### Pose Library Ingestion

The active pose ingestion feature includes:

- `compute_skeleton_features(landmarks) -> dict`
- `ingest_pose_image(poses_json_path, pose_id, image_path, overwrite=False) -> dict`
- `build_embedding_document(pose) -> str`
- `scripts/ingest_pose_image.py`

Stored poses may or may not have `landmark_template`. Ingestion must not overwrite an existing `landmark_template` unless `overwrite=True`.

`ingest_pose_image(...)` writes these fields back into the matching pose in `app/pose_library/poses.json`:

- `landmark_template`: MediaPipe 33-landmark template, source image metadata, quality metrics, raw landmarks, and `derived_features`.
- `scene_analysis`: scene classifier result with `primary_category`, `candidates`, and `model_name`.
- `embedding_document`: plain text assembled from pose metadata, scene analysis, skeleton features, and instructions.

The `landmark_template` coordinate space is normalized image coordinates, origin at image top-left, with `x`/`y` in `[0.0, 1.0]`.

Do not implement recommendation logic or LLM suggestions unless the user asks.

## Development Rules

Follow the repository instructions in `AGENTS.md`.

- Use the existing `venv` for Python dependency installs and updates.
- Whenever dependencies change, update `requirement.txt` promptly.
- If `pyproject.toml` dependencies change, keep `requirement.txt` in sync.
- For new files or folders, decide whether they belong in `.gitignore`.
- Do not commit secrets, tokens, local caches, downloaded model files, uploaded images, or local environment files.
- Avoid broad refactors unless the user explicitly asks for them.
- The current refactor explicitly removed old legacy/recommender and pose card code from the active tree.
- Prefer focused tests for changed behavior.
- `images/`, `venv/`, model caches, and local prompt files should remain ignored and uncommitted.

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

Current baseline after the MVP refactor: `11 passed`. Some environments emit urllib3/LibreSSL and MediaPipe dependency warnings; those warnings are currently expected.

Run focused tests:

```bash
venv/bin/python -m pytest tests/test_analyze_image.py
venv/bin/python -m pytest tests/test_skeleton_features.py
venv/bin/python -m pytest tests/test_pose_library_updater.py
```

Inspect scene classification for one image:

```bash
venv/bin/python scripts/test_scene_classifier.py images/full/full_01.jpg
```

Ingest one reference image into the pose library:

```bash
venv/bin/python scripts/ingest_pose_image.py \
  --pose-id street_side_hair_touch_002 \
  --image images/example.jpg \
  --poses-json app/pose_library/poses.json \
  --overwrite
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
