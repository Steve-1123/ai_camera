# AI Handoff

This file is for future AI coding agents working on this repository. Keep changes small, explicit, and easy to review.

## Current MVP

`ai_camera` is now centered on one HTTP business path:

```text
HTTP -> image path analysis -> pose analysis + scene analysis -> optional DB storage
```

The active endpoint is:

- `POST /analyze_image_path`: accepts an image path plus `should_store`; always returns pose + scene analysis when image analysis succeeds.

## Active Structure

```text
app/
  main.py
  schemas.py
  core/
    config.py
    database.py
  models/
    image.py
    pose.py
    pose_keypoint.py
  repositories/
    image_repository.py
    pose_repository.py
  services/
    image_analysis_service.py
    pose_library_service.py
  vision/
    image_loader.py
    pose_estimator.py
    scene_classifier.py
  pose_library/
    loader.py
    poses.json
    updater.py
  features/
    skeleton_features.py
```

## Main Flow

- `app/main.py::analyze_image_path_endpoint`
- `app/services/image_analysis_service.py::analyze_image_path`
- `app/vision/pose_estimator.py::PoseEstimator.estimate`
- `app/vision/scene_classifier.py::SceneClassifier.classify`
- Optional storage:
  - `app/repositories/image_repository.py`
  - `app/repositories/pose_repository.py`
  - `app/models/image.py`
  - `app/models/pose.py`
  - `app/models/pose_keypoint.py`

Scene analysis is returned by the API but intentionally not stored in MySQL yet.

Scene analysis now uses CLIP only. The previous heuristic RGB fallback was removed because it was redundant and less accurate for semantic background recognition.

## Database

Use MySQL through `DATABASE_URL`:

```bash
export DATABASE_URL='mysql+pymysql://user:password@localhost:3306/pose_app?charset=utf8mb4'
```

Create tables:

```bash
venv/bin/python -m app.main init-db
```

MVP uses `Base.metadata.create_all(engine)`. Alembic is not required yet.

## Commands

```bash
venv/bin/pip install -r requirement.txt
venv/bin/uvicorn app.main:app --reload
venv/bin/python -m pytest
```

Current test baseline: `17 passed`. Some local environments emit urllib3/LibreSSL and MediaPipe dependency warnings; those are expected unless dependency versions change.

Analyze a path through HTTP:

```bash
curl -X POST 'http://127.0.0.1:8000/analyze_image_path' \
  -H 'Content-Type: application/json' \
  -d '{"image_url":"images/example.jpg","should_store":true}'
```

CLI path using the same service:

```bash
venv/bin/python -m app.main analyze --image-url images/example.jpg
```

## Rules

- Keep `requirement.txt` and `pyproject.toml` in sync.
- Do not commit `venv/`, images, model caches, secrets, or local prompt files.
- Keep the `vision` layer free of DB writes.
- Keep repositories free of model inference.
- Keep scene/background analysis out of MySQL until schema is confirmed.
- Keep CLIP as the only scene classifier unless the maintainer explicitly asks for another model.
- `scripts/` are optional manual tools and `tests/` are active regression coverage; do not delete either just because they are not runtime modules.
- Prefer small, test-covered changes because commits are reviewed by Claude.
