# ai-camera

Narrow MVP FastAPI backend for image upload analysis and stored pose ingestion.

The current API accepts one uploaded image and returns:

- basic image metadata
- MediaPipe pose landmarks when a person is detected
- CLIP scene classification with a heuristic fallback
- stored pose ingestion into `app/pose_library/poses.json`

Recommendation APIs and old recommender modules are intentionally not part of the current MVP.

## Install

```bash
venv/bin/pip install -r requirement.txt
```

## Run

```bash
venv/bin/uvicorn app.main:app --reload
```

## Test

```bash
venv/bin/python -m pytest
```

## Endpoints

`GET /health`

`POST /analyze_image`

```bash
curl -X POST "http://127.0.0.1:8000/analyze_image" \
  -F "image=@/path/to/photo.jpg"
```

## Ingest Pose Image

```bash
venv/bin/python scripts/ingest_pose_image.py \
  --pose-id street_side_hair_touch_002 \
  --image images/example.jpg \
  --poses-json app/pose_library/poses.json \
  --overwrite
```
