# ai-camera

Narrow MVP FastAPI backend for image upload analysis.

The current API accepts one uploaded image and returns:

- basic image metadata
- MediaPipe pose landmarks when a person is detected
- a lightweight heuristic scene classification result

Legacy pose recommendation code is kept under `app/legacy/` and is not imported by the default API.

## Install

```bash
pip install -e ".[dev]"
```

## Run

```bash
uvicorn app.main:app --reload
```

## Test

```bash
pytest
```

## Endpoints

`GET /health`

`POST /analyze_image`

```bash
curl -X POST "http://127.0.0.1:8000/analyze_image" \
  -F "image=@/path/to/photo.jpg"
```

