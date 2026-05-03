# ai-photo-pose

MVP FastAPI backend for an AI photography pose recommendation app.

The current implementation is intentionally rule-based. Image analysis is mocked through explicit scene, style, and person context fields in the request.

## Run

```bash
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

## Test

```bash
pytest
```

## Endpoint

`POST /recommend_pose`

```json
{
  "scene": {
    "scene_tags": ["urban", "street"],
    "style_tags": ["editorial", "confident"]
  },
  "person": {
    "age_group": "adult",
    "body_type": "average",
    "comfort_level": "medium",
    "mobility_notes": []
  }
}
```
