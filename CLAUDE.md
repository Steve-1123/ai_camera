# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`ai_camera` — FastAPI backend. Single business path: image path → pose analysis (MediaPipe) + scene classification (CLIP) → optional MySQL storage.

## Quick Commands

```bash
venv/bin/pip install -r requirement.txt
venv/bin/uvicorn app.main:app --reload
venv/bin/python -m pytest
venv/bin/python -m pytest tests/test_analyze_image.py  # single file

# DB init
DATABASE_URL='mysql+pymysql://user:pass@localhost:3306/pose_app?charset=utf8mb4' \
  venv/bin/python -m app.main init-db

# Main API
curl -X POST 'http://127.0.0.1:8000/analyze_image_path' \
  -H 'Content-Type: application/json' \
  -d '{"image_url":"images/example.jpg","should_store":true}'
```

## Document Map

Before coding, read the docs relevant to the task:

| Document | When to read |
|---|---|
| `AGENTS.md` | Always first — rules all agents must follow |
| `README.md` | Project overview, install, run, test |
| `docs/ARCHITECTURE.md` | Layer boundaries, file responsibilities. Required when touching directory structure. |
| `docs/SCHEMAS.md` | API I/O, DB tables, JSON fields. Required when touching schemas or storage. |
| `docs/AI_HANDOFF.md` | Current state, recent changes, what's not done. Required when continuing someone else's work. |
| `docs/ROADMAP.md` | Current phase and next steps. Required when making product-direction changes. |
| `docs/DECISIONS.md` | Technical decisions and rationale. Required when considering tech changes. |

## Key Constraints (from AGENTS.md)

- Single main entry point: `POST /analyze_image_path` → `app.services.image_analysis_service.analyze_image_path`
- `vision/` must not touch DB. `repositories/` must not call model inference.
- Scene analysis is returned but NOT stored in MySQL.
- Only 3 DB tables: `app_images`, `poses`, `pose_keypoints`.
- Do NOT add recommendation APIs, LLM suggestions, real-time camera, or pose card rendering unless explicitly asked.
- Do not modify this file unless explicitly asked.