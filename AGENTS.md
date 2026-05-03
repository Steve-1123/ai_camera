# Repository Guidelines

## Project Structure & Module Organization

This repository is a small FastAPI backend for rule-based photography pose recommendations. Application code lives in `app/`. The API entry point is `app/main.py`, shared Pydantic models are in `app/schemas.py`, recommendation logic is in `app/recommender/`, and pose data loading plus the JSON library are in `app/pose_library/`. Tests live in `tests/` and mirror the main behavior areas: API, pose library loading, and recommendation rules.

## Build, Test, and Development Commands

- `pip install -e ".[dev]"`: install the package in editable mode with test dependencies.
- `uvicorn app.main:app --reload`: run the local FastAPI development server.
- `pytest`: run the full test suite configured by `pyproject.toml`.
- `pytest tests/test_api.py`: run a focused test module while working on one behavior area.

## Coding Style & Naming Conventions

Use Python 3.9+ syntax and keep modules simple and typed. Follow the existing style: 4-space indentation, explicit imports from `app.*`, snake_case functions and variables, PascalCase classes, and descriptive Pydantic schema names such as `RecommendationRequest`. Keep JSON data in `app/pose_library/poses.json` consistently formatted and validated by the loader tests before changing recommender behavior.

## Testing Guidelines

The project uses `pytest`; test discovery is limited to `tests/` by `pyproject.toml`. Name new test files `test_*.py` and test functions `test_*`. Add or update tests when changing schema validation, pose loading, scoring, filtering, or API responses. Prefer focused assertions that describe observable behavior rather than implementation details.

## Commit & Pull Request Guidelines

The current history only contains the initial backend commit, so use concise, imperative commit messages going forward, for example `Add pose library validation` or `Update recommendation scoring`. Pull requests should include a short summary, testing performed, and any API or JSON schema changes. Include example requests or screenshots only when the behavior is user-visible through the API docs or responses.

## Security & Configuration Tips

Do not commit secrets, tokens, or local environment files. Keep runtime configuration explicit in code or documented in `README.md` until a dedicated settings module is introduced. Treat `poses.json` as source data: review large additions for malformed JSON, duplicate IDs, and unsafe or inaccessible pose instructions.
