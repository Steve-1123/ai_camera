from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str


def get_settings() -> Settings:
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not configured. Example: "
            "mysql+pymysql://user:password@localhost:3306/pose_app?charset=utf8mb4"
        )
    return Settings(database_url=database_url)
