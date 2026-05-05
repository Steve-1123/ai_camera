from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


def create_app_engine(database_url: str | None = None):
    url = database_url or get_settings().database_url
    return create_engine(url, pool_pre_ping=True, future=True)


def create_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=create_app_engine(database_url), expire_on_commit=False, class_=Session)


SessionLocal = create_session_factory


def init_db(database_url: str | None = None) -> None:
    from app.models import image, pose, pose_keypoint  # noqa: F401

    engine = create_app_engine(database_url)
    Base.metadata.create_all(engine)


@contextmanager
def session_scope(database_url: str | None = None) -> Iterator[Session]:
    session = create_session_factory(database_url)()
    try:
        yield session
    finally:
        session.close()
