from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base

_ENGINE: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _ensure_sqlite_schema(engine: Engine) -> None:
    """Backfill small schema gaps in existing local SQLite databases.

    The app currently relies on per-user memory rows. Older databases created
    before `MemoryItem.user_id` existed will otherwise crash on memory lookup.
    """
    if engine.dialect.name != "sqlite":
        return

    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if "memory_items" not in tables:
        return

    memory_columns = {col["name"] for col in inspector.get_columns("memory_items")}
    if "user_id" not in memory_columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE memory_items ADD COLUMN user_id INTEGER"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_memory_items_user_id ON memory_items (user_id)"))


def init_db(database_url: str) -> None:
    global _ENGINE, _SessionLocal
    if _ENGINE is not None and _SessionLocal is not None:
        return

    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}

    _ENGINE = create_engine(database_url, connect_args=connect_args, pool_pre_ping=True)
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_ENGINE)

    Base.metadata.create_all(bind=_ENGINE)
    _ensure_sqlite_schema(_ENGINE)


def get_db() -> Generator[Session, None, None]:
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() on startup.")
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()
