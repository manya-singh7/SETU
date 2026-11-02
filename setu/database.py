"""
Database session management and engine initialization using SQLModel.

Stores all bridge entities, upload metadata, analysis results, and
engineer triage items in a local SQLite database for zero-friction local setup.
"""

from typing import Generator
from sqlmodel import Session, SQLModel, create_engine
from setu.config import settings

# SQLite connection args for multi-threaded FastAPI access
connect_args = {"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
engine = create_engine(settings.DATABASE_URL, echo=False, connect_args=connect_args)


def create_db_and_tables() -> None:
    """Create all SQLModel tables on startup."""
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency for yielding database sessions."""
    with Session(engine) as session:
        yield session
