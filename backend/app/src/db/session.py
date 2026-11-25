from collections.abc import Iterator

from fastapi import Request
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def create_session_factory(
    database_url: str,
) -> tuple[Engine, sessionmaker[Session]]:
    engine = create_engine(database_url, echo=False)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, session_factory


def get_db(request: Request) -> Iterator[Session]:
    session_factory = getattr(request.app.state, "db_session_factory", None)
    if session_factory is None:
        raise RuntimeError("Database session factory is not initialized.")

    session = session_factory()
    try:
        yield session
    finally:
        session.close()
