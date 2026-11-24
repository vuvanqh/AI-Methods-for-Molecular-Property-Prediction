from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# relative: the app runs as `src.*` (from backend/app) but alembic imports it as `app.src.*` (from backend/)
from ..core.config import settings

engine = create_engine(settings.database_url, echo=False)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
