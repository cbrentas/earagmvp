from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False)


class Base(DeclarativeBase):
    pass


def initialize_database():
    from app import models  # noqa: F401; register the models
    with engine.begin() as conn:
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
    Base.metadata.create_all(bind=engine)


def get_db():
    with SessionLocal() as db:
        yield db
