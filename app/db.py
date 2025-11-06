from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


_engine_cache = {}
_SessionFactory = None


def get_engine(db_url: str):
    if db_url not in _engine_cache:
        _engine_cache[db_url] = create_engine(db_url, echo=False, future=True)
    return _engine_cache[db_url]


def get_session_factory(db_url: str):
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(
            bind=get_engine(db_url),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
    return _SessionFactory

