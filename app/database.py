from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings

# asyncpg is the async PostgreSQL driver
engine = create_async_engine(
    settings.DATABASE_URL,   # postgresql+asyncpg://user:pass@localhost/db
    echo=True,               # logs SQL queries — remove in production
)

# Session factory — every request gets its own session
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # object attributes stay accessible after commit
)

class Base(DeclarativeBase):
    pass

# Dependency — used in every route that needs a DB session
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise