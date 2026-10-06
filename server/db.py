"""反饋持久化（Phase 4）：SQLAlchemy sync engine + feedback 表。

- DATABASE_URL 未設或連不上 → db_ok=False，/feedback 回 503，前端降級本地暫存（demo 不開天窗）
- 表結構在 lifespan create_all（demo 級別，不引 alembic）
"""
import os
from datetime import datetime, timezone

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://dance:dance@localhost:5432/dance_feedback",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(index=True)
    url: Mapped[str]
    kind: Mapped[str]                      # youtube | mp4 | file
    media_time: Mapped[float]
    predicted: Mapped[str]                 # 當下預測舞種（含 unknown）
    confidence: Mapped[float]
    correct: Mapped[str | None] = mapped_column(default=None)  # 👍 = NULL；👎 = 正確類別
    verdict: Mapped[str]                   # up | down
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))


def init_db() -> bool:
    """建表；回傳 DB 是否可用。"""
    try:
        Base.metadata.create_all(engine)
        return True
    except Exception:
        return False


def db_ok() -> bool:
    try:
        return inspect(engine).has_table("feedback")
    except Exception:
        return False
