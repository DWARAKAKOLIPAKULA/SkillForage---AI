from sqlalchemy import String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base
import uuid

class Goal(Base):
    __tablename__ = "goals"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    title: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=True)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)

    user: Mapped["User"] = relationship("User", back_populates="goals")