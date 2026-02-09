"""Share model for content sharing."""

from sqlalchemy import Column, String, Integer, DateTime, Enum
from sqlalchemy.sql import func
import enum
from datetime import datetime

from .base import Base


class SharePlatform(str, enum.Enum):
    """Sharing platforms."""
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    LINKEDIN = "linkedin"
    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"
    COPY_LINK = "copy_link"
    OTHER = "other"


class Share(Base):
    """Model for content shares."""
    
    __tablename__ = "shares"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    content_type = Column(String, nullable=False, index=True)  # 'chat', 'comment', etc.
    content_id = Column(String, nullable=False, index=True)
    platform = Column(Enum(SharePlatform), nullable=False, index=True)
    created_at = Column(DateTime, default=func.now(), index=True)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "content_type": self.content_type,
            "content_id": self.content_id,
            "platform": self.platform.value if self.platform else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }







