"""
Collaboration Models
Collaboration session related Pydantic models
"""

from datetime import datetime
from typing import List, Dict, Any
from uuid import uuid4

from pydantic import BaseModel, Field

from .enums import SessionStatus

class CollaborationSession(BaseModel):
    """Collaboration session model"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    project_id: str
    session_name: str
    creator_id: str
    participants: List[str] = Field(default_factory=list)
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    settings: Dict[str, Any] = Field(default_factory=dict)

class CollaborationMessage(BaseModel):
    """Collaboration message model"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    user_id: str
    message_type: str = Field(..., pattern="^(text|edit|comment|cursor|selection)$")
    content: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)

class CollaborationEvent(BaseModel):
    """Collaboration event model"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    user_id: str
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)







