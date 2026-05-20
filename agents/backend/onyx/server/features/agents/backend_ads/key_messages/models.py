from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum

class MessageType(str, Enum):
    MARKETING = "marketing"
    EDUCATIONAL = "educational"
    PROMOTIONAL = "promotional"
    INFORMATIONAL = "informational"
    CALL_TO_ACTION = "call_to_action"

class MessageTone(str, Enum):
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"
    CONVERSATIONAL = "conversational"

class KeyMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    message_type: MessageType = MessageType.INFORMATIONAL
    tone: MessageTone = MessageTone.PROFESSIONAL
    target_audience: Optional[str] = None
    context: Optional[str] = None
    keywords: List[str] = []
    max_length: Optional[int] = None

class GeneratedResponse(BaseModel):
    id: str
    original_message: str
    response: str
    message_type: MessageType
    tone: MessageTone
    created_at: datetime
    word_count: int
    character_count: int
    keywords_used: List[str] = []
    sentiment_score: Optional[float] = None
    readability_score: Optional[float] = None

class KeyMessageResponse(BaseModel):
    success: bool
    data: Optional[GeneratedResponse] = None
    error: Optional[str] = None
    processing_time: float
    suggestions: List[str] = [] 