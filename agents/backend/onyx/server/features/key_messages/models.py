from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Key Messages models for Onyx.
"""

class MessageType(str, Enum):
    """Types of messages that can be generated."""
    MARKETING = "marketing"
    EDUCATIONAL = "educational"
    PROMOTIONAL = "promotional"
    INFORMATIONAL = "informational"
    CALL_TO_ACTION = "call_to_action"
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    WEBSITE = "website"

class MessageTone(str, Enum):
    """Tones for message generation."""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"
    CONVERSATIONAL = "conversational"
    ENTHUSIASTIC = "enthusiastic"
    URGENT = "urgent"
    CALM = "calm"

class KeyMessageRequest(BaseModel):
    """Request model for key message generation."""
    message: str = Field(..., min_length=1, max_length=10000, description="The original message to process")
    message_type: MessageType = Field(MessageType.INFORMATIONAL, description="Type of message to generate")
    tone: MessageTone = Field(MessageTone.PROFESSIONAL, description="Tone for the generated message")
    target_audience: Optional[str] = Field(None, description="Target audience description")
    context: Optional[str] = Field(None, description="Additional context for message generation")
    keywords: List[str] = Field(default_factory=list, description="Keywords to include in the message")
    max_length: Optional[int] = Field(None, description="Maximum length for the generated message")
    brand_voice: Optional[Dict[str, Any]] = Field(None, description="Brand voice settings")
    industry: Optional[str] = Field(None, description="Industry context")
    call_to_action: Optional[str] = Field(None, description="Specific call to action")

    @validator('message')
    def validate_message(cls, v) -> bool:
        """Validate message with guard clauses."""
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        
        if len(v) > 10000:
            raise ValueError("Message too long (max 10000 characters)")
        
        if len(v) < 1:
            raise ValueError("Message too short (min 1 character)")
        
        return v.strip()

    @validator('target_audience')
    def validate_target_audience(cls, v) -> Optional[Dict[str, Any]]:
        """Validate target audience."""
        if v is not None and len(v) > 500:
            raise ValueError("Target audience description too long (max 500 characters)")
        return v

    @validator('context')
    def validate_context(cls, v) -> bool:
        """Validate context."""
        if v is not None and len(v) > 2000:
            raise ValueError("Context too long (max 2000 characters)")
        return v

    @validator('keywords')
    def validate_keywords(cls, v) -> bool:
        """Validate keywords."""
        if len(v) > 20:
            raise ValueError("Too many keywords (max 20)")
        
        for keyword in v:
            if not keyword or not keyword.strip():
                raise ValueError("Keywords cannot be empty")
            
            if len(keyword) > 50:
                raise ValueError("Keyword too long (max 50 characters)")
        
        return [kw.strip() for kw in v]

    @validator('max_length')
    def validate_max_length(cls, v) -> bool:
        """Validate max length."""
        if v is not None:
            if v <= 0:
                raise ValueError("Max length must be positive")
            
            if v > 5000:
                raise ValueError("Max length too large (max 5000)")
        
        return v

    @validator('brand_voice')
    def validate_brand_voice(cls, v) -> bool:
        """Validate brand voice settings."""
        if v is not None and not isinstance(v, dict):
            raise ValueError("Brand voice must be a dictionary")
        
        if v is not None and len(v) > 20:
            raise ValueError("Too many brand voice settings (max 20)")
        
        return v

    @validator('industry')
    def validate_industry(cls, v) -> bool:
        """Validate industry."""
        if v is not None and len(v) > 100:
            raise ValueError("Industry description too long (max 100 characters)")
        return v

    @validator('call_to_action')
    def validate_call_to_action(cls, v) -> bool:
        """Validate call to action."""
        if v is not None and len(v) > 200:
            raise ValueError("Call to action too long (max 200 characters)")
        return v

class GeneratedResponse(BaseModel):
    """Response model for generated content."""
    id: str = Field(..., description="Unique identifier for the response")
    original_message: str = Field(..., description="Original input message")
    response: str = Field(..., description="Generated response")
    message_type: MessageType = Field(..., description="Type of generated message")
    tone: MessageTone = Field(..., description="Tone used in generation")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    word_count: int = Field(..., description="Number of words in response")
    character_count: int = Field(..., description="Number of characters in response")
    keywords_used: List[str] = Field(default_factory=list, description="Keywords successfully used")
    sentiment_score: Optional[float] = Field(None, description="Sentiment analysis score")
    readability_score: Optional[float] = Field(None, description="Readability score")
    processing_time: float = Field(..., description="Time taken to generate response")
    suggestions: List[str] = Field(default_factory=list, description="Additional suggestions")

    @validator('id')
    def validate_id(cls, v) -> bool:
        """Validate response ID."""
        if not v or not v.strip():
            raise ValueError("Response ID cannot be empty")
        return v

    @validator('response')
    def validate_response(cls, v) -> bool:
        """Validate generated response."""
        if not v or not v.strip():
            raise ValueError("Generated response cannot be empty")
        
        if len(v) > 10000:
            raise ValueError("Generated response too long (max 10000 characters)")
        
        return v

    @validator('word_count')
    def validate_word_count(cls, v) -> bool:
        """Validate word count."""
        if v < 0:
            raise ValueError("Word count cannot be negative")
        return v

    @validator('character_count')
    def validate_character_count(cls, v) -> bool:
        """Validate character count."""
        if v < 0:
            raise ValueError("Character count cannot be negative")
        return v

    @validator('processing_time')
    def validate_processing_time(cls, v) -> bool:
        """Validate processing time."""
        if v < 0:
            raise ValueError("Processing time cannot be negative")
        
        if v > 300:  # 5 minutes max
            raise ValueError("Processing time too high (max 300 seconds)")
        
        return v

    @validator('sentiment_score')
    def validate_sentiment_score(cls, v) -> bool:
        """Validate sentiment score."""
        if v is not None and (v < -1 or v > 1):
            raise ValueError("Sentiment score must be between -1 and 1")
        return v

    @validator('readability_score')
    def validate_readability_score(cls, v) -> bool:
        """Validate readability score."""
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Readability score must be between 0 and 1")
        return v

class KeyMessageResponse(BaseModel):
    """Main response model for key message operations."""
    success: bool = Field(..., description="Operation success status")
    data: Optional[GeneratedResponse] = Field(None, description="Generated response data")
    error: Optional[str] = Field(None, description="Error message if any")
    processing_time: float = Field(..., description="Total processing time")
    suggestions: List[str] = Field(default_factory=list, description="Additional suggestions")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    @validator('processing_time')
    def validate_processing_time(cls, v) -> bool:
        """Validate processing time."""
        if v < 0:
            raise ValueError("Processing time cannot be negative")
        return v

    @validator('error')
    def validate_error(cls, v) -> bool:
        """Validate error message."""
        if v is not None and len(v) > 1000:
            raise ValueError("Error message too long (max 1000 characters)")
        return v

class MessageAnalysis(BaseModel):
    """Analysis results for a message."""
    sentiment: str = Field(..., description="Sentiment analysis result")
    tone_consistency: float = Field(..., description="Tone consistency score")
    clarity_score: float = Field(..., description="Message clarity score")
    engagement_potential: float = Field(..., description="Engagement potential score")
    keyword_optimization: float = Field(..., description="Keyword optimization score")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")

    @validator('sentiment')
    def validate_sentiment(cls, v) -> bool:
        """Validate sentiment."""
        valid_sentiments = ['positive', 'negative', 'neutral', 'mixed']
        if v not in valid_sentiments:
            raise ValueError(f"Invalid sentiment. Must be one of: {valid_sentiments}")
        return v

    @validator('tone_consistency', 'clarity_score', 'engagement_potential', 'keyword_optimization')
    def validate_scores(cls, v) -> bool:
        """Validate score values."""
        if v < 0 or v > 1:
            raise ValueError("Score must be between 0 and 1")
        return v

    @validator('suggestions')
    def validate_suggestions(cls, v) -> bool:
        """Validate suggestions."""
        if len(v) > 10:
            raise ValueError("Too many suggestions (max 10)")
        
        for suggestion in v:
            if not suggestion or not suggestion.strip():
                raise ValueError("Suggestions cannot be empty")
            
            if len(suggestion) > 200:
                raise ValueError("Suggestion too long (max 200 characters)")
        
        return [s.strip() for s in v]

class BatchKeyMessageRequest(BaseModel):
    """Request model for batch key message generation."""
    messages: List[KeyMessageRequest] = Field(..., description="List of messages to process")
    batch_size: Optional[int] = Field(10, description="Maximum batch size for processing")

    @validator('messages')
    def validate_messages(cls, v) -> bool:
        """Validate messages list."""
        if not v:
            raise ValueError("Messages list cannot be empty")
        
        if len(v) > 100:
            raise ValueError("Too many messages (max 100)")
        
        return v

    @validator('batch_size')
    def validate_batch_size(cls, v) -> bool:
        """Validate batch size."""
        if v is not None:
            if v <= 0:
                raise ValueError("Batch size must be positive")
            
            if v > 50:
                raise ValueError("Batch size too large (max 50)")
        
        return v

class BatchKeyMessageResponse(BaseModel):
    """Response model for batch operations."""
    success: bool = Field(..., description="Overall operation success")
    results: List[KeyMessageResponse] = Field(..., description="Individual results")
    total_processed: int = Field(..., description="Total messages processed")
    failed_count: int = Field(..., description="Number of failed operations")
    processing_time: float = Field(..., description="Total processing time")

    @validator('total_processed')
    def validate_total_processed(cls, v) -> bool:
        """Validate total processed count."""
        if v < 0:
            raise ValueError("Total processed cannot be negative")
        return v

    @validator('failed_count')
    def validate_failed_count(cls, v) -> bool:
        """Validate failed count."""
        if v < 0:
            raise ValueError("Failed count cannot be negative")
        return v

    @validator('processing_time')
    def validate_processing_time(cls, v) -> bool:
        """Validate processing time."""
        if v < 0:
            raise ValueError("Processing time cannot be negative")
        return v 