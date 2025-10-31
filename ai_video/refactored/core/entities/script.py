from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID
from pydantic import Field, field_validator
from .base import AggregateRoot
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Script Entity
============

Script entity representing generated scripts with timing and synchronization data.
"""





class ScriptTone(str, Enum):
    """Script tone options."""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    ENERGETIC = "energetic"
    CALM = "calm"
    AUTHORITATIVE = "authoritative"
    HUMOROUS = "humorous"
    INSPIRATIONAL = "inspirational"


class ScriptStatus(str, Enum):
    """Script generation status."""
    DRAFT = "draft"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class Script(AggregateRoot):
    """
    Script entity for video content.
    
    Scripts represent the textual content that will be spoken by
    the AI avatar, including timing and synchronization data.
    """
    
    # Basic information
    title: str = Field(..., min_length=1, max_length=100, description="Script title")
    content: str = Field(..., min_length=10, description="Script content")
    
    # Generation parameters
    tone: ScriptTone = Field(default=ScriptTone.PROFESSIONAL, description="Script tone")
    language: str = Field(default="es", min_length=2, max_length=5, description="Script language")
    target_duration: int = Field(default=60, ge=10, le=300, description="Target duration in seconds")
    
    # User and ownership
    user_id: UUID = Field(..., description="Script owner ID")
    creator_id: UUID = Field(..., description="Script creator ID")
    
    # Status and processing
    status: ScriptStatus = Field(default=ScriptStatus.DRAFT, description="Script status")
    generation_parameters: Dict = Field(
        default={},
        description="Parameters used for script generation"
    )
    
    # Timing and synchronization
    word_count: Optional[int] = Field(None, ge=0, description="Total word count")
    estimated_duration: Optional[float] = Field(None, ge=0, description="Estimated duration in seconds")
    speaking_rate: float = Field(default=150, ge=50, le=300, description="Words per minute")
    
    # Segments and timing
    segments: List[Dict] = Field(
        default=[],
        description="Script segments with timing information"
    )
    pauses: List[Dict] = Field(
        default=[],
        description="Pause points in the script"
    )
    
    # Metadata
    keywords: List[str] = Field(default=[], description="Script keywords")
    tags: List[str] = Field(default=[], description="Script tags")
    is_public: bool = Field(default=False, description="Public script flag")
    
    # Usage statistics
    usage_count: int = Field(default=0, description="Number of times used")
    rating: float = Field(default=0.0, ge=0.0, le=5.0, description="Average rating")
    rating_count: int = Field(default=0, description="Number of ratings")
    
    # Customization
    custom_instructions: Optional[str] = Field(None, description="Custom generation instructions")
    include_pauses: bool = Field(default=True, description="Include natural pauses")
    emphasis_points: List[Dict] = Field(
        default=[],
        description="Points of emphasis in the script"
    )
    
    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate script title."""
        if not v.strip():
            raise ValueError("Script title cannot be empty")
        return v.strip()
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate script content."""
        if not v.strip():
            raise ValueError("Script content cannot be empty")
        return v.strip()
    
    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Validate language code."""
        valid_languages = ["es", "en", "fr", "de", "it", "pt", "ja", "ko", "zh"]
        if v not in valid_languages:
            raise ValueError(f"Unsupported language: {v}")
        return v
    
    def _validate_entity(self) -> None:
        """Validate script business rules."""
        if self.status == ScriptStatus.COMPLETED and not self.segments:
            raise ValueError("Completed scripts must have segments")
        
        if self.estimated_duration and self.estimated_duration > self.target_duration * 1.5:
            raise ValueError("Estimated duration exceeds target duration by more than 50%")
    
    def start_generation(self) -> None:
        """Start script generation."""
        self.status = ScriptStatus.GENERATING
        self.mark_as_dirty()
    
    def complete_generation(self, segments: List[Dict], word_count: int) -> None:
        """Complete script generation."""
        self.status = ScriptStatus.COMPLETED
        self.segments = segments
        self.word_count = word_count
        self.estimated_duration = self._calculate_duration()
        self.mark_as_dirty()
    
    def fail_generation(self, error_message: str) -> None:
        """Mark script generation as failed."""
        self.status = ScriptStatus.FAILED
        self.generation_parameters["error"] = error_message
        self.mark_as_dirty()
    
    def add_segment(self, text: str, start_time: float, end_time: float, emphasis: bool = False) -> None:
        """Add a segment to the script."""
        segment = {
            "text": text,
            "start_time": start_time,
            "end_time": end_time,
            "emphasis": emphasis,
            "duration": end_time - start_time,
        }
        self.segments.append(segment)
        self.mark_as_dirty()
    
    def add_pause(self, position: int, duration: float) -> None:
        """Add a pause to the script."""
        pause = {
            "position": position,
            "duration": duration,
            "type": "natural"
        }
        self.pauses.append(pause)
        self.mark_as_dirty()
    
    def add_emphasis_point(self, word: str, position: int, intensity: float = 1.0) -> None:
        """Add an emphasis point to the script."""
        emphasis = {
            "word": word,
            "position": position,
            "intensity": intensity,
        }
        self.emphasis_points.append(emphasis)
        self.mark_as_dirty()
    
    def increment_usage(self) -> None:
        """Increment usage count."""
        self.usage_count += 1
        self.mark_as_dirty()
    
    def add_rating(self, rating: float) -> None:
        """Add a rating to the script."""
        if not 0.0 <= rating <= 5.0:
            raise ValueError("Rating must be between 0.0 and 5.0")
        
        # Calculate new average rating
        total_rating = self.rating * self.rating_count + rating
        self.rating_count += 1
        self.rating = total_rating / self.rating_count
        
        self.mark_as_dirty()
    
    def _calculate_duration(self) -> float:
        """Calculate estimated duration based on segments."""
        if not self.segments:
            return 0.0
        
        total_duration = sum(segment["duration"] for segment in self.segments)
        
        # Add pause durations
        pause_duration = sum(pause["duration"] for pause in self.pauses)
        
        return total_duration + pause_duration
    
    def get_word_count(self) -> int:
        """Get total word count."""
        if self.word_count is not None:
            return self.word_count
        
        # Calculate from content
        return len(self.content.split())
    
    def get_reading_time(self) -> float:
        """Get estimated reading time based on speaking rate."""
        word_count = self.get_word_count()
        return (word_count / self.speaking_rate) * 60
    
    def optimize_for_duration(self, target_duration: float) -> None:
        """Optimize script for target duration."""
        current_duration = self.get_reading_time()
        
        if current_duration > target_duration:
            # Need to shorten
            target_words = int((target_duration / 60) * self.speaking_rate)
            words = self.content.split()
            
            if len(words) > target_words:
                # Truncate content
                self.content = " ".join(words[:target_words])
                self.word_count = target_words
                self.mark_as_dirty()
        
        elif current_duration < target_duration * 0.8:
            # Need to expand
            target_words = int((target_duration / 60) * self.speaking_rate)
            current_words = self.get_word_count()
            
            if current_words < target_words:
                # Add placeholder content
                additional_words = target_words - current_words
                placeholder = " " * additional_words
                self.content += placeholder
                self.word_count = target_words
                self.mark_as_dirty()
    
    def to_summary_dict(self) -> Dict:
        """Convert to summary dictionary for listings."""
        return {
            "id": str(self.id),
            "title": self.title,
            "tone": self.tone.value,
            "language": self.language,
            "word_count": self.get_word_count(),
            "estimated_duration": self.estimated_duration,
            "status": self.status.value,
            "usage_count": self.usage_count,
            "rating": self.rating,
        }
    
    def get_timing_data(self) -> Dict:
        """Get timing data for video synchronization."""
        return {
            "segments": self.segments,
            "pauses": self.pauses,
            "emphasis_points": self.emphasis_points,
            "total_duration": self.estimated_duration,
            "speaking_rate": self.speaking_rate,
        }
    
    def clone_for_user(self, user_id: UUID, title: str) -> "Script":
        """Create a clone of this script for a specific user."""
        return Script(
            title=title,
            content=self.content,
            tone=self.tone,
            language=self.language,
            target_duration=self.target_duration,
            user_id=user_id,
            creator_id=user_id,
            speaking_rate=self.speaking_rate,
            keywords=self.keywords.copy(),
            tags=self.tags.copy(),
            is_public=False,
            custom_instructions=self.custom_instructions,
            include_pauses=self.include_pauses,
        ) 