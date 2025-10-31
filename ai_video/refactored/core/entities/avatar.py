from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

from datetime import datetime
from enum import Enum
from typing import Dict, Optional
from uuid import UUID
from pydantic import Field, field_validator
from .base import AggregateRoot
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Avatar Entity
============

Avatar entity representing AI-generated avatars with voice and appearance configuration.
"""





class AvatarGender(str, Enum):
    """Avatar gender options."""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class AvatarStyle(str, Enum):
    """Avatar style options."""
    REALISTIC = "realistic"
    CARTOON = "cartoon"
    ANIME = "anime"
    BUSINESS = "business"
    CASUAL = "casual"
    CREATIVE = "creative"


class AvatarStatus(str, Enum):
    """Avatar status."""
    GENERATING = "generating"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"


class Avatar(AggregateRoot):
    """
    AI Avatar entity for video generation.
    
    Avatars represent AI-generated characters with voice synthesis
    and appearance customization capabilities.
    """
    
    # Basic information
    name: str = Field(..., min_length=1, max_length=50, description="Avatar name")
    description: Optional[str] = Field(None, max_length=200, description="Avatar description")
    
    # Appearance configuration
    gender: AvatarGender = Field(..., description="Avatar gender")
    style: AvatarStyle = Field(..., description="Avatar style")
    age_range: str = Field(
        default="25-35",
        description="Avatar age range",
        examples=["18-25", "25-35", "35-45", "45-60"]
    )
    ethnicity: Optional[str] = Field(
        None,
        description="Avatar ethnicity",
        examples=["caucasian", "asian", "hispanic", "african", "mixed"]
    )
    
    # Voice configuration
    voice_settings: Dict = Field(
        default={
            "language": "es",
            "accent": "neutral",
            "speed": 1.0,
            "pitch": 1.0,
            "voice_id": None
        },
        description="Voice synthesis settings"
    )
    
    # Media assets
    preview_image_url: Optional[str] = Field(None, description="Avatar preview image")
    preview_video_url: Optional[str] = Field(None, description="Avatar preview video")
    model_file_url: Optional[str] = Field(None, description="Avatar model file")
    
    # Status and metadata
    status: AvatarStatus = Field(default=AvatarStatus.GENERATING, description="Avatar status")
    creator_id: UUID = Field(..., description="Avatar creator ID")
    is_public: bool = Field(default=False, description="Public avatar flag")
    
    # Usage statistics
    usage_count: int = Field(default=0, description="Number of times used")
    total_duration: float = Field(default=0.0, description="Total video duration created")
    
    # Customization
    custom_appearance: Optional[Dict] = Field(
        None,
        description="Custom appearance settings"
    )
    outfit: Optional[str] = Field(
        None,
        description="Avatar outfit style",
        examples=["business", "casual", "formal", "creative"]
    )
    
    # Technical metadata
    model_version: str = Field(default="1.0", description="Avatar model version")
    generation_parameters: Dict = Field(
        default={},
        description="Parameters used for avatar generation"
    )
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate avatar name."""
        if not v.strip():
            raise ValueError("Avatar name cannot be empty")
        return v.strip()
    
    @field_validator("age_range")
    @classmethod
    def validate_age_range(cls, v: str) -> str:
        """Validate age range format."""
        valid_ranges = ["18-25", "25-35", "35-45", "45-60", "60+"]
        if v not in valid_ranges:
            raise ValueError(f"Invalid age range: {v}")
        return v
    
    @field_validator("voice_settings")
    @classmethod
    def validate_voice_settings(cls, v: Dict) -> Dict:
        """Validate voice settings."""
        required_keys = ["language", "accent", "speed", "pitch"]
        for key in required_keys:
            if key not in v:
                raise ValueError(f"Missing required voice setting: {key}")
        
        if not 0.5 <= v["speed"] <= 2.0:
            raise ValueError("Voice speed must be between 0.5 and 2.0")
        
        if not 0.5 <= v["pitch"] <= 2.0:
            raise ValueError("Voice pitch must be between 0.5 and 2.0")
        
        return v
    
    def _validate_entity(self) -> None:
        """Validate avatar business rules."""
        if self.status == AvatarStatus.ACTIVE and not self.model_file_url:
            raise ValueError("Active avatars must have a model file")
        
        if self.is_public and not self.preview_image_url:
            raise ValueError("Public avatars must have a preview image")
    
    def activate(self) -> None:
        """Activate the avatar."""
        if not self.model_file_url:
            raise ValueError("Cannot activate avatar without model file")
        
        self.status = AvatarStatus.ACTIVE
        self.mark_as_dirty()
    
    def deactivate(self) -> None:
        """Deactivate the avatar."""
        self.status = AvatarStatus.INACTIVE
        self.mark_as_dirty()
    
    def mark_error(self, error_message: str) -> None:
        """Mark avatar as having an error."""
        self.status = AvatarStatus.ERROR
        self.generation_parameters["error"] = error_message
        self.mark_as_dirty()
    
    def increment_usage(self, duration: float = 0.0) -> None:
        """Increment usage statistics."""
        self.usage_count += 1
        self.total_duration += duration
        self.mark_as_dirty()
    
    def update_voice_settings(self, settings: Dict) -> None:
        """Update voice settings."""
        # Validate new settings
        self.voice_settings.update(settings)
        self._validate_voice_settings(self.voice_settings)
        self.mark_as_dirty()
    
    def set_custom_appearance(self, appearance: Dict) -> None:
        """Set custom appearance settings."""
        self.custom_appearance = appearance
        self.mark_as_dirty()
    
    def is_available_for_user(self, user_id: UUID) -> bool:
        """Check if avatar is available for a specific user."""
        if self.status != AvatarStatus.ACTIVE:
            return False
        
        # Public avatars are available to everyone
        if self.is_public:
            return True
        
        # Private avatars only for creator
        return self.creator_id == user_id
    
    def get_voice_config(self) -> Dict:
        """Get voice configuration for synthesis."""
        return {
            "voice_id": self.voice_settings.get("voice_id"),
            "language": self.voice_settings["language"],
            "accent": self.voice_settings["accent"],
            "speed": self.voice_settings["speed"],
            "pitch": self.voice_settings["pitch"],
        }
    
    def to_summary_dict(self) -> Dict:
        """Convert to summary dictionary for listings."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "gender": self.gender.value,
            "style": self.style.value,
            "preview_image_url": self.preview_image_url,
            "is_public": self.is_public,
            "usage_count": self.usage_count,
            "voice_settings": self.voice_settings,
        }
    
    def clone_for_user(self, user_id: UUID, name: str) -> "Avatar":
        """Create a clone of this avatar for a specific user."""
        return Avatar(
            name=name,
            description=f"Clone of {self.name}",
            gender=self.gender,
            style=self.style,
            age_range=self.age_range,
            ethnicity=self.ethnicity,
            voice_settings=self.voice_settings.copy(),
            creator_id=user_id,
            is_public=False,
            custom_appearance=self.custom_appearance.copy() if self.custom_appearance else None,
            outfit=self.outfit,
        ) 