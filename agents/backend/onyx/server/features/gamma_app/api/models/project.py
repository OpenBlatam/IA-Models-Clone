"""
Project Models
Project-related Pydantic models
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from .validators import sanitize_optional_string_field

class Project(BaseModel):
    """Project model"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    owner_id: str
    collaborators: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    settings: Dict[str, Any] = Field(default_factory=dict)
    is_public: bool = False

class ProjectCreate(BaseModel):
    """Project creation model"""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    is_public: bool = False
    settings: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('name', 'description')
    @classmethod
    def sanitize_string_fields(cls, v: Optional[str]) -> Optional[str]:
        """Sanitize string fields"""
        return sanitize_optional_string_field(v)

class ProjectUpdate(BaseModel):
    """Project update model"""
    name: Optional[str] = None
    description: Optional[str] = None
    is_public: Optional[bool] = None
    settings: Optional[Dict[str, Any]] = None







