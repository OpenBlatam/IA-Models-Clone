from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from dataclasses import dataclass
from typing import Optional
from uuid import UUID
import re
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Author Value Object
==================

Value object for post author with validation and business rules.
"""



@dataclass(frozen=True)
class Author:
    """
    Author value object with validation and business rules.
    
    This value object encapsulates author information and ensures
    data integrity.
    """
    
    id: UUID
    name: str
    email: Optional[str] = None
    company: Optional[str] = None
    title: Optional[str] = None
    linkedin_profile: Optional[str] = None
    
    def __post_init__(self) -> Any:
        """Validate author data after initialization."""
        self._validate_author()
    
    def _validate_author(self) -> None:
        """Validate author data according to business rules."""
        if not self.name or not self.name.strip():
            raise ValueError("Author name cannot be empty")
        
        if len(self.name.strip()) < 2:
            raise ValueError("Author name must be at least 2 characters long")
        
        if len(self.name) > 100:
            raise ValueError("Author name cannot exceed 100 characters")
        
        if self.email and not self._is_valid_email(self.email):
            raise ValueError("Invalid email format")
        
        if self.linkedin_profile and not self._is_valid_linkedin_profile(self.linkedin_profile):
            raise ValueError("Invalid LinkedIn profile URL")
    
    def _is_valid_email(self, email: str) -> bool:
        """Validate email format."""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(email_pattern, email))
    
    def _is_valid_linkedin_profile(self, profile_url: str) -> bool:
        """Validate LinkedIn profile URL."""
        linkedin_pattern = r'^https?://(?:www\.)?linkedin\.com/in/[a-zA-Z0-9-]+/?$'
        return bool(re.match(linkedin_pattern, profile_url))
    
    def get_display_name(self) -> str:
        """Get display name for the author."""
        if self.title and self.company:
            return f"{self.name} - {self.title} at {self.company}"
        elif self.title:
            return f"{self.name} - {self.title}"
        elif self.company:
            return f"{self.name} - {self.company}"
        else:
            return self.name
    
    def get_linkedin_url(self) -> Optional[str]:
        """Get LinkedIn profile URL."""
        return self.linkedin_profile
    
    def has_complete_profile(self) -> bool:
        """Check if author has a complete profile."""
        return all([
            self.name,
            self.email,
            self.company,
            self.title,
            self.linkedin_profile
        ])
    
    def get_profile_completeness(self) -> float:
        """Calculate profile completeness percentage."""
        fields = [
            self.name,
            self.email,
            self.company,
            self.title,
            self.linkedin_profile
        ]
        
        completed_fields = sum(1 for field in fields if field)
        return (completed_fields / len(fields)) * 100
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "email": self.email,
            "company": self.company,
            "title": self.title,
            "linkedin_profile": self.linkedin_profile,
            "display_name": self.get_display_name(),
            "profile_completeness": self.get_profile_completeness()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Author':
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]),
            name=data["name"],
            email=data.get("email"),
            company=data.get("company"),
            title=data.get("title"),
            linkedin_profile=data.get("linkedin_profile")
        )
    
    def __str__(self) -> str:
        return self.get_display_name()
    
    def __repr__(self) -> str:
        return f"Author(id={self.id}, name='{self.name}')" 