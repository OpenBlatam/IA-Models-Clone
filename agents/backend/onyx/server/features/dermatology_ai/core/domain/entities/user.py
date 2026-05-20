"""
User domain entity.

Represents a user in the dermatology AI system with their
preferences, skin type, and profile information.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

from .enums import SkinType


@dataclass
class User:
    """
    User domain entity.
    
    Represents a user with their profile information, skin type,
    and preferences for personalized recommendations.
    """
    
    id: str
    email: str
    name: Optional[str] = None
    skin_type: Optional[SkinType] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def update_preferences(self, preferences: Dict[str, Any]) -> None:
        """
        Update user preferences.
        
        Merges new preferences with existing ones and updates
        the updated_at timestamp.
        
        Args:
            preferences: Dictionary of preference key-value pairs
        """
        self.preferences.update(preferences)
        self.updated_at = datetime.utcnow()










