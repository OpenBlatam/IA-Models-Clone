"""
Wardrobe Models
===============

Modelos de datos para vestimenta.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from .base import BaseModel, TimestampMixin


class DressCode(str, Enum):
    """Códigos de vestimenta."""
    FORMAL = "formal"
    BUSINESS_CASUAL = "business_casual"
    CASUAL = "casual"
    SMART_CASUAL = "smart_casual"
    BLACK_TIE = "black_tie"
    COCKTAIL = "cocktail"
    STREETWEAR = "streetwear"
    ARTISTIC = "artistic"
    CUSTOM = "custom"


class Season(str, Enum):
    """Estaciones."""
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"
    ALL_SEASON = "all_season"


@dataclass
class WardrobeItemModel(BaseModel, TimestampMixin):
    """Modelo de item de vestimenta."""
    name: str
    category: str
    color: str
    brand: Optional[str] = None
    size: Optional[str] = None
    season: Season = Season.ALL_SEASON
    dress_codes: List[DressCode] = field(default_factory=list)
    notes: Optional[str] = None
    image_url: Optional[str] = None
    last_worn: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        base_dict = super().to_dict()
        timestamp_dict = TimestampMixin.to_dict(self)
        
        return {
            **base_dict,
            **timestamp_dict,
            "name": self.name,
            "category": self.category,
            "color": self.color,
            "brand": self.brand,
            "size": self.size,
            "season": self.season.value,
            "dress_codes": [dc.value for dc in self.dress_codes],
            "notes": self.notes,
            "image_url": self.image_url,
            "last_worn": self.last_worn.isoformat() if self.last_worn else None
        }


@dataclass
class OutfitModel(BaseModel, TimestampMixin):
    """Modelo de outfit."""
    name: str
    items: List[str]  # IDs de items
    dress_code: DressCode
    occasion: str
    season: Season = Season.ALL_SEASON
    notes: Optional[str] = None
    image_url: Optional[str] = None
    last_worn: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        base_dict = super().to_dict()
        timestamp_dict = TimestampMixin.to_dict(self)
        
        return {
            **base_dict,
            **timestamp_dict,
            "name": self.name,
            "items": self.items,
            "dress_code": self.dress_code.value,
            "occasion": self.occasion,
            "season": self.season.value,
            "notes": self.notes,
            "image_url": self.image_url,
            "last_worn": self.last_worn.isoformat() if self.last_worn else None
        }




