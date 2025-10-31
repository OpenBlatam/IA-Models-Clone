from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Literal
from dataclasses import dataclass, field
from datetime import datetime
from .base_types import (
from .model_field import ModelField, FieldConfig
from .model_schema import ModelSchema, SchemaConfig
from .base_model import OnyxBaseModel
from .cache_mixin import CacheMixin
from .validation_mixin import ValidationMixin
from .event_mixin import EventMixin
from .index_mixin import IndexMixin
from .permission_mixin import PermissionMixin
from .status_mixin import StatusMixin
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Brand Kit Model - Onyx Integration
Model for managing brand kit data with enhanced capabilities and Onyx integration.
"""
    CACHE_TTL, VALIDATION_TIMEOUT,
    ModelId, ModelKey, ModelValue,
    ValidationType, CacheType, EventType,
    StatusType, CategoryType, PermissionType
)

T = TypeVar('T')

class BrandKitColor(ModelField):
    """Field for brand kit color with Onyx integration."""
    
    def __init__(
        self,
        name: str,
        hex: str,
        category: Literal["primary", "secondary", "accent", "neutral"] = "primary",
        description: Optional[str] = None,
        required: bool = True,
        default: Optional[str] = None,
        validation: Optional[Dict[str, Any]] = None,
        cache: Optional[Dict[str, Any]] = None,
        events: Optional[Dict[str, Any]] = None,
        index: Optional[Dict[str, Any]] = None,
        permissions: Optional[Dict[str, Any]] = None,
        status: Optional[Dict[str, Any]] = None
    ):
        """Initialize brand kit color field."""
        super().__init__(
            name=name,
            type=str,
            description=description or f"Brand kit {category} color",
            required=required,
            default=default,
            validation=validation or {
                "type": ValidationType.COLOR,
                "format": "hex",
                "timeout": VALIDATION_TIMEOUT,
                "rules": {
                    "format": "^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$",
                    "message": "Color must be a valid hex code"
                }
            },
            cache=cache or {
                "type": CacheType.COLOR,
                "ttl": CACHE_TTL,
                "key_prefix": f"color_{category}"
            },
            events=events or {
                "type": EventType.COLOR,
                "on_change": True,
                "notify": True
            },
            index=index or {
                "type": "color",
                "searchable": True,
                "category": category
            },
            permissions=permissions or {
                "type": PermissionType.COLOR,
                "read": True,
                "write": True,
                "roles": ["admin", "designer"]
            },
            status=status or {
                "type": StatusType.COLOR,
                "active": True,
                "category": category
            }
        )
        self.hex = hex
        self.category = category

class BrandKitTypography(ModelField):
    """Field for brand kit typography with Onyx integration."""
    
    def __init__(
        self,
        name: str,
        font_family: str,
        style: str,
        category: Literal["heading", "body", "display", "mono"] = "body",
        weights: List[int] = field(default_factory=lambda: [400, 700]),
        description: Optional[str] = None,
        required: bool = True,
        default: Optional[str] = None,
        validation: Optional[Dict[str, Any]] = None,
        cache: Optional[Dict[str, Any]] = None,
        events: Optional[Dict[str, Any]] = None,
        index: Optional[Dict[str, Any]] = None,
        permissions: Optional[Dict[str, Any]] = None,
        status: Optional[Dict[str, Any]] = None
    ):
        """Initialize brand kit typography field."""
        super().__init__(
            name=name,
            type=str,
            description=description or f"Brand kit {category} typography",
            required=required,
            default=default,
            validation=validation or {
                "type": ValidationType.TYPOGRAPHY,
                "timeout": VALIDATION_TIMEOUT,
                "rules": {
                    "font_family": "required",
                    "weights": "required",
                    "message": "Typography must have font family and weights"
                }
            },
            cache=cache or {
                "type": CacheType.TYPOGRAPHY,
                "ttl": CACHE_TTL,
                "key_prefix": f"typography_{category}"
            },
            events=events or {
                "type": EventType.TYPOGRAPHY,
                "on_change": True,
                "notify": True
            },
            index=index or {
                "type": "typography",
                "searchable": True,
                "category": category
            },
            permissions=permissions or {
                "type": PermissionType.TYPOGRAPHY,
                "read": True,
                "write": True,
                "roles": ["admin", "designer"]
            },
            status=status or {
                "type": StatusType.TYPOGRAPHY,
                "active": True,
                "category": category
            }
        )
        self.font_family = font_family
        self.style = style
        self.category = category
        self.weights = weights

class BrandKitVoice(ModelField):
    """Field for brand kit voice with Onyx integration."""
    
    def __init__(
        self,
        name: str,
        tone: Literal["professional", "casual", "friendly", "authoritative", "humorous", "formal"] = "professional",
        style: Literal["conversational", "technical", "storytelling", "persuasive", "educational"] = "conversational",
        personality_traits: List[str] = field(default_factory=list),
        industry_terms: List[str] = field(default_factory=list),
        description: Optional[str] = None,
        required: bool = True,
        default: Optional[str] = None,
        validation: Optional[Dict[str, Any]] = None,
        cache: Optional[Dict[str, Any]] = None,
        events: Optional[Dict[str, Any]] = None,
        index: Optional[Dict[str, Any]] = None,
        permissions: Optional[Dict[str, Any]] = None,
        status: Optional[Dict[str, Any]] = None
    ):
        """Initialize brand kit voice field."""
        super().__init__(
            name=name,
            type=str,
            description=description or "Brand kit voice",
            required=required,
            default=default,
            validation=validation or {
                "type": ValidationType.VOICE,
                "timeout": VALIDATION_TIMEOUT,
                "rules": {
                    "tone": "required",
                    "style": "required",
                    "message": "Voice must have tone and style"
                }
            },
            cache=cache or {
                "type": CacheType.VOICE,
                "ttl": CACHE_TTL,
                "key_prefix": "voice"
            },
            events=events or {
                "type": EventType.VOICE,
                "on_change": True,
                "notify": True
            },
            index=index or {
                "type": "voice",
                "searchable": True,
                "categories": ["tone", "style", "personality"]
            },
            permissions=permissions or {
                "type": PermissionType.VOICE,
                "read": True,
                "write": True,
                "roles": ["admin", "content"]
            },
            status=status or {
                "type": StatusType.VOICE,
                "active": True
            }
        )
        self.tone = tone
        self.style = style
        self.personality_traits = personality_traits
        self.industry_terms = industry_terms

class BrandKitSchema(ModelSchema):
    """Schema for brand kit with Onyx integration."""
    
    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        fields: Optional[Dict[str, ModelField]] = None,
        validation: Optional[Dict[str, Any]] = None,
        cache: Optional[Dict[str, Any]] = None,
        events: Optional[Dict[str, Any]] = None,
        index: Optional[Dict[str, Any]] = None,
        permissions: Optional[Dict[str, Any]] = None,
        status: Optional[Dict[str, Any]] = None
    ):
        """Initialize brand kit schema."""
        super().__init__(
            name=name,
            description=description or "Brand kit schema",
            fields=fields or {},
            validation=validation or {
                "type": ValidationType.SCHEMA,
                "timeout": VALIDATION_TIMEOUT,
                "rules": {
                    "colors": "required",
                    "typography": "required",
                    "voice": "required",
                    "message": "Brand kit must have colors, typography, and voice"
                }
            },
            cache=cache or {
                "type": CacheType.SCHEMA,
                "ttl": CACHE_TTL,
                "key_prefix": "brand_kit"
            },
            events=events or {
                "type": EventType.SCHEMA,
                "on_change": True,
                "notify": True
            },
            index=index or {
                "type": "schema",
                "searchable": True,
                "categories": ["brand", "design", "content"]
            },
            permissions=permissions or {
                "type": PermissionType.SCHEMA,
                "read": True,
                "write": True,
                "roles": ["admin", "designer", "content"]
            },
            status=status or {
                "type": StatusType.SCHEMA,
                "active": True
            }
        )

class BrandKit(
    OnyxBaseModel,
    CacheMixin,
    ValidationMixin,
    EventMixin,
    IndexMixin,
    PermissionMixin,
    StatusMixin
):
    """Brand kit model with enhanced Onyx integration."""
    
    def __init__(
        self,
        id: Optional[str] = None,
        name: str = "",
        description: Optional[str] = None,
        colors: Optional[List[BrandKitColor]] = None,
        typography: Optional[Dict[str, BrandKitTypography]] = None,
        voice: Optional[BrandKitVoice] = None,
        logo: Optional[str] = None,
        mission: Optional[str] = None,
        vision: Optional[str] = None,
        values: Optional[List[str]] = None,
        target_audience: Optional[List[str]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        validation: Optional[Dict[str, Any]] = None,
        cache: Optional[Dict[str, Any]] = None,
        events: Optional[Dict[str, Any]] = None,
        index: Optional[Dict[str, Any]] = None,
        permissions: Optional[Dict[str, Any]] = None,
        status: Optional[Dict[str, Any]] = None
    ):
        """Initialize brand kit."""
        super().__init__(
            id=id,
            name=name,
            description=description,
            created_at=created_at,
            updated_at=updated_at,
            validation=validation or {
                "type": ValidationType.BRAND_KIT,
                "timeout": VALIDATION_TIMEOUT,
                "rules": {
                    "name": "required",
                    "colors": "required",
                    "typography": "required",
                    "voice": "required",
                    "message": "Brand kit must have name, colors, typography, and voice"
                }
            },
            cache=cache or {
                "type": CacheType.BRAND_KIT,
                "ttl": CACHE_TTL,
                "key_prefix": "brand_kit"
            },
            events=events or {
                "type": EventType.BRAND_KIT,
                "on_change": True,
                "notify": True
            },
            index=index or {
                "type": "brand_kit",
                "searchable": True,
                "categories": ["brand", "design", "content"]
            },
            permissions=permissions or {
                "type": PermissionType.BRAND_KIT,
                "read": True,
                "write": True,
                "roles": ["admin", "designer", "content"]
            },
            status=status or {
                "type": StatusType.BRAND_KIT,
                "active": True
            }
        )
        self.colors = colors or []
        self.typography = typography or {}
        self.voice = voice
        self.logo = logo
        self.mission = mission
        self.vision = vision
        self.values = values or []
        self.target_audience = target_audience or []
    
    def add_color(self, color: BrandKitColor) -> None:
        """Add a color to the brand kit."""
        self.colors.append(color)
        self._trigger_event("color_added", {
            "color": color.get_data(),
            "timestamp": datetime.utcnow().isoformat()
        })
        self._clear_cache("colors")
    
    def remove_color(self, color: BrandKitColor) -> None:
        """Remove a color from the brand kit."""
        if color in self.colors:
            self.colors.remove(color)
            self._trigger_event("color_removed", {
                "color": color.get_data(),
                "timestamp": datetime.utcnow().isoformat()
            })
            self._clear_cache("colors")
    
    def set_typography(self, name: str, typography: BrandKitTypography) -> None:
        """Set typography for a specific use case."""
        self.typography[name] = typography
        self._trigger_event("typography_updated", {
            "name": name,
            "typography": typography.get_data(),
            "timestamp": datetime.utcnow().isoformat()
        })
        self._clear_cache("typography")
    
    def get_typography(self, name: str) -> Optional[BrandKitTypography]:
        """Get typography for a specific use case."""
        cache_key = f"typography_{name}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached
        
        typography = self.typography.get(name)
        if typography:
            self._set_cache(cache_key, typography)
        return typography
    
    def set_voice(self, voice: BrandKitVoice) -> None:
        """Set the brand voice."""
        self.voice = voice
        self._trigger_event("voice_updated", {
            "voice": voice.get_data(),
            "timestamp": datetime.utcnow().isoformat()
        })
        self._clear_cache("voice")
    
    def add_value(self, value: str) -> None:
        """Add a brand value."""
        if value not in self.values:
            self.values.append(value)
            self._trigger_event("value_added", {
                "value": value,
                "timestamp": datetime.utcnow().isoformat()
            })
            self._clear_cache("values")
    
    def remove_value(self, value: str) -> None:
        """Remove a brand value."""
        if value in self.values:
            self.values.remove(value)
            self._trigger_event("value_removed", {
                "value": value,
                "timestamp": datetime.utcnow().isoformat()
            })
            self._clear_cache("values")
    
    def add_target_audience(self, audience: str) -> None:
        """Add a target audience segment."""
        if audience not in self.target_audience:
            self.target_audience.append(audience)
            self._trigger_event("audience_added", {
                "audience": audience,
                "timestamp": datetime.utcnow().isoformat()
            })
            self._clear_cache("audience")
    
    def remove_target_audience(self, audience: str) -> None:
        """Remove a target audience segment."""
        if audience in self.target_audience:
            self.target_audience.remove(audience)
            self._trigger_event("audience_removed", {
                "audience": audience,
                "timestamp": datetime.utcnow().isoformat()
            })
            self._clear_cache("audience")
    
    def get_data(self) -> Dict[str, Any]:
        """Get brand kit data."""
        cache_key = f"brand_kit_data_{self.id}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached
        
        data = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "colors": [color.get_data() for color in self.colors],
            "typography": {name: typo.get_data() for name, typo in self.typography.items()},
            "voice": self.voice.get_data() if self.voice else None,
            "logo": self.logo,
            "mission": self.mission,
            "vision": self.vision,
            "values": self.values,
            "target_audience": self.target_audience,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        self._set_cache(cache_key, data)
        return data
    
    @classmethod
    def from_data(cls, data: Dict[str, Any]) -> BrandKit:
        """Create brand kit from data."""
        colors = [BrandKitColor(**color) for color in data.get("colors", [])]
        typography = {
            name: BrandKitTypography(**typo)
            for name, typo in data.get("typography", {}).items()
        }
        voice = BrandKitVoice(**data["voice"]) if data.get("voice") else None
        
        return cls(
            id=data.get("id"),
            name=data.get("name", ""),
            description=data.get("description"),
            colors=colors,
            typography=typography,
            voice=voice,
            logo=data.get("logo"),
            mission=data.get("mission"),
            vision=data.get("vision"),
            values=data.get("values", []),
            target_audience=data.get("target_audience", []),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None
        ) 