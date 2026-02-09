from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

from .model import BrandKit
from typing import Dict, List, Optional
from uuid import UUID
from ...utils.base_model import OnyxBaseModel
from ...utils.model_decorators import validate_model, cache_model, log_operations
from pydantic import Field, validator
from datetime import datetime
import logging

from typing import Any, List, Dict, Optional
import asyncio
# In-memory store for demonstration
_brand_kit_store: Dict[UUID, BrandKit] = {}

def save_brand_kit(brand_kit: BrandKit) -> None:
    """Save a BrandKit to the store."""
    _brand_kit_store[brand_kit.id] = brand_kit

def get_brand_kit(brand_kit_id: UUID) -> Optional[BrandKit]:
    """Retrieve a BrandKit by its UUID."""
    return _brand_kit_store.get(brand_kit_id)

def delete_brand_kit(brand_kit_id: UUID) -> None:
    """Delete a BrandKit by its UUID."""
    _brand_kit_store.pop(brand_kit_id, None)

def list_brand_kits() -> List[BrandKit]:
    """List all BrandKits in the store."""
    return list(_brand_kit_store.values())

class BrandKit(OnyxBaseModel):
    """
    Brand kit model for Onyx: validated, extensible, and compatible with OnyxBaseModel.
    """
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=2, max_length=128, description="Brand name")
    description: Optional[str] = Field(None, max_length=512)
    colors: List[Dict[str, Any]] = Field(default_factory=list)
    typography: List[Dict[str, Any]] = Field(default_factory=list)
    voice: List[Dict[str, Any]] = Field(default_factory=list)
    values: List[str] = Field(default_factory=list)
    target_audience: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    locale: Optional[str] = Field(None, description="Locale for i18n")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("name")
    def name_not_empty(cls, v) -> Any:
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()

    @validator("colors", "typography", "voice", "values", pre=True)
    def list_or_empty(cls, v) -> List[Any]:
        return v or []

    @validator("target_audience", "metadata", pre=True)
    def dict_or_empty(cls, v) -> Any:
        return v or {}

    @dataclass
class Config:
        orm_mode = True
        validate_assignment = True
        extra = "forbid"

    @validate_model(validate_types=True, validate_custom=True)
    @cache_model(key_field="id")
    @log_operations(logging.getLogger(__name__))
    def save(self, user_context: Optional[Dict[str, Any]] = None) -> None:
        super().save(user_context=user_context)

    @validate_model(validate_types=True, validate_custom=True)
    @cache_model(key_field="id")
    @log_operations(logging.getLogger(__name__))
    def delete(self, user_context: Optional[Dict[str, Any]] = None) -> None:
        super().delete(user_context=user_context)

    @validate_model(validate_types=True, validate_custom=True)
    @cache_model(key_field="id")
    @log_operations(logging.getLogger(__name__))
    def restore(self, user_context: Optional[Dict[str, Any]] = None) -> None:
        super().restore(user_context=user_context) 