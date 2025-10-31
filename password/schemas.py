from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import structlog
from pydantic import Field, field_validator, ConfigDict, BaseModel
from uuid6 import uuid7, UUID
import orjson

from typing import Any, List, Dict, Optional
import logging
import asyncio
logger = structlog.get_logger()

class ORJSONModel(BaseModel):
    model_config = ConfigDict(json_loads=orjson.loads, json_dumps=orjson.dumps)

class PasswordCreate(ORJSONModel):
    """Schema for creating a Password (input)."""
    value: str = Field(..., min_length=8, max_length=128)
    description: str | None = None

    @field_validator('value')
    def password_strong(cls, v) -> Any:
        if not v or not v.strip():
            logger.error("PasswordCreate value validation failed", value=v)
            raise ValueError("Password must not be empty")
        if len(v) < 8:
            logger.error("PasswordCreate value validation failed: too short", value=v)
            raise ValueError("Password must be at least 8 characters")
        return v

    def __post_init_post_parse__(self) -> Any:
        logger.info("PasswordCreate instantiated", description=self.description)

class PasswordRead(ORJSONModel):
    """Schema for reading a Password (output)."""
    id: UUID
    description: str | None

    def __post_init_post_parse__(self) -> Any:
        logger.info("PasswordRead instantiated", id=str(self.id), description=self.description) 