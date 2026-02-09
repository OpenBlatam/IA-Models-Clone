from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import structlog
from pydantic import Field, field_validator, ConfigDict, BaseModel
from typing import Optional, Dict
from uuid import UUID
import orjson
from uuid6 import uuid7

from typing import Any, List, Dict, Optional
import logging
import asyncio
logger = structlog.get_logger()

class ORJSONModel(BaseModel):
    model_config = ConfigDict(json_loads=orjson.loads, json_dumps=orjson.dumps)

class ToolCreate(ORJSONModel):
    """Schema for creating a Tool (input)."""
    name: str = Field(..., min_length=2, max_length=128)
    config: dict | None = Field(default_factory=dict)

    @field_validator('name')
    def name_not_empty(cls, v) -> Any:
        if not v or not v.strip():
            logger.error("ToolCreate name validation failed", value=v)
            raise ValueError("Name must not be empty")
        return v

    @field_validator('config')
    def config_is_dict(cls, v) -> Any:
        if not isinstance(v, dict):
            logger.error("ToolCreate config validation failed", value=v)
            raise ValueError("Config must be a dict")
        return v

    def __post_init_post_parse__(self) -> Any:
        logger.info("ToolCreate instantiated", name=self.name)

class ToolRead(ORJSONModel):
    """Schema for reading a Tool (output)."""
    id: UUID
    name: str
    config: dict | None

    def __post_init_post_parse__(self) -> Any:
        logger.info("ToolRead instantiated", id=str(self.id), name=self.name) 