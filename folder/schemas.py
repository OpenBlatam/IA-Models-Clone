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

class FolderCreate(ORJSONModel):
    """Schema for creating a Folder (input)."""
    name: str = Field(..., min_length=2, max_length=128)
    parent_id: UUID | None = None

    @field_validator('name')
    def name_not_empty(cls, v) -> Any:
        if not v or not v.strip():
            logger.error("FolderCreate name validation failed", value=v)
            raise ValueError("Name must not be empty")
        return v

    def __post_init_post_parse__(self) -> Any:
        logger.info("FolderCreate instantiated", name=self.name)

class FolderRead(ORJSONModel):
    """Schema for reading a Folder (output)."""
    id: UUID
    name: str
    parent_id: UUID | None

    def __post_init_post_parse__(self) -> Any:
        logger.info("FolderRead instantiated", id=str(self.id), name=self.name) 