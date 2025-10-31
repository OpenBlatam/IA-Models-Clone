from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import structlog
from pydantic import Field, field_validator, ConfigDict, BaseModel
from typing import Optional, List, Dict
from uuid import UUID
import orjson
from uuid6 import uuid7

from typing import Any, List, Dict, Optional
import logging
import asyncio
logger = structlog.get_logger()

class ORJSONModel(BaseModel):
    model_config = ConfigDict(json_loads=orjson.loads, json_dumps=orjson.dumps)

class DocumentSetCreate(ORJSONModel):
    """Schema for creating a DocumentSet (input)."""
    name: str = Field(..., min_length=2, max_length=128)
    documents: List[str] = Field(default_factory=list)
    metadata: Dict = Field(default_factory=dict)

    @field_validator('name')
    def name_not_empty(cls, v) -> Any:
        if not v or not v.strip():
            logger.error("DocumentSetCreate name validation failed", value=v)
            raise ValueError("Name must not be empty")
        return v

    def __post_init_post_parse__(self) -> Any:
        logger.info("DocumentSetCreate instantiated", name=self.name)

class DocumentSetRead(ORJSONModel):
    """Schema for reading a DocumentSet (output)."""
    id: UUID
    name: str
    documents: List[str]
    metadata: Dict

    def __post_init_post_parse__(self) -> Any:
        logger.info("DocumentSetRead instantiated", id=str(self.id), name=self.name) 