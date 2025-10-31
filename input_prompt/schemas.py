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

class InputPromptCreate(ORJSONModel):
    """Schema for creating an InputPrompt (input)."""
    prompt: str = Field(..., min_length=1)
    metadata: dict | None = Field(default_factory=dict)

    @field_validator('prompt')
    def prompt_not_empty(cls, v) -> Any:
        if not v or not v.strip():
            logger.error("InputPromptCreate prompt validation failed", value=v)
            raise ValueError("Prompt must not be empty")
        return v

    def __post_init_post_parse__(self) -> Any:
        logger.info("InputPromptCreate instantiated", prompt=self.prompt)

class InputPromptRead(ORJSONModel):
    """Schema for reading an InputPrompt (output)."""
    id: UUID
    prompt: str
    metadata: dict | None

    def __post_init_post_parse__(self) -> Any:
        logger.info("InputPromptRead instantiated", id=str(self.id), prompt=self.prompt) 