from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

from uuid import UUID, uuid4
from pydantic import BaseModel, Field, validator, field_validator, ConfigDict, model_validator
from typing import Dict, Optional, Any
from datetime import datetime
import logging
import structlog
import orjson
from transformers import PreTrainedTokenizerBase
import pandas as pd
import numpy as np

from onyx.db.models import InputPrompt
from onyx.utils.logger import setup_logger
from onyx.core.models import OnyxBaseModel
from uuid6 import uuid7
from agents.backend.onyx.server.features.utils.ml_data_pipeline import send_training_example_kafka

from typing import Any, List, Dict, Optional
import asyncio
logger = structlog.get_logger()


class CreateInputPromptRequest(BaseModel):
    prompt: str
    content: str
    is_public: bool


class UpdateInputPromptRequest(BaseModel):
    prompt: str
    content: str
    active: bool


class InputPromptResponse(BaseModel):
    id: int
    prompt: str
    content: str
    active: bool


class InputPromptSnapshot(BaseModel):
    id: int
    prompt: str
    content: str
    active: bool
    user_id: UUID | None
    is_public: bool

    @classmethod
    def from_model(cls, input_prompt: InputPrompt) -> "InputPromptSnapshot":
        return InputPromptSnapshot(
            id=input_prompt.id,
            prompt=input_prompt.prompt,
            content=input_prompt.content,
            active=input_prompt.active,
            user_id=input_prompt.user_id,
            is_public=input_prompt.is_public,
        )


class ORJSONModel(OnyxBaseModel):
    model_config = ConfigDict(json_loads=orjson.loads, json_dumps=orjson.dumps)


class InputPrompt(ORJSONModel):
    __slots__ = (
        'id', 'prompt', 'metadata', 'created_at', 'updated_at', 'created_by', 'updated_by',
        'source', 'version', 'trace_id', 'is_deleted'
    )
    id: UUID = Field(default_factory=uuid7)
    prompt: str = Field(..., min_length=1, description="Texto del prompt")
    metadata: dict = Field(default_factory=dict, description="Metadatos adicionales")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str | None = None
    updated_by: str | None = None
    source: str | None = None
    version: int = 1
    trace_id: str | None = None
    is_deleted: bool = False

    @field_validator('prompt')
    def prompt_not_empty(cls, v) -> Any:
        if not v or not v.strip():
            logger.error("InputPrompt prompt validation failed", value=v)
            raise ValueError("Prompt must not be empty")
        return v

    @field_validator('metadata', mode="before")
    @classmethod
    def dict_or_empty(cls, v) -> Any:
        return v or {}

    @model_validator(mode="after")
    def check_prompt_and_metadata(self) -> Any:
        if self.prompt and self.metadata and "lang" in self.metadata and not self.metadata["lang"]:
            logger.warning("Prompt metadata 'lang' should not be empty", prompt=self.prompt)
        if self.created_at > self.updated_at:
            logger.warning("created_at is after updated_at", id=str(self.id))
        return self

    def audit_log(self) -> Any:
        return {
            "id": str(self.id),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "updated_by": self.updated_by,
            "source": self.source,
            "version": self.version,
            "trace_id": self.trace_id,
            "is_deleted": self.is_deleted,
        }

    def update(self, **kwargs) -> Any:
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.updated_at = datetime.utcnow()
        self.version += 1
        logger.info("InputPrompt updated", id=str(self.id), version=self.version, trace_id=self.trace_id)

    def soft_delete(self) -> Any:
        self.is_deleted = True
        self.update()
        logger.info("InputPrompt soft deleted", id=str(self.id), trace_id=self.trace_id)

    def restore(self) -> Any:
        self.is_deleted = False
        self.update()
        logger.info("InputPrompt restored", id=str(self.id), trace_id=self.trace_id)

    def to_dict(self) -> Any:
        return self.model_dump()

    def to_json(self) -> Any:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str):
        
    """from_json function."""
return cls.model_validate_json(data)

    def to_training_example(self) -> Any:
        return {
            "input": self.prompt,
            "metadata": self.metadata,
        }

    @classmethod
    def from_training_example(cls, example: dict):
        
    """from_training_example function."""
return cls(prompt=example["input"], metadata=example.get("metadata", {}))

    def tokenize(self, tokenizer: 'PreTrainedTokenizerBase', max_length: int = 128):
        """Tokeniza el prompt usando un tokenizer de HuggingFace."""
        return tokenizer(
            self.prompt,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

    def to_pandas(self) -> Any:
        """Convierte el modelo a un DataFrame de pandas (una sola fila)."""
        return pd.DataFrame([{**self.to_dict()}])

    def to_numpy(self) -> Any:
        """Convierte el modelo a un array de numpy (solo los valores principales)."""
        return np.array([self.prompt])

    def __post_init_post_parse__(self) -> Any:
        logger.info("InputPrompt instantiated", id=str(self.id), prompt=self.prompt)

    def send_to_kafka(self, topic="ml_training_examples", bootstrap_servers=None) -> Any:
        """
        Envía este ejemplo a un topic de Kafka para el pipeline ML/LLM automatizado.
        """
        send_training_example_kafka(self, topic=topic, bootstrap_servers=bootstrap_servers)

    @dataclass
class Config:
        frozen = True
        validate_assignment = True
