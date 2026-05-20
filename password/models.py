from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

from pydantic import BaseModel, Field, validator
from typing import Optional
from ...utils.base_model import OnyxBaseModel
from pydantic import field_validator, ConfigDict, model_validator
from uuid import UUID, uuid4
from datetime import datetime
import logging
import structlog
import orjson
from uuid6 import uuid7
from agents.backend.onyx.server.features.utils.ml_data_pipeline import send_training_example_kafka

from typing import Any, List, Dict, Optional
import asyncio
logger = structlog.get_logger()


class UserResetRequest(BaseModel):
    user_email: str


class UserResetResponse(BaseModel):
    user_id: str
    new_password: str


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


class PasswordResetRequest(OnyxBaseModel):
    """Password reset request model with OnyxBaseModel, Pydantic v2, and orjson serialization."""
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID = Field(...)
    token: str = Field(..., min_length=1)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    @field_validator("token")
    @classmethod
    def token_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Token cannot be empty")
        return v.strip()


class ORJSONModel(OnyxBaseModel):
    model_config = ConfigDict(json_loads=orjson.loads, json_dumps=orjson.dumps)


class Password(ORJSONModel):
    __slots__ = (
        'id', 'value', 'description', 'created_at', 'updated_at', 'created_by', 'updated_by',
        'source', 'version', 'trace_id', 'is_deleted'
    )
    id: UUID = Field(default_factory=uuid7)
    value: str = Field(..., min_length=8, max_length=128, description="Contraseña segura")
    description: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str | None = None
    updated_by: str | None = None
    source: str | None = None
    version: int = 1
    trace_id: str | None = None
    is_deleted: bool = False

    def __repr__(self) -> str:
        return f"<Password id={self.id} description={self.description!r}>"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Password):
            return False
        return self.id == other.id and self.value == other.value

    @field_validator('value')
    def password_strong(cls, v) -> Any:
        if not v or not v.strip():
            logger.error("Password value validation failed", value=v)
            raise ValueError("Password must not be empty")
        if len(v) < 8:
            logger.error("Password value validation failed: too short", value=v)
            raise ValueError("Password must be at least 8 characters")
        return v

    @model_validator(mode="after")
    def check_value_and_description(self) -> Any:
        if self.value and self.description and self.value in (self.description or ""):
            logger.warning("Description should not contain the password value", value=self.value)
        if self.created_at > self.updated_at:
            logger.warning("created_at is after updated_at", id=str(self.id))
        return self

    def __post_init_post_parse__(self) -> Any:
        logger.info("Password instantiated", id=str(self.id), description=self.description)

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
        logger.info("Password updated", id=str(self.id), version=self.version, trace_id=self.trace_id)

    def soft_delete(self) -> Any:
        self.is_deleted = True
        self.update()
        logger.info("Password soft deleted", id=str(self.id), trace_id=self.trace_id)

    def restore(self) -> Any:
        self.is_deleted = False
        self.update()
        logger.info("Password restored", id=str(self.id), trace_id=self.trace_id)

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
            "input": self.value,
            "metadata": {"description": self.description},
        }

    @classmethod
    def from_training_example(cls, example: dict):
        
    """from_training_example function."""
return cls(value=example["input"], description=example["metadata"].get("description"))

    def send_to_kafka(self, topic="ml_training_examples", bootstrap_servers=None) -> Any:
        """
        Envía este ejemplo a un topic de Kafka para el pipeline ML/LLM automatizado.
        """
        send_training_example_kafka(self, topic=topic, bootstrap_servers=bootstrap_servers)

    # Ejemplo de uso:
    # pwd = Password(value="supersegura123")
    # pwd.send_to_kafka(topic="ml_training_examples", bootstrap_servers=["localhost:9092"])

    @dataclass
class Config:
        frozen = True
        validate_assignment = True
