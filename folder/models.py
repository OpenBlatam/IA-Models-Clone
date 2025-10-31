from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

from uuid import UUID, uuid4
from pydantic import BaseModel, Field, validator, field_validator, ConfigDict, model_validator
from typing import List, Optional
from datetime import datetime
import logging
import structlog
import orjson

from onyx.server.query_and_chat.models import ChatSessionDetails
from onyx.core.models import OnyxBaseModel
from agents.backend.onyx.server.features.utils.ml_data_pipeline import send_training_example_kafka

from typing import Any, List, Dict, Optional
import asyncio
logger = structlog.get_logger()


class UserFolderSnapshot(BaseModel):
    folder_id: int
    folder_name: str | None
    display_priority: int
    chat_sessions: list[ChatSessionDetails]


class GetUserFoldersResponse(BaseModel):
    folders: list[UserFolderSnapshot]


class FolderCreationRequest(BaseModel):
    folder_name: str | None = None


class FolderUpdateRequest(BaseModel):
    folder_name: str | None = None


class FolderChatSessionRequest(BaseModel):
    chat_session_id: UUID


class DeleteFolderOptions(BaseModel):
    including_chats: bool = False


class ORJSONModel(OnyxBaseModel):
    model_config = ConfigDict(json_loads=orjson.loads, json_dumps=orjson.dumps)


class Folder(ORJSONModel):
    __slots__ = (
        'id', 'name', 'parent_id', 'created_at', 'updated_at', 'created_by', 'updated_by',
        'source', 'version', 'trace_id', 'is_deleted'
    )
    id: UUID = Field(default_factory=uuid7)
    name: str = Field(..., min_length=2, max_length=128, description="Nombre de la carpeta")
    parent_id: UUID | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str | None = None
    updated_by: str | None = None
    source: str | None = None
    version: int = 1
    trace_id: str | None = None
    is_deleted: bool = False

    @field_validator('name')
    def name_not_empty(cls, v) -> Any:
        if not v or not v.strip():
            logger.error("Folder name validation failed", value=v)
            raise ValueError("Name must not be empty")
        return v

    @model_validator(mode="after")
    def check_name_and_parent(self) -> Any:
        if self.parent_id and self.id == self.parent_id:
            logger.warning("Folder cannot be its own parent", id=str(self.id))
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
        logger.info("Folder updated", id=str(self.id), version=self.version, trace_id=self.trace_id)

    def soft_delete(self) -> Any:
        self.is_deleted = True
        self.update()
        logger.info("Folder soft deleted", id=str(self.id), trace_id=self.trace_id)

    def restore(self) -> Any:
        self.is_deleted = False
        self.update()
        logger.info("Folder restored", id=str(self.id), trace_id=self.trace_id)

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
            "input": self.name,
            "metadata": {"parent_id": str(self.parent_id) if self.parent_id else None},
        }

    @classmethod
    def from_training_example(cls, example: dict):
        
    """from_training_example function."""
return cls(name=example["input"], parent_id=example["metadata"].get("parent_id"))

    def send_to_kafka(self, topic="ml_training_examples", bootstrap_servers=None) -> Any:
        """
        Envía este ejemplo a un topic de Kafka para el pipeline ML/LLM automatizado.
        """
        send_training_example_kafka(self, topic=topic, bootstrap_servers=bootstrap_servers)

    # Ejemplo de uso:
    # folder = Folder(name="Carpeta 1")
    # folder.send_to_kafka(topic="ml_training_examples", bootstrap_servers=["localhost:9092"])

    @dataclass
class Config:
        frozen = True
        validate_assignment = True
