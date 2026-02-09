from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import Field, field_validator, root_validator, ConfigDict, model_validator
from ...utils.base_model import OnyxBaseModel
from uuid import UUID, uuid4
from ...utils.model_repository import ModelRepository
from ...utils.model_service import ModelService
from ...utils.model_decorators import validate_model, cache_model, log_operations
import logging
import structlog
import orjson
from uuid6 import uuid7
from agents.backend.onyx.server.features.utils.ml_data_pipeline import send_training_example_kafka
from typing import Any, List, Dict, Optional
import asyncio
"""
Tool Models - Onyx Integration
Enhanced models for tools with advanced features.
"""

_repository = ModelRepository()
_service = ModelService()

logger = structlog.get_logger()

class ORJSONModel(OnyxBaseModel):
    model_config = ConfigDict(json_loads=orjson.loads, json_dumps=orjson.dumps)

class Header(OnyxBaseModel):
    """HTTP header definition for tool integrations."""
    id: UUID = Field(default_factory=uuid4)
    key: str = Field(..., min_length=1, max_length=128)
    value: str = Field(..., min_length=1, max_length=512)
    description: Optional[str] = None
    is_required: bool = False
    is_secret: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("key", "value")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

    @validate_model(validate_types=True, validate_custom=True)
    @cache_model(key_field="key")
    @log_operations(logging.getLogger(__name__))
    def save(self, user_context: Optional[Dict[str, Any]] = None) -> None:
        _service.create_model(self.__class__.__name__, self.to_dict(), self.id)
        _repository._service = _service
        _repository._service.register_model(self.__class__.__name__, self.__class__)
        _repository._service.register_schema(self.__class__.__name__, self._schema)
        _repository._service.create_model(self.__class__.__name__, self.to_dict(), self.id)

class ToolDefinition(OnyxBaseModel):
    """Definition of a tool, including schemas and metadata."""
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=2, max_length=128)
    version: str
    description: str
    category: str
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    headers: List[Header] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Tool name cannot be empty")
        return v.strip()

    @field_validator("version")
    def check_version(cls, v) -> Any:
        parts = v.split(".")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError("Version must be in format X.Y.Z")
        return v

    @field_validator("category")
    def check_category(cls, v) -> Any:
        allowed = ["api", "utility", "integration", "custom"]
        if v not in allowed:
            raise ValueError(f"Category must be one of: {', '.join(allowed)}")
        return v

    @field_validator("headers", mode="before")
    @classmethod
    def list_or_empty(cls, v) -> List[Any]:
        return v or []

    @field_validator("input_schema", "output_schema", "parameters", "metadata", mode="before")
    @classmethod
    def dict_or_empty(cls, v) -> Any:
        return v or {}

    def get_required_headers(self) -> List[Header]:
        return [h for h in self.headers if h.is_required]

    def get_secret_headers(self) -> List[Header]:
        return [h for h in self.headers if h.is_secret]

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        # TODO: Implement schema validation
        return True

    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        # TODO: Implement schema validation
        return True

class ToolSnapshot(OnyxBaseModel):
    """Snapshot of a tool definition at a point in time."""
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=2, max_length=128)
    version: str
    description: str
    category: str
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    headers: List[Header] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Tool name cannot be empty")
        return v.strip()

    @field_validator("headers", mode="before")
    @classmethod
    def list_or_empty(cls, v) -> List[Any]:
        return v or []

    @field_validator("input_schema", "output_schema", "parameters", "metadata", mode="before")
    @classmethod
    def dict_or_empty(cls, v) -> Any:
        return v or {}

    @classmethod
    def from_definition(cls, definition: ToolDefinition) -> "ToolSnapshot":
        now = datetime.utcnow().isoformat()
        return cls(
            id=str(uuid4()),
            name=definition.name,
            version=definition.version,
            description=definition.description,
            category=definition.category,
            input_schema=definition.input_schema,
            output_schema=definition.output_schema,
            headers=definition.headers,
            parameters=definition.parameters,
            metadata=definition.metadata,
            created_at=now,
            updated_at=now
        )

    def to_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            version=self.version,
            description=self.description,
            category=self.category,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            headers=self.headers,
            parameters=self.parameters,
            metadata=self.metadata
        )

class CustomToolCreate(OnyxBaseModel):
    """Request model for creating a custom tool."""
    id: UUID = Field(default_factory=uuid4)
    definition: ToolDefinition
    is_public: bool = False
    owner_id: Optional[UUID] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("metadata", mode="before")
    @classmethod
    def dict_or_empty(cls, v) -> Any:
        return v or {}

class ToolUpdate(OnyxBaseModel):
    """Request model for updating a tool definition."""
    id: UUID = Field(default_factory=uuid4)
    name: Optional[str] = None
    version: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    headers: Optional[List[Header]] = None
    parameters: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("Tool name cannot be empty")
        return v.strip() if v else v

    @field_validator("headers", mode="before")
    @classmethod
    def list_or_empty(cls, v) -> List[Any]:
        return v or []

    @field_validator("input_schema", "output_schema", "parameters", "metadata", mode="before")
    @classmethod
    def dict_or_empty(cls, v) -> Any:
        return v or {}

class Tool(ORJSONModel):
    __slots__ = (
        'id', 'name', 'config', 'created_at', 'updated_at', 'created_by', 'updated_by',
        'source', 'version', 'trace_id', 'is_deleted'
    )
    id: UUID = Field(default_factory=uuid7)
    name: str = Field(..., min_length=2, max_length=128, description="Nombre de la herramienta")
    config: dict = Field(default_factory=dict, description="Configuración de la herramienta")
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
            logger.error("Tool name validation failed", value=v)
            raise ValueError("Name must not be empty")
        return v

    @field_validator('config', mode="before")
    @classmethod
    def dict_or_empty(cls, v) -> Any:
        return v or {}

    @model_validator(mode="after")
    def check_timestamps(self) -> Any:
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
        logger.info("Tool updated", id=str(self.id), version=self.version, trace_id=self.trace_id)

    def soft_delete(self) -> Any:
        self.is_deleted = True
        self.update()
        logger.info("Tool soft deleted", id=str(self.id), trace_id=self.trace_id)

    def restore(self) -> Any:
        self.is_deleted = False
        self.update()
        logger.info("Tool restored", id=str(self.id), trace_id=self.trace_id)

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
            "metadata": self.config,
        }

    @classmethod
    def from_training_example(cls, example: dict):
        
    """from_training_example function."""
return cls(name=example["input"], config=example.get("metadata", {}))

    def send_to_kafka(self, topic="ml_training_examples", bootstrap_servers=None) -> Any:
        """
        Envía este ejemplo a un topic de Kafka para el pipeline ML/LLM automatizado.
        """
        send_training_example_kafka(self, topic=topic, bootstrap_servers=bootstrap_servers)

    @dataclass
class Config:
        frozen = True
        validate_assignment = True

# Example usage:
"""
# Create header
header = Header(
    key="Authorization",
    value="Bearer token123",
    description="API authentication token",
    is_required=True,
    is_secret=True
)

# Create tool definition
tool_def = ToolDefinition(
    name="Weather API",
    version="1.0.0",
    description="Get weather information for a location",
    category="api",
    input_schema={
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "units": {"type": "string", "enum": ["metric", "imperial"]}
        },
        "required": ["location"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "temperature": {"type": "number"},
            "humidity": {"type": "number"},
            "description": {"type": "string"}
        }
    },
    headers=[header],
    parameters={
        "base_url": "https://api.weather.com",
        "timeout": 30
    }
)

# Create tool snapshot
snapshot = ToolSnapshot.from_definition(tool_def)

# Create custom tool
custom_tool = CustomToolCreate(
    definition=tool_def,
    is_public=True,
    owner_id="user123"
)

# Update tool
tool_update = ToolUpdate(
    name="Weather API v2",
    version="2.0.0",
    description="Enhanced weather information API"
)

# Index models
redis_indexer = RedisIndexer()
tool_def.index(redis_indexer)
snapshot.index(redis_indexer)
custom_tool.index(redis_indexer)
"""
