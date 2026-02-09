from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

from datetime import datetime
from uuid import UUID, uuid4
from typing import Dict, List, Optional, Any
import logging
import structlog
import orjson

from pydantic import BaseModel, Field, validator, field_validator, ConfigDict, model_validator

from onyx.context.search.enums import RecencyBiasSetting
from onyx.db.models import Persona
from onyx.db.models import PersonaLabel
from onyx.db.models import Prompt
from onyx.db.models import StarterMessage
from onyx.server.features.document_set.models import DocumentSet
from onyx.server.features.tool.models import ToolSnapshot
from onyx.server.models import MinimalUserSnapshot
from onyx.utils.logger import setup_logger
from onyx.core.models import OnyxBaseModel
from uuid6 import uuid7
from agents.backend.onyx.server.features.utils.ml_data_pipeline import send_training_example_kafka


from typing import Any, List, Dict, Optional
import asyncio
logger = structlog.get_logger()


class PromptSnapshot(BaseModel):
    id: int
    name: str
    description: str
    system_prompt: str
    task_prompt: str
    include_citations: bool
    datetime_aware: bool
    default_prompt: bool
    # Not including persona info, not needed

    @classmethod
    def from_model(cls, prompt: Prompt) -> "PromptSnapshot":
        if prompt.deleted:
            raise ValueError("Prompt has been deleted")

        return PromptSnapshot(
            id=prompt.id,
            name=prompt.name,
            description=prompt.description,
            system_prompt=prompt.system_prompt,
            task_prompt=prompt.task_prompt,
            include_citations=prompt.include_citations,
            datetime_aware=prompt.datetime_aware,
            default_prompt=prompt.default_prompt,
        )


# More minimal request for generating a persona prompt
class GenerateStarterMessageRequest(BaseModel):
    name: str
    description: str
    instructions: str
    document_set_ids: list[int]
    generation_count: int


class PersonaUpsertRequest(BaseModel):
    name: str
    description: str
    system_prompt: str
    task_prompt: str
    datetime_aware: bool
    document_set_ids: list[int]
    num_chunks: float
    include_citations: bool
    is_public: bool
    recency_bias: RecencyBiasSetting
    prompt_ids: list[int]
    llm_filter_extraction: bool
    llm_relevance_filter: bool
    llm_model_provider_override: str | None = None
    llm_model_version_override: str | None = None
    starter_messages: list[StarterMessage] | None = None
    # For Private Personas, who should be able to access these
    users: list[UUID] = Field(default_factory=list)
    groups: list[int] = Field(default_factory=list)
    # e.g. ID of SearchTool or ImageGenerationTool or <USER_DEFINED_TOOL>
    tool_ids: list[int]
    icon_color: str | None = None
    icon_shape: int | None = None
    remove_image: bool | None = None
    uploaded_image_id: str | None = None  # New field for uploaded image
    search_start_date: datetime | None = None
    label_ids: list[int] | None = None
    is_default_persona: bool = False
    display_priority: int | None = None
    user_file_ids: list[int] | None = None
    user_folder_ids: list[int] | None = None


class PersonaSnapshot(BaseModel):
    id: int
    name: str
    description: str
    is_public: bool
    is_visible: bool
    icon_shape: int | None
    icon_color: str | None
    uploaded_image_id: str | None
    user_file_ids: list[int]
    user_folder_ids: list[int]
    display_priority: int | None
    is_default_persona: bool
    builtin_persona: bool
    starter_messages: list[StarterMessage] | None
    tools: list[ToolSnapshot]
    labels: list["PersonaLabelSnapshot"]
    owner: MinimalUserSnapshot | None
    users: list[MinimalUserSnapshot]
    groups: list[int]
    document_sets: list[DocumentSet]
    llm_model_provider_override: str | None
    llm_model_version_override: str | None
    num_chunks: float | None

    @classmethod
    def from_model(cls, persona: Persona) -> "PersonaSnapshot":
        return PersonaSnapshot(
            id=persona.id,
            name=persona.name,
            description=persona.description,
            is_public=persona.is_public,
            is_visible=persona.is_visible,
            icon_shape=persona.icon_shape,
            icon_color=persona.icon_color,
            uploaded_image_id=persona.uploaded_image_id,
            user_file_ids=[file.id for file in persona.user_files],
            user_folder_ids=[folder.id for folder in persona.user_folders],
            display_priority=persona.display_priority,
            is_default_persona=persona.is_default_persona,
            builtin_persona=persona.builtin_persona,
            starter_messages=persona.starter_messages,
            tools=[ToolSnapshot.from_model(tool) for tool in persona.tools],
            labels=[PersonaLabelSnapshot.from_model(label) for label in persona.labels],
            owner=(
                MinimalUserSnapshot(id=persona.user.id, email=persona.user.email)
                if persona.user
                else None
            ),
            users=[
                MinimalUserSnapshot(id=user.id, email=user.email)
                for user in persona.users
            ],
            groups=[user_group.id for user_group in persona.groups],
            document_sets=[
                DocumentSet.from_model(document_set_model)
                for document_set_model in persona.document_sets
            ],
            llm_model_provider_override=persona.llm_model_provider_override,
            llm_model_version_override=persona.llm_model_version_override,
            num_chunks=persona.num_chunks,
        )


# Model with full context on perona's internal settings
# This is used for flows which need to know all settings
class FullPersonaSnapshot(PersonaSnapshot):
    search_start_date: datetime | None = None
    prompts: list[PromptSnapshot] = Field(default_factory=list)
    llm_relevance_filter: bool = False
    llm_filter_extraction: bool = False

    @classmethod
    def from_model(
        cls, persona: Persona, allow_deleted: bool = False
    ) -> "FullPersonaSnapshot":
        if persona.deleted:
            error_msg = f"Persona with ID {persona.id} has been deleted"
            if not allow_deleted:
                raise ValueError(error_msg)
            else:
                logger.warning(error_msg)

        return FullPersonaSnapshot(
            id=persona.id,
            name=persona.name,
            description=persona.description,
            is_public=persona.is_public,
            is_visible=persona.is_visible,
            icon_shape=persona.icon_shape,
            icon_color=persona.icon_color,
            uploaded_image_id=persona.uploaded_image_id,
            user_file_ids=[file.id for file in persona.user_files],
            user_folder_ids=[folder.id for folder in persona.user_folders],
            display_priority=persona.display_priority,
            is_default_persona=persona.is_default_persona,
            builtin_persona=persona.builtin_persona,
            starter_messages=persona.starter_messages,
            users=[
                MinimalUserSnapshot(id=user.id, email=user.email)
                for user in persona.users
            ],
            groups=[user_group.id for user_group in persona.groups],
            tools=[ToolSnapshot.from_model(tool) for tool in persona.tools],
            labels=[PersonaLabelSnapshot.from_model(label) for label in persona.labels],
            owner=(
                MinimalUserSnapshot(id=persona.user.id, email=persona.user.email)
                if persona.user
                else None
            ),
            document_sets=[
                DocumentSet.from_model(document_set_model)
                for document_set_model in persona.document_sets
            ],
            num_chunks=persona.num_chunks,
            search_start_date=persona.search_start_date,
            prompts=[PromptSnapshot.from_model(prompt) for prompt in persona.prompts],
            llm_relevance_filter=persona.llm_relevance_filter,
            llm_filter_extraction=persona.llm_filter_extraction,
            llm_model_provider_override=persona.llm_model_provider_override,
            llm_model_version_override=persona.llm_model_version_override,
        )


class PromptTemplateResponse(BaseModel):
    final_prompt_template: str


class PersonaSharedNotificationData(BaseModel):
    persona_id: int


class ImageGenerationToolStatus(BaseModel):
    is_available: bool


class PersonaLabelCreate(BaseModel):
    name: str


class PersonaLabelResponse(BaseModel):
    id: int
    name: str

    @classmethod
    def from_model(cls, category: PersonaLabel) -> "PersonaLabelResponse":
        return PersonaLabelResponse(
            id=category.id,
            name=category.name,
        )


class PersonaLabelSnapshot(BaseModel):
    id: int
    name: str

    @classmethod
    def from_model(cls, label: PersonaLabel) -> "PersonaLabelSnapshot":
        return PersonaLabelSnapshot(
            id=label.id,
            name=label.name,
        )


class ORJSONModel(OnyxBaseModel):
    model_config = ConfigDict(json_loads=orjson.loads, json_dumps=orjson.dumps)


class Persona(ORJSONModel):
    __slots__ = (
        'id', 'name', 'description', 'attributes', 'created_at', 'updated_at', 'created_by', 'updated_by',
        'source', 'version', 'trace_id', 'is_deleted'
    )
    id: UUID = Field(default_factory=uuid7)
    name: str = Field(..., min_length=2, max_length=128, description="Nombre completo de la persona")
    description: str | None = None
    attributes: dict = Field(default_factory=dict, description="Atributos adicionales")
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
            logger.error("Persona name validation failed", value=v)
            raise ValueError("Name must not be empty")
        return v

    @field_validator('attributes', mode="before")
    @classmethod
    def dict_or_empty(cls, v) -> Any:
        return v or {}

    @model_validator(mode="after")
    def check_name_and_description(self) -> Any:
        if self.name and self.description and self.name in (self.description or ""):
            logger.warning("Description should not contain the name", name=self.name)
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
        logger.info("Persona updated", id=str(self.id), version=self.version, trace_id=self.trace_id)

    def soft_delete(self) -> Any:
        self.is_deleted = True
        self.update()
        logger.info("Persona soft deleted", id=str(self.id), trace_id=self.trace_id)

    def restore(self) -> Any:
        self.is_deleted = False
        self.update()
        logger.info("Persona restored", id=str(self.id), trace_id=self.trace_id)

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
            "output": self.description,
            "metadata": self.attributes,
        }

    @classmethod
    def from_training_example(cls, example: dict):
        
    """from_training_example function."""
return cls(name=example["input"], description=example.get("output"), attributes=example.get("metadata", {}))

    def __post_init_post_parse__(self) -> Any:
        logger.info("Persona instantiated", id=str(self.id), name=self.name)

    def send_to_kafka(self, topic="ml_training_examples", bootstrap_servers=None) -> Any:
        """
        Envía este ejemplo a un topic de Kafka para el pipeline ML/LLM automatizado.
        """
        send_training_example_kafka(self, topic=topic, bootstrap_servers=bootstrap_servers)

    # Ejemplo de uso:
    # persona = Persona(name="Juan", description="Ejemplo")
    # persona.send_to_kafka(topic="ml_training_examples", bootstrap_servers=["localhost:9092"])

    @dataclass
class Config:
        frozen = True
        validate_assignment = True
