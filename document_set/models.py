from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from datetime import datetime
import structlog
import orjson

from onyx.db.models import DocumentSet as DocumentSetDBModel
from onyx.server.documents.models import ConnectorCredentialPairDescriptor
from onyx.server.documents.models import ConnectorSnapshot
from onyx.server.documents.models import CredentialSnapshot
from onyx.core.models import OnyxBaseModel
from agents.backend.onyx.server.features.utils.ml_data_pipeline import send_training_example_kafka

logger = structlog.get_logger()

class ORJSONModel(OnyxBaseModel):
    model_config = ConfigDict(json_loads=orjson.loads, json_dumps=orjson.dumps)

class DocumentSetCreationRequest(BaseModel):
    name: str
    description: str
    cc_pair_ids: list[int]
    is_public: bool
    # For Private Document Sets, who should be able to access these
    users: list[UUID] = Field(default_factory=list)
    groups: list[int] = Field(default_factory=list)


class DocumentSetUpdateRequest(BaseModel):
    id: int
    description: str
    cc_pair_ids: list[int]
    is_public: bool
    # For Private Document Sets, who should be able to access these
    users: list[UUID]
    groups: list[int]


class CheckDocSetPublicRequest(BaseModel):
    """Note that this does not mean that the Document Set itself is to be viewable by everyone
    Rather, this refers to the CC-Pairs in the Document Set, and if every CC-Pair is public
    """

    document_set_ids: list[int]


class CheckDocSetPublicResponse(BaseModel):
    is_public: bool


class DocumentSet(ORJSONModel):
    __slots__ = (
        'id', 'name', 'documents', 'metadata', 'created_at', 'updated_at', 'created_by', 'updated_by',
        'source', 'version', 'trace_id', 'is_deleted'
    )
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=2, max_length=128, description="Nombre del set de documentos")
    documents: list[str] = Field(default_factory=list, description="Lista de documentos")
    metadata: dict = Field(default_factory=dict, description="Metadatos adicionales")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str | None = None
    updated_by: str | None = None
    source: str | None = None
    version: int = 1
    trace_id: str | None = None
    is_deleted: bool = False
    # Campos ampliados para compatibilidad con modelo de base de datos existente
    description: Optional[str] = None
    cc_pair_descriptors: List[ConnectorCredentialPairDescriptor] = Field(default_factory=list)
    is_up_to_date: Optional[bool] = None
    is_public: Optional[bool] = None
    users: List[UUID] = Field(default_factory=list)
    groups: List[int] = Field(default_factory=list)

    @field_validator('name')
    def name_not_empty(cls, v) -> Any:
        if not v or not v.strip():
            logger.error("DocumentSet name validation failed", value=v)
            raise ValueError("Name must not be empty")
        return v

    @field_validator('documents', mode="before")
    @classmethod
    def list_or_empty(cls, v) -> List[Any]:
        return v or []

    @field_validator('metadata', mode="before")
    @classmethod
    def dict_or_empty(cls, v) -> Any:
        return v or {}

    @model_validator(mode="after")
    def check_documents_and_metadata(self) -> Any:
        if self.documents and not isinstance(self.metadata, dict):
            logger.warning("Metadata should be a dict if documents exist", documents=self.documents)
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
        logger.info("DocumentSet updated", id=str(self.id), version=self.version, trace_id=self.trace_id)

    def soft_delete(self) -> Any:
        self.is_deleted = True
        self.update()
        logger.info("DocumentSet soft deleted", id=str(self.id), trace_id=self.trace_id)

    def restore(self) -> Any:
        self.is_deleted = False
        self.update()
        logger.info("DocumentSet restored", id=str(self.id), trace_id=self.trace_id)

    def to_dict(self) -> Any:
        return self.model_dump()

    def to_json(self) -> Any:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str) -> "DocumentSet":
        return cls.model_validate_json(data)

    def to_training_example(self) -> Any:
        return {
            "input": self.name,
            "output": self.documents,
            "metadata": self.metadata,
        }

    @classmethod
    def from_training_example(cls, example: dict) -> "DocumentSet":
        return cls(
            name=example.get("input", ""),
            documents=example.get("output", []) or [],
            metadata=example.get("metadata", {}) or {},
        )

    @classmethod
    def from_model(cls, document_set_model: DocumentSetDBModel) -> "DocumentSet":
        return cls(
            id=document_set_model.id,
            name=document_set_model.name,
            description=document_set_model.description,
            cc_pair_descriptors=[
                ConnectorCredentialPairDescriptor(
                    id=cc_pair.id,
                    name=cc_pair.name,
                    connector=ConnectorSnapshot.from_connector_db_model(
                        cc_pair.connector
                    ),
                    credential=CredentialSnapshot.from_credential_db_model(
                        cc_pair.credential
                    ),
                    access_type=cc_pair.access_type,
                )
                for cc_pair in document_set_model.connector_credential_pairs
            ],
            is_up_to_date=document_set_model.is_up_to_date,
            is_public=document_set_model.is_public,
            users=[user.id for user in document_set_model.users],
            groups=[group.id for group in document_set_model.groups],
        )

    def send_to_kafka(self, topic="ml_training_examples", bootstrap_servers=None) -> Any:
        """
        Envía este ejemplo a un topic de Kafka para el pipeline ML/LLM automatizado.
        """
        send_training_example_kafka(self, topic=topic, bootstrap_servers=bootstrap_servers)

    # Ejemplo de uso:
    # ds = DocumentSet(name="Set 1", documents=["doc1", "doc2"])
    # ds.send_to_kafka(topic="ml_training_examples", bootstrap_servers=["localhost:9092"])

    model_config = ConfigDict(validate_assignment=True)
