from typing import Optional, List, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from ..constants import (
    MAX_TITLE_LENGTH,
    MAX_DESCRIPTION_LENGTH,
    MAX_CHAT_CONTENT_LENGTH,
    MAX_TAG_LENGTH,
    MIN_TITLE_LENGTH,
    MIN_CHAT_CONTENT_LENGTH,
    MAX_TAGS_PER_CHAT,
)


class PublishChatRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    title: str = Field(
        ...,
        min_length=MIN_TITLE_LENGTH,
        max_length=MAX_TITLE_LENGTH,
        description="Título del chat",
        examples=["Mi increíble chat sobre IA"]
    )
    description: Optional[str] = Field(
        None,
        max_length=MAX_DESCRIPTION_LENGTH,
        description="Descripción opcional del chat",
        examples=["Un chat sobre inteligencia artificial y machine learning"]
    )
    chat_content: str = Field(
        ...,
        min_length=MIN_CHAT_CONTENT_LENGTH,
        max_length=MAX_CHAT_CONTENT_LENGTH,
        description="Contenido del chat (JSON o texto)",
        examples=["{\"messages\": [...]}"]
    )
    tags: Optional[List[str]] = Field(
        None,
        max_length=MAX_TAGS_PER_CHAT,
        description=f"Tags del chat (máximo {MAX_TAGS_PER_CHAT})",
        examples=[["ai", "machine-learning", "chat"]]
    )
    is_public: bool = Field(
        True,
        description="Si el chat es público o privado"
    )
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Title cannot be empty")
        return v.strip()
    
    @field_validator('description')
    @classmethod
    def validate_description(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
            if not v:
                return None
            if len(v) > MAX_DESCRIPTION_LENGTH:
                raise ValueError(f"Description cannot exceed {MAX_DESCRIPTION_LENGTH} characters")
        return v
    
    @field_validator('chat_content')
    @classmethod
    def validate_chat_content(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Chat content cannot be empty")
        if len(v) > MAX_CHAT_CONTENT_LENGTH:
            raise ValueError(f"Chat content cannot exceed {MAX_CHAT_CONTENT_LENGTH} characters")
        return v.strip()
    
    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        
        if len(v) > MAX_TAGS_PER_CHAT:
            raise ValueError(f"Maximum {MAX_TAGS_PER_CHAT} tags allowed")
        
        sanitized = []
        seen = set()
        for tag in v:
            if tag:
                tag_clean = tag.strip().lower()[:MAX_TAG_LENGTH]
                if tag_clean and tag_clean not in seen:
                    sanitized.append(tag_clean)
                    seen.add(tag_clean)
        
        return sanitized if sanitized else None
    
    @model_validator(mode='after')
    def validate_model(self):
        if not self.chat_content.strip():
            raise ValueError("Chat content cannot be empty or only whitespace")
        return self


class RemixChatRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    original_chat_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="ID del chat original",
        examples=["123e4567-e89b-12d3-a456-426614174000"]
    )
    title: str = Field(
        ...,
        min_length=MIN_TITLE_LENGTH,
        max_length=MAX_TITLE_LENGTH,
        description="Título del remix",
        examples=["Remix: Mi increíble chat sobre IA"]
    )
    description: Optional[str] = Field(
        None,
        max_length=MAX_DESCRIPTION_LENGTH,
        description="Descripción del remix"
    )
    chat_content: str = Field(
        ...,
        min_length=MIN_CHAT_CONTENT_LENGTH,
        max_length=MAX_CHAT_CONTENT_LENGTH,
        description="Contenido del remix"
    )
    tags: Optional[List[str]] = Field(
        None,
        max_length=MAX_TAGS_PER_CHAT,
        description=f"Tags del remix (máximo {MAX_TAGS_PER_CHAT})"
    )
    
    @field_validator('original_chat_id')
    @classmethod
    def validate_original_chat_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Original chat ID cannot be empty")
        return v.strip()
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        return PublishChatRequest.validate_title(v)
    
    @field_validator('description')
    @classmethod
    def validate_description(cls, v: Optional[str]) -> Optional[str]:
        return PublishChatRequest.validate_description(v)
    
    @field_validator('chat_content')
    @classmethod
    def validate_chat_content(cls, v: str) -> str:
        return PublishChatRequest.validate_chat_content(v)
    
    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        return PublishChatRequest.validate_tags(v)


class VoteRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    chat_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="ID del chat",
        examples=["123e4567-e89b-12d3-a456-426614174000"]
    )
    vote_type: Literal["upvote", "downvote"] = Field(
        "upvote",
        description="Tipo de voto: upvote o downvote"
    )
    
    @field_validator('chat_id')
    @classmethod
    def validate_chat_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Chat ID cannot be empty")
        return v.strip()


class SearchRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    query: Optional[str] = Field(
        None,
        max_length=MAX_TITLE_LENGTH,
        description="Texto de búsqueda",
        examples=["inteligencia artificial"]
    )
    tags: Optional[List[str]] = Field(
        None,
        max_length=MAX_TAGS_PER_CHAT,
        description=f"Filtrar por tags (máximo {MAX_TAGS_PER_CHAT})"
    )
    user_id: Optional[str] = Field(
        None,
        min_length=1,
        max_length=100,
        description="Filtrar por usuario"
    )
    sort_by: Literal["score", "created_at", "vote_count", "remix_count"] = Field(
        "score",
        description="Campo por el cual ordenar"
    )
    order: Literal["asc", "desc"] = Field(
        "desc",
        description="Orden ascendente o descendente"
    )
    page: int = Field(
        1,
        ge=1,
        le=1000,
        description="Número de página (1-indexed)"
    )
    page_size: int = Field(
        20,
        ge=1,
        le=100,
        description="Tamaño de página (máximo 100)"
    )
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
            if not v:
                return None
            if len(v) > MAX_TITLE_LENGTH:
                raise ValueError(f"Query cannot exceed {MAX_TITLE_LENGTH} characters")
        return v
    
    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        
        if len(v) > MAX_TAGS_PER_CHAT:
            raise ValueError(f"Maximum {MAX_TAGS_PER_CHAT} tags allowed for filtering")
        
        sanitized = []
        seen = set()
        for tag in v:
            if tag:
                tag_clean = tag.strip().lower()[:MAX_TAG_LENGTH]
                if tag_clean and tag_clean not in seen:
                    sanitized.append(tag_clean)
                    seen.add(tag_clean)
        
        return sanitized if sanitized else None
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
            if not v:
                return None
        return v
    
    @model_validator(mode='after')
    def validate_model(self):
        return self


class UpdateChatRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    title: Optional[str] = Field(
        None,
        min_length=MIN_TITLE_LENGTH,
        max_length=MAX_TITLE_LENGTH,
        description="Nuevo título del chat",
        examples=["Mi chat actualizado sobre IA"]
    )
    description: Optional[str] = Field(
        None,
        max_length=MAX_DESCRIPTION_LENGTH,
        description="Nueva descripción del chat"
    )
    tags: Optional[List[str]] = Field(
        None,
        max_length=MAX_TAGS_PER_CHAT,
        description=f"Nueva lista de tags (máximo {MAX_TAGS_PER_CHAT})"
    )
    is_public: Optional[bool] = Field(
        None,
        description="Cambiar visibilidad del chat"
    )
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            return PublishChatRequest.validate_title(v)
        return v
    
    @field_validator('description')
    @classmethod
    def validate_description(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            return PublishChatRequest.validate_description(v)
        return v
    
    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            return PublishChatRequest.validate_tags(v)
        return v
    
    @model_validator(mode='after')
    def validate_model(self):
        if not any([self.title, self.description, self.tags is not None, self.is_public is not None]):
            raise ValueError("At least one field must be provided for update")
        return self


class CommentRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    content: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Contenido del comentario",
        examples=["¡Excelente chat! Muy útil."]
    )
    parent_comment_id: Optional[str] = Field(
        None,
        description="ID del comentario padre (para respuestas)"
    )
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Comment content cannot be empty")
        if len(v.strip()) > 2000:
            raise ValueError("Comment content cannot exceed 2000 characters")
        return v.strip()


class BulkOperationRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    chat_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Lista de IDs de chats (máximo 100)",
        examples=[["chat1", "chat2", "chat3"]]
    )
    operation: Literal["delete", "feature", "unfeature", "make_public", "make_private"] = Field(
        ...,
        description="Tipo de operación a realizar"
    )
    
    @field_validator('chat_ids')
    @classmethod
    def validate_chat_ids(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Chat IDs list cannot be empty")
        if len(v) > 100:
            raise ValueError("Maximum 100 chat IDs allowed")
        
        sanitized = [chat_id.strip() for chat_id in v if chat_id and chat_id.strip()]
        if not sanitized:
            raise ValueError("No valid chat IDs provided")
        
        seen = set()
        unique = []
        for chat_id in sanitized:
            if chat_id not in seen:
                seen.add(chat_id)
                unique.append(chat_id)
        
        return unique


class ExportRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    chat_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Lista de IDs de chats a exportar"
    )
    format: Literal["json", "csv", "xml"] = Field(
        "json",
        description="Formato de exportación"
    )
    include_stats: bool = Field(
        False,
        description="Incluir estadísticas en la exportación"
    )
    
    @field_validator('chat_ids')
    @classmethod
    def validate_chat_ids(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Chat IDs list cannot be empty")
        if len(v) > 100:
            raise ValueError("Maximum 100 chat IDs allowed")
        
        sanitized = [chat_id.strip() for chat_id in v if chat_id and chat_id.strip()]
        if not sanitized:
            raise ValueError("No valid chat IDs provided")
        
        seen = set()
        unique = []
        for chat_id in sanitized:
            if chat_id not in seen:
                seen.add(chat_id)
                unique.append(chat_id)
        
        return unique


class NotificationRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    user_id: str = Field(..., min_length=1, description="ID del usuario a notificar")
    type: Literal["remix", "vote", "comment", "feature", "system"] = Field(
        ...,
        description="Tipo de notificación"
    )
    title: str = Field(..., min_length=1, max_length=200, description="Título de la notificación")
    message: str = Field(..., min_length=1, max_length=1000, description="Mensaje de la notificación")
    chat_id: Optional[str] = Field(None, description="ID del chat relacionado")
    link: Optional[str] = Field(None, description="Link adicional")


class ReportRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    chat_id: str = Field(..., min_length=1, description="ID del chat a reportar")
    reason: Literal["spam", "inappropriate", "copyright", "other"] = Field(
        ...,
        description="Razón del reporte"
    )
    description: Optional[str] = Field(
        None,
        max_length=1000,
        description="Descripción adicional del reporte"
    )
    
    @field_validator('description')
    @classmethod
    def validate_description(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
            if not v:
                return None
        return v


class FilterRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    min_score: Optional[float] = Field(None, ge=0, description="Score mínimo")
    max_score: Optional[float] = Field(None, ge=0, description="Score máximo")
    min_votes: Optional[int] = Field(None, ge=0, description="Número mínimo de votos")
    min_remixes: Optional[int] = Field(None, ge=0, description="Número mínimo de remixes")
    min_views: Optional[int] = Field(None, ge=0, description="Número mínimo de visualizaciones")
    featured_only: bool = Field(False, description="Solo chats destacados")
    public_only: bool = Field(True, description="Solo chats públicos")
    date_from: Optional[datetime] = Field(None, description="Fecha de inicio")
    date_to: Optional[datetime] = Field(None, description="Fecha de fin")
    
    @model_validator(mode='after')
    def validate_dates(self):
        if self.date_from and self.date_to:
            if self.date_from > self.date_to:
                raise ValueError("date_from must be before date_to")
        return self
    
    @model_validator(mode='after')
    def validate_scores(self):
        if self.min_score is not None and self.max_score is not None:
            if self.min_score > self.max_score:
                raise ValueError("min_score must be less than or equal to max_score")
        return self

