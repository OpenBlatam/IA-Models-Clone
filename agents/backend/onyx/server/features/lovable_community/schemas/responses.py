from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field

from .requests import PublishChatRequest


class PublishedChatResponse(BaseModel):
    id: str
    user_id: str
    title: str
    description: Optional[str]
    chat_content: str
    tags: Optional[List[str]]
    vote_count: int
    remix_count: int
    view_count: int
    score: float
    is_public: bool
    is_featured: bool
    created_at: datetime
    updated_at: datetime
    original_chat_id: Optional[str]
    has_user_voted: Optional[bool] = None
    user_vote_type: Optional[str] = None


class ChatListResponse(BaseModel):
    chats: List[PublishedChatResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class RemixResponse(BaseModel):
    id: str
    original_chat_id: str
    remix_chat_id: str
    user_id: str
    created_at: datetime
    original_chat: Optional[PublishedChatResponse] = None
    remix_chat: Optional[PublishedChatResponse] = None


class VoteResponse(BaseModel):
    id: str
    chat_id: str
    user_id: str
    vote_type: str
    created_at: datetime


class ChatStatsResponse(BaseModel):
    chat_id: str = Field(..., description="ID del chat")
    vote_count: int = Field(..., ge=0, description="Número total de votos")
    remix_count: int = Field(..., ge=0, description="Número de remixes")
    view_count: int = Field(..., ge=0, description="Número de visualizaciones")
    score: float = Field(..., description="Score de ranking")
    rank: Optional[int] = Field(None, ge=1, description="Posición en ranking")
    upvote_count: Optional[int] = Field(None, ge=0, description="Número de upvotes")
    downvote_count: Optional[int] = Field(None, ge=0, description="Número de downvotes")
    engagement_rate: Optional[float] = Field(None, ge=0, description="Tasa de engagement")


class CommentResponse(BaseModel):
    id: str = Field(..., description="ID del comentario")
    chat_id: str = Field(..., description="ID del chat")
    user_id: str = Field(..., description="ID del usuario que comentó")
    content: str = Field(..., description="Contenido del comentario")
    parent_comment_id: Optional[str] = Field(None, description="ID del comentario padre")
    likes_count: int = Field(0, ge=0, description="Número de likes")
    replies_count: int = Field(0, ge=0, description="Número de respuestas")
    created_at: datetime = Field(..., description="Fecha de creación")
    updated_at: datetime = Field(..., description="Fecha de última actualización")


class UserProfileResponse(BaseModel):
    user_id: str = Field(..., description="ID del usuario")
    total_chats: int = Field(0, ge=0, description="Total de chats publicados")
    total_remixes: int = Field(0, ge=0, description="Total de remixes creados")
    total_votes: int = Field(0, ge=0, description="Total de votos dados")
    average_score: Optional[float] = Field(None, ge=0, description="Score promedio de chats")
    top_chat_id: Optional[str] = Field(None, description="ID del chat con mayor score")
    joined_at: Optional[datetime] = Field(None, description="Fecha de registro")


class TrendingChatsResponse(BaseModel):
    period: Literal["hour", "day", "week", "month"] = Field(..., description="Período de tiempo")
    chats: List[PublishedChatResponse] = Field(..., description="Lista de chats trending")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp de generación")


class BulkOperationResponse(BaseModel):
    operation: str = Field(..., description="Tipo de operación realizada")
    total_requested: int = Field(..., ge=0, description="Total de chats solicitados")
    successful: int = Field(..., ge=0, description="Número de operaciones exitosas")
    failed: int = Field(..., ge=0, description="Número de operaciones fallidas")
    failed_chat_ids: List[str] = Field(default_factory=list, description="IDs de chats que fallaron")
    errors: List[str] = Field(default_factory=list, description="Mensajes de error")


class AnalyticsResponse(BaseModel):
    total_chats: int = Field(..., ge=0, description="Total de chats publicados")
    total_users: int = Field(..., ge=0, description="Total de usuarios únicos")
    total_votes: int = Field(..., ge=0, description="Total de votos")
    total_remixes: int = Field(..., ge=0, description="Total de remixes")
    total_views: int = Field(..., ge=0, description="Total de visualizaciones")
    average_score: float = Field(..., ge=0, description="Score promedio")
    top_tags: List[Dict[str, Any]] = Field(default_factory=list, description="Tags más populares")
    period: Optional[str] = Field(None, description="Período de análisis")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp de generación")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Tipo de error")
    message: str = Field(..., description="Mensaje descriptivo del error")
    details: Optional[Dict[str, Any]] = Field(None, description="Detalles adicionales del error")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp del error")
    path: Optional[str] = Field(None, description="Ruta del endpoint")


class SuccessResponse(BaseModel):
    success: bool = Field(True, description="Indica si la operación fue exitosa")
    message: str = Field(..., description="Mensaje descriptivo")
    data: Optional[Dict[str, Any]] = Field(None, description="Datos adicionales")


class NotificationResponse(BaseModel):
    id: str = Field(..., description="ID de la notificación")
    user_id: str = Field(..., description="ID del usuario")
    type: str = Field(..., description="Tipo de notificación")
    title: str = Field(..., description="Título")
    message: str = Field(..., description="Mensaje")
    chat_id: Optional[str] = Field(None, description="ID del chat relacionado")
    link: Optional[str] = Field(None, description="Link adicional")
    read: bool = Field(False, description="Si la notificación fue leída")
    created_at: datetime = Field(..., description="Fecha de creación")


class ReportResponse(BaseModel):
    id: str = Field(..., description="ID del reporte")
    chat_id: str = Field(..., description="ID del chat reportado")
    user_id: str = Field(..., description="ID del usuario que reportó")
    reason: str = Field(..., description="Razón del reporte")
    description: Optional[str] = Field(None, description="Descripción adicional")
    status: Literal["pending", "reviewed", "resolved", "dismissed"] = Field(
        "pending",
        description="Estado del reporte"
    )
    created_at: datetime = Field(..., description="Fecha de creación")


class FeaturedChatsResponse(BaseModel):
    chats: List[PublishedChatResponse] = Field(..., description="Lista de chats destacados")
    total: int = Field(..., ge=0, description="Total de chats destacados")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp de generación")


class UserActivityResponse(BaseModel):
    user_id: str = Field(..., description="ID del usuario")
    recent_chats: List[PublishedChatResponse] = Field(default_factory=list, description="Chats recientes")
    recent_remixes: List[RemixResponse] = Field(default_factory=list, description="Remixes recientes")
    recent_votes: List[VoteResponse] = Field(default_factory=list, description="Votos recientes")
    activity_timeline: List[Dict[str, Any]] = Field(default_factory=list, description="Timeline de actividad")


class HealthCheckResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Estado del servicio")
    version: str = Field(..., description="Versión de la API")
    database: Literal["connected", "disconnected"] = Field(..., description="Estado de la base de datos")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp del check")

