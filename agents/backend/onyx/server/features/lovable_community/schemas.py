"""
Schemas Pydantic para validación de requests y responses (backward compatibility)

Este archivo mantiene compatibilidad hacia atrás importando desde el módulo schemas/.
Los schemas están ahora organizados en:
- schemas/requests.py: Todos los request schemas
- schemas/responses.py: Todos los response schemas

Para nuevas importaciones, use:
    from .schemas import PublishChatRequest, PublishedChatResponse, etc.
"""

# Import all schemas from the modular structure for backward compatibility
from .schemas.requests import (
    PublishChatRequest,
    RemixChatRequest,
    VoteRequest,
    SearchRequest,
    UpdateChatRequest,
    CommentRequest,
    BulkOperationRequest,
    ExportRequest,
    NotificationRequest,
    ReportRequest,
    FilterRequest,
)

from .schemas.responses import (
    PublishedChatResponse,
    ChatListResponse,
    RemixResponse,
    VoteResponse,
    ChatStatsResponse,
    CommentResponse,
    UserProfileResponse,
    TrendingChatsResponse,
    BulkOperationResponse,
    AnalyticsResponse,
    ErrorResponse,
    SuccessResponse,
    NotificationResponse,
    ReportResponse,
    FeaturedChatsResponse,
    UserActivityResponse,
    HealthCheckResponse,
)

# Re-export all for backward compatibility
__all__ = [
    # Requests
    "PublishChatRequest",
    "RemixChatRequest",
    "VoteRequest",
    "SearchRequest",
    "UpdateChatRequest",
    "CommentRequest",
    "BulkOperationRequest",
    "ExportRequest",
    "NotificationRequest",
    "ReportRequest",
    "FilterRequest",
    # Responses
    "PublishedChatResponse",
    "ChatListResponse",
    "RemixResponse",
    "VoteResponse",
    "ChatStatsResponse",
    "CommentResponse",
    "UserProfileResponse",
    "TrendingChatsResponse",
    "BulkOperationResponse",
    "AnalyticsResponse",
    "ErrorResponse",
    "SuccessResponse",
    "NotificationResponse",
    "ReportResponse",
    "FeaturedChatsResponse",
    "UserActivityResponse",
    "HealthCheckResponse",
]
