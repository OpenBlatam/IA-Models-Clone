"""
API Models
Pydantic models organized by domain
"""

from .enums import (
    ContentType,
    OutputFormat,
    DesignStyle,
    UserRole,
    SessionStatus
)
from .base import BaseResponse, ErrorResponse
from .user import User, UserCreate, UserUpdate
from .project import Project, ProjectCreate, ProjectUpdate
from .content import ContentRequest, ContentResponse, ContentUpdate
from .collaboration import (
    CollaborationSession,
    CollaborationMessage,
    CollaborationEvent
)
from .export import ExportRequest, ExportResponse
from .analytics import (
    AnalyticsRequest,
    AnalyticsResponse,
    DashboardData,
    ContentPerformance,
    CollaborationStats
)
from .common import (
    Template,
    TemplateCreate,
    SearchRequest,
    SearchResponse,
    Notification,
    NotificationCreate,
    UserSettings,
    SystemSettings,
    HealthCheck,
    ServiceStatus,
    WebSocketMessage,
    WebSocketResponse
)

__all__ = [
    "ContentType",
    "OutputFormat",
    "DesignStyle",
    "UserRole",
    "SessionStatus",
    "BaseResponse",
    "ErrorResponse",
    "User",
    "UserCreate",
    "UserUpdate",
    "Project",
    "ProjectCreate",
    "ProjectUpdate",
    "ContentRequest",
    "ContentResponse",
    "ContentUpdate",
    "CollaborationSession",
    "CollaborationMessage",
    "CollaborationEvent",
    "ExportRequest",
    "ExportResponse",
    "AnalyticsRequest",
    "AnalyticsResponse",
    "DashboardData",
    "ContentPerformance",
    "CollaborationStats",
    "Template",
    "TemplateCreate",
    "SearchRequest",
    "SearchResponse",
    "Notification",
    "NotificationCreate",
    "UserSettings",
    "SystemSettings",
    "HealthCheck",
    "ServiceStatus",
    "WebSocketMessage",
    "WebSocketResponse",
]







