"""
API module for Robot Maintenance AI.
Exports common components for API development.
"""

from .maintenance_api import router, create_maintenance_app
from .base_router import BaseRouter, create_base_router
from .schemas import (
    MaintenanceQuestionRequest,
    ProcedureRequest,
    DiagnosisRequest,
    PredictionRequest,
    ChecklistRequest,
    ScheduleRequest
)
from .dependencies import (
    get_tutor,
    get_conversation_manager,
    get_rate_limiter,
    check_rate_limit
)
from .responses import success_response, error_response, paginated_response
from .exceptions import (
    MaintenanceAPIException,
    ValidationError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    TimeoutError,
    InternalServerError
)

__all__ = [
    # Main router
    "router",
    "create_maintenance_app",
    # Base Router
    "BaseRouter",
    "create_base_router",
    # Schemas
    "MaintenanceQuestionRequest",
    "ProcedureRequest",
    "DiagnosisRequest",
    "PredictionRequest",
    "ChecklistRequest",
    "ScheduleRequest",
    # Dependencies
    "get_tutor",
    "get_conversation_manager",
    "get_rate_limiter",
    "check_rate_limit",
    # Responses
    "success_response",
    "error_response",
    "paginated_response",
    # Exceptions
    "MaintenanceAPIException",
    "ValidationError",
    "NotFoundError",
    "RateLimitError",
    "ServiceUnavailableError",
    "TimeoutError",
    "InternalServerError",
]





