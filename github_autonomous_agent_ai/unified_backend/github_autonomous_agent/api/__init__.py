"""API modules."""

from api.utils import handle_api_errors, validate_github_token, create_error_response
from api.dependencies import get_storage, get_github_client, get_task_processor, get_worker_manager
from api.validators import validate_repository, validate_instruction, validate_task_id
from api.schemas import (
    CreateTaskRequest,
    TaskResponse,
    RepositoryInfoRequest,
    RepositoryInfoResponse,
    AgentControlRequest,
    AgentStatusResponse,
    WorkerMetricsResponse,
    AgentMetricsResponse
)
from api.response_models import (
    ErrorResponse,
    SuccessResponse,
    HealthResponse
)

__all__ = [
    "handle_api_errors",
    "validate_github_token",
    "create_error_response",
    "get_storage",
    "get_github_client",
    "get_task_processor",
    "get_worker_manager",
    "validate_repository",
    "validate_instruction",
    "validate_task_id",
    "CreateTaskRequest",
    "TaskResponse",
    "RepositoryInfoRequest",
    "RepositoryInfoResponse",
    "AgentControlRequest",
    "AgentStatusResponse",
    "WorkerMetricsResponse",
    "AgentMetricsResponse",
    "ErrorResponse",
    "SuccessResponse",
    "HealthResponse",
]
