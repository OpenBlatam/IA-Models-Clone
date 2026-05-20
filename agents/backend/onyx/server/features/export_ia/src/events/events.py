"""
Event definitions for Export IA event sourcing.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from ..domains.export.value_objects import ExportFormat, DocumentType, QualityLevel


@dataclass
class BaseEvent(ABC):
    """Base class for all events."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    aggregate_id: str = ""
    event_type: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set event type after initialization."""
        if not self.event_type:
            self.event_type = self.__class__.__name__


# Export Events
@dataclass
class ExportRequested(BaseEvent):
    """Event raised when an export is requested."""
    content: Dict[str, Any] = field(default_factory=dict)
    format: ExportFormat = ExportFormat.PDF
    document_type: DocumentType = DocumentType.REPORT
    quality_level: QualityLevel = QualityLevel.PROFESSIONAL
    user_id: Optional[str] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ExportStarted(BaseEvent):
    """Event raised when export processing starts."""
    task_id: str = ""
    estimated_duration: Optional[float] = None


@dataclass
class ExportProgressUpdated(BaseEvent):
    """Event raised when export progress is updated."""
    task_id: str = ""
    progress: float = 0.0
    current_step: str = ""
    estimated_remaining: Optional[float] = None


@dataclass
class ExportCompleted(BaseEvent):
    """Event raised when export is completed successfully."""
    task_id: str = ""
    file_path: str = ""
    file_size: int = 0
    quality_score: Optional[float] = None
    processing_time: float = 0.0
    warnings: List[str] = field(default_factory=list)


@dataclass
class ExportFailed(BaseEvent):
    """Event raised when export fails."""
    task_id: str = ""
    error_message: str = ""
    error_code: Optional[str] = None
    retry_count: int = 0


@dataclass
class ExportCancelled(BaseEvent):
    """Event raised when export is cancelled."""
    task_id: str = ""
    reason: str = ""
    cancelled_by: Optional[str] = None


# Task Events
@dataclass
class TaskCreated(BaseEvent):
    """Event raised when a task is created."""
    task_id: str = ""
    task_type: str = ""
    priority: int = 0
    scheduled_at: Optional[datetime] = None


@dataclass
class TaskUpdated(BaseEvent):
    """Event raised when a task is updated."""
    task_id: str = ""
    status: str = ""
    progress: float = 0.0
    updated_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskAssigned(BaseEvent):
    """Event raised when a task is assigned."""
    task_id: str = ""
    assigned_to: str = ""
    assigned_by: Optional[str] = None


@dataclass
class TaskCompleted(BaseEvent):
    """Event raised when a task is completed."""
    task_id: str = ""
    completion_time: datetime = field(default_factory=datetime.now)
    result: Dict[str, Any] = field(default_factory=dict)


# Quality Events
@dataclass
class QualityValidationStarted(BaseEvent):
    """Event raised when quality validation starts."""
    task_id: str = ""
    validation_type: str = ""


@dataclass
class QualityValidated(BaseEvent):
    """Event raised when quality validation is completed."""
    task_id: str = ""
    overall_score: float = 0.0
    formatting_score: float = 0.0
    content_score: float = 0.0
    accessibility_score: float = 0.0
    professional_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class QualityEnhancementApplied(BaseEvent):
    """Event raised when quality enhancement is applied."""
    task_id: str = ""
    enhancement_type: str = ""
    improvements: List[str] = field(default_factory=list)
    confidence_score: float = 0.0


# User Events
@dataclass
class UserAuthenticated(BaseEvent):
    """Event raised when user is authenticated."""
    user_id: str = ""
    username: str = ""
    authentication_method: str = ""
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class UserSessionStarted(BaseEvent):
    """Event raised when user session starts."""
    user_id: str = ""
    session_id: str = ""
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class UserSessionEnded(BaseEvent):
    """Event raised when user session ends."""
    user_id: str = ""
    session_id: str = ""
    duration: float = 0.0
    reason: str = ""


@dataclass
class UserPermissionGranted(BaseEvent):
    """Event raised when user permission is granted."""
    user_id: str = ""
    permission: str = ""
    granted_by: str = ""
    expires_at: Optional[datetime] = None


@dataclass
class UserPermissionRevoked(BaseEvent):
    """Event raised when user permission is revoked."""
    user_id: str = ""
    permission: str = ""
    revoked_by: str = ""
    reason: str = ""


# System Events
@dataclass
class SystemHealthCheck(BaseEvent):
    """Event raised during system health check."""
    service_name: str = ""
    status: str = "healthy"
    response_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    disk_usage: float = 0.0


@dataclass
class SystemAlert(BaseEvent):
    """Event raised when system alert is triggered."""
    alert_type: str = ""
    severity: str = "info"
    message: str = ""
    service_name: Optional[str] = None
    threshold: Optional[float] = None
    current_value: Optional[float] = None


@dataclass
class SystemMaintenance(BaseEvent):
    """Event raised during system maintenance."""
    maintenance_type: str = ""
    description: str = ""
    scheduled_start: datetime = field(default_factory=datetime.now)
    scheduled_end: Optional[datetime] = None
    affected_services: List[str] = field(default_factory=list)


# Plugin Events
@dataclass
class PluginLoaded(BaseEvent):
    """Event raised when plugin is loaded."""
    plugin_name: str = ""
    plugin_type: str = ""
    version: str = ""
    author: str = ""


@dataclass
class PluginEnabled(BaseEvent):
    """Event raised when plugin is enabled."""
    plugin_name: str = ""
    enabled_by: Optional[str] = None


@dataclass
class PluginDisabled(BaseEvent):
    """Event raised when plugin is disabled."""
    plugin_name: str = ""
    disabled_by: Optional[str] = None
    reason: str = ""


@dataclass
class PluginError(BaseEvent):
    """Event raised when plugin error occurs."""
    plugin_name: str = ""
    error_message: str = ""
    error_type: str = ""
    stack_trace: Optional[str] = None


# Workflow Events
@dataclass
class WorkflowStarted(BaseEvent):
    """Event raised when workflow starts."""
    workflow_id: str = ""
    workflow_name: str = ""
    triggered_by: Optional[str] = None
    input_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowStepCompleted(BaseEvent):
    """Event raised when workflow step is completed."""
    workflow_id: str = ""
    step_id: str = ""
    step_name: str = ""
    result: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0


@dataclass
class WorkflowCompleted(BaseEvent):
    """Event raised when workflow is completed."""
    workflow_id: str = ""
    workflow_name: str = ""
    success: bool = True
    total_duration: float = 0.0
    output_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowFailed(BaseEvent):
    """Event raised when workflow fails."""
    workflow_id: str = ""
    workflow_name: str = ""
    error_message: str = ""
    failed_step: Optional[str] = None
    retry_count: int = 0


# Analytics Events
@dataclass
class MetricRecorded(BaseEvent):
    """Event raised when metric is recorded."""
    metric_name: str = ""
    metric_value: float = 0.0
    metric_unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceThresholdExceeded(BaseEvent):
    """Event raised when performance threshold is exceeded."""
    metric_name: str = ""
    current_value: float = 0.0
    threshold: float = 0.0
    service_name: Optional[str] = None
    severity: str = "warning"


@dataclass
class UsageStatisticsUpdated(BaseEvent):
    """Event raised when usage statistics are updated."""
    user_id: Optional[str] = None
    service_name: str = ""
    usage_count: int = 0
    total_duration: float = 0.0
    period_start: datetime = field(default_factory=datetime.now)
    period_end: datetime = field(default_factory=datetime.now)




