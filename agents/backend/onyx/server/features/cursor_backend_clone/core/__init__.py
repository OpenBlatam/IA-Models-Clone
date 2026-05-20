"""
Core modules for Cursor Agent 24/7
"""

from .agent import CursorAgent
from .domain.agent import AgentStatus, AgentConfig, Task
from .task_executor import TaskExecutor, TaskStatus, ExecutionResult
from .command_listener import CommandListener
from .services.persistent_service import PersistentService
from .services.file_watcher import FileWatcher
from .command_executor import CommandExecutor
from .infrastructure.messaging.websocket import WebSocketManager
from .infrastructure.messaging.notifications import NotificationManager, NotificationLevel, Notification
from .infrastructure.monitoring.metrics import MetricsCollector, Timer
from .infrastructure.monitoring.health import HealthChecker, HealthStatus, HealthCheck
from .utils.rate_limiting.rate_limiter import RateLimiter, TaskRateLimiter
from .services.exporters import DataExporter
from .infrastructure.scheduling.scheduler import TaskScheduler, ScheduleType, ScheduledTask
from .infrastructure.persistence.backup import BackupManager
from .infrastructure.plugins.plugins import PluginManager, BasePlugin
from .infrastructure.security.auth import AuthManager, User, Role
from .infrastructure.caching.cache import Cache, CommandCache, CacheEntry
from .utils.templates.templates import TemplateManager, CommandTemplate
from .utils.validation.validators import CommandValidator, InputValidator, ValidationResult
from .infrastructure.messaging.event_bus import EventBus, EventType, Event
from .infrastructure.clustering.cluster import ClusterManager, ClusterNode, NodeStatus

# AI Components
try:
    from .ai.ai_processor import AIProcessor, ProcessedCommand, CommandIntent
except ImportError:
    AIProcessor = None
    ProcessedCommand = None
    CommandIntent = None

try:
    from .ai.embeddings import EmbeddingStore
except ImportError:
    EmbeddingStore = None

try:
    from .ai.pattern_learner import PatternLearner
except ImportError:
    PatternLearner = None

try:
    from .ai.llm_pipeline import LLMPipeline, LLMConfig, FineTuner
except ImportError:
    LLMPipeline = None
    LLMConfig = None
    FineTuner = None

try:
    from .gradio_interface import GradioInterface
except ImportError:
    GradioInterface = None

try:
    from .utils.middleware.middleware import (
        LoggingMiddleware,
        RateLimitMiddleware,
        SecurityHeadersMiddleware,
        ErrorHandlingMiddleware
    )
    from .utils.logging.logger_config import setup_logging, get_logger
except ImportError:
    LoggingMiddleware = None
    RateLimitMiddleware = None
    SecurityHeadersMiddleware = None
    ErrorHandlingMiddleware = None
    setup_logging = None
    get_logger = None

try:
    from .utils.alerts.alerting import AlertManager, AlertRule, Alert, AlertSeverity, AlertCondition
except ImportError:
    AlertManager = None
    AlertRule = None
    Alert = None
    AlertSeverity = None
    AlertCondition = None

try:
    from .utils.config.config_manager import ConfigManager
except ImportError:
    ConfigManager = None

__all__ = [
    "CursorAgent",
    "AgentStatus",
    "AgentConfig",
    "Task",
    "TaskExecutor",
    "TaskStatus",
    "ExecutionResult",
    "CommandListener",
    "PersistentService",
    "FileWatcher",
    "CommandExecutor",
    "WebSocketManager",
    "NotificationManager",
    "NotificationLevel",
    "Notification",
    "MetricsCollector",
    "Timer",
    "HealthChecker",
    "HealthStatus",
    "HealthCheck",
    "RateLimiter",
    "TaskRateLimiter",
    "DataExporter",
    "TaskScheduler",
    "ScheduleType",
    "ScheduledTask",
    "BackupManager",
    "PluginManager",
    "BasePlugin",
    "AuthManager",
    "User",
    "Role",
    "Cache",
    "CommandCache",
    "CacheEntry",
    "TemplateManager",
    "CommandTemplate",
    "CommandValidator",
    "InputValidator",
    "ValidationResult",
    "EventBus",
    "EventType",
    "Event",
    "ClusterManager",
    "ClusterNode",
    "NodeStatus",
    "AlertManager",
    "AlertRule",
    "Alert",
    "AlertSeverity",
    "AlertCondition",
    "ConfigManager",
    "LoggingMiddleware",
    "RateLimitMiddleware",
    "SecurityHeadersMiddleware",
    "ErrorHandlingMiddleware",
    "setup_logging",
    "get_logger",
    "AIProcessor",
    "ProcessedCommand",
    "CommandIntent",
    "EmbeddingStore",
    "PatternLearner",
    "LLMPipeline",
    "LLMConfig",
    "FineTuner",
    "GradioInterface",
]
