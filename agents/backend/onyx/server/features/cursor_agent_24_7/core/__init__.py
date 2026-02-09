"""
Core modules for Cursor Agent 24/7
"""

from .agent import CursorAgent, AgentStatus, AgentConfig, Task
from .task_executor import TaskExecutor, TaskStatus, ExecutionResult
from .command_listener import CommandListener
from .persistent_service import PersistentService
from .file_watcher import FileWatcher
from .command_executor import CommandExecutor
from .websocket_handler import WebSocketManager
from .notifications import NotificationManager, NotificationLevel, Notification
from .metrics import MetricsCollector, Timer
from .health_check import HealthChecker, HealthStatus, HealthCheck
from .rate_limiter import RateLimiter, TaskRateLimiter
from .exporters import DataExporter
from .scheduler import TaskScheduler, ScheduleType, ScheduledTask
from .backup import BackupManager
from .plugins import PluginManager, BasePlugin
from .auth import AuthManager, User, Role
from .cache import Cache, CommandCache, CacheEntry
from .templates import TemplateManager, CommandTemplate
from .validators import CommandValidator, InputValidator, ValidationResult
from .event_bus import EventBus, EventType, Event
from .cluster import ClusterManager, ClusterNode, NodeStatus

# AI Components
try:
    from .ai_processor import AIProcessor, ProcessedCommand, CommandIntent
except ImportError:
    AIProcessor = None
    ProcessedCommand = None
    CommandIntent = None

try:
    from .embeddings import EmbeddingStore
except ImportError:
    EmbeddingStore = None

try:
    from .pattern_learner import PatternLearner
except ImportError:
    PatternLearner = None

try:
    from .llm_pipeline import LLMPipeline, LLMConfig, FineTuner
except ImportError:
    LLMPipeline = None
    LLMConfig = None
    FineTuner = None

try:
    from .gradio_interface import GradioInterface
except ImportError:
    GradioInterface = None

try:
    from .middleware import (
        LoggingMiddleware,
        RateLimitMiddleware,
        SecurityHeadersMiddleware,
        ErrorHandlingMiddleware
    )
    from .logger_config import setup_logging, get_logger
except ImportError:
    LoggingMiddleware = None
    RateLimitMiddleware = None
    SecurityHeadersMiddleware = None
    ErrorHandlingMiddleware = None
    setup_logging = None
    get_logger = None

try:
    from .alerting import AlertManager, AlertRule, Alert, AlertSeverity, AlertCondition
except ImportError:
    AlertManager = None
    AlertRule = None
    Alert = None
    AlertSeverity = None
    AlertCondition = None

try:
    from .config_manager import ConfigManager
except ImportError:
    ConfigManager = None

# Perplexity query processing system
try:
    from .perplexity import (
        PerplexityProcessor,
        QueryType,
        QueryTypeDetector,
        ResponseFormatter,
        PromptBuilder,
        CitationManager,
        SearchResult,
        ProcessedQuery,
        PerplexityValidator,
        ValidationIssue,
        ValidationLevel
    )
except ImportError:
    PerplexityProcessor = None
    QueryType = None
    QueryTypeDetector = None
    ResponseFormatter = None
    PromptBuilder = None
    CitationManager = None
    SearchResult = None
    ProcessedQuery = None
    PerplexityValidator = None
    ValidationIssue = None
    ValidationLevel = None

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
    "PerplexityProcessor",
    "QueryType",
    "QueryTypeDetector",
    "ResponseFormatter",
    "PromptBuilder",
    "CitationManager",
    "SearchResult",
    "ProcessedQuery",
    "PerplexityValidator",
    "ValidationIssue",
    "ValidationLevel",
]
