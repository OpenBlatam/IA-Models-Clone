"""
Core module for AI Tutor Educacional.
"""

from .tutor import AITutor
from .conversation_manager import ConversationManager
from .learning_analyzer import LearningAnalyzer
from .cache_manager import CacheManager
from .rate_limiter import RateLimiter
from .metrics_collector import MetricsCollector, TutorMetrics
from .quiz_generator import QuizGenerator
from .report_generator import ReportGenerator
from .gamification import GamificationSystem, BadgeType, Badge
from .evaluator import AnswerEvaluator, EvaluationResult
from .recommendation_engine import RecommendationEngine
from .notification_system import NotificationSystem, NotificationType, Notification
from .dashboard_analytics import DashboardAnalytics
from .database import DatabaseManager
from .auth import AuthManager, User
from .webhooks import WebhookManager, WebhookEvent, Webhook
from .lms_integration import LMSIntegration, LMSType
from .api_versioning import APIVersionManager, APIVersion
from .advanced_validation import AdvancedValidator, ValidationError
from .advanced_logging import AdvancedLogger, StructuredFormatter
from .data_export import DataExporter
from .performance_optimizer import PerformanceOptimizer, timing_decorator
from .batch_processor import BatchProcessor
from .error_handler import ErrorHandler, ErrorSeverity
from .scheduler import TaskScheduler, TaskPriority
from .security_manager import SecurityManager
from .monitoring import SystemMonitor, SystemHealth
from .backup_manager import BackupManager
from .analytics_engine import AnalyticsEngine
from .content_generator import ContentGenerator, ContentType
from .adaptive_learning import AdaptiveLearningEngine, LearningStyle, DifficultyLevel
from .collaboration import CollaborationManager, CollaborationType
from .assessment_system import AssessmentSystem, AssessmentType, QuestionType
from .feedback_system import FeedbackSystem, FeedbackType

__all__ = [
    "AITutor",
    "ConversationManager",
    "LearningAnalyzer",
    "CacheManager",
    "RateLimiter",
    "MetricsCollector",
    "TutorMetrics",
    "QuizGenerator",
    "ReportGenerator",
    "GamificationSystem",
    "BadgeType",
    "Badge",
    "AnswerEvaluator",
    "EvaluationResult",
    "RecommendationEngine",
    "NotificationSystem",
    "NotificationType",
    "Notification",
    "DashboardAnalytics",
    "DatabaseManager",
    "AuthManager",
    "User",
    "WebhookManager",
    "WebhookEvent",
    "Webhook",
    "LMSIntegration",
    "LMSType",
    "APIVersionManager",
    "APIVersion",
    "AdvancedValidator",
    "ValidationError",
    "AdvancedLogger",
    "StructuredFormatter",
    "DataExporter",
    "PerformanceOptimizer",
    "timing_decorator",
    "BatchProcessor",
    "ErrorHandler",
    "ErrorSeverity",
    "TaskScheduler",
    "TaskPriority",
    "SecurityManager",
    "SystemMonitor",
    "SystemHealth",
    "BackupManager",
    "AnalyticsEngine",
    "ContentGenerator",
    "ContentType",
    "AdaptiveLearningEngine",
    "LearningStyle",
    "DifficultyLevel",
    "CollaborationManager",
    "CollaborationType",
    "AssessmentSystem",
    "AssessmentType",
    "QuestionType",
    "FeedbackSystem",
    "FeedbackType",
]
