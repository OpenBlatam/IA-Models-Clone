"""Utility modules for AI Project Generator"""

from .github_integration import GitHubIntegration
from .test_generator import TestGenerator
from .cicd_generator import CICDGenerator
from .export_generator import ExportGenerator
from .validator import ProjectValidator
from .deployment_generator import DeploymentGenerator
from .project_cloner import ProjectCloner
from .template_manager import TemplateManager
from .search_engine import ProjectSearchEngine
from .cache_manager import CacheManager
from .webhook_manager import WebhookManager
from .rate_limiter import RateLimiter
from .auth_manager import AuthManager
from .metrics_collector import MetricsCollector
from .backup_manager import BackupManager
from .dashboard_generator import DashboardGenerator
from .api_versioning import APIVersionManager
from .health_checker import AdvancedHealthChecker
from .notification_service import NotificationService
from .plugin_system import PluginSystem
from .event_system import EventSystem
from .logging_config import AdvancedLoggingConfig
from .performance_optimizer import (
    ProjectCache,
    ParallelProjectProcessor,
    GenerationOptimizer,
    SmartBatchProcessor,
    cached_generation,
    get_project_cache,
    get_generation_optimizer
)
from .realtime_streaming import (
    StreamEventType,
    StreamEvent,
    StreamManager,
    ProjectStreamer,
    QueueStreamer,
    StatsStreamer,
    get_stream_manager
)
from .performance_analyzer import (
    PerformanceAnalyzer,
    TimePredictor,
    ResourceMonitor,
    get_performance_analyzer,
    get_time_predictor,
    get_resource_monitor
)
from .auto_scaler import (
    AutoScaler,
    ScaleAction,
    ScalingDecision,
    get_auto_scaler
)
from .intelligent_alerts import (
    IntelligentAlertSystem,
    Alert,
    AlertSeverity,
    AlertStatus,
    get_alert_system
)
from .code_optimizer import (
    CodeOptimizer,
    OptimizationType,
    OptimizationSuggestion,
    get_code_optimizer
)
from .code_quality_analyzer import (
    CodeQualityAnalyzer,
    QualityMetric,
    QualityScore,
    FunctionMetrics,
    get_code_quality_analyzer
)
from .auto_test_generator import (
    AutoTestGenerator,
    TestType,
    TestCase,
    get_auto_test_generator
)
from .dependency_analyzer import (
    DependencyAnalyzer,
    DependencyType,
    Dependency,
    get_dependency_analyzer
)
from .auto_refactor import (
    AutoRefactor,
    RefactorType,
    RefactorSuggestion,
    get_auto_refactor
)
from .performance_analyzer import (
    PerformanceAnalyzer,
    PerformanceIssue,
    PerformanceIssueReport,
    get_performance_analyzer_code
)
from .advanced_bug_detector import (
    AdvancedBugDetector,
    BugType,
    BugReport,
    get_advanced_bug_detector
)
from .architecture_analyzer import (
    ArchitectureAnalyzer,
    ArchitecturePattern,
    ArchitectureIssue,
    get_architecture_analyzer
)
from .code_standards_validator import (
    CodeStandardsValidator,
    StandardType,
    StandardViolation,
    get_code_standards_validator
)
from .design_suggester import (
    DesignSuggester,
    DesignSuggestionType,
    DesignSuggestion,
    get_design_suggester
)
from .analytics_engine import AnalyticsEngine
from .recommendation_engine import RecommendationEngine
from .project_versioning import ProjectVersioning
from .collaboration_system import CollaborationSystem
from .auto_documentation import AutoDocumentation
from .alert_system import AlertSystem, AlertLevel
from .scheduler import TaskScheduler
from .import_export import AdvancedImportExport
from .ml_predictor import MLPredictor
from .auto_optimizer import AutoOptimizer
from .advanced_testing import AdvancedTesting
from .auto_deployment import AutoDeployment
from .advanced_security import AdvancedSecurity
from .code_quality_analyzer import CodeQualityAnalyzer
from .intelligent_suggestions import IntelligentSuggestions
from .benchmark_system import BenchmarkSystem
from .advanced_metrics import AdvancedMetrics

__all__ = [
    "GitHubIntegration",
    "TestGenerator",
    "CICDGenerator",
    "ExportGenerator",
    "ProjectValidator",
    "DeploymentGenerator",
    "ProjectCloner",
    "TemplateManager",
    "ProjectSearchEngine",
    "CacheManager",
    "WebhookManager",
    "RateLimiter",
    "AuthManager",
    "MetricsCollector",
    "BackupManager",
    "DashboardGenerator",
    "APIVersionManager",
    "AdvancedHealthChecker",
    "NotificationService",
    "PluginSystem",
    "EventSystem",
    "AdvancedLoggingConfig",
    # Performance optimization
    "ProjectCache",
    "ParallelProjectProcessor",
    "GenerationOptimizer",
    "SmartBatchProcessor",
    "cached_generation",
    "get_project_cache",
    "get_generation_optimizer",
    # Real-time streaming
    "StreamEventType",
    "StreamEvent",
    "StreamManager",
    "ProjectStreamer",
    "QueueStreamer",
    "StatsStreamer",
    "get_stream_manager",
    # Analytics
    "AnalyticsEngine",
    # Recommendations
    "RecommendationEngine",
    # Versioning
    "ProjectVersioning",
    # Collaboration
    "CollaborationSystem",
    # Documentation
    "AutoDocumentation",
    # Alerts
    "AlertSystem",
    "AlertLevel",
    # Scheduling
    "TaskScheduler",
    # Import/Export
    "AdvancedImportExport",
    # ML
    "MLPredictor",
    # Auto Optimization
    "AutoOptimizer",
    # Advanced Testing
    "AdvancedTesting",
    # Auto Deployment
    "AutoDeployment",
    # Advanced Security
    "AdvancedSecurity",
    # Code Quality
    "CodeQualityAnalyzer",
    # Intelligent Suggestions
    "IntelligentSuggestions",
    # Benchmark System
    "BenchmarkSystem",
    # Advanced Metrics
    "AdvancedMetrics",
    # Performance Analysis
    "PerformanceAnalyzer",
    "TimePredictor",
    "ResourceMonitor",
    "get_performance_analyzer",
    "get_time_predictor",
    "get_resource_monitor",
    # Auto Scaling
    "AutoScaler",
    "ScaleAction",
    "ScalingDecision",
    "get_auto_scaler",
    # Intelligent Alerts
    "IntelligentAlertSystem",
    "Alert",
    "AlertSeverity",
    "AlertStatus",
    "get_alert_system",
    # Code Optimization
    "CodeOptimizer",
    "OptimizationType",
    "OptimizationSuggestion",
    "get_code_optimizer",
    # Code Quality Analysis
    "CodeQualityAnalyzer",
    "QualityMetric",
    "QualityScore",
    "FunctionMetrics",
    "get_code_quality_analyzer",
    # Auto Test Generation
    "AutoTestGenerator",
    "TestType",
    "TestCase",
    "get_auto_test_generator",
    # Dependency Analysis
    "DependencyAnalyzer",
    "DependencyType",
    "Dependency",
    "get_dependency_analyzer",
    # Auto Refactor
    "AutoRefactor",
    "RefactorType",
    "RefactorSuggestion",
    "get_auto_refactor",
    # Performance Analysis
    "PerformanceAnalyzer",
    "PerformanceIssue",
    "PerformanceIssueReport",
    "get_performance_analyzer_code",
    # Advanced Bug Detection
    "AdvancedBugDetector",
    "BugType",
    "BugReport",
    "get_advanced_bug_detector",
    # Architecture Analysis
    "ArchitectureAnalyzer",
    "ArchitecturePattern",
    "ArchitectureIssue",
    "get_architecture_analyzer",
    # Code Standards Validation
    "CodeStandardsValidator",
    "StandardType",
    "StandardViolation",
    "get_code_standards_validator",
    # Design Suggestions
    "DesignSuggester",
    "DesignSuggestionType",
    "DesignSuggestion",
    "get_design_suggester",
    # Security Analysis
    "AdvancedSecurityAnalyzer",
    "SecurityVulnerability",
    "SecurityIssue",
    "get_advanced_security_analyzer",
    # Code Smell Detection
    "CodeSmellDetector",
    "CodeSmellType",
    "CodeSmell",
    "get_code_smell_detector",
    # Cognitive Complexity
    "CognitiveComplexityAnalyzer",
    "ComplexityMetric",
    "get_cognitive_complexity_analyzer",
    # Deep Learning Generator
    "DeepLearningGenerator",
    "ModelType",
    "TrainingConfig",
    "ModelArchitecture",
    "TrainingSetup",
    "get_deep_learning_generator",
]

