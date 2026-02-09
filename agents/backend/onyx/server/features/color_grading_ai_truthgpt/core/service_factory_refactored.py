"""
Refactored Service Factory for Color Grading AI
================================================

Improved service factory with better organization and dependency management.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from ..config.color_grading_config import ColorGradingConfig
from ..services.video_processor import VideoProcessor
from ..services.image_processor import ImageProcessor
from ..services.color_analyzer import ColorAnalyzer
from ..services.color_matcher import ColorMatcher
from ..services.template_manager import TemplateManager
from ..services.cache_unified import UnifiedCache, CacheStrategy
from ..services.batch_processor import BatchProcessor
from ..services.webhook_manager import WebhookManager
from ..services.metrics_collector import MetricsCollector
from ..services.comparison_generator import ComparisonGenerator
from ..services.lut_manager import LUTManager
from ..services.queue_unified import UnifiedQueue
from ..services.parameter_exporter import ParameterExporter
from ..services.history_manager import HistoryManager
from ..services.performance_optimizer import PerformanceOptimizer
from ..services.video_quality_analyzer import VideoQualityAnalyzer
from ..services.preset_manager import PresetManager
from ..services.backup_manager import BackupManager
from ..services.notification_service import NotificationService
from ..services.version_manager import VersionManager
from ..services.cloud_integration import CloudIntegrationManager
from ..services.recommendation_engine import RecommendationEngine
from ..services.workflow_manager import WorkflowManager
from ..services.collaboration_manager import CollaborationManager
from ..services.analytics_service import AnalyticsService
from ..services.event_bus import EventBus
from ..services.ml_optimizer import MLOptimizer
from ..services.security_manager import SecurityManager
from ..services.optimization_engine import OptimizationEngine
from ..services.telemetry_service import TelemetryService
from ..services.circuit_breaker import CircuitBreaker
from ..services.retry_manager import RetryManager
from ..services.load_balancer import LoadBalancer, LoadBalanceStrategy
from ..services.feature_flags import FeatureFlagManager
from ..services.rate_limiter import RateLimiter, RateLimitConfig, RateLimitAlgorithm
from ..services.throttle_manager import ThrottleManager, ThrottleConfig
from ..services.backpressure import BackpressureManager, BackpressureConfig
from ..services.health_monitor import HealthMonitor
from ..services.graceful_shutdown import GracefulShutdownManager
from ..services.lifecycle_manager import LifecycleManager
from ..services.audit_logger import AuditLogger
from ..services.compliance_manager import ComplianceManager
from ..services.experiment_manager import ExperimentManager
from ..services.analytics_dashboard import AnalyticsDashboard
from ..services.performance_monitor import PerformanceMonitor
from ..services.unified_performance_manager import UnifiedPerformanceManager, PerformanceMode
from ..services.advanced_caching import AdvancedCache, CacheStrategy as AdvancedCacheStrategy
from ..services.performance_profiler import PerformanceProfiler, ProfilerMode
from ..services.service_discovery import ServiceDiscovery
from ..services.config_validator import ConfigValidator, ValidationLevel
from ..services.metrics_aggregator import MetricsAggregator
from ..services.unified_optimization_system import UnifiedOptimizationSystem, OptimizationMode
from ..services.unified_analytics_system import UnifiedAnalyticsSystem
from ..services.service_orchestrator import ServiceOrchestrator, OrchestrationStrategy
from ..services.event_scheduler import EventScheduler, ScheduleType
from ..services.api_gateway import APIGateway, RouteMethod
from ..services.data_pipeline import DataPipeline, PipelineStage
from ..services.unified_resource_manager import UnifiedResourceManager, ResourceType
from ..services.unified_communication_system import UnifiedCommunicationSystem, CommunicationChannel
from ..services.unified_batch_system import UnifiedBatchSystem, BatchMode
from ..services.test_runner import TestRunner
from ..services.documentation_generator import DocumentationGenerator
from ..services.transformation_engine import TransformationEngine
from ..services.performance_benchmark import PerformanceBenchmark
from ..services.unified_processing_system import UnifiedProcessingSystem, MediaType
from ..services.unified_workflow_system import UnifiedWorkflowSystem, WorkflowMode
from ..services.unified_export_system import UnifiedExportSystem, ExportFormat
from ..services.validation_framework import ValidationFramework, ValidationLevel as ValidationFrameworkLevel
from ..services.monitoring_dashboard import MonitoringDashboard
from ..services.error_recovery import ErrorRecoverySystem, RecoveryStrategy
from ..services.cost_optimizer import CostOptimizer, CostType
from ..services.security_auditor import SecurityAuditor, SecurityLevel
from ..services.unified_security_system import UnifiedSecuritySystem, SecurityMode
from ..services.unified_monitoring_system import UnifiedMonitoringSystem, MonitoringMode
from ..services.ai_model_manager import AIModelManager
from ..services.feature_toggle import FeatureToggleSystem, ToggleType
from ..services.cache_warming import CacheWarmingSystem, WarmingStrategy

logger = logging.getLogger(__name__)


class RefactoredServiceFactory:
    """
    Refactored service factory with improved organization.
    
    Improvements:
    - Better categorization
    - Dependency injection
    - Lazy loading
    - Service groups
    """
    
    def __init__(self, config: ColorGradingConfig, output_dirs: Dict[str, Path]):
        """
        Initialize refactored service factory.
        
        Args:
            config: Configuration
            output_dirs: Output directories dictionary
        """
        self.config = config
        self.output_dirs = output_dirs
        self._services: Dict[str, Any] = {}
        self._initialized = False
    
    def _get_storage_path(self, subdir: str) -> str:
        """Get storage subdirectory path."""
        return str(self.output_dirs["storage"] / subdir)
    
    def initialize_all(self) -> Dict[str, Any]:
        """
        Initialize all services in correct order.
        
        Returns:
            Dictionary of all services
        """
        if self._initialized:
            return self._services
        
        # 1. Core infrastructure (no dependencies)
        self._init_infrastructure()
        
        # 2. Processing services (no dependencies)
        self._init_processing()
        
        # 3. Management services (may depend on infrastructure)
        self._init_management()
        
        # 4. Support services (may depend on management)
        self._init_support()
        
        # 5. Advanced services (depend on multiple services)
        self._init_advanced()
        
        self._initialized = True
        return self._services
    
    def _init_infrastructure(self):
        """Initialize infrastructure services."""
        self._services.update({
            "event_bus": EventBus(),
            "security_manager": SecurityManager(),
            "telemetry_service": TelemetryService(
                storage_dir=self._get_storage_path("telemetry")
            ),
            "ml_optimizer": MLOptimizer(),
            "optimization_engine": OptimizationEngine(),
            # Unified security system (consolidates security services)
            "unified_security_system": UnifiedSecuritySystem(
                security_mode=SecurityMode.STANDARD
            ),
        })
    
    def _init_processing(self):
        """Initialize processing services."""
        # Individual processors (for backward compatibility)
        self._services.update({
            "video_processor": VideoProcessor(
                ffmpeg_path=self.config.video_processing.ffmpeg_path
            ),
            "image_processor": ImageProcessor(),
            "color_analyzer": ColorAnalyzer(
                histogram_bins=self.config.color_analysis.histogram_bins,
                color_space=self.config.color_analysis.color_space
            ),
            "color_matcher": ColorMatcher(),
            "video_quality_analyzer": VideoQualityAnalyzer(),
        })
        
        # Unified processing system (consolidates all processors)
        self._services["unified_processing_system"] = UnifiedProcessingSystem(
            ffmpeg_path=self.config.video_processing.ffmpeg_path,
            histogram_bins=self.config.color_analysis.histogram_bins,
            color_space=self.config.color_analysis.color_space
        )
    
    def _init_management(self):
        """Initialize management services."""
        self._services.update({
            "cache_manager": UnifiedCache(
                cache_dir=str(self.output_dirs["cache"]),
                ttl=self.config.cache_ttl if self.config.enable_cache else 0,
                max_size=1000,
                strategy=CacheStrategy.LRU,
                redis_url=getattr(self.config, "redis_url", None)
            ),
            "template_manager": TemplateManager(
                templates_dir=self._get_storage_path("templates")
            ),
            "preset_manager": PresetManager(
                presets_dir=self._get_storage_path("presets")
            ),
            "lut_manager": LUTManager(
                luts_dir=self._get_storage_path("luts")
            ),
            "history_manager": HistoryManager(
                history_dir=self._get_storage_path("history")
            ),
            "version_manager": VersionManager(
                versions_dir=self._get_storage_path("versions")
            ),
            "backup_manager": BackupManager(
                backup_dir=self._get_storage_path("backups")
            ),
            # Unified resource manager (consolidates all resource managers)
            "unified_resource_manager": UnifiedResourceManager(
                base_dir=str(self.output_dirs["storage"]),
                templates_dir=self._get_storage_path("templates"),
                presets_dir=self._get_storage_path("presets"),
                luts_dir=self._get_storage_path("luts"),
                versions_dir=self._get_storage_path("versions"),
                history_dir=self._get_storage_path("history"),
                backups_dir=self._get_storage_path("backups")
            ),
            # AI Model Manager
            "ai_model_manager": AIModelManager(
                models_dir=self._get_storage_path("models")
            ),
        })
    
    def _init_support(self):
        """Initialize support services."""
        self._services.update({
            "task_queue": UnifiedQueue(
                max_workers=self.config.max_parallel_tasks,
                storage_dir=self._get_storage_path("queue")
            ),
            "batch_processor": BatchProcessor(
                max_parallel=self.config.max_parallel_tasks
            ),
            "webhook_manager": WebhookManager(),
            "notification_service": NotificationService(),
            # Unified batch system (consolidates batch processing)
            "unified_batch_system": UnifiedBatchSystem(
                max_parallel=self.config.max_parallel_tasks,
                default_mode=BatchMode.BASIC
            ),
            # Unified communication system (consolidates communication services)
            "unified_communication_system": UnifiedCommunicationSystem(),
            "metrics_collector": MetricsCollector(
                metrics_dir=self._get_storage_path("metrics")
            ),
            "performance_monitor": PerformanceMonitor(),
            "performance_optimizer": PerformanceOptimizer(),
            "unified_performance_manager": UnifiedPerformanceManager(
                mode=PerformanceMode.OPTIMIZATION,
                enable_profiling=False
            ),
            "performance_profiler": PerformanceProfiler(mode=ProfilerMode.SIMPLE),
            "advanced_cache": AdvancedCache(
                max_size=1000,
                strategy=AdvancedCacheStrategy.LRU
            ),
            "comparison_generator": ComparisonGenerator(),
            "parameter_exporter": ParameterExporter(),
            # Unified export system (consolidates export services)
            "unified_export_system": UnifiedExportSystem(
                output_dir=self._get_storage_path("exports")
            ),
            "collaboration_manager": CollaborationManager(),
            "workflow_manager": WorkflowManager(
                workflows_dir=self._get_storage_path("workflows")
            ),
            "cloud_integration": CloudIntegrationManager(),
            # Resilience patterns
            "circuit_breaker": CircuitBreaker("default"),
            "retry_manager": RetryManager(),
            "load_balancer": LoadBalancer(strategy=LoadBalanceStrategy.LEAST_LOAD),
            "feature_flags": FeatureFlagManager(),
            # Feature toggle system
            "feature_toggle": FeatureToggleSystem(),
            # Traffic control
            "rate_limiter": RateLimiter(
                config=RateLimitConfig(
                    max_requests=100,
                    window_seconds=60.0,
                    algorithm=RateLimitAlgorithm.TOKEN_BUCKET
                )
            ),
            "throttle_manager": ThrottleManager(
                config=ThrottleConfig(
                    max_concurrent=10,
                    max_queue_size=100
                )
            ),
            "backpressure_manager": BackpressureManager(
                config=BackpressureConfig()
            ),
            # Lifecycle management
            "health_monitor": HealthMonitor(),
            "graceful_shutdown": GracefulShutdownManager(shutdown_timeout=30.0),
            "lifecycle_manager": LifecycleManager(),
            # Unified monitoring system (consolidates monitoring services)
            "unified_monitoring_system": UnifiedMonitoringSystem(
                monitoring_mode=MonitoringMode.STANDARD
            ),
            # Additional services
            "validation_framework": ValidationFramework(),
            "monitoring_dashboard": MonitoringDashboard(),
            "error_recovery": ErrorRecoverySystem(),
            "cost_optimizer": CostOptimizer(),
            "security_auditor": SecurityAuditor(),
            # Compliance & Audit
            "audit_logger": AuditLogger(audit_dir=self._get_storage_path("audit")),
            "compliance_manager": ComplianceManager(),
            # Experimentation & Analytics
            "experiment_manager": ExperimentManager(),
            "analytics_dashboard": AnalyticsDashboard(),
            # Service Management
            "service_discovery": ServiceDiscovery(),
            "config_validator": ConfigValidator(validation_level=ValidationLevel.STRICT),
            "metrics_aggregator": MetricsAggregator(window_size=100),
        })
    
    def _init_advanced(self):
        """Initialize advanced services with dependencies."""
        # Recommendation engine needs other services
        self._services["recommendation_engine"] = RecommendationEngine(
            template_manager=self._services["template_manager"],
            history_manager=self._services["history_manager"],
            metrics_collector=self._services["metrics_collector"]
        )
        
        # Analytics service needs other services
        self._services["analytics_service"] = AnalyticsService(
            metrics_collector=self._services["metrics_collector"],
            history_manager=self._services["history_manager"]
        )
        
        # Unified optimization system
        self._services["unified_optimization_system"] = UnifiedOptimizationSystem(
            default_mode=OptimizationMode.FULL
        )
        
        # Unified analytics system needs other services
        self._services["unified_analytics_system"] = UnifiedAnalyticsSystem(
            metrics_collector=self._services["metrics_collector"],
            history_manager=self._services["history_manager"],
            telemetry_storage_dir=self._get_storage_path("telemetry"),
            metrics_dir=self._get_storage_path("metrics")
        )
        
        # Service orchestrator (needs all services)
        self._services["service_orchestrator"] = ServiceOrchestrator(
            services=self._services
        )
        
        # Event scheduler
        self._services["event_scheduler"] = EventScheduler()
        
        # API Gateway
        self._services["api_gateway"] = APIGateway()
        
        # Data pipeline
        self._services["data_pipeline"] = DataPipeline(name="default")
        
        # Unified workflow system (consolidates workflow services)
        self._services["unified_workflow_system"] = UnifiedWorkflowSystem(
            workflows_dir=self._get_storage_path("workflows"),
            services=self._services
        )
        
        # Testing and development tools
        self._services["test_runner"] = TestRunner()
        self._services["documentation_generator"] = DocumentationGenerator(
            output_dir=str(self.output_dirs["storage"] / "docs")
        )
        self._services["transformation_engine"] = TransformationEngine()
        self._services["performance_benchmark"] = PerformanceBenchmark()
        
        # Cache warming system (needs cache_manager, initialized after management)
        self._services["cache_warming"] = CacheWarmingSystem(
            cache_system=self._services.get("cache_manager")
        )
    
    def get_service(self, name: str) -> Any:
        """
        Get service by name.
        
        Args:
            name: Service name
            
        Returns:
            Service instance
        """
        if not self._initialized:
            self.initialize_all()
        
        if name not in self._services:
            raise ValueError(f"Service not found: {name}")
        
        return self._services[name]
    
    def get_all_services(self) -> Dict[str, Any]:
        """Get all services."""
        if not self._initialized:
            self.initialize_all()
        
        return self._services.copy()
    
    def get_services_by_category(self, category: str) -> Dict[str, Any]:
        """
        Get services by category.
        
        Categories:
        - infrastructure
        - processing
        - management
        - support
        - advanced
        """
        categories = {
            "infrastructure": ["event_bus", "security_manager", "telemetry_service"],
            "processing": [
                "video_processor", "image_processor", "color_analyzer",
                "color_matcher", "video_quality_analyzer",
                "unified_processing_system"
            ],
            "management": [
                "cache_manager", "template_manager", "preset_manager",
                "lut_manager", "history_manager", "version_manager",
                "backup_manager"
            ],
            "support": [
                "task_queue", "batch_processor", "webhook_manager",
                "notification_service", "metrics_collector",
                "performance_monitor", "performance_optimizer",
                "unified_performance_manager", "performance_profiler",
                "comparison_generator", "parameter_exporter",
                "collaboration_manager", "workflow_manager",
                "cloud_integration", "advanced_cache"
            ],
            "advanced": [
                "recommendation_engine", "analytics_service",
                "unified_optimization_system", "unified_analytics_system",
                "service_orchestrator", "event_scheduler",
                "api_gateway", "data_pipeline",
                "unified_resource_manager", "unified_communication_system",
                "unified_batch_system", "unified_workflow_system",
                "unified_export_system", "unified_security_system",
                "unified_monitoring_system",
            ],
            "security": [
                "security_manager", "unified_security_system",
                "security_auditor", "validation_framework",
            ],
            "monitoring": [
                "health_monitor", "unified_monitoring_system",
                "monitoring_dashboard", "performance_monitor",
            ],
            "development": [
                "test_runner", "documentation_generator",
                "transformation_engine", "performance_benchmark",
            ],
            "ai_ml": [
                "ai_model_manager", "ml_optimizer",
                "prediction_engine", "auto_tuner",
            ],
            "features": [
                "feature_flags", "feature_toggle",
            ],
            "optimization": [
                "cost_optimizer", "error_recovery",
                "cache_warming",
            ],
            "infrastructure_advanced": [
                "ml_optimizer", "optimization_engine"
            ],
            "resilience": [
                "circuit_breaker", "retry_manager",
                "load_balancer", "feature_flags"
            ],
            "traffic_control": [
                "rate_limiter", "throttle_manager",
                "backpressure_manager"
            ],
            "lifecycle": [
                "health_monitor", "graceful_shutdown",
                "lifecycle_manager"
            ],
            "compliance": [
                "audit_logger", "compliance_manager"
            ],
            "experimentation": [
                "experiment_manager", "analytics_dashboard"
            ],
            "performance": [
                "performance_monitor", "performance_optimizer",
                "unified_performance_manager", "performance_profiler"
            ],
            "caching": [
                "cache_manager", "advanced_cache"
            ],
            "service_management": [
                "service_discovery", "config_validator", "metrics_aggregator"
            ],
        }
        
        service_names = categories.get(category, [])
        return {
            name: self.get_service(name)
            for name in service_names
            if name in self._services
        }

