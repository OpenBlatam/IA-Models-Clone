"""
Service Factory for Color Grading AI
=====================================

Factory pattern for creating and managing services.

Note: Consider using RefactoredServiceFactory for better organization.
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
from ..services.cache_unified import UnifiedCache
from ..services.batch_processor import BatchProcessor
from ..services.webhook_manager import WebhookManager
from ..services.metrics_collector import MetricsCollector
from ..services.comparison_generator import ComparisonGenerator
from ..services.lut_manager import LUTManager
from ..services.queue_unified import UnifiedQueue, TaskPriority, RetryStrategy
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

logger = logging.getLogger(__name__)


class ServiceFactory:
    """
    Factory for creating and managing services.
    
    Centralizes service initialization and provides dependency injection.
    """
    
    def __init__(self, config: ColorGradingConfig, output_dirs: Dict[str, Path]):
        """
        Initialize service factory.
        
        Args:
            config: Configuration
            output_dirs: Output directories dictionary
        """
        self.config = config
        self.output_dirs = output_dirs
        self._services: Dict[str, Any] = {}
    
    def create_processing_services(self) -> Dict[str, Any]:
        """Create processing services."""
        if "processing" not in self._services:
            self._services["processing"] = {
                "video_processor": VideoProcessor(
                    ffmpeg_path=self.config.video_processing.ffmpeg_path
                ),
                "image_processor": ImageProcessor(),
                "color_analyzer": ColorAnalyzer(
                    histogram_bins=self.config.color_analysis.histogram_bins,
                    color_space=self.config.color_analysis.color_space
                ),
                "color_matcher": ColorMatcher(),
            }
        return self._services["processing"]
    
    def create_management_services(self) -> Dict[str, Any]:
        """Create management services."""
        if "management" not in self._services:
            self._services["management"] = {
                "template_manager": TemplateManager(
                    templates_dir=str(self.output_dirs["storage"] / "templates")
                ),
                "lut_manager": LUTManager(
                    luts_dir=str(self.output_dirs["storage"] / "luts")
                ),
                "cache_manager": UnifiedCache(
                    cache_dir=str(self.output_dirs["cache"]),
                    ttl=self.config.cache_ttl if self.config.enable_cache else 0,
                    redis_url=getattr(self.config, "redis_url", None)
                ),
                "history_manager": HistoryManager(
                    history_dir=str(self.output_dirs["storage"] / "history")
                ),
            }
        return self._services["management"]
    
    def create_support_services(self) -> Dict[str, Any]:
        """Create support services."""
        if "support" not in self._services:
            self._services["support"] = {
                "batch_processor": BatchProcessor(
                    max_parallel=self.config.max_parallel_tasks
                ),
                "webhook_manager": WebhookManager(),
                "metrics_collector": MetricsCollector(
                    metrics_dir=str(self.output_dirs["storage"] / "metrics")
                ),
                "comparison_generator": ComparisonGenerator(),
                "task_queue": UnifiedQueue(
                    max_workers=self.config.max_parallel_tasks,
                    storage_dir=str(self.output_dirs["storage"] / "queue")
                ),
                "parameter_exporter": ParameterExporter(),
                "performance_optimizer": PerformanceOptimizer(),
                "video_quality_analyzer": VideoQualityAnalyzer(),
                "preset_manager": PresetManager(
                    presets_dir=str(self.output_dirs["storage"] / "presets")
                ),
                "backup_manager": BackupManager(
                    backup_dir=str(self.output_dirs["storage"] / "backups")
                ),
                "notification_service": NotificationService(),
                "version_manager": VersionManager(
                    versions_dir=str(self.output_dirs["storage"] / "versions")
                ),
                "cloud_integration": CloudIntegrationManager(),
                "workflow_manager": WorkflowManager(
                    workflows_dir=str(self.output_dirs["storage"] / "workflows")
                ),
                "collaboration_manager": CollaborationManager(),
            }
        return self._services["support"]
    
    def create_recommendation_engine(self) -> RecommendationEngine:
        """Create recommendation engine (needs other services)."""
        management = self.create_management_services()
        support = self.create_support_services()
        
        return RecommendationEngine(
            template_manager=management.get("template_manager"),
            history_manager=management.get("history_manager"),
            metrics_collector=support.get("metrics_collector")
        )
    
    def get_all_services(self) -> Dict[str, Any]:
        """Get all services."""
        processing = self.create_processing_services()
        management = self.create_management_services()
        support = self.create_support_services()
        recommendation_engine = self.create_recommendation_engine()
        
        return {
            **processing,
            **management,
            **support,
            "recommendation_engine": recommendation_engine,
        }

