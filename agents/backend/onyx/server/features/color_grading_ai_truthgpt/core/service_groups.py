"""
Service Groups for Color Grading AI
====================================

Logical grouping of services for better organization.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProcessingGroup:
    """Processing services group."""
    video_processor: Any
    image_processor: Any
    color_analyzer: Any
    color_matcher: Any
    video_quality_analyzer: Any


@dataclass
class ManagementGroup:
    """Management services group."""
    template_manager: Any
    preset_manager: Any
    lut_manager: Any
    cache_manager: Any
    history_manager: Any
    version_manager: Any
    backup_manager: Any


@dataclass
class InfrastructureGroup:
    """Infrastructure services group."""
    event_bus: Any
    security_manager: Any
    telemetry_service: Any
    task_queue: Any
    cloud_integration: Any


@dataclass
class AnalyticsGroup:
    """Analytics services group."""
    metrics_collector: Any
    performance_monitor: Any
    performance_optimizer: Any
    analytics_service: Any


@dataclass
class IntelligenceGroup:
    """Intelligence services group."""
    recommendation_engine: Any
    ml_optimizer: Any
    optimization_engine: Any


@dataclass
class CollaborationGroup:
    """Collaboration services group."""
    webhook_manager: Any
    notification_service: Any
    collaboration_manager: Any
    workflow_manager: Any


class ServiceGroups:
    """
    Service groups for organized access.
    
    Provides logical grouping of services for cleaner code.
    """
    
    def __init__(self, services: Dict[str, Any]):
        """
        Initialize service groups.
        
        Args:
            services: Dictionary of all services
        """
        self.processing = ProcessingGroup(
            video_processor=services.get("video_processor"),
            image_processor=services.get("image_processor"),
            color_analyzer=services.get("color_analyzer"),
            color_matcher=services.get("color_matcher"),
            video_quality_analyzer=services.get("video_quality_analyzer"),
        )
        
        self.management = ManagementGroup(
            template_manager=services.get("template_manager"),
            preset_manager=services.get("preset_manager"),
            lut_manager=services.get("lut_manager"),
            cache_manager=services.get("cache_manager"),
            history_manager=services.get("history_manager"),
            version_manager=services.get("version_manager"),
            backup_manager=services.get("backup_manager"),
        )
        
        self.infrastructure = InfrastructureGroup(
            event_bus=services.get("event_bus"),
            security_manager=services.get("security_manager"),
            telemetry_service=services.get("telemetry_service"),
            task_queue=services.get("task_queue"),
            cloud_integration=services.get("cloud_integration"),
        )
        
        self.analytics = AnalyticsGroup(
            metrics_collector=services.get("metrics_collector"),
            performance_monitor=services.get("performance_monitor"),
            performance_optimizer=services.get("performance_optimizer"),
            analytics_service=services.get("analytics_service"),
        )
        
        self.intelligence = IntelligenceGroup(
            recommendation_engine=services.get("recommendation_engine"),
            ml_optimizer=services.get("ml_optimizer"),
            optimization_engine=services.get("optimization_engine"),
        )
        
        self.collaboration = CollaborationGroup(
            webhook_manager=services.get("webhook_manager"),
            notification_service=services.get("notification_service"),
            collaboration_manager=services.get("collaboration_manager"),
            workflow_manager=services.get("workflow_manager"),
        )
        
        # Support services (not grouped)
        self.batch_processor = services.get("batch_processor")
        self.comparison_generator = services.get("comparison_generator")
        self.parameter_exporter = services.get("parameter_exporter")
    
    def get_all(self) -> Dict[str, Any]:
        """Get all services as flat dictionary."""
        return {
            # Processing
            "video_processor": self.processing.video_processor,
            "image_processor": self.processing.image_processor,
            "color_analyzer": self.processing.color_analyzer,
            "color_matcher": self.processing.color_matcher,
            "video_quality_analyzer": self.processing.video_quality_analyzer,
            # Management
            "template_manager": self.management.template_manager,
            "preset_manager": self.management.preset_manager,
            "lut_manager": self.management.lut_manager,
            "cache_manager": self.management.cache_manager,
            "history_manager": self.management.history_manager,
            "version_manager": self.management.version_manager,
            "backup_manager": self.management.backup_manager,
            # Infrastructure
            "event_bus": self.infrastructure.event_bus,
            "security_manager": self.infrastructure.security_manager,
            "telemetry_service": self.infrastructure.telemetry_service,
            "task_queue": self.infrastructure.task_queue,
            "cloud_integration": self.infrastructure.cloud_integration,
            # Analytics
            "metrics_collector": self.analytics.metrics_collector,
            "performance_monitor": self.analytics.performance_monitor,
            "performance_optimizer": self.analytics.performance_optimizer,
            "analytics_service": self.analytics.analytics_service,
            # Intelligence
            "recommendation_engine": self.intelligence.recommendation_engine,
            "ml_optimizer": self.intelligence.ml_optimizer,
            "optimization_engine": self.intelligence.optimization_engine,
            # Collaboration
            "webhook_manager": self.collaboration.webhook_manager,
            "notification_service": self.collaboration.notification_service,
            "collaboration_manager": self.collaboration.collaboration_manager,
            "workflow_manager": self.collaboration.workflow_manager,
            # Support
            "batch_processor": self.batch_processor,
            "comparison_generator": self.comparison_generator,
            "parameter_exporter": self.parameter_exporter,
        }




