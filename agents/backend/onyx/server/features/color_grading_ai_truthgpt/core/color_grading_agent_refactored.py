"""
Refactored Color Grading Agent
==============================

Improved agent with better organization and service groups.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..infrastructure.openrouter_client import OpenRouterClient
from ..infrastructure.truthgpt_client import TruthGPTClient
from ..config.color_grading_config import ColorGradingConfig
from .system_prompts_builder import SystemPromptsBuilder
from .helpers import create_output_directories
from .service_factory_refactored import RefactoredServiceFactory
from .service_groups import ServiceGroups
from .grading_orchestrator import GradingOrchestrator
from .exceptions import (
    MediaNotFoundError,
    InvalidParametersError,
    TemplateNotFoundError
)
from ..services.queue_unified import TaskPriority
from .plugin_manager import PluginManager

logger = logging.getLogger(__name__)


class RefactoredColorGradingAgent:
    """
    Refactored color grading agent with improved organization.
    
    Improvements:
    - Service groups for organized access
    - Cleaner code structure
    - Better separation of concerns
    - Simplified service access
    """
    
    def __init__(
        self,
        config: Optional[ColorGradingConfig] = None,
        max_parallel_tasks: int = 5,
        output_dir: str = "color_grading_output",
        debug: bool = False,
    ):
        """
        Initialize refactored color grading agent.
        
        Args:
            config: Configuration
            max_parallel_tasks: Maximum parallel tasks
            output_dir: Output directory
            debug: Debug mode
        """
        self.config = config or ColorGradingConfig()
        self.config.validate()
        
        # Infrastructure clients
        self.openrouter_client = OpenRouterClient(api_key=self.config.openrouter.api_key)
        self.truthgpt_client = TruthGPTClient(
            config=self.config.truthgpt.to_dict() if self.config.truthgpt else {}
        )
        
        # Output directories
        self.output_dir = Path(output_dir)
        self.output_dirs = create_output_directories(
            self.output_dir,
            ["results", "tasks", "storage", "previews", "cache"]
        )
        
        # Configuration
        self.max_parallel_tasks = max_parallel_tasks
        self.debug = debug
        self.running = False
        
        # System prompts
        self.system_prompts = SystemPromptsBuilder.build_all_prompts()
        
        # Service factory (refactored)
        self.service_factory = RefactoredServiceFactory(self.config, self.output_dirs)
        self.services = self.service_factory.get_all_services()
        
        # Service groups (organized access)
        self.groups = ServiceGroups(self.services)
        
        # Grading orchestrator
        self.orchestrator = GradingOrchestrator(
            services=self.services,
            config=self.config,
            output_dirs=self.output_dirs
        )
        
        # Plugin manager
        self.plugin_manager = PluginManager(
            plugins_dir=str(self.output_dirs["storage"] / "plugins")
        )
        
        # Auth manager
        from .auth_manager import AuthManager
        self.auth_manager = AuthManager()
        
        logger.info(f"Initialized RefactoredColorGradingAgent with {max_parallel_tasks} max parallel tasks")
    
    # Convenience properties for backward compatibility
    @property
    def video_processor(self):
        """Video processor."""
        return self.groups.processing.video_processor
    
    @property
    def image_processor(self):
        """Image processor."""
        return self.groups.processing.image_processor
    
    @property
    def color_analyzer(self):
        """Color analyzer."""
        return self.groups.processing.color_analyzer
    
    @property
    def color_matcher(self):
        """Color matcher."""
        return self.groups.processing.color_matcher
    
    @property
    def template_manager(self):
        """Template manager."""
        return self.groups.management.template_manager
    
    @property
    def cache_manager(self):
        """Cache manager."""
        return self.groups.management.cache_manager
    
    @property
    def batch_processor(self):
        """Batch processor."""
        return self.groups.batch_processor
    
    @property
    def webhook_manager(self):
        """Webhook manager."""
        return self.groups.collaboration.webhook_manager
    
    @property
    def metrics_collector(self):
        """Metrics collector."""
        return self.groups.analytics.metrics_collector
    
    @property
    def comparison_generator(self):
        """Comparison generator."""
        return self.groups.comparison_generator
    
    @property
    def lut_manager(self):
        """LUT manager."""
        return self.groups.management.lut_manager
    
    @property
    def task_queue(self):
        """Task queue."""
        return self.groups.infrastructure.task_queue
    
    @property
    def parameter_exporter(self):
        """Parameter exporter."""
        return self.groups.parameter_exporter
    
    @property
    def history_manager(self):
        """History manager."""
        return self.groups.management.history_manager
    
    @property
    def performance_optimizer(self):
        """Performance optimizer."""
        return self.groups.analytics.performance_optimizer
    
    @property
    def video_quality_analyzer(self):
        """Video quality analyzer."""
        return self.groups.processing.video_quality_analyzer
    
    @property
    def preset_manager(self):
        """Preset manager."""
        return self.groups.management.preset_manager
    
    @property
    def backup_manager(self):
        """Backup manager."""
        return self.groups.management.backup_manager
    
    @property
    def notification_service(self):
        """Notification service."""
        return self.groups.collaboration.notification_service
    
    @property
    def version_manager(self):
        """Version manager."""
        return self.groups.management.version_manager
    
    @property
    def cloud_integration(self):
        """Cloud integration."""
        return self.groups.infrastructure.cloud_integration
    
    # Main methods (same as original)
    async def grade_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        template_name: Optional[str] = None,
        reference_image: Optional[str] = None,
        reference_video: Optional[str] = None,
        color_params: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Apply color grading to video."""
        if not Path(video_path).exists():
            raise MediaNotFoundError(f"Video not found: {video_path}")
        
        if not output_path:
            output_path = str(self.output_dirs["results"] / f"graded_{Path(video_path).stem}.mp4")
        
        try:
            params = await self.orchestrator.resolve_color_parameters(
                template_name=template_name,
                reference_image=reference_image,
                reference_video=reference_video,
                color_params=color_params,
                description=description,
                media_path=video_path
            )
        except ValueError as e:
            raise InvalidParametersError(str(e))
        
        async def _process():
            result_path = await self.video_processor.apply_color_grading(
                video_path,
                output_path,
                params,
                codec=self.config.video_processing.codec,
                quality=self.config.video_processing.quality
            )
            return {
                "success": True,
                "output_path": result_path,
                "color_params": params,
                "input_path": video_path,
            }
        
        return await self.orchestrator.process_with_tracking(
            operation="grade_video",
            processor_func=_process,
            input_path=video_path,
            template_name=template_name
        )
    
    async def grade_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        template_name: Optional[str] = None,
        reference_image: Optional[str] = None,
        color_params: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Apply color grading to image."""
        if not Path(image_path).exists():
            raise MediaNotFoundError(f"Image not found: {image_path}")
        
        if not output_path:
            output_path = str(self.output_dirs["results"] / f"graded_{Path(image_path).stem}.jpg")
        
        try:
            params = await self.orchestrator.resolve_color_parameters(
                template_name=template_name,
                reference_image=reference_image,
                color_params=color_params,
                description=description,
                media_path=image_path
            )
        except ValueError as e:
            raise InvalidParametersError(str(e))
        
        async def _process():
            result_path = await self.image_processor.apply_color_grading(
                image_path,
                output_path,
                params
            )
            return {
                "success": True,
                "output_path": result_path,
                "color_params": params,
                "input_path": image_path,
            }
        
        return await self.orchestrator.process_with_tracking(
            operation="grade_image",
            processor_func=_process,
            input_path=image_path,
            template_name=template_name
        )
    
    async def analyze_media(self, media_path: str, media_type: str = "auto") -> Dict[str, Any]:
        """Analyze color properties of media."""
        if media_type == "auto":
            ext = Path(media_path).suffix.lower()
            media_type = "video" if ext in [".mp4", ".mov", ".avi", ".mkv"] else "image"
        
        if media_type == "image":
            return await self.color_analyzer.analyze_image(media_path)
        else:
            frames_dir = self.output_dirs["cache"] / "frames" / Path(media_path).stem
            frames = await self.video_processor.extract_frames(
                media_path,
                str(frames_dir),
                interval=self.config.color_analysis.keyframe_interval
            )
            return await self.color_analyzer.analyze_video_frames(
                frames,
                scene_threshold=self.config.color_analysis.scene_detection_threshold
            )
    
    async def list_templates(self, category: Optional[str] = None, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """List available templates."""
        templates = self.template_manager.list_templates(category=category, tags=tags)
        return [t.to_dict() for t in templates]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics."""
        return self.metrics_collector.get_stats()
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get system resource statistics."""
        return self.performance_optimizer.get_resource_stats()
    
    async def close(self):
        """Close agent and cleanup resources."""
        await self.openrouter_client.close()
        await self.truthgpt_client.close()
        await self.webhook_manager.close()
        logger.info("RefactoredColorGradingAgent closed")




