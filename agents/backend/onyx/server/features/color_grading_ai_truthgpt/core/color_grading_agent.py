"""
Color Grading Agent - Main class for Color Grading AI with SAM3 architecture
============================================================================

Refactored with:
- Service Factory pattern for service management
- Grading Orchestrator for operation coordination
- Better separation of concerns
- Improved error handling
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..infrastructure.openrouter_client import OpenRouterClient
from ..infrastructure.truthgpt_client import TruthGPTClient
from ..config.color_grading_config import ColorGradingConfig
from .system_prompts_builder import SystemPromptsBuilder
from .helpers import create_output_directories, create_message
from .service_factory import ServiceFactory
from .grading_orchestrator import GradingOrchestrator
from .exceptions import (
    MediaNotFoundError,
    InvalidParametersError,
    TemplateNotFoundError
)
from ..services.queue_unified import UnifiedQueue, TaskPriority

logger = logging.getLogger(__name__)


class ColorGradingAgent:
    """
    Autonomous agent for color grading based on SAM3 architecture.
    
    Features:
    - Continuous 24/7 operation
    - Parallel task execution
    - OpenRouter LLM integration
    - TruthGPT optimization
    - Automatic color grading
    - Video and image processing
    - Template support
    - Color matching from references
    """
    
    def __init__(
        self,
        config: Optional[ColorGradingConfig] = None,
        max_parallel_tasks: int = 5,
        output_dir: str = "color_grading_output",
        debug: bool = False,
    ):
        """
        Initialize color grading agent.
        
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
        
        # Service factory
        self.service_factory = ServiceFactory(self.config, self.output_dirs)
        self.services = self.service_factory.get_all_services()
        
        # Grading orchestrator
        self.orchestrator = GradingOrchestrator(
            services=self.services,
            config=self.config,
            output_dirs=self.output_dirs
        )
        
        # Expose commonly used services for convenience
        self.video_processor = self.services["video_processor"]
        self.image_processor = self.services["image_processor"]
        self.color_analyzer = self.services["color_analyzer"]
        self.color_matcher = self.services["color_matcher"]
        self.template_manager = self.services["template_manager"]
        self.cache_manager = self.services["cache_manager"]
        self.batch_processor = self.services["batch_processor"]
        self.webhook_manager = self.services["webhook_manager"]
        self.metrics_collector = self.services["metrics_collector"]
        self.comparison_generator = self.services["comparison_generator"]
        self.lut_manager = self.services["lut_manager"]
        self.task_queue = self.services["task_queue"]
        self.parameter_exporter = self.services["parameter_exporter"]
        self.history_manager = self.services["history_manager"]
        self.performance_optimizer = self.services["performance_optimizer"]
        self.video_quality_analyzer = self.services.get("video_quality_analyzer")
        self.preset_manager = self.services.get("preset_manager")
        self.backup_manager = self.services.get("backup_manager")
        self.notification_service = self.services.get("notification_service")
        self.version_manager = self.services.get("version_manager")
        self.cloud_integration = self.services.get("cloud_integration")
        
        # Plugin manager (not in service factory)
        self.plugin_manager = PluginManager(
            plugins_dir=str(self.output_dirs["storage"] / "plugins")
        )
        
        # Auth manager (not in service factory)
        from .auth_manager import AuthManager
        self.auth_manager = AuthManager()
        
        logger.info(f"Initialized ColorGradingAgent with {max_parallel_tasks} max parallel tasks")
    
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
        """
        Apply color grading to video.
        
        Args:
            video_path: Path to input video
            output_path: Path to output video (auto-generated if None)
            template_name: Name of template to apply
            reference_image: Path to reference image for color matching
            reference_video: Path to reference video for color matching
            color_params: Direct color parameters
            description: Text description of desired look
            
        Returns:
            Dictionary with result information
            
        Raises:
            MediaNotFoundError: If video file not found
            InvalidParametersError: If parameters are invalid
        """
        if not Path(video_path).exists():
            raise MediaNotFoundError(f"Video not found: {video_path}")
        
        # Determine output path
        if not output_path:
            output_path = str(self.output_dirs["results"] / f"graded_{Path(video_path).stem}.mp4")
        
        # Resolve color parameters
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
        
        # Process with tracking
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
        """
        Apply color grading to image.
        
        Args:
            image_path: Path to input image
            output_path: Path to output image (auto-generated if None)
            template_name: Name of template to apply
            reference_image: Path to reference image for color matching
            color_params: Direct color parameters
            description: Text description of desired look
            
        Returns:
            Dictionary with result information
            
        Raises:
            MediaNotFoundError: If image file not found
            InvalidParametersError: If parameters are invalid
        """
        if not Path(image_path).exists():
            raise MediaNotFoundError(f"Image not found: {image_path}")
        
        # Determine output path
        if not output_path:
            output_path = str(self.output_dirs["results"] / f"graded_{Path(image_path).stem}.jpg")
        
        # Resolve color parameters
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
        
        # Process with tracking
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
    
    async def analyze_media(
        self,
        media_path: str,
        media_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Analyze color properties of media.
        
        Args:
            media_path: Path to image or video
            media_type: "image", "video", or "auto"
            
        Returns:
            Dictionary with analysis results
        """
        if media_type == "auto":
            ext = Path(media_path).suffix.lower()
            media_type = "video" if ext in [".mp4", ".mov", ".avi", ".mkv"] else "image"
        
        if media_type == "image":
            return await self.color_analyzer.analyze_image(media_path)
        else:
            # Extract frames and analyze
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
    
    async def list_templates(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List available templates.
        
        Args:
            category: Filter by category
            tags: Filter by tags
            
        Returns:
            List of template dictionaries
        """
        templates = self.template_manager.list_templates(category=category, tags=tags)
        return [t.to_dict() for t in templates]
    
    async def create_preview(
        self,
        video_path: str,
        duration: float = 10.0,
        resolution: str = "720p"
    ) -> str:
        """
        Create preview clip.
        
        Args:
            video_path: Path to video
            duration: Preview duration in seconds
            resolution: Preview resolution
            
        Returns:
            Path to preview video
        """
        output_path = str(self.output_dirs["previews"] / f"preview_{Path(video_path).stem}.mp4")
        return await self.video_processor.create_preview(
            video_path,
            output_path,
            duration,
            resolution
        )
    
    async def process_batch(
        self,
        items: List[Dict[str, Any]],
        media_type: str = "video"
    ) -> str:
        """
        Process batch of media files.
        
        Args:
            items: List of items with input_path, output_path, parameters
            media_type: "video" or "image"
            
        Returns:
            Batch job ID
        """
        processor_func = self.grade_video if media_type == "video" else self.grade_image
        
        job = await self.batch_processor.process_batch(
            items,
            processor_func
        )
        
        return job.id
    
    async def get_batch_status(self, job_id: str) -> Dict[str, Any]:
        """Get batch job status."""
        return self.batch_processor.get_job_status(job_id)
    
    def register_webhook(self, url: str, events: List[str], secret: Optional[str] = None):
        """Register webhook for notifications."""
        from ..services.webhook_manager import Webhook
        webhook = Webhook(url=url, events=events, secret=secret)
        self.webhook_manager.register(webhook)
    
    async def create_comparison(
        self,
        before_path: str,
        after_path: str,
        output_path: str,
        style: str = "side_by_side"
    ) -> str:
        """Create before/after comparison."""
        return await self.comparison_generator.create_image_comparison(
            before_path,
            after_path,
            output_path,
            style
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics."""
        return self.metrics_collector.get_stats()
    
    async def enqueue_task(
        self,
        task_type: str,
        parameters: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> str:
        """Enqueue task for async processing."""
        return await self.task_queue.enqueue(task_type, parameters, priority)
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status from queue."""
        task = await self.task_queue.get_task(task_id)
        if not task:
            return {"error": "Task not found"}
        return task.to_dict()
    
    def export_parameters(
        self,
        params: Dict[str, Any],
        output_path: str,
        format: str = "all"
    ) -> Dict[str, str]:
        """Export color parameters to various formats."""
        if format == "all":
            return self.parameter_exporter.export_all_formats(params, output_path)
        elif format == "json":
            return {"json": self.parameter_exporter.export_json(params, output_path)}
        elif format == "drx":
            return {"drx": self.parameter_exporter.export_davinci_resolve(params, output_path)}
        elif format == "cube":
            return {"cube": self.parameter_exporter.export_lut_cube(params, output_path)}
        else:
            raise InvalidParametersError(f"Unsupported format: {format}")
    
    def get_history(
        self,
        operation: Optional[str] = None,
        template: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get processing history."""
        entries = self.history_manager.search(
            operation=operation,
            template=template,
            limit=limit
        )
        return [entry.to_dict() for entry in entries]
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get system resource statistics."""
        return self.performance_optimizer.get_resource_stats()
    
    async def analyze_video_quality(self, video_path: str) -> Dict[str, Any]:
        """Analyze video quality metrics."""
        return await self.video_quality_analyzer.analyze_quality(video_path)
    
    def create_preset(
        self,
        name: str,
        description: str,
        color_params: Dict[str, Any],
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Create a color grading preset."""
        return self.preset_manager.create_preset(
            name, description, color_params, category, tags
        )
    
    def list_presets(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        favorites_only: bool = False
    ) -> List[Dict[str, Any]]:
        """List color grading presets."""
        presets = self.preset_manager.list_presets(category, tags, favorites_only)
        return [p.to_dict() for p in presets]
    
    def create_backup(
        self,
        source_dirs: List[str],
        backup_name: Optional[str] = None
    ) -> str:
        """Create a backup."""
        return self.backup_manager.create_backup(source_dirs, backup_name)
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups."""
        return self.backup_manager.list_backups()
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List available plugins."""
        return self.plugin_manager.list_plugins()
    
    async def execute_plugin(
        self,
        plugin_name: str,
        media_path: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a plugin."""
        return await self.plugin_manager.execute_plugin(plugin_name, media_path, params)
    
    async def close(self):
        """Close agent and cleanup resources."""
        await self.openrouter_client.close()
        await self.truthgpt_client.close()
        await self.webhook_manager.close()
        logger.info("ColorGradingAgent closed")
