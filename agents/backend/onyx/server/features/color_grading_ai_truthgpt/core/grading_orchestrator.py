"""
Grading Orchestrator for Color Grading AI
==========================================

Orchestrates color grading operations with proper error handling and metrics.
"""

import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class GradingOrchestrator:
    """
    Orchestrates color grading operations.
    
    Handles:
    - Parameter resolution
    - Cache management
    - Metrics tracking
    - History recording
    - Webhook notifications
    """
    
    def __init__(
        self,
        services: Dict[str, Any],
        config: Any,
        output_dirs: Dict[str, Path]
    ):
        """
        Initialize orchestrator.
        
        Args:
            services: Service dictionary
            config: Configuration
            output_dirs: Output directories
        """
        self.services = services
        self.config = config
        self.output_dirs = output_dirs
    
    async def resolve_color_parameters(
        self,
        template_name: Optional[str] = None,
        reference_image: Optional[str] = None,
        reference_video: Optional[str] = None,
        color_params: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        media_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Resolve color parameters from various sources.
        
        Args:
            template_name: Template name
            reference_image: Reference image path
            reference_video: Reference video path
            color_params: Direct color parameters
            description: Text description
            media_path: Media path for context
            
        Returns:
            Color parameters dictionary
        """
        if color_params:
            return color_params
        
        if template_name:
            return self.services["template_manager"].get_template_color_params(template_name)
        
        if reference_image and media_path:
            match_result = await self.services["color_matcher"].match_from_reference_image(
                media_path, reference_image
            )
            return match_result["color_params"]
        
        if reference_video and media_path:
            match_result = await self.services["color_matcher"].match_from_reference_video(
                media_path, reference_video
            )
            return match_result["color_params"]
        
        if description and media_path:
            # Use LLM to generate parameters
            return await self._generate_params_from_description(description, media_path)
        
        raise ValueError("Must provide template_name, reference, color_params, or description")
    
    async def _generate_params_from_description(
        self,
        description: str,
        media_path: str
    ) -> Dict[str, Any]:
        """
        Generate parameters from description using LLM.
        
        Note: This requires access to OpenRouter client which should be passed
        through the services or config. For now, using color_matcher as fallback.
        """
        result = await self.services["color_matcher"].generate_grading_from_description(description)
        return result["color_params"]
    
    async def process_with_tracking(
        self,
        operation: str,
        processor_func,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process with full tracking (cache, metrics, history, webhooks).
        
        Args:
            operation: Operation name
            processor_func: Processing function
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Processing result
        """
        # Extract input path for cache key
        input_path = kwargs.get("input_path") or (args[0] if args else None)
        if not input_path:
            raise ValueError("Input path required")
        
        # Check cache
        if self.config.enable_cache:
            file_hash = await self.services["cache_manager"].get_file_hash(input_path)
            cached_result = await self.services["cache_manager"].get(
                file_hash,
                prefix=f"{operation}_grading"
            )
            if cached_result:
                logger.info(f"Using cached result for {input_path}")
                return cached_result
        
        # Track metrics
        start_time = time.time()
        file_size = Path(input_path).stat().st_size if Path(input_path).exists() else 0
        
        try:
            # Process
            result = await processor_func(*args, **kwargs)
            
            duration = time.time() - start_time
            output_path = result.get("output_path")
            output_size = Path(output_path).stat().st_size if output_path and Path(output_path).exists() else None
            
            # Cache result
            if self.config.enable_cache:
                await self.services["cache_manager"].set(
                    file_hash,
                    result,
                    prefix=f"{operation}_grading"
                )
            
            # Record metrics
            self.services["metrics_collector"].record(
                operation=operation,
                media_type="video" if operation == "grade_video" else "image",
                duration=duration,
                success=True,
                file_size=file_size,
                output_size=output_size,
                template_used=kwargs.get("template_name")
            )
            
            # Add to history
            self.services["history_manager"].add(
                input_path=input_path,
                output_path=output_path or "",
                operation=operation,
                color_params=result.get("color_params", {}),
                template_used=kwargs.get("template_name"),
                duration=duration,
                file_size=file_size,
                output_size=output_size,
                success=True
            )
            
            # Send webhook
            await self.services["webhook_manager"].send("completed", result)
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            # Record failure
            self.services["metrics_collector"].record(
                operation=operation,
                media_type="video" if operation == "grade_video" else "image",
                duration=duration,
                success=False,
                file_size=file_size,
                error=error_msg,
                template_used=kwargs.get("template_name")
            )
            
            # Add to history
            self.services["history_manager"].add(
                input_path=input_path,
                output_path="",
                operation=operation,
                color_params={},
                template_used=kwargs.get("template_name"),
                duration=duration,
                file_size=file_size,
                success=False,
                error=error_msg
            )
            
            # Send webhook
            await self.services["webhook_manager"].send("failed", {"error": error_msg})
            
            raise

