"""
Service Handler for Enhancer Agent
==================================
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field

from .common import create_message
from .service_config import ServiceType, ServiceConfig, ServiceConfigRegistry
from ..infrastructure.openrouter_client import OpenRouterClient
from ..infrastructure.truthgpt_client import TruthGPTClient
from ..utils.validators import FileValidator, ParameterValidator, ValidationError
from ..utils.file_helpers import get_mime_type
from ..core.video_processor import VideoProcessor

logger = logging.getLogger(__name__)


@dataclass
class ServiceResult:
    """Result from a service request."""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    tokens_used: int = 0
    model: str = ""
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {**self.data}
        result["tokens_used"] = self.tokens_used
        result["model"] = self.model
        if self.timestamp:
            result["timestamp"] = self.timestamp
        if self.error:
            result["error"] = self.error
        return result
    
    @classmethod
    def from_response(
        cls,
        response: Dict[str, Any],
        config: ServiceConfig
    ) -> "ServiceResult":
        """Create from API response."""
        data = {
            config.response_key: response.get("response", ""),
        }
        
        return cls(
            success=True,
            data=data,
            tokens_used=response.get("tokens_used", 0),
            model=response.get("model", ""),
            timestamp=datetime.now().isoformat() if config.include_timestamp else None,
        )
    
    @classmethod
    def error_result(cls, error: str) -> "ServiceResult":
        """Create error result."""
        return cls(success=False, error=error)


# Import handlers from services module
from .services import (
    BaseServiceHandler,
    EnhanceImageHandler,
    EnhanceVideoHandler,
    UpscaleHandler,
    DenoiseHandler,
    RestoreHandler,
    ColorCorrectionHandler
)


class ServiceHandlerRegistry:
    """Registry for service handlers."""
    
    _handlers: Dict[ServiceType, BaseServiceHandler] = {}
    
    @classmethod
    def register(cls, handler: BaseServiceHandler):
        """Register a handler."""
        cls._handlers[handler.service_type] = handler
    
    @classmethod
    def get(cls, service_type: ServiceType) -> Optional[BaseServiceHandler]:
        """Get handler for service type."""
        return cls._handlers.get(service_type)
    
    @classmethod
    def register_defaults(cls):
        """Register default handlers."""
        cls.register(EnhanceImageHandler())
        cls.register(EnhanceVideoHandler())
        cls.register(UpscaleHandler())
        cls.register(DenoiseHandler())
        cls.register(RestoreHandler())
        cls.register(ColorCorrectionHandler())


# Initialize default handlers
ServiceHandlerRegistry.register_defaults()


class ServiceHandler:
    """
    Handles service requests with common patterns.
    """
    
    def __init__(
        self,
        openrouter_client: OpenRouterClient,
        truthgpt_client: TruthGPTClient,
        system_prompts: Dict[str, str],
        config: Any,
        video_processor: Optional[VideoProcessor] = None
    ):
        self.openrouter_client = openrouter_client
        self.truthgpt_client = truthgpt_client
        self.system_prompts = system_prompts
        self.config = config
        self.video_processor = video_processor
    
    async def execute_request(
        self,
        prompt: str,
        config: ServiceConfig,
        context: Optional[Dict[str, Any]] = None
    ) -> ServiceResult:
        """Execute a service request."""
        try:
            # Check if we have an image path in context for vision models
            image_path = context.get("image_path") if context else None
            
            # Optimize prompt with TruthGPT
            optimized_prompt = await self.truthgpt_client.optimize_query(prompt)
            
            system_prompt = self.system_prompts.get(config.system_prompt_key, self.system_prompts["default"])
            
            # If we have an image, use vision model
            if image_path:
                try:
                    # Validate image file
                    FileValidator.validate_image_file(
                        image_path,
                        max_size_mb=self.config.max_file_size_mb,
                        allowed_extensions=self.config.allowed_image_formats
                    )
                    
                    # Get MIME type
                    mime_type = get_mime_type(image_path)
                    
                    # Use vision model
                    response = await self.openrouter_client.process_image(
                        model=self.config.openrouter.model,
                        image_path=image_path,
                        prompt=optimized_prompt,
                        system_prompt=system_prompt,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                        mime_type=mime_type,
                    )
                    
                    return ServiceResult.from_response(response, config)
                    
                except ValidationError as e:
                    logger.warning(f"Image validation failed, falling back to text-only: {e}")
                    # Fall through to text-only mode
            
            # Text-only mode (no image or image validation failed)
            messages = [
                create_message("system", system_prompt),
                create_message("user", optimized_prompt),
            ]
            
            # Call OpenRouter
            response = await self.openrouter_client.chat_completion(
                model=self.config.openrouter.model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            
            return ServiceResult.from_response(response, config)
            
        except Exception as e:
            logger.error(f"Error executing service request: {e}", exc_info=True)
            return ServiceResult.error_result(str(e))
    
    async def handle(self, service_type: ServiceType, parameters: Dict[str, Any]) -> ServiceResult:
        """Handle a service request."""
        handler = ServiceHandlerRegistry.get(service_type)
        if not handler:
            return ServiceResult.error_result(f"Unknown service type: {service_type}")
        
        # Build context with file path and analysis
        context = {}
        file_path = parameters.get("file_path")
        if file_path:
            # Determine file type
            try:
                file_type = FileValidator.get_file_type(file_path)
                if file_type == "image":
                    context["image_path"] = file_path
                elif file_type == "video" and self.video_processor:
                    # Analyze video if processor available
                    try:
                        video_analysis = await self.video_processor.analyze_video(file_path)
                        parameters["video_analysis"] = video_analysis
                        context["video_analysis"] = video_analysis
                    except Exception as e:
                        logger.warning(f"Could not analyze video: {e}")
            except Exception as e:
                logger.warning(f"Could not determine file type: {e}")
        
        # Build prompt
        prompt = handler.build_prompt(parameters)
        
        # Execute with context
        return await self.execute_request(prompt, handler.config, context)

