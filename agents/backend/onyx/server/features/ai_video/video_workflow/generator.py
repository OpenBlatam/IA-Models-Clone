from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

from typing import Any, Dict
from datetime import datetime
from onyx.utils.logger import setup_logger
from onyx.utils.telemetry import TelemetryLogger
from ...models import VideoRequest, VideoResponse
from ...core.exceptions import AIVideoError, ValidationError
from .core.workflow import OnyxVideoWorkflow
from .core.models import GeneratorStatus
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Onyx Video Workflow - Video Generator

Video generator with advanced features using Onyx's infrastructure.
"""


# Onyx imports

# Local imports


class OnyxVideoGenerator:
    """
    Onyx video generator with advanced features.
    
    Provides video generation capabilities using Onyx's infrastructure
    with support for different video types and styles.
    """
    
    def __init__(self, workflow_type: str = "default"):
        
    """__init__ function."""
self.logger = setup_logger("onyx_video_generator")
        self.workflow = OnyxVideoWorkflow(workflow_type)
        self.telemetry = TelemetryLogger()
        self.workflow_type = workflow_type
    
    async def initialize(self) -> None:
        """Initialize the video generator."""
        await self.workflow.initialize()
    
    async def generate_video(self, request: VideoRequest) -> VideoResponse:
        """Generate video using Onyx workflow."""
        try:
            # Validate request
            await self._validate_request(request)
            
            # Process request
            result = await self.workflow.process_request(request)
            
            # Create response
            response = VideoResponse(
                request_id=request.request_id,
                status="completed",
                output_url=result.get("output_url", "generated_video.mp4"),
                metadata={
                    **result,
                    "workflow": f"onyx_{self.workflow_type}",
                    "generated_at": datetime.now().isoformat()
                }
            )
            
            # Log telemetry
            self.telemetry.log_info("video_generated", {
                "request_id": request.request_id,
                "user_id": request.user_id,
                "quality": request.quality,
                "duration": request.duration,
                "workflow_type": self.workflow_type
            })
            
            return response
            
        except Exception as e:
            self.logger.error(f"Video generation failed: {e}")
            raise AIVideoError(f"Video generation failed: {e}")
    
    async def generate_video_with_vision(self, request: VideoRequest, image_data: bytes) -> VideoResponse:
        """Generate video with vision capabilities."""
        try:
            # Validate request
            await self._validate_request(request)
            
            # Process with vision
            result = await self.workflow.process_with_vision(request, image_data)
            
            # Create response
            response = VideoResponse(
                request_id=request.request_id,
                status="completed",
                output_url="vision_generated_video.mp4",
                metadata={
                    **result,
                    "workflow": f"onyx_vision_{self.workflow_type}",
                    "generated_at": datetime.now().isoformat()
                }
            )
            
            # Log telemetry
            self.telemetry.log_info("vision_video_generated", {
                "request_id": request.request_id,
                "user_id": request.user_id,
                "image_size": len(image_data),
                "workflow_type": self.workflow_type
            })
            
            return response
            
        except Exception as e:
            self.logger.error(f"Vision video generation failed: {e}")
            raise AIVideoError(f"Vision video generation failed: {e}")
    
    async async def _validate_request(self, request: VideoRequest) -> None:
        """Validate video request."""
        if not request.input_text.strip():
            raise ValidationError("Input text cannot be empty")
        
        if request.duration <= 0 or request.duration > 600:
            raise ValidationError("Duration must be between 1 and 600 seconds")
        
        if request.quality not in ["low", "medium", "high", "ultra"]:
            raise ValidationError("Invalid quality setting")
        
        if not request.output_format:
            raise ValidationError("Output format must be specified")
    
    async def get_generator_status(self) -> GeneratorStatus:
        """Get generator status."""
        workflow_status = await self.workflow.get_workflow_status()
        
        return GeneratorStatus(
            workflow_status=workflow_status,
            telemetry_enabled=True
        )
    
    async def cleanup(self) -> None:
        """Cleanup generator resources."""
        await self.workflow.cleanup()
    
    def get_workflow_type(self) -> str:
        """Get the current workflow type."""
        return self.workflow_type
    
    async def switch_workflow_type(self, new_type: str) -> None:
        """Switch to a different workflow type."""
        try:
            # Cleanup current workflow
            await self.cleanup()
            
            # Create new workflow
            self.workflow = OnyxVideoWorkflow(new_type)
            self.workflow_type = new_type
            
            # Initialize new workflow
            await self.workflow.initialize()
            
            self.logger.info(f"Switched to workflow type: {new_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to switch workflow type: {e}")
            raise AIVideoError(f"Workflow type switch failed: {e}")
    
    async def get_available_workflow_types(self) -> Dict[str, str]:
        """Get available workflow types and their descriptions."""
        return {
            "default": "Standard video generation workflow",
            "vision": "Vision-based video generation workflow",
            "quick": "Quick video generation workflow",
            "advanced": "Advanced video generation with additional processing"
        } 