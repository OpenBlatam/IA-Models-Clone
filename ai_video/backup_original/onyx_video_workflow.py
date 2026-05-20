from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import json
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from onyx.llm.factory import get_default_llms, get_default_llm_with_vision
from onyx.llm.interfaces import LLM
from onyx.llm.utils import get_max_input_tokens_from_llm_provider
from onyx.utils.logger import setup_logger
from onyx.utils.threadpool_concurrency import ThreadSafeDict, run_functions_in_parallel, FunctionCall
from onyx.utils.timing import time_function
from onyx.utils.retry_wrapper import retry_wrapper
from onyx.utils.telemetry import TelemetryLogger
from onyx.utils.text_processing import clean_text, extract_keywords
from onyx.utils.gpu_utils import get_gpu_info, is_gpu_available
from onyx.utils.file import get_file_extension, get_file_size
from onyx.core.functions import process_document, format_response, handle_error
from onyx.db.engine import get_session_with_current_tenant
from .models import VideoRequest, VideoResponse, PluginConfig
from .core.exceptions import AIVideoError, PluginError, ValidationError
from .core.onyx_integration import OnyxIntegrationManager, onyx_integration
from typing import Any, List, Dict, Optional
import logging
"""
Onyx Video Workflow

Adapted video workflow that leverages Onyx's LLM infrastructure,
threading utilities, and performance optimizations for AI video generation.
"""


# Onyx imports

# Local imports

logger = setup_logger(__name__)


@dataclass
class OnyxVideoStep:
    """Represents a step in the Onyx video workflow."""
    name: str
    description: str
    llm_prompt: str
    required: bool = True
    timeout: int = 60
    retry_attempts: int = 3
    dependencies: List[str] = field(default_factory=list)


@dataclass
class OnyxVideoContext:
    """Context for Onyx video generation."""
    request: VideoRequest
    llm: Optional[LLM] = None
    vision_llm: Optional[LLM] = None
    gpu_available: bool = False
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class OnyxVideoWorkflow:
    """
    Onyx-adapted video workflow.
    
    Uses Onyx's LLM infrastructure, threading utilities, and performance
    optimizations for efficient video generation.
    """
    
    def __init__(self) -> Any:
        self.logger = setup_logger("onyx_video_workflow")
        self.telemetry = TelemetryLogger()
        self.cache: ThreadSafeDict[str, Any] = ThreadSafeDict()
        
        # Define workflow steps
        self.steps = [
            OnyxVideoStep(
                name="content_analysis",
                description="Analyze input content and extract key themes",
                llm_prompt="Analyze the following content and extract key themes, emotions, and visual elements: {input_text}",
                timeout=30
            ),
            OnyxVideoStep(
                name="script_generation",
                description="Generate video script based on content analysis",
                llm_prompt="Based on the content analysis, generate a compelling video script with scenes, dialogue, and visual descriptions: {content_analysis}",
                dependencies=["content_analysis"],
                timeout=60
            ),
            OnyxVideoStep(
                name="storyboard_creation",
                description="Create storyboard from script",
                llm_prompt="Convert the video script into a detailed storyboard with scene descriptions, camera angles, and visual elements: {script}",
                dependencies=["script_generation"],
                timeout=90
            ),
            OnyxVideoStep(
                name="visual_style_definition",
                description="Define visual style and aesthetic",
                llm_prompt="Based on the content and storyboard, define the visual style, color palette, and aesthetic direction: {storyboard}",
                dependencies=["storyboard_creation"],
                timeout=45
            ),
            OnyxVideoStep(
                name="video_generation",
                description="Generate final video",
                llm_prompt="Generate the final video based on the storyboard and visual style: {storyboard} {visual_style}",
                dependencies=["storyboard_creation", "visual_style_definition"],
                timeout=300
            )
        ]
    
    async def initialize(self) -> None:
        """Initialize the Onyx video workflow."""
        try:
            self.logger.info("Initializing Onyx video workflow")
            
            # Initialize Onyx integration
            await onyx_integration.initialize()
            
            # Check GPU availability
            gpu_available = is_gpu_available()
            if gpu_available:
                gpu_info = get_gpu_info()
                self.logger.info(f"GPU available: {gpu_info}")
            else:
                self.logger.info("GPU not available, using CPU")
            
            self.logger.info("Onyx video workflow initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Workflow initialization failed: {e}")
            raise AIVideoError(f"Workflow initialization failed: {e}")
    
    @retry_wrapper(max_attempts=3, backoff_factor=2)
    async async def process_request(self, request: VideoRequest) -> VideoResponse:
        """Process video request using Onyx workflow."""
        try:
            self.logger.info(f"Processing video request: {request.request_id}")
            
            # Create context
            context = await self._create_context(request)
            
            # Execute workflow steps
            with time_function("onyx_video_workflow"):
                result = await self._execute_workflow(context)
            
            # Create response
            response = VideoResponse(
                request_id=request.request_id,
                status="completed",
                output_url=result.get("output_url", "generated_video.mp4"),
                metadata={
                    **result,
                    "workflow": "onyx",
                    "gpu_used": context.gpu_available,
                    "generated_at": datetime.now().isoformat()
                }
            )
            
            self.logger.info(f"Video request completed: {request.request_id}")
            return response
            
        except Exception as e:
            self.logger.error(f"Video request failed: {request.request_id} - {e}")
            raise AIVideoError(f"Video processing failed: {e}")
    
    async def _create_context(self, request: VideoRequest) -> OnyxVideoContext:
        """Create workflow context."""
        try:
            # Get LLM instances
            llm = await onyx_integration.llm_manager.get_default_llm()
            vision_llm = await onyx_integration.llm_manager.get_vision_llm()
            
            # Check GPU availability
            gpu_available = is_gpu_available()
            
            context = OnyxVideoContext(
                request=request,
                llm=llm,
                vision_llm=vision_llm,
                gpu_available=gpu_available,
                metadata={
                    "user_id": request.user_id,
                    "quality": request.quality,
                    "duration": request.duration,
                    "output_format": request.output_format
                }
            )
            
            return context
            
        except Exception as e:
            self.logger.error(f"Context creation failed: {e}")
            raise AIVideoError(f"Context creation failed: {e}")
    
    async def _execute_workflow(self, context: OnyxVideoContext) -> Dict[str, Any]:
        """Execute workflow steps."""
        try:
            results = {}
            
            # Execute steps in dependency order
            for step in self._get_ordered_steps():
                self.logger.info(f"Executing step: {step.name}")
                
                # Check dependencies
                if not self._check_dependencies(step, results):
                    raise AIVideoError(f"Dependencies not met for step: {step.name}")
                
                # Execute step
                with time_function(f"step_{step.name}"):
                    step_result = await self._execute_step(step, context, results)
                
                results[step.name] = step_result
                self.logger.info(f"Step completed: {step.name}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            raise AIVideoError(f"Workflow execution failed: {e}")
    
    def _get_ordered_steps(self) -> List[OnyxVideoStep]:
        """Get steps in dependency order."""
        # Simple topological sort for dependencies
        ordered_steps = []
        completed_steps = set()
        
        while len(ordered_steps) < len(self.steps):
            for step in self.steps:
                if step.name in completed_steps:
                    continue
                
                # Check if all dependencies are completed
                if all(dep in completed_steps for dep in step.dependencies):
                    ordered_steps.append(step)
                    completed_steps.add(step.name)
            
            # If no progress, there's a circular dependency
            if len(ordered_steps) == len(completed_steps):
                break
        
        return ordered_steps
    
    def _check_dependencies(self, step: OnyxVideoStep, results: Dict[str, Any]) -> bool:
        """Check if step dependencies are met."""
        return all(dep in results for dep in step.dependencies)
    
    async def _execute_step(self, step: OnyxVideoStep, context: OnyxVideoContext, results: Dict[str, Any]) -> Any:
        """Execute a single workflow step."""
        try:
            # Prepare prompt with context
            prompt = self._prepare_prompt(step, context, results)
            
            # Execute with retry
            @retry_wrapper(max_attempts=step.retry_attempts, backoff_factor=2)
            async def execute_with_timeout():
                
    """execute_with_timeout function."""
return await asyncio.wait_for(
                    self._execute_llm_step(prompt, context),
                    timeout=step.timeout
                )
            
            result = await execute_with_timeout()
            
            # Cache result
            self.cache[f"{context.request.request_id}_{step.name}"] = {
                "result": result,
                "timestamp": time.time(),
                "step": step.name
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Step execution failed: {step.name} - {e}")
            if step.required:
                raise AIVideoError(f"Required step failed: {step.name} - {e}")
            else:
                return None
    
    def _prepare_prompt(self, step: OnyxVideoStep, context: OnyxVideoContext, results: Dict[str, Any]) -> str:
        """Prepare prompt for step execution."""
        prompt = step.llm_prompt
        
        # Replace placeholders with actual values
        replacements = {
            "{input_text}": context.request.input_text,
            "{content_analysis}": results.get("content_analysis", ""),
            "{script}": results.get("script_generation", ""),
            "{storyboard}": results.get("storyboard_creation", ""),
            "{visual_style}": results.get("visual_style_definition", ""),
            "{user_id}": context.request.user_id,
            "{quality}": context.request.quality,
            "{duration}": str(context.request.duration),
            "{output_format}": context.request.output_format
        }
        
        for placeholder, value in replacements.items():
            prompt = prompt.replace(placeholder, str(value))
        
        return prompt
    
    async def _execute_llm_step(self, prompt: str, context: OnyxVideoContext) -> str:
        """Execute LLM step."""
        try:
            # Use vision LLM if available and appropriate
            llm = context.vision_llm if context.vision_llm and "visual" in prompt.lower() else context.llm
            
            if llm is None:
                raise AIVideoError("No LLM available")
            
            # Generate response
            response = await llm.agenerate(prompt)
            result = response.generations[0][0].text
            
            # Clean and validate result
            cleaned_result = clean_text(result)
            if not cleaned_result.strip():
                raise AIVideoError("Empty LLM response")
            
            return cleaned_result
            
        except Exception as e:
            self.logger.error(f"LLM step execution failed: {e}")
            raise AIVideoError(f"LLM execution failed: {e}")
    
    async def process_with_vision(self, request: VideoRequest, image_data: bytes) -> VideoResponse:
        """Process video request with vision capabilities."""
        try:
            self.logger.info(f"Processing vision request: {request.request_id}")
            
            # Create context
            context = await self._create_context(request)
            
            if context.vision_llm is None:
                raise AIVideoError("Vision LLM not available")
            
            # Generate vision-based content
            vision_prompt = f"Analyze this image and create a video script: {request.input_text}"
            vision_result = await onyx_integration.llm_manager.generate_with_vision(
                vision_prompt, image_data
            )
            
            # Create response
            response = VideoResponse(
                request_id=request.request_id,
                status="completed",
                output_url="vision_generated_video.mp4",
                metadata={
                    "vision_analysis": vision_result,
                    "workflow": "onyx_vision",
                    "gpu_used": context.gpu_available,
                    "generated_at": datetime.now().isoformat()
                }
            )
            
            self.logger.info(f"Vision request completed: {request.request_id}")
            return response
            
        except Exception as e:
            self.logger.error(f"Vision request failed: {request.request_id} - {e}")
            raise AIVideoError(f"Vision processing failed: {e}")
    
    async def get_workflow_status(self) -> Dict[str, Any]:
        """Get workflow status."""
        try:
            status = {
                "workflow": "onyx_video",
                "steps": [step.name for step in self.steps],
                "gpu_available": is_gpu_available(),
                "gpu_info": get_gpu_info() if is_gpu_available() else None,
                "cache_size": len(self.cache),
                "onyx_integration": True,
                "timestamp": datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Status check failed: {e}")
            return {"error": str(e)}
    
    async def cleanup(self) -> None:
        """Cleanup workflow resources."""
        try:
            self.cache.clear()
            self.logger.info("Workflow cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


class OnyxVideoGenerator:
    """
    Onyx video generator with advanced features.
    
    Provides video generation capabilities using Onyx's infrastructure
    with support for different video types and styles.
    """
    
    def __init__(self) -> Any:
        self.logger = setup_logger("onyx_video_generator")
        self.workflow = OnyxVideoWorkflow()
        self.telemetry = TelemetryLogger()
    
    async def initialize(self) -> None:
        """Initialize the video generator."""
        await self.workflow.initialize()
    
    async def generate_video(self, request: VideoRequest) -> VideoResponse:
        """Generate video using Onyx workflow."""
        try:
            # Validate request
            await self._validate_request(request)
            
            # Process request
            response = await self.workflow.process_request(request)
            
            # Log telemetry
            self.telemetry.log_info("video_generated", {
                "request_id": request.request_id,
                "user_id": request.user_id,
                "quality": request.quality,
                "duration": request.duration
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
            response = await self.workflow.process_with_vision(request, image_data)
            
            # Log telemetry
            self.telemetry.log_info("vision_video_generated", {
                "request_id": request.request_id,
                "user_id": request.user_id,
                "image_size": len(image_data)
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
    
    async def get_generator_status(self) -> Dict[str, Any]:
        """Get generator status."""
        workflow_status = await self.workflow.get_workflow_status()
        
        return {
            "generator": "onyx_video",
            "workflow_status": workflow_status,
            "telemetry_enabled": True,
            "timestamp": datetime.now().isoformat()
        }
    
    async def cleanup(self) -> None:
        """Cleanup generator resources."""
        await self.workflow.cleanup()


# Global instances
onyx_video_workflow = OnyxVideoWorkflow()
onyx_video_generator = OnyxVideoGenerator()


# Utility functions
async def initialize_onyx_video_system() -> None:
    """Initialize Onyx video system."""
    await onyx_video_generator.initialize()


async def generate_onyx_video(request: VideoRequest) -> VideoResponse:
    """Generate video using Onyx system."""
    return await onyx_video_generator.generate_video(request)


async def generate_onyx_video_with_vision(request: VideoRequest, image_data: bytes) -> VideoResponse:
    """Generate video with vision using Onyx system."""
    return await onyx_video_generator.generate_video_with_vision(request, image_data)


def get_onyx_video_status() -> Dict[str, Any]:
    """Get Onyx video system status."""
    return asyncio.run(onyx_video_generator.get_generator_status()) 