from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import time
from typing import Any, Dict, List, Optional
from datetime import datetime
from onyx.utils.logger import setup_logger
from onyx.utils.threadpool_concurrency import ThreadSafeDict
from onyx.utils.timing import time_function
from onyx.utils.retry_wrapper import retry_wrapper
from onyx.utils.telemetry import TelemetryLogger
from onyx.utils.text_processing import clean_text
from onyx.utils.gpu_utils import get_gpu_info, is_gpu_available
from .models import (
from ...core.exceptions import AIVideoError
from ...core.onyx_integration import onyx_integration
        from ..steps import get_workflow_steps_by_type
from typing import Any, List, Dict, Optional
import logging
"""
Onyx Video Workflow - Core Workflow Engine

Main workflow engine for video generation using Onyx's infrastructure.
"""


# Onyx imports

# Local imports
    OnyxVideoStep, 
    OnyxVideoContext, 
    WorkflowExecutionResult,
    StepExecutionResult,
    WorkflowStatus
)


class OnyxVideoWorkflow:
    """
    Onyx-adapted video workflow.
    
    Uses Onyx's LLM infrastructure, threading utilities, and performance
    optimizations for efficient video generation.
    """
    
    def __init__(self, workflow_type: str = "default"):
        
    """__init__ function."""
self.logger = setup_logger("onyx_video_workflow")
        self.telemetry = TelemetryLogger()
        self.cache: ThreadSafeDict[str, Any] = ThreadSafeDict()
        self.workflow_type = workflow_type
        
        # Import steps dynamically to avoid circular imports
        self.steps = get_workflow_steps_by_type(workflow_type)
    
    async def initialize(self) -> None:
        """Initialize the Onyx video workflow."""
        try:
            self.logger.info(f"Initializing Onyx video workflow (type: {self.workflow_type})")
            
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
    async async def process_request(self, request: Any) -> Dict[str, Any]:
        """Process video request using Onyx workflow."""
        try:
            self.logger.info(f"Processing video request: {request.request_id}")
            
            # Create context
            context = await self._create_context(request)
            
            # Execute workflow steps
            with time_function("onyx_video_workflow"):
                result = await self._execute_workflow(context)
            
            self.logger.info(f"Video request completed: {request.request_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Video request failed: {request.request_id} - {e}")
            raise AIVideoError(f"Video processing failed: {e}")
    
    async def _create_context(self, request: Any) -> OnyxVideoContext:
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
    
    async def process_with_vision(self, request: Any, image_data: bytes) -> Dict[str, Any]:
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
            
            result = {
                "vision_analysis": vision_result,
                "workflow": "onyx_vision",
                "gpu_used": context.gpu_available,
                "generated_at": datetime.now().isoformat()
            }
            
            self.logger.info(f"Vision request completed: {request.request_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Vision request failed: {request.request_id} - {e}")
            raise AIVideoError(f"Vision processing failed: {e}")
    
    async def get_workflow_status(self) -> WorkflowStatus:
        """Get workflow status."""
        try:
            status = WorkflowStatus(
                steps=[step.name for step in self.steps],
                gpu_available=is_gpu_available(),
                gpu_info=get_gpu_info() if is_gpu_available() else None,
                cache_size=len(self.cache)
            )
            
            return status
            
        except Exception as e:
            self.logger.error(f"Status check failed: {e}")
            return WorkflowStatus()
    
    async def cleanup(self) -> None:
        """Cleanup workflow resources."""
        try:
            self.cache.clear()
            self.logger.info("Workflow cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}") 