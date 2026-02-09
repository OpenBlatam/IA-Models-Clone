"""
Clothing Change Service
=======================

Main service that orchestrates clothing changes using OpenRouter, TruthGPT, and ComfyUI.

This service coordinates the entire workflow:
1. Prompt optimization with OpenRouter (optional)
2. Enhancement with TruthGPT (optional)
3. Execution via ComfyUI workflow
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from infrastructure.openrouter_client import get_openrouter_client
from infrastructure.truthgpt_client import TruthGPTClient
from services.comfyui_service import ComfyUIService
from services.metrics_service import get_metrics_service
from services.webhook_service import get_webhook_service, WebhookEvent
from config.settings import get_settings

logger = logging.getLogger(__name__)


# Constants
class ServiceStatus(str, Enum):
    """Service execution status"""
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    PROCESSING = "processing"


# Default values
DEFAULT_GUIDANCE_SCALE = 50.0
DEFAULT_NUM_STEPS = 12
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted, deformed, bad anatomy"


@dataclass
class ClothingChangeRequest:
    """Request parameters for clothing change"""
    image_url: str
    clothing_description: str
    mask_url: Optional[str] = None
    character_name: Optional[str] = None
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    num_steps: int = DEFAULT_NUM_STEPS
    seed: Optional[int] = None
    optimize_prompt: bool = True


@dataclass
class ClothingChangeResult:
    """Result of clothing change operation"""
    success: bool
    prompt_id: Optional[str] = None
    original_prompt: Optional[str] = None
    optimized_prompt: Optional[str] = None
    error: Optional[str] = None
    openrouter_used: bool = False
    truthgpt_used: bool = False
    workflow_status: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None


class ClothingChangeService:
    """
    Main service for character clothing changes.
    
    Orchestrates the complete workflow:
    - Prompt optimization via OpenRouter
    - Enhancement via TruthGPT
    - Execution via ComfyUI
    """
    
    def __init__(self):
        """Initialize the service with all required clients"""
        self.settings = get_settings()
        self.openrouter_client = self._initialize_openrouter()
        self.truthgpt_client = self._initialize_truthgpt()
        self.comfyui_service = ComfyUIService()
        self.metrics_service = get_metrics_service()
        self.webhook_service = get_webhook_service()
    
    def _initialize_openrouter(self) -> Optional[Any]:
        """Initialize OpenRouter client if enabled"""
        if not self.settings.openrouter_enabled:
            logger.info("OpenRouter is disabled")
            return None
        
        try:
            client = get_openrouter_client()
            logger.info("OpenRouter client initialized")
            return client
        except Exception as e:
            logger.warning(f"Failed to initialize OpenRouter client: {e}")
            return None
    
    def _initialize_truthgpt(self) -> Optional[TruthGPTClient]:
        """Initialize TruthGPT client if enabled"""
        if not self.settings.truthgpt_enabled:
            logger.info("TruthGPT is disabled")
            return None
        
        try:
            client = TruthGPTClient({
                "truthgpt_endpoint": self.settings.truthgpt_endpoint,
                "timeout": self.settings.truthgpt_timeout
            })
            logger.info("TruthGPT client initialized")
            return client
        except Exception as e:
            logger.warning(f"Failed to initialize TruthGPT client: {e}")
            return None
    
    async def change_clothing(
        self,
        image_url: str,
        clothing_description: str,
        mask_url: Optional[str] = None,
        character_name: Optional[str] = None,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        num_steps: int = DEFAULT_NUM_STEPS,
        seed: Optional[int] = None,
        optimize_prompt: bool = True
    ) -> Dict[str, Any]:
        """
        Change character clothing with AI assistance.
        
        This method orchestrates the complete workflow:
        1. Validates input parameters
        2. Optimizes prompt with OpenRouter (if enabled)
        3. Enhances with TruthGPT (if enabled)
        4. Executes ComfyUI workflow
        
        Args:
            image_url: URL or path to input image
            clothing_description: Description of desired clothing
            mask_url: URL or path to mask image (optional)
            character_name: Name of character (for context)
            negative_prompt: Negative prompt for generation
            guidance_scale: Guidance scale (default: 50.0)
            num_steps: Number of inference steps (default: 12)
            seed: Random seed for reproducibility
            optimize_prompt: Whether to optimize prompt with OpenRouter
            
        Returns:
            Dictionary with execution results:
            - success: bool - Whether operation succeeded
            - prompt_id: str - ComfyUI prompt ID
            - original_prompt: str - Original user prompt
            - optimized_prompt: str - Final optimized prompt
            - openrouter_used: bool - Whether OpenRouter was used
            - truthgpt_used: bool - Whether TruthGPT was used
            - workflow_status: str - ComfyUI workflow status
            - settings: dict - Generation settings used
            - error: str - Error message if failed
            
        Raises:
            ValueError: If required parameters are invalid
        """
        # Validate inputs
        self._validate_inputs(image_url, clothing_description, guidance_scale, num_steps)
        
        start_time = time.time()
        try:
            # Step 1: Optimize prompt with OpenRouter
            final_prompt, openrouter_used = await self._optimize_prompt(
                clothing_description=clothing_description,
                character_name=character_name,
                optimize=optimize_prompt
            )
            
            # Step 2: Enhance with TruthGPT
            final_prompt, truthgpt_used = await self._enhance_with_truthgpt(
                prompt=final_prompt,
                image_url=image_url,
                clothing_description=clothing_description,
                character_name=character_name
            )
            
            # Step 3: Execute ComfyUI workflow
            workflow_result = await self._execute_workflow(
                image_url=image_url,
                mask_url=mask_url,
                clothing_description=clothing_description,
                prompt=final_prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                seed=seed
            )
            
            duration = time.time() - start_time
            result = self._build_success_response(
                prompt_id=workflow_result.get("prompt_id"),
                original_prompt=clothing_description,
                optimized_prompt=final_prompt,
                openrouter_used=openrouter_used,
                truthgpt_used=truthgpt_used,
                workflow_status=workflow_result.get("status"),
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                seed=seed
            )
            
            # Record metrics
            self.metrics_service.record_operation(
                operation_type="clothing_change",
                success=True,
                duration=duration,
                prompt_id=workflow_result.get("prompt_id"),
                openrouter_used=openrouter_used,
                truthgpt_used=truthgpt_used
            )
            
            # Send webhook notification
            await self._send_workflow_webhook(
                event_type="workflow_completed",
                prompt_id=workflow_result.get("prompt_id"),
                data=result
            )
            
            return result
            
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            duration = time.time() - start_time
            error_result = self._build_error_response(str(e), clothing_description)
            
            # Record metrics
            self.metrics_service.record_operation(
                operation_type="clothing_change",
                success=False,
                duration=duration,
                error=str(e)
            )
            
            # Send webhook notification for failure
            await self._send_workflow_webhook(
                event_type="workflow_failed",
                prompt_id=None,
                data={"error": str(e), "clothing_description": clothing_description}
            )
            
            return error_result
        except Exception as e:
            logger.error(f"Error changing clothing: {e}", exc_info=True)
            duration = time.time() - start_time
            error_result = self._build_error_response(str(e), clothing_description)
            
            # Record metrics
            self.metrics_service.record_operation(
                operation_type="clothing_change",
                success=False,
                duration=duration,
                error=str(e)
            )
            
            # Send webhook notification for failure
            await self._send_workflow_webhook(
                event_type="workflow_failed",
                prompt_id=None,
                data={"error": str(e), "clothing_description": clothing_description}
            )
            
            return error_result
    
    def _validate_inputs(
        self,
        image_url: str,
        clothing_description: str,
        guidance_scale: float,
        num_steps: int
    ) -> None:
        """Validate input parameters"""
        if not image_url or not image_url.strip():
            raise ValueError("image_url is required and cannot be empty")
        
        if not clothing_description or not clothing_description.strip():
            raise ValueError("clothing_description is required and cannot be empty")
        
        if not (1.0 <= guidance_scale <= 100.0):
            raise ValueError(f"guidance_scale must be between 1.0 and 100.0, got {guidance_scale}")
        
        if not (1 <= num_steps <= 100):
            raise ValueError(f"num_steps must be between 1 and 100, got {num_steps}")
    
    async def _optimize_prompt(
        self,
        clothing_description: str,
        character_name: Optional[str],
        optimize: bool
    ) -> Tuple[str, bool]:
        """
        Optimize prompt with OpenRouter if enabled.
        
        Returns:
            Tuple of (optimized_prompt, was_used)
        """
        if not optimize or not self.openrouter_client:
            return clothing_description, False
        
        try:
            logger.info("Optimizing prompt with OpenRouter...")
            context = {
                "clothing_description": clothing_description,
                "character_name": character_name,
                "task": "character_clothing_inpainting"
            }
            
            optimized = await self.openrouter_client.optimize_prompt(
                original_prompt=clothing_description,
                context=context,
                model=self.settings.openrouter_model
            )
            
            logger.info(f"Prompt optimized: {clothing_description[:50]}... -> {optimized[:50]}...")
            return optimized, True
            
        except Exception as e:
            logger.warning(f"OpenRouter optimization failed, using original prompt: {e}")
            return clothing_description, False
    
    async def _enhance_with_truthgpt(
        self,
        prompt: str,
        image_url: str,
        clothing_description: str,
        character_name: Optional[str]
    ) -> Tuple[str, bool]:
        """
        Enhance prompt with TruthGPT if enabled.
        
        Returns:
            Tuple of (enhanced_prompt, was_used)
        """
        if not self.truthgpt_client:
            return prompt, False
        
        try:
            logger.info("Enhancing prompt with TruthGPT...")
            context = {
                "image_url": image_url,
                "clothing_description": clothing_description,
                "character_name": character_name
            }
            
            result = await self.truthgpt_client.process_with_truthgpt(
                query=prompt,
                context=context
            )
            
            if result.get("truthgpt_enhanced"):
                enhanced = result.get("result", prompt)
                if isinstance(enhanced, str) and enhanced.strip():
                    logger.info("TruthGPT enhancement applied")
                    return enhanced, True
            
            return prompt, False
            
        except Exception as e:
            logger.warning(f"TruthGPT enhancement failed, using original prompt: {e}")
            return prompt, False
    
    async def _execute_workflow(
        self,
        image_url: str,
        mask_url: Optional[str],
        clothing_description: str,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        num_steps: int,
        seed: Optional[int]
    ) -> Dict[str, Any]:
        """Execute ComfyUI workflow"""
        logger.info("Executing ComfyUI workflow...")
        
        return await self.comfyui_service.execute_clothing_change(
            image_url=image_url,
            mask_url=mask_url,
            clothing_description=clothing_description,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            seed=seed
        )
    
    def _build_success_response(
        self,
        prompt_id: Optional[str],
        original_prompt: str,
        optimized_prompt: str,
        openrouter_used: bool,
        truthgpt_used: bool,
        workflow_status: Optional[str],
        guidance_scale: float,
        num_steps: int,
        seed: Optional[int]
    ) -> Dict[str, Any]:
        """Build success response dictionary"""
        return {
            "success": True,
            "prompt_id": prompt_id,
            "original_prompt": original_prompt,
            "optimized_prompt": optimized_prompt,
            "openrouter_used": openrouter_used,
            "truthgpt_used": truthgpt_used,
            "workflow_status": workflow_status,
            "settings": {
                "guidance_scale": guidance_scale,
                "num_steps": num_steps,
                "seed": seed
            }
        }
    
    def _build_error_response(
        self,
        error_message: str,
        original_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build error response dictionary"""
        response = {
            "success": False,
            "error": error_message
        }
        
        if original_prompt:
            response["original_prompt"] = original_prompt
        
        return response
    
    async def get_status(self, prompt_id: str) -> Dict[str, Any]:
        """
        Get status of a workflow execution.
        
        Args:
            prompt_id: ComfyUI prompt ID to check
            
        Returns:
            Dictionary with status information:
            - prompt_id: str - The prompt ID checked
            - queue_status: dict - ComfyUI queue status
            - error: str - Error message if failed
        """
        if not prompt_id or not prompt_id.strip():
            return self._build_error_response("prompt_id is required")
        
        try:
            queue_status = await self.comfyui_service.get_queue_status()
            
            return {
                "prompt_id": prompt_id,
                "queue_status": queue_status,
                "status": ServiceStatus.PROCESSING.value
            }
            
        except Exception as e:
            logger.error(f"Error getting status for prompt_id {prompt_id}: {e}", exc_info=True)
            return {
                "prompt_id": prompt_id,
                "error": str(e),
                "status": ServiceStatus.FAILED.value
            }
    
    async def face_swap(
        self,
        image_url: str,
        face_url: str,
        mask_url: Optional[str] = None,
        character_name: Optional[str] = None,
        prompt: Optional[str] = None,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        num_steps: int = DEFAULT_NUM_STEPS,
        seed: Optional[int] = None,
        optimize_prompt: bool = True
    ) -> Dict[str, Any]:
        """
        Swap face in an image that's being processed with inpainting.
        
        This method orchestrates face swap:
        1. Optimizes prompt with OpenRouter (if enabled)
        2. Enhances with TruthGPT (if enabled)
        3. Executes ComfyUI workflow with face swap
        
        Args:
            image_url: URL or path to input image (the image in inpainting)
            face_url: URL or path to face image to swap in
            mask_url: URL or path to mask image (optional)
            character_name: Name of character (for context)
            prompt: Custom prompt (if not provided, uses default face swap prompt)
            negative_prompt: Negative prompt
            guidance_scale: Guidance scale (default: 50.0)
            num_steps: Number of inference steps (default: 12)
            seed: Random seed
            optimize_prompt: Whether to optimize prompt with OpenRouter
            
        Returns:
            Dictionary with execution results
        """
        # Validate inputs
        if not image_url or not image_url.strip():
            return self._build_error_response("image_url is required")
        if not face_url or not face_url.strip():
            return self._build_error_response("face_url is required for face swap")
        
        # Default prompt for face swap
        default_prompt = "best quality, face swap, high quality portrait, realistic face, detailed facial features"
        final_prompt = prompt or default_prompt
        
        start_time = time.time()
        try:
            # Step 1: Optimize prompt with OpenRouter
            if optimize_prompt and self.openrouter_client:
                logger.info("Optimizing face swap prompt with OpenRouter...")
                context = {
                    "task": "face_swap",
                    "character_name": character_name,
                    "image_url": image_url,
                    "face_url": face_url
                }
                optimized, openrouter_used = await self._optimize_prompt(
                    clothing_description=final_prompt,
                    character_name=character_name,
                    optimize=optimize_prompt
                )
                if openrouter_used:
                    final_prompt = optimized
            else:
                openrouter_used = False
            
            # Step 2: Enhance with TruthGPT
            final_prompt, truthgpt_used = await self._enhance_with_truthgpt(
                prompt=final_prompt,
                image_url=image_url,
                clothing_description=f"face swap with {face_url}",
                character_name=character_name
            )
            
            # Step 3: Execute ComfyUI workflow with face swap
            workflow_result = await self.comfyui_service.execute_face_swap(
                image_url=image_url,
                face_url=face_url,
                mask_url=mask_url,
                prompt=final_prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                seed=seed
            )
            
            duration = time.time() - start_time
            result = {
                "success": True,
                "prompt_id": workflow_result.get("prompt_id"),
                "original_prompt": prompt or default_prompt,
                "optimized_prompt": final_prompt,
                "openrouter_used": openrouter_used,
                "truthgpt_used": truthgpt_used,
                "workflow_status": workflow_result.get("status"),
                "face_swap": True,
                "face_url": face_url,
                "settings": {
                    "guidance_scale": guidance_scale,
                    "num_steps": num_steps,
                    "seed": seed
                }
            }
            
            # Record metrics
            self.metrics_service.record_operation(
                operation_type="face_swap",
                success=True,
                duration=duration,
                prompt_id=workflow_result.get("prompt_id"),
                openrouter_used=openrouter_used,
                truthgpt_used=truthgpt_used
            )
            
            # Send webhook notification
            await self._send_workflow_webhook(
                event_type="workflow_completed",
                prompt_id=workflow_result.get("prompt_id"),
                data=result
            )
            
            return result
            
        except ValueError as e:
            logger.error(f"Validation error in face swap: {e}")
            duration = time.time() - start_time
            error_result = self._build_error_response(str(e))
            
            # Record metrics
            self.metrics_service.record_operation(
                operation_type="face_swap",
                success=False,
                duration=duration,
                error=str(e)
            )
            
            # Send webhook notification for failure
            await self._send_workflow_webhook(
                event_type="workflow_failed",
                prompt_id=None,
                data={"error": str(e), "operation": "face_swap"}
            )
            
            return error_result
        except Exception as e:
            logger.error(f"Error in face swap: {e}", exc_info=True)
            duration = time.time() - start_time
            error_result = self._build_error_response(str(e))
            
            # Record metrics
            self.metrics_service.record_operation(
                operation_type="face_swap",
                success=False,
                duration=duration,
                error=str(e)
            )
            
            # Send webhook notification for failure
            await self._send_workflow_webhook(
                event_type="workflow_failed",
                prompt_id=None,
                data={"error": str(e), "operation": "face_swap"}
            )
            
            return error_result
    
    async def get_analytics(self) -> Dict[str, Any]:
        """
        Get analytics from all services.
        
        Returns:
            Dictionary with analytics from all enabled services:
            - openrouter_enabled: bool
            - truthgpt_enabled: bool
            - comfyui_url: str
            - truthgpt: dict - TruthGPT analytics if enabled
        """
        analytics = {
            "openrouter_enabled": self.openrouter_client is not None,
            "truthgpt_enabled": self.truthgpt_client is not None,
            "comfyui_url": self.settings.comfyui_api_url,
            "services": {
                "openrouter": {
                    "enabled": self.openrouter_client is not None,
                    "model": self.settings.openrouter_model if self.openrouter_client else None
                },
                "truthgpt": {
                    "enabled": self.truthgpt_client is not None,
                    "endpoint": self.settings.truthgpt_endpoint if self.truthgpt_client else None
                },
                "comfyui": {
                    "url": self.settings.comfyui_api_url,
                    "workflow_path": self.settings.comfyui_workflow_path
                }
            }
        }
        
        # Get TruthGPT analytics if available
        if self.truthgpt_client:
            try:
                truthgpt_analytics = await self.truthgpt_client.get_analytics()
                analytics["truthgpt"] = truthgpt_analytics
            except Exception as e:
                logger.warning(f"Failed to get TruthGPT analytics: {e}")
                analytics["truthgpt"] = {"error": str(e)}
        
        return analytics

