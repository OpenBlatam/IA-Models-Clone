"""
Tensor Workflow Processor
========================

Handles tensor and workflow operations, including generation,
listing, and ComfyUI workflow creation.
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from ...core.utils.generator_manager import GeneratorManager
from ...config.clothing_changer_config import ClothingChangerConfig

logger = logging.getLogger(__name__)


class TensorWorkflowProcessor:
    """Processes tensor and workflow operations."""
    
    def __init__(
        self,
        generator_manager: Optional[GeneratorManager],
        config: ClothingChangerConfig,
        initialize_model_fn: callable,
    ):
        """
        Initialize Tensor Workflow Processor.
        
        Args:
            generator_manager: Generator manager instance
            config: Configuration instance
            initialize_model_fn: Function to initialize model
        """
        self.generator_manager = generator_manager
        self.config = config
        self.initialize_model_fn = initialize_model_fn
    
    def create_workflow(
        self,
        tensor_path: Path,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Create ComfyUI workflow JSON from tensor.
        
        Args:
            tensor_path: Path to safe tensor
            prompt: Generation prompt
            negative_prompt: Negative prompt
            num_inference_steps: Inference steps
            guidance_scale: Guidance scale
            
        Returns:
            Dict with workflow info
        """
        if not self.generator_manager:
            raise RuntimeError("Generator manager not initialized")
        
        self.generator_manager.ensure_generator_initialized()
        
        workflow_path = self.generator_manager.create_workflow(
            tensor_path=tensor_path,
            prompt=prompt,
            negative_prompt=negative_prompt or self.config.default_negative_prompt,
            num_inference_steps=num_inference_steps or self.config.default_num_inference_steps,
            guidance_scale=guidance_scale or self.config.default_guidance_scale,
        )
        
        return {
            "workflow_path": str(workflow_path),
            "created": True,
        }
    
    def list_tensors(
        self,
        model,
        use_deepseek_fallback: bool
    ) -> List[Dict[str, Any]]:
        """
        List all generated safe tensors.
        
        Args:
            model: Model instance
            use_deepseek_fallback: Whether using DeepSeek fallback
            
        Returns:
            List of tensor info dicts
        """
        # Check if tensor generation is possible
        if not GeneratorManager.can_generate_tensors(model, use_deepseek_fallback):
            return []
        
        if not self.generator_manager:
            return []
        
        self.generator_manager.ensure_generator_initialized()
        
        return self.generator_manager.list_tensors()
    
    def can_generate_tensors(
        self,
        model,
        use_deepseek_fallback: bool
    ) -> bool:
        """
        Check if tensor generation is possible.
        
        Args:
            model: Model instance
            use_deepseek_fallback: Whether using DeepSeek fallback
            
        Returns:
            True if tensor generation is possible
        """
        return GeneratorManager.can_generate_tensors(model, use_deepseek_fallback)

