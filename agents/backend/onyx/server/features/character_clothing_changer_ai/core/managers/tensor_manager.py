"""
Tensor Manager
=============

Handles ComfyUI tensor generation and management.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from ...models.comfyui_tensor_generator import ComfyUITensorGenerator
from ...config.clothing_changer_config import ClothingChangerConfig

logger = logging.getLogger(__name__)


class TensorManager:
    """Manages ComfyUI tensor operations."""
    
    def __init__(
        self,
        generator: ComfyUITensorGenerator,
        config: ClothingChangerConfig
    ):
        """
        Initialize Tensor Manager.
        
        Args:
            generator: ComfyUI tensor generator
            config: Configuration instance
        """
        self.generator = generator
        self.config = config
    
    def generate_tensor(
        self,
        original_image: Union[str, Path],
        clothing_description: str,
        changed_image,
        character_name: Optional[str] = None,
        output_filename: Optional[str] = None,
    ) -> Path:
        """
        Generate ComfyUI safe tensor from clothing change.
        
        Args:
            original_image: Original character image
            clothing_description: Clothing description
            changed_image: Changed image
            character_name: Optional character name
            output_filename: Optional custom output filename
            
        Returns:
            Path to generated tensor
        """
        return self.generator.generate_from_clothing_change(
            original_image=original_image,
            clothing_description=clothing_description,
            changed_image=changed_image,
            character_name=character_name,
            output_filename=output_filename,
        )
    
    def create_workflow(
        self,
        tensor_path: Union[str, Path],
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> Path:
        """
        Create ComfyUI workflow JSON from tensor.
        
        Args:
            tensor_path: Path to safe tensor
            prompt: Generation prompt
            negative_prompt: Negative prompt
            num_inference_steps: Inference steps
            guidance_scale: Guidance scale
            
        Returns:
            Path to workflow JSON
        """
        return self.generator.create_comfyui_workflow_json(
            tensor_path=tensor_path,
            prompt=prompt,
            negative_prompt=negative_prompt or self.config.default_negative_prompt,
            num_inference_steps=num_inference_steps or self.config.default_num_inference_steps,
            guidance_scale=guidance_scale or self.config.default_guidance_scale,
        )
    
    def list_tensors(self) -> List[Dict[str, Any]]:
        """
        List all generated safe tensors.
        
        Returns:
            List of tensor info dicts
        """
        return self.generator.list_generated_tensors()


