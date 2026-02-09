"""
Metadata Builder Utilities
===========================

Utilities for building and managing metadata for safe tensors.
"""

from datetime import datetime
from typing import Dict, Any, Optional
import torch


class MetadataBuilder:
    """Helper class for building metadata dictionaries."""
    
    @staticmethod
    def build_embedding_metadata(
        character_name: Optional[str],
        num_images: int,
        embedding: torch.Tensor,
        model_id: str,
        device: str,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build metadata dictionary for character embedding.
        
        Args:
            character_name: Character name
            num_images: Number of images used
            embedding: Embedding tensor
            model_id: Model ID used
            device: Device used
            additional_metadata: Optional additional metadata
            
        Returns:
            Complete metadata dictionary
        """
        metadata = {
            "character_name": character_name or "unknown",
            "num_images": num_images,
            "embedding_dim": embedding.shape[0],
            "created_at": datetime.now().isoformat(),
            "model_id": model_id,
            "device": str(device),
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        return metadata
    
    @staticmethod
    def build_workflow_metadata(
        prompt_template: str,
        negative_prompt: Optional[str],
        num_inference_steps: int,
        guidance_scale: float,
        character_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build metadata dictionary for workflow tensor.
        
        Args:
            prompt_template: Prompt template
            negative_prompt: Negative prompt
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            character_metadata: Optional character metadata
            
        Returns:
            Workflow metadata dictionary
        """
        metadata = {
            "prompt_template": prompt_template,
            "negative_prompt": negative_prompt or "",
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "created_at": datetime.now().isoformat(),
        }
        
        if character_metadata:
            metadata["character_metadata"] = character_metadata
        
        return metadata

