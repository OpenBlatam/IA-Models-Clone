"""
ComfyUI Tensor Generator
========================

Utilities for generating ComfyUI-compatible safe tensors from clothing changes.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import json
import logging
from safetensors.torch import save_file, load_file
from datetime import datetime
from PIL import Image
import numpy as np

from .flux2_clothing_model import Flux2ClothingChangerModel

logger = logging.getLogger(__name__)


class ComfyUITensorGenerator:
    """
    Generator for ComfyUI-compatible safe tensors from clothing changes.
    
    Creates safe tensor files that can be directly used in ComfyUI workflows
    for character clothing replacement.
    """
    
    def __init__(
        self,
        model: Optional[Flux2ClothingChangerModel] = None,
        output_dir: Union[str, Path] = "./comfyui_tensors",
    ):
        """
        Initialize ComfyUI Tensor Generator.
        
        Args:
            model: Flux2ClothingChangerModel instance (optional)
            output_dir: Directory to save safe tensors
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ComfyUITensorGenerator initialized with output_dir: {self.output_dir}")
    
    def generate_from_clothing_change(
        self,
        original_image: Union[Image.Image, str, Path],
        clothing_description: str,
        changed_image: Optional[Union[Image.Image, str, Path]] = None,
        character_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        output_filename: Optional[str] = None,
    ) -> Path:
        """
        Generate ComfyUI-compatible safe tensor from clothing change.
        
        Args:
            original_image: Original character image
            clothing_description: Description of new clothing
            changed_image: Image with changed clothing (will be generated if not provided)
            character_name: Optional character name
            metadata: Optional additional metadata
            output_filename: Optional custom output filename
            
        Returns:
            Path to generated safe tensor file
        """
        if self.model is None:
            raise ValueError("Model not initialized. Set model or create new one.")
        
        # Generate changed image if not provided
        if changed_image is None:
            logger.info("Generating clothing change...")
            if isinstance(original_image, (str, Path)):
                original_pil = Image.open(original_image).convert("RGB")
            else:
                original_pil = original_image
            
            changed_pil = self.model.change_clothing(
                image=original_pil,
                clothing_description=clothing_description,
            )
        else:
            if isinstance(changed_image, (str, Path)):
                changed_pil = Image.open(changed_image).convert("RGB")
            else:
                changed_pil = changed_image
        
        # Encode character features
        character_embedding = self.model.encode_character(original_image)
        
        # Encode clothing description
        clothing_embedding = self.model.encode_clothing_description(clothing_description)
        
        # Prepare data for ComfyUI
        comfyui_data = {
            # Character embedding
            "character_embedding": character_embedding.cpu(),
            
            # Clothing embedding
            "clothing_embedding": clothing_embedding.cpu(),
            
            # Combined embedding for ComfyUI
            "combined_embedding": torch.cat([character_embedding, clothing_embedding], dim=0).cpu(),
        }
        
        # Prepare metadata
        full_metadata = {
            "character_name": character_name or "unknown",
            "clothing_description": clothing_description,
            "created_at": datetime.now().isoformat(),
            "model_id": self.model.model_id,
            "device": str(self.model.device),
            "comfyui_version": "1.0.0",
            "tensor_format": "safetensors",
        }
        
        if metadata:
            full_metadata.update(metadata)
        
        # Generate filename
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            char_name_safe = (character_name or "character").replace(" ", "_").lower()
            clothing_safe = clothing_description[:30].replace(" ", "_").lower()
            output_filename = f"comfyui_{char_name_safe}_{clothing_safe}_{timestamp}.safetensors"
        
        output_path = self.output_dir / output_filename
        
        # Save safe tensor
        save_file(comfyui_data, str(output_path))
        
        # Save metadata
        metadata_path = output_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(full_metadata, f, indent=2)
        
        # Save images for reference
        images_dir = self.output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Save original image
        original_path = images_dir / f"{output_path.stem}_original.png"
        if isinstance(original_image, (str, Path)):
            import shutil
            shutil.copy(original_image, original_path)
        else:
            original_image.save(original_path)
        
        # Save changed image
        changed_path = images_dir / f"{output_path.stem}_changed.png"
        changed_pil.save(changed_path)
        
        logger.info(f"ComfyUI safe tensor generated: {output_path}")
        return output_path
    
    def generate_workflow_tensor(
        self,
        tensor_path: Union[str, Path],
        workflow_config: Dict[str, Any],
        output_filename: Optional[str] = None,
    ) -> Path:
        """
        Generate workflow-ready safe tensor with ComfyUI workflow configuration.
        
        Args:
            tensor_path: Path to base safe tensor
            workflow_config: ComfyUI workflow configuration
            output_filename: Optional custom output filename
            
        Returns:
            Path to workflow safe tensor
        """
        tensor_path = Path(tensor_path)
        
        # Load base tensor
        data = load_file(str(tensor_path))
        
        # Add workflow config
        workflow_data = {
            **data,
            "workflow_config": workflow_config,
        }
        
        # Generate filename
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"comfyui_workflow_{timestamp}.safetensors"
        
        output_path = self.output_dir / output_filename
        
        # Save workflow tensor
        save_file(workflow_data, str(output_path))
        
        # Save workflow JSON separately for ComfyUI
        workflow_json_path = output_path.with_suffix(".json")
        with open(workflow_json_path, "w") as f:
            json.dump(workflow_config, f, indent=2)
        
        logger.info(f"Workflow safe tensor generated: {output_path}")
        return output_path
    
    def create_comfyui_workflow_json(
        self,
        tensor_path: Union[str, Path],
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        output_filename: Optional[str] = None,
    ) -> Path:
        """
        Create ComfyUI workflow JSON file.
        
        Args:
            tensor_path: Path to safe tensor
            prompt: Generation prompt
            negative_prompt: Negative prompt
            num_inference_steps: Inference steps
            guidance_scale: Guidance scale
            output_filename: Optional custom output filename
            
        Returns:
            Path to workflow JSON file
        """
        tensor_path = Path(tensor_path)
        
        # Load metadata
        metadata_path = tensor_path.with_suffix(".json")
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        
        # Create ComfyUI workflow structure
        workflow = {
            "workflow": {
                "1": {
                    "inputs": {
                        "text": prompt,
                        "clip": ["4", 0],
                    },
                    "class_type": "CLIPTextEncode",
                },
                "2": {
                    "inputs": {
                        "text": negative_prompt or "",
                        "clip": ["4", 0],
                    },
                    "class_type": "CLIPTextEncode",
                },
                "3": {
                    "inputs": {
                        "seed": 42,
                        "steps": num_inference_steps,
                        "cfg": guidance_scale,
                        "sampler_name": "euler",
                        "scheduler": "normal",
                        "denoise": 1.0,
                        "model": ["4", 0],
                        "positive": ["1", 0],
                        "negative": ["2", 0],
                        "latent_image": ["5", 0],
                    },
                    "class_type": "KSampler",
                },
                "4": {
                    "inputs": {
                        "model_name": "flux2-dev.safetensors",
                    },
                    "class_type": "CheckpointLoaderSimple",
                },
                "5": {
                    "inputs": {
                        "width": 1024,
                        "height": 1024,
                        "batch_size": 1,
                    },
                    "class_type": "EmptyLatentImage",
                },
                "6": {
                    "inputs": {
                        "samples": ["3", 0],
                        "vae": ["4", 1],
                    },
                    "class_type": "VAEDecode",
                },
                "7": {
                    "inputs": {
                        "filename_prefix": "comfyui_clothing_change",
                        "images": ["6", 0],
                    },
                    "class_type": "SaveImage",
                },
            },
            "metadata": metadata,
        }
        
        # Generate filename
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"comfyui_workflow_{timestamp}.json"
        
        output_path = self.output_dir / output_filename
        
        # Save workflow JSON
        with open(output_path, "w") as f:
            json.dump(workflow, f, indent=2)
        
        logger.info(f"ComfyUI workflow JSON generated: {output_path}")
        return output_path
    
    def list_generated_tensors(self) -> List[Dict[str, Any]]:
        """
        List all generated safe tensors with metadata.
        
        Returns:
            List of dicts with tensor info
        """
        tensors = []
        
        for tensor_path in self.output_dir.glob("*.safetensors"):
            try:
                # Load metadata if exists
                metadata_path = tensor_path.with_suffix(".json")
                metadata = None
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                
                tensors.append({
                    "path": str(tensor_path),
                    "filename": tensor_path.name,
                    "size": tensor_path.stat().st_size,
                    "metadata": metadata,
                })
            except Exception as e:
                logger.warning(f"Error reading tensor {tensor_path}: {e}")
        
        return tensors


