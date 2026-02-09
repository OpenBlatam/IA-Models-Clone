"""
Safe Tensor Generator for Character Consistency
=================================================

Utilities for generating and managing safe tensors for character consistency workflows.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import json
import logging
from safetensors.torch import save_file, load_file
from datetime import datetime

from .flux2_character_model import Flux2CharacterConsistencyModel
from .utils.filename_generator import FilenameGenerator
from .utils.metadata_builder import MetadataBuilder

logger = logging.getLogger(__name__)


class SafeTensorGenerator:
    """
    Generator for safe tensors from character consistency embeddings.
    
    Creates safe tensor files that can be used in workflows for maintaining
    character consistency across image generations.
    """
    
    def __init__(
        self,
        model: Optional[Flux2CharacterConsistencyModel] = None,
        output_dir: Union[str, Path] = "./character_embeddings",
    ):
        """
        Initialize Safe Tensor Generator.
        
        Args:
            model: Flux2CharacterConsistencyModel instance (optional, can be created later)
            output_dir: Directory to save safe tensors
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"SafeTensorGenerator initialized with output_dir: {self.output_dir}")
    
    def generate_from_images(
        self,
        images: Union[
            List[Union[str, Path]],
            str,
            Path
        ],
        character_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        output_filename: Optional[str] = None,
    ) -> Path:
        """
        Generate safe tensor from one or multiple images.
        
        Args:
            images: Single image path or list of image paths
            character_name: Optional name for the character
            metadata: Optional additional metadata
            output_filename: Optional custom output filename
            
        Returns:
            Path to generated safe tensor file
        """
        if self.model is None:
            raise ValueError("Model not initialized. Set model or create new one.")
        
        image_list = self._normalize_images(images)
        embedding = self._generate_embedding(image_list)
        full_metadata = self._build_metadata(character_name, len(image_list), embedding, metadata)
        output_path = self._resolve_output_path(character_name, output_filename)
        
        self.model.save_embedding(embedding, output_path, full_metadata)
        logger.info(f"Safe tensor generated: {output_path}")
        return output_path
    
    def _normalize_images(
        self,
        images: Union[List[Union[str, Path]], str, Path]
    ) -> List[Union[str, Path]]:
        """Normalize images input to list format."""
        return [images] if isinstance(images, (str, Path)) else images
    
    def _generate_embedding(
        self,
        image_list: List[Union[str, Path]]
    ) -> torch.Tensor:
        """Generate embedding from image list."""
        logger.info(f"Generating character embedding from {len(image_list)} image(s)")
        return self.model(image_list)
    
    def _build_metadata(
        self,
        character_name: Optional[str],
        num_images: int,
        embedding: torch.Tensor,
        additional_metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build complete metadata dictionary."""
        return MetadataBuilder.build_embedding_metadata(
            character_name=character_name,
            num_images=num_images,
            embedding=embedding,
            model_id=self.model.model_id,
            device=str(self.model.device),
            additional_metadata=additional_metadata
        )
    
    def _resolve_output_path(
        self,
        character_name: Optional[str],
        output_filename: Optional[str]
    ) -> Path:
        """Resolve output path for safe tensor."""
        if output_filename is None:
            output_filename = FilenameGenerator.generate_embedding_filename(character_name)
        return self.output_dir / output_filename
    
    def generate_workflow_tensor(
        self,
        embedding: torch.Tensor,
        workflow_config: Dict[str, Any],
        output_filename: Optional[str] = None,
    ) -> Path:
        """
        Generate safe tensor optimized for workflow usage.
        
        Args:
            embedding: Character embedding tensor
            workflow_config: Workflow configuration (prompts, settings, etc.)
            output_filename: Optional custom output filename
            
        Returns:
            Path to generated workflow-safe tensor file
        """
        # Prepare data for workflow
        workflow_data = {
            "character_embedding": embedding.cpu(),
            "workflow_config": workflow_config,
        }
        
        # Generate filename
        if output_filename is None:
            output_filename = FilenameGenerator.generate_workflow_filename()
        
        output_path = self.output_dir / output_filename
        
        # Save safe tensor
        save_file(workflow_data, str(output_path))
        
        # Save workflow config as JSON
        config_path = output_path.with_suffix(".json")
        with open(config_path, "w") as f:
            json.dump(workflow_config, f, indent=2)
        
        logger.info(f"Workflow safe tensor generated: {output_path}")
        return output_path
    
    def load_workflow_tensor(
        self,
        tensor_path: Union[str, Path],
        device: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Load workflow safe tensor.
        
        Args:
            tensor_path: Path to safe tensor file
            device: Device to load tensor on
            
        Returns:
            Tuple of (embedding tensor, workflow config, metadata)
        """
        tensor_path = Path(tensor_path)
        
        # Load safe tensor
        data = load_file(str(tensor_path))
        embedding = data["character_embedding"]
        workflow_config = data.get("workflow_config", {})
        
        # Move to device if specified
        if device:
            embedding = embedding.to(device)
        
        # Load metadata if exists
        metadata_path = tensor_path.with_suffix(".json")
        metadata = None
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        
        return embedding, workflow_config, metadata
    
    def batch_generate(
        self,
        image_groups: List[List[Union[str, Path]]],
        character_names: Optional[List[str]] = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Path]:
        """
        Generate multiple safe tensors in batch.
        
        Args:
            image_groups: List of image groups (each group is a list of image paths)
            character_names: Optional list of character names
            metadata_list: Optional list of metadata dicts
            
        Returns:
            List of paths to generated safe tensors
        """
        if self.model is None:
            raise ValueError("Model not initialized.")
        
        results = [
            self.generate_from_images(
                images=images,
                character_name=self._get_item(character_names, i),
                metadata=self._get_item(metadata_list, i),
            )
            for i, images in enumerate(image_groups)
        ]
        
        logger.info(f"Batch generated {len(results)} safe tensors")
        return results
    
    @staticmethod
    def _get_item(items: Optional[List[Any]], index: int) -> Optional[Any]:
        """Safely get item from list by index."""
        return items[index] if items and index < len(items) else None
    
    def create_workflow_template(
        self,
        embedding_path: Union[str, Path],
        prompt_template: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        output_filename: Optional[str] = None,
    ) -> Path:
        """
        Create a workflow template from an existing embedding.
        
        Args:
            embedding_path: Path to character embedding safe tensor
            prompt_template: Prompt template with {character} placeholder
            negative_prompt: Negative prompt
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            output_filename: Optional custom output filename
            
        Returns:
            Path to workflow template safe tensor
        """
        # Load embedding
        embedding, metadata = Flux2CharacterConsistencyModel.load_embedding(
            embedding_path,
            device=str(self.model.device) if self.model else None
        )
        
        # Create workflow config
        workflow_config = MetadataBuilder.build_workflow_metadata(
            prompt_template=prompt_template,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            character_metadata=metadata,
        )
        
        # Generate workflow tensor
        return self.generate_workflow_tensor(
            embedding=embedding,
            workflow_config=workflow_config,
            output_filename=output_filename,
        )
    
    def list_generated_tensors(self) -> List[Dict[str, Any]]:
        """
        List all generated safe tensors with metadata.
        
        Returns:
            List of dicts with tensor info
        """
        return [
            self._build_tensor_info(tensor_path)
            for tensor_path in self.output_dir.glob("*.safetensors")
        ]
    
    def _build_tensor_info(self, tensor_path: Path) -> Dict[str, Any]:
        """Build info dictionary for a tensor file."""
        try:
            metadata = self._load_tensor_metadata(tensor_path)
            return {
                "path": str(tensor_path),
                "filename": tensor_path.name,
                "size": tensor_path.stat().st_size,
                "metadata": metadata,
            }
        except Exception as e:
            logger.warning(f"Error reading tensor {tensor_path}: {e}")
            return {
                "path": str(tensor_path),
                "filename": tensor_path.name,
                "size": tensor_path.stat().st_size if tensor_path.exists() else 0,
                "metadata": None,
                "error": str(e),
            }
    
    @staticmethod
    def _load_tensor_metadata(tensor_path: Path) -> Optional[Dict[str, Any]]:
        """Load metadata for a tensor file."""
        metadata_path = tensor_path.with_suffix(".json")
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, "r") as f:
            return json.load(f)

