"""
Character Consistency Service
=============================

Main service for character consistency operations.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from PIL import Image
import torch

from ..models.flux2_character_model import Flux2CharacterConsistencyModel
from ..models.safe_tensor_generator import SafeTensorGenerator
from ..config.character_consistency_config import CharacterConsistencyConfig

logger = logging.getLogger(__name__)


class CharacterConsistencyService:
    """
    Main service for character consistency operations.
    
    Handles model initialization, image processing, and safe tensor generation.
    """
    
    def __init__(self, config: Optional[CharacterConsistencyConfig] = None):
        """
        Initialize Character Consistency Service.
        
        Args:
            config: Configuration instance (optional, will create default if not provided)
        """
        self.config = config or CharacterConsistencyConfig.from_env()
        self.config.validate()
        
        self.model: Optional[Flux2CharacterConsistencyModel] = None
        self.generator: Optional[SafeTensorGenerator] = None
        
        logger.info("CharacterConsistencyService initialized")
    
    def initialize_model(self) -> None:
        """Initialize the Flux2 model."""
        if self.model is not None:
            logger.warning("Model already initialized")
            return
        
        logger.info("Initializing Flux2 Character Consistency Model...")
        
        try:
            self.model = self._create_model()
            self.generator = self._create_generator()
            logger.info("Model initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def _create_model(self) -> Flux2CharacterConsistencyModel:
        """Create and configure the Flux2 model."""
        return Flux2CharacterConsistencyModel(
            model_id=self.config.model_id,
            device=self.config.device,
            dtype=self._get_dtype(),
            enable_optimizations=self.config.enable_optimizations,
            embedding_dim=self.config.embedding_dim,
        )
    
    def _get_dtype(self) -> Optional[torch.dtype]:
        """Get torch dtype from config."""
        if self.config.dtype == "float16":
            return torch.float16
        elif self.config.dtype == "float32":
            return torch.float32
        return None
    
    def _create_generator(self) -> SafeTensorGenerator:
        """Create safe tensor generator."""
        return SafeTensorGenerator(
            model=self.model,
            output_dir=self.config.output_dir,
        )
    
    def _ensure_model_initialized(self) -> None:
        """Ensure model is initialized."""
        if self.model is None:
            self.initialize_model()
    
    def _ensure_generator_initialized(self) -> None:
        """Ensure generator is initialized."""
        if self.generator is None:
            self._ensure_model_initialized()
            self.generator = self._create_generator()
    
    def generate_character_embedding(
        self,
        images: Union[List[Union[str, Path, Image.Image]], str, Path, Image.Image],
        character_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        save_tensor: bool = True,
        output_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate character consistency embedding from images.
        
        Args:
            images: Single image or list of images (paths or PIL Images)
            character_name: Optional character name
            metadata: Optional additional metadata
            save_tensor: Whether to save as safe tensor
            output_filename: Optional custom output filename
            
        Returns:
            Dict with embedding info and path if saved
        """
        self._ensure_model_initialized()
        
        image_list = self._normalize_images(images)
        logger.info(f"Generating character embedding from {len(image_list)} image(s)")
        
        embedding = self.model(image_list)
        result = self._build_result_dict(embedding, len(image_list), character_name)
        
        if save_tensor:
            result.update(self._save_embedding_tensor(
                image_list, character_name, metadata, output_filename
            ))
        
        return result
    
    @staticmethod
    def _normalize_images(
        images: Union[List[Union[str, Path, Image.Image]], str, Path, Image.Image]
    ) -> List[Union[str, Path, Image.Image]]:
        """Normalize images input to list format."""
        return images if isinstance(images, list) else [images]
    
    @staticmethod
    def _build_result_dict(
        embedding: torch.Tensor, 
        num_images: int, 
        character_name: Optional[str]
    ) -> Dict[str, Any]:
        """Build result dictionary from embedding."""
        return {
            "embedding_shape": list(embedding.shape),
            "embedding_dim": embedding.shape[0],
            "num_images": num_images,
            "character_name": character_name,
        }
    
    def _save_embedding_tensor(
        self,
        images: List[Union[str, Path, Image.Image]],
        character_name: Optional[str],
        metadata: Optional[Dict[str, Any]],
        output_filename: Optional[str],
    ) -> Dict[str, Any]:
        """Save embedding tensor and return save info."""
        self._ensure_generator_initialized()
        
        output_path = self.generator.generate_from_images(
            images=images,
            character_name=character_name,
            metadata=metadata,
            output_filename=output_filename,
        )
        
        return {
            "saved_path": str(output_path),
            "saved": True,
        }
    
    def create_workflow_tensor(
        self,
        embedding_path: Union[str, Path],
        prompt_template: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Create workflow-ready safe tensor from existing embedding.
        
        Args:
            embedding_path: Path to character embedding safe tensor
            prompt_template: Prompt template with {character} placeholder
            negative_prompt: Negative prompt
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            
        Returns:
            Dict with workflow tensor info
        """
        self._ensure_generator_initialized()
        
        output_path = self.generator.create_workflow_template(
            embedding_path=embedding_path,
            prompt_template=prompt_template,
            negative_prompt=negative_prompt or "",
            num_inference_steps=num_inference_steps or self.config.default_num_inference_steps,
            guidance_scale=guidance_scale or self.config.default_guidance_scale,
        )
        
        return {
            "workflow_tensor_path": str(output_path),
            "created": True,
        }
    
    def batch_generate(
        self,
        image_groups: List[List[Union[str, Path, Image.Image]]],
        character_names: Optional[List[str]] = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple character embeddings in batch.
        
        Args:
            image_groups: List of image groups
            character_names: Optional list of character names
            metadata_list: Optional list of metadata dicts
            
        Returns:
            List of result dicts
        """
        self._ensure_model_initialized()
        self._ensure_generator_initialized()
        
        results = []
        for i, images in enumerate(image_groups):
            char_name = self._get_character_name(character_names, i)
            metadata = self._get_metadata(metadata_list, i)
            
            result = self.generate_character_embedding(
                images=images,
                character_name=char_name,
                metadata=metadata,
                save_tensor=True,
            )
            results.append(result)
        
        return results
    
    @staticmethod
    def _get_character_name(
        character_names: Optional[List[str]], 
        index: int
    ) -> Optional[str]:
        """Get character name at index if available."""
        return character_names[index] if character_names and index < len(character_names) else None
    
    @staticmethod
    def _get_metadata(
        metadata_list: Optional[List[Dict[str, Any]]], 
        index: int
    ) -> Optional[Dict[str, Any]]:
        """Get metadata at index if available."""
        return metadata_list[index] if metadata_list and index < len(metadata_list) else None
    
    def list_embeddings(self) -> List[Dict[str, Any]]:
        """
        List all generated character embeddings.
        
        Returns:
            List of embedding info dicts
        """
        self._ensure_generator_initialized()
        return self.generator.list_generated_tensors()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self.model is None:
            return {"status": "not_initialized"}
        return self.model.get_model_info()
    
    def close(self) -> None:
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Service closed and resources cleaned up")
