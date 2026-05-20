"""
Clothing Changer Service
========================

Main service for clothing change operations.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from PIL import Image
import torch
import numpy as np

from ..models.flux2_clothing_model import Flux2ClothingChangerModel
from ..models.deepseek_clothing_model import DeepSeekClothingModel
from ..models.comfyui_tensor_generator import ComfyUITensorGenerator
from ..models.prompt_enhancer import PromptEnhancer, ClothingStyleAnalyzer
from ..models.embedding_cache import EmbeddingCache
from ..models.quality_metrics import QualityMetrics
from ..config.clothing_changer_config import ClothingChangerConfig
from .factories.model_factory import ModelFactory
from .utils.result_builder import ResultBuilder
from .utils.model_selector import ModelSelector
from .utils.image_loader import ImageLoader
from .utils.generator_manager import GeneratorManager
from .utils.model_initializer import ModelInitializer
from .utils.service_utilities import ServiceUtilities
from .processors.clothing_change_processor import ClothingChangeProcessor
from .processors.tensor_workflow_processor import TensorWorkflowProcessor
from .managers.service_initializer import ServiceInitializer
from .managers.resource_cleaner import ResourceCleaner
from .processors.batch_processor import BatchProcessor

logger = logging.getLogger(__name__)


class ClothingChangerService:
    """
    Main service for clothing change operations.
    
    Handles model initialization, image processing, and ComfyUI tensor generation.
    """
    
    def __init__(self, config: Optional[ClothingChangerConfig] = None):
        """
        Initialize Clothing Changer Service.
        
        Args:
            config: Configuration instance (optional, will create default if not provided)
        """
        self.config = config or ClothingChangerConfig.from_env()
        self.config.validate()
        
        # Initialize model
        self.model: Optional[Flux2ClothingChangerModel] = None
        self.deepseek_model: Optional[DeepSeekClothingModel] = None
        self.generator: Optional[ComfyUITensorGenerator] = None
        self.use_deepseek_fallback: bool = False
        
        # Initialize utilities
        self.prompt_enhancer = PromptEnhancer()
        self.style_analyzer = ClothingStyleAnalyzer()
        self.quality_metrics = QualityMetrics()
        
        # Initialize cache if enabled
        cache_dir = getattr(self.config, "cache_dir", "./embedding_cache")
        self.cache = EmbeddingCache(cache_dir=cache_dir) if getattr(self.config, "enable_cache", True) else None
        
        # Initialize service utilities
        self.service_utilities = ServiceUtilities(
            prompt_enhancer=self.prompt_enhancer,
            style_analyzer=self.style_analyzer,
            quality_metrics=self.quality_metrics,
            cache=self.cache,
        )
        
        # Initialize service initializer
        self.service_initializer = ServiceInitializer(self.config)
        
        # Initialize model selector (will be set after model initialization)
        self.model_selector: Optional[ModelSelector] = None
        self.generator_manager: Optional[GeneratorManager] = None
        self.clothing_change_processor: Optional[ClothingChangeProcessor] = None
        self.tensor_workflow_processor: Optional[TensorWorkflowProcessor] = None
        
        logger.info("ClothingChangerService initialized")
    
    def initialize_model(self) -> None:
        """Initialize the Flux2 model, with DeepSeek fallback."""
        # Initialize models
        self.model, self.deepseek_model, self.use_deepseek_fallback = (
            self.service_initializer.initialize_models(
                self.model,
                self.deepseek_model
            )
        )
        
        # Initialize generator if Flux2 model is available
        self.generator = self.service_initializer.initialize_generator(self.model)
        
        if self.deepseek_model and not self.model:
            logger.info("DeepSeek model initialized as fallback (no tensor generator)")
        
        # Initialize model selector
        self.model_selector = self.service_initializer.create_model_selector(
            self.model,
            self.deepseek_model,
            self.use_deepseek_fallback
        )
        
        # Initialize generator manager
        self.generator_manager = self.service_initializer.create_generator_manager(
            self.generator,
            self.model,
            self.initialize_model
        )
        
        # Initialize generator manager
        self.generator_manager = GeneratorManager(
            generator=self.generator,
            config=self.config,
            model=self.model,
            initialize_model_fn=self.initialize_model
        )
        
        # Initialize clothing change processor
        self.clothing_change_processor = ClothingChangeProcessor(
            model_selector=self.model_selector,
            generator_manager=self.generator_manager,
            quality_metrics=self.quality_metrics,
            config=self.config,
            initialize_model_fn=self.initialize_model,
        )
        
        # Initialize tensor workflow processor
        self.tensor_workflow_processor = TensorWorkflowProcessor(
            generator_manager=self.generator_manager,
            config=self.config,
            initialize_model_fn=self.initialize_model,
        )
    
    def change_clothing(
        self,
        image: Union[str, Path, Image.Image],
        clothing_description: str,
        character_name: Optional[str] = None,
        mask: Optional[Union[Image.Image, np.ndarray]] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        strength: Optional[float] = None,
        save_tensor: bool = True,
        output_filename: Optional[str] = None,
        style: Optional[str] = None,
        quality_level: str = "high",
        enhance_prompt: bool = True,
        calculate_metrics: bool = True,
    ) -> Dict[str, Any]:
        """
        Change clothing in character image.
        
        Args:
            image: Input character image
            clothing_description: Description of new clothing
            character_name: Optional character name
            mask: Optional mask for clothing area
            prompt: Optional full prompt
            negative_prompt: Negative prompt
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            strength: Inpainting strength
            save_tensor: Whether to save as ComfyUI safe tensor
            output_filename: Optional custom output filename
            
        Returns:
            Dict with result info and paths
        """
        # Initialize processor if needed
        if self.clothing_change_processor is None:
            if self.model_selector is None:
                self.model_selector = ModelSelector(
                    flux2_model=self.model,
                    deepseek_model=self.deepseek_model,
                    use_deepseek_fallback=self.use_deepseek_fallback,
                    config=self.config
                )
            
            self.clothing_change_processor = ClothingChangeProcessor(
                model_selector=self.model_selector,
                generator_manager=self.generator_manager,
                quality_metrics=self.quality_metrics,
                config=self.config,
                initialize_model_fn=self.initialize_model,
            )
        
        # Update model selector with current models
        self.model_selector.flux2_model = self.model
        self.model_selector.deepseek_model = self.deepseek_model
        self.model_selector.use_deepseek_fallback = self.use_deepseek_fallback
        
        # Process clothing change
        return self.clothing_change_processor.process(
            image=image,
            clothing_description=clothing_description,
            character_name=character_name,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            save_tensor=save_tensor,
            output_filename=output_filename,
            style=style,
            calculate_metrics=calculate_metrics,
        )
    
    def create_comfyui_workflow(
        self,
        tensor_path: Union[str, Path],
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
        if self.tensor_workflow_processor is None:
            self.tensor_workflow_processor = TensorWorkflowProcessor(
                generator_manager=self.generator_manager,
                config=self.config,
                initialize_model_fn=self.initialize_model,
            )
        
        return self.tensor_workflow_processor.create_workflow(
            tensor_path=Path(tensor_path),
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
    
    def list_tensors(self) -> List[Dict[str, Any]]:
        """
        List all generated safe tensors.
        
        Returns:
            List of tensor info dicts
        """
        if self.tensor_workflow_processor is None:
            self.tensor_workflow_processor = TensorWorkflowProcessor(
                generator_manager=self.generator_manager,
                config=self.config,
                initialize_model_fn=self.initialize_model,
            )
        
        return self.tensor_workflow_processor.list_tensors(
            self.model,
            self.use_deepseek_fallback
        )
    
    def batch_change_clothing(
        self,
        image_clothing_pairs: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Change clothing for multiple images in batch.
        
        Args:
            image_clothing_pairs: List of dicts with 'image' and 'clothing_description' keys
            **kwargs: Additional arguments passed to change_clothing
        
        Returns:
            List of result dicts
        """
        if self.model is None and self.deepseek_model is None:
            self.initialize_model()
        
        processor = BatchProcessor(self.change_clothing)
        return processor.process_batch(image_clothing_pairs, **kwargs)
    
    def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Validate a prompt and get suggestions.
        
        Args:
            prompt: Prompt to validate
            
        Returns:
            Validation results
        """
        return self.service_utilities.validate_prompt(prompt)
    
    def analyze_clothing_style(self, description: str) -> Dict[str, Any]:
        """
        Analyze clothing description to extract style information.
        
        Args:
            description: Clothing description
            
        Returns:
            Style analysis
        """
        return self.service_utilities.analyze_clothing_style(description)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        return self.service_utilities.get_cache_stats()
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self.service_utilities.clear_cache()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return self.service_utilities.get_model_info(
            self.model,
            self.deepseek_model,
            self.use_deepseek_fallback
        )
    
    def close(self) -> None:
        """Clean up resources."""
        ResourceCleaner.cleanup_all(
            self.model,
            self.deepseek_model,
            self.generator
        )
        
        # Clear references
        self.model = None
        self.generator = None
        self.deepseek_model = None

