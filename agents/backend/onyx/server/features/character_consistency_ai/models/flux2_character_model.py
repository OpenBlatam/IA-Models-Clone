"""
Flux2 Character Consistency Model
==================================

Model based on Flux2 architecture for generating character consistency embeddings.
Processes one or multiple images and extracts consistent character features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Union, Tuple
from pathlib import Path
import logging
import json
from PIL import Image
import numpy as np
from safetensors.torch import save_file, load_file

try:
    from diffusers import FluxTransformer2DModel
    from transformers import CLIPImageProcessor, CLIPVisionModel
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logging.warning("Diffusers not available, some features may be limited")

from .constants import (
    DEFAULT_MODEL_ID,
    DEFAULT_CLIP_MODEL_ID,
    DEFAULT_EMBEDDING_DIM,
    MIN_INTERMEDIATE_SIZE_MULTIPLIER,
    MIN_ATTENTION_HEADS,
    ATTENTION_HEAD_DIM,
    DROPOUT_RATE,
    CROSS_ATTENTION_BLEND_WEIGHT,
    FUSION_BASE_WEIGHT,
    FUSION_METHODS_COUNT,
    ATTENTION_SLICE_SIZE,
)

from .helpers import (
    ImageProcessor,
    FeaturePooler,
    DeviceManager,
    EmbeddingIO,
    ModelInitializer,
    ModelOptimizer,
    ImageQualityValidator,
    EmbeddingQualityValidator,
    ImageQualityMetrics,
    EmbeddingMetrics,
    retry_on_failure,
)

logger = logging.getLogger(__name__)


class CharacterEncoder(nn.Module):
    """Encodes CLIP features into character embeddings."""
    
    def __init__(self, clip_hidden_size: int, embedding_dim: int):
        super().__init__()
        intermediate_size = max(2048, embedding_dim * MIN_INTERMEDIATE_SIZE_MULTIPLIER)
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(clip_hidden_size, intermediate_size),
            nn.LayerNorm(intermediate_size),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
        )
        
        self.character_encoder = nn.Sequential(
            nn.Linear(intermediate_size, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        
        self.residual_proj = nn.Linear(clip_hidden_size, embedding_dim)
    
    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        """Encode pooled features with residual connection."""
        extracted = self.feature_extractor(pooled_features)
        encoded = self.character_encoder(extracted)
        residual = self.residual_proj(pooled_features)
        return encoded + residual


class MultiImageAggregator(nn.Module):
    """Aggregates multiple image embeddings into consistent character embedding."""
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        num_heads = min(MIN_ATTENTION_HEADS, embedding_dim // ATTENTION_HEAD_DIM)
        
        self.aggregator = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=DROPOUT_RATE,
        )
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=DROPOUT_RATE,
        )
        
        self.fusion_weights = nn.Parameter(torch.ones(FUSION_METHODS_COUNT) / FUSION_METHODS_COUNT)
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize fusion weights."""
        with torch.no_grad():
            self.fusion_weights.data = torch.ones_like(self.fusion_weights.data) / FUSION_METHODS_COUNT
    
    def forward(self, stacked_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Aggregate multiple embeddings.
        
        Args:
            stacked_embeddings: [1, num_images, embedding_dim]
            
        Returns:
            Aggregated embedding [1, embedding_dim]
        """
        num_images = stacked_embeddings.size(1)
        
        # Self-attention aggregation
        mean_query = stacked_embeddings.mean(dim=1, keepdim=True)
        attn_output, _ = self.aggregator(
            query=mean_query,
            key=stacked_embeddings,
            value=stacked_embeddings,
        )
        attn_aggregated = attn_output.squeeze(1)
        
        # Cross-attention between images
        if num_images > 1:
            cross_aggregated = self._apply_cross_attention(stacked_embeddings, num_images)
        else:
            cross_aggregated = attn_aggregated
        
        # Statistical aggregation
        mean_aggregated = stacked_embeddings.mean(dim=1)
        max_aggregated = stacked_embeddings.max(dim=1)[0]
        
        # Weighted fusion
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = (
            weights[0] * mean_aggregated +
            weights[1] * max_aggregated +
            weights[2] * attn_aggregated
        )
        
        # Blend cross-attention if available
        if num_images > 1:
            fused = FUSION_BASE_WEIGHT * fused + CROSS_ATTENTION_BLEND_WEIGHT * cross_aggregated
        
        return fused
    
    def _apply_cross_attention(
        self, 
        stacked_embeddings: torch.Tensor, 
        num_images: int
    ) -> torch.Tensor:
        """Apply cross-attention between images."""
        cross_outputs = []
        for i in range(num_images):
            query = stacked_embeddings[:, i:i+1, :]
            cross_out, _ = self.cross_attention(
                query=query,
                key=stacked_embeddings,
                value=stacked_embeddings,
            )
            cross_outputs.append(cross_out.squeeze(1))
        return torch.stack(cross_outputs, dim=0).mean(dim=0)


class ConsistencyProjector(nn.Module):
    """Projects aggregated embeddings to consistency space."""
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.final_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Project and normalize embedding."""
        projected = self.projector(embedding)
        return self.final_norm(projected)




class Flux2CharacterConsistencyModel(nn.Module):
    """
    Flux2-based model for character consistency.
    
    Extracts consistent character features from one or multiple images
    and generates embeddings that can be used for maintaining character
    consistency across different generations.
    """
    
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        enable_optimizations: bool = True,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        validate_images: bool = True,
        enhance_images: bool = False,
    ):
        """
        Initialize Flux2 Character Consistency Model.
        
        Args:
            model_id: HuggingFace model ID for Flux2
            device: Device to run on (cuda/cpu/auto)
            dtype: Data type (float16/float32)
            enable_optimizations: Enable memory optimizations
            embedding_dim: Dimension of character embeddings
            validate_images: Enable image quality validation
            enhance_images: Enable automatic image enhancement
        """
        super().__init__()
        
        self.model_id = model_id
        self.embedding_dim = embedding_dim
        self.device = DeviceManager.setup_device(device)
        self.dtype = DeviceManager.setup_dtype(self.device, dtype)
        self.enable_optimizations = enable_optimizations
        self.validate_images = validate_images
        self.enhance_images = enhance_images
        
        # Statistics
        self.stats = {
            "images_processed": 0,
            "images_validated": 0,
            "images_enhanced": 0,
            "validation_failures": 0,
            "total_time": 0.0,
        }
        
        self._build_model()
        logger.info(f"Flux2CharacterConsistencyModel initialized on {self.device}")
    
    def _build_model(self) -> None:
        """Build the model architecture."""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "Diffusers library is required. Install with: pip install diffusers transformers"
            )
        
        try:
            self._load_base_models()
            self._build_components()
            self._initialize_weights()
            self._move_to_device()
            
            if self.enable_optimizations:
                ModelOptimizer.apply_optimizations(self.transformer, self.device)
            
            self.eval()
        
        except Exception as e:
            logger.error(f"Error building model: {e}")
            raise RuntimeError(f"Failed to build Flux2 model: {e}")
    
    def _load_base_models(self) -> None:
        """Load base Flux2 and CLIP models."""
        self.transformer = FluxTransformer2DModel.from_pretrained(
            self.model_id,
            subfolder="transformer",
            torch_dtype=self.dtype,
        )
        
        self.clip_processor = CLIPImageProcessor.from_pretrained(
            DEFAULT_CLIP_MODEL_ID
        )
        self.clip_vision = CLIPVisionModel.from_pretrained(
            DEFAULT_CLIP_MODEL_ID,
            torch_dtype=self.dtype,
        )
    
    def _build_components(self) -> None:
        """Build model components."""
        clip_hidden_size = self.clip_vision.config.hidden_size
        
        self.character_encoder = CharacterEncoder(clip_hidden_size, self.embedding_dim)
        self.aggregator = MultiImageAggregator(self.embedding_dim)
        self.consistency_projector = ConsistencyProjector(self.embedding_dim)
    
    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        ModelInitializer.initialize_weights(self)
    
    def _move_to_device(self) -> None:
        """Move all components to device."""
        self.transformer = ModelInitializer.move_to_device(self.transformer, self.device)
        self.clip_vision = ModelInitializer.move_to_device(self.clip_vision, self.device)
        self.character_encoder = ModelInitializer.move_to_device(self.character_encoder, self.device)
        self.aggregator = ModelInitializer.move_to_device(self.aggregator, self.device)
        self.consistency_projector = ModelInitializer.move_to_device(self.consistency_projector, self.device)
    
    def _convert_to_pil_for_enhancement(
        self,
        image: Union[Image.Image, str, Path, np.ndarray]
    ) -> Optional[Image.Image]:
        """Convert image to PIL for enhancement."""
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        return None
    
    def preprocess_image(
        self, 
        image: Union[Image.Image, str, Path, np.ndarray]
    ) -> torch.Tensor:
        """Preprocess image for model input."""
        return ImageProcessor.process_with_clip(
            image, self.clip_processor, self.device
        )
    
    @retry_on_failure(max_attempts=3, delay=0.5)
    def encode_image(
        self, 
        image: Union[Image.Image, str, Path, np.ndarray]
    ) -> torch.Tensor:
        """
        Encode a single image into character embedding.
        
        Args:
            image: Input image
            
        Returns:
            Character embedding tensor [embedding_dim]
        """
        # Validate image if enabled
        if self.validate_images:
            metrics = ImageQualityValidator.validate_image(image)
            if not metrics.is_valid:
                error_msg = f"Image validation failed: {', '.join(metrics.errors)}"
                logger.error(error_msg)
                self.stats["validation_failures"] += 1
                raise ValueError(error_msg)
            
            if metrics.warnings:
                logger.warning(f"Image quality warnings: {', '.join(metrics.warnings)}")
            
            self.stats["images_validated"] += 1
            
            # Enhance image if enabled
            if self.enhance_images and metrics.warnings:
                pil_image = self._convert_to_pil_for_enhancement(image)
                if pil_image:
                    image = ImageQualityValidator.enhance_image(pil_image)
                    self.stats["images_enhanced"] += 1
        
        with torch.no_grad():
            pixel_values = self.preprocess_image(image)
            clip_outputs = self.clip_vision(pixel_values=pixel_values)
            image_features = clip_outputs.last_hidden_state
            
            pooled_features = FeaturePooler.pool_clip_features(image_features)
            character_embedding = self.character_encoder(pooled_features)
            character_embedding = character_embedding.squeeze(0)
            
            self.stats["images_processed"] += 1
            
            return character_embedding
    
    def encode_multiple_images(
        self,
        images: List[Union[Image.Image, str, Path, np.ndarray]]
    ) -> torch.Tensor:
        """
        Encode multiple images and aggregate into consistent character embedding.
        
        Args:
            images: List of input images
            
        Returns:
            Aggregated character embedding tensor [embedding_dim]
        """
        if not images:
            raise ValueError("At least one image is required")
        
        with torch.no_grad():
            embeddings = [self.encode_image(img) for img in images]
            stacked_embeddings = torch.stack(embeddings, dim=0).unsqueeze(0)
            
            aggregated = self.aggregator(stacked_embeddings)
            consistency_embedding = self.consistency_projector(aggregated)
            
            return consistency_embedding.squeeze(0)
    
    def forward(
        self,
        images: Union[
            Image.Image,
            str,
            Path,
            np.ndarray,
            List[Union[Image.Image, str, Path, np.ndarray]]
        ]
    ) -> torch.Tensor:
        """
        Forward pass - generate character consistency embedding.
        
        Args:
            images: Single image or list of images
            
        Returns:
            Character consistency embedding tensor
        """
        if isinstance(images, list):
            return self.encode_multiple_images(images)
        return self.encode_image(images)
    
    def save_embedding(
        self,
        embedding: torch.Tensor,
        output_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save character embedding as safe tensor.
        
        Args:
            embedding: Character embedding tensor
            output_path: Path to save safe tensor
            metadata: Optional metadata to include
        """
        EmbeddingIO.save_embedding(
            embedding=embedding,
            output_path=Path(output_path),
            metadata=metadata
        )
    
    @classmethod
    def load_embedding(
        cls,
        embedding_path: Union[str, Path],
        device: Optional[str] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Load character embedding from safe tensor.
        
        Args:
            embedding_path: Path to safe tensor file
            device: Device to load tensor on
            
        Returns:
            Tuple of (embedding tensor, metadata dict)
        """
        device_obj = torch.device(device) if device else None
        return EmbeddingIO.load_embedding(
            embedding_path=Path(embedding_path),
            device=device_obj
        )
    
    def validate_embedding(self, embedding: torch.Tensor) -> EmbeddingMetrics:
        """
        Validate embedding quality and return metrics.
        
        Args:
            embedding: Embedding tensor to validate
            
        Returns:
            EmbeddingMetrics with quality assessment
        """
        return EmbeddingQualityValidator.validate_embedding(embedding)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        model_info = ModelInitializer.get_model_info(self)
        
        # Calculate model size
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 * 1024)
        
        # Calculate average time per image
        avg_time = (
            self.stats["total_time"] / self.stats["images_processed"]
            if self.stats["images_processed"] > 0 else 0.0
        )
        
        return {
            "model_id": self.model_id,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "embedding_dim": self.embedding_dim,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "optimizations_enabled": self.enable_optimizations,
            "model_size_mb": round(model_size_mb, 2),
            "optimizations_enabled": self.enable_optimizations,
            "validate_images": self.validate_images,
            "enhance_images": self.enhance_images,
            "statistics": {
                "images_processed": self.stats["images_processed"],
                "images_validated": self.stats["images_validated"],
                "images_enhanced": self.stats["images_enhanced"],
                "validation_failures": self.stats["validation_failures"],
                "average_time_per_image": round(avg_time, 4),
            },
            "features": {
                "residual_connections": True,
                "cross_attention": True,
                "weighted_fusion": True,
                "enhanced_pooling": True,
                "modular_architecture": True,
                "image_validation": self.validate_images,
                "image_enhancement": self.enhance_images,
            }
        }
    
    def compute_similarity(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding tensor
            embedding2: Second embedding tensor
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Normalize embeddings
        emb1_norm = F.normalize(embedding1, p=2, dim=-1)
        emb2_norm = F.normalize(embedding2, p=2, dim=-1)
        
        # Compute cosine similarity
        similarity = torch.sum(emb1_norm * emb2_norm, dim=-1).item()
        
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
