from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import time
from enum import Enum
from torch.utils.data import Dataset, DataLoader
import warnings
from abc import ABC, abstractmethod
from diffusion_models import (
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Advanced Diffusion Pipelines
Comprehensive implementation of various diffusion pipelines including
StableDiffusionPipeline, StableDiffusionXLPipeline, and custom pipelines.
"""


# Import our diffusion models
    DiffusionModel, DiffusionConfig, SchedulerType, 
    DiffusionTrainer, DiffusionAnalyzer, DiffusionType
)


class PipelineType(Enum):
    """Types of diffusion pipelines."""
    STABLE_DIFFUSION = "stable_diffusion"
    STABLE_DIFFUSION_XL = "stable_diffusion_xl"
    CUSTOM_DIFFUSION = "custom_diffusion"
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"
    INPAINTING = "inpainting"
    CONTROL_NET = "control_net"


@dataclass
class PipelineConfig:
    """Configuration for diffusion pipelines."""
    # Pipeline type
    pipeline_type: PipelineType = PipelineType.STABLE_DIFFUSION
    
    # Model configuration
    model_config: DiffusionConfig = None
    
    # Text processing
    use_text_conditioning: bool = True
    text_encoder_dim: int = 768
    max_text_length: int = 77
    
    # Image processing
    use_image_conditioning: bool = False
    image_size: int = 512
    in_channels: int = 3
    
    # Control features
    use_control_net: bool = False
    control_net_channels: int = 3
    
    # Advanced features
    use_classifier_free_guidance: bool = True
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    eta: float = 0.0
    
    # Performance
    use_mixed_precision: bool = True
    use_attention_slicing: bool = False
    use_memory_efficient_attention: bool = False


class TextEncoder(nn.Module):
    """Text encoder for conditioning diffusion models."""
    
    def __init__(self, config: PipelineConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        
        # Simple text encoder (in practice, this would be CLIP or similar)
        self.embedding = nn.Embedding(config.max_text_length, config.text_encoder_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.text_encoder_dim,
                nhead=8,
                dim_feedforward=config.text_encoder_dim * 4,
                dropout=0.1
            ),
            num_layers=6
        )
        self.final_projection = nn.Linear(config.text_encoder_dim, config.model_config.hidden_size)
    
    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Encode text tokens."""
        # Embed tokens
        embeddings = self.embedding(text_tokens)
        
        # Apply transformer
        encoded = self.transformer(embeddings)
        
        # Global pooling
        pooled = encoded.mean(dim=1)
        
        # Project to model dimension
        projected = self.final_projection(pooled)
        
        return projected


class ImageEncoder(nn.Module):
    """Image encoder for conditioning diffusion models."""
    
    def __init__(self, config: PipelineConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        
        # Simple image encoder (in practice, this would be a pre-trained encoder)
        self.encoder = nn.Sequential(
            nn.Conv2d(config.in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, config.model_config.hidden_size)
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images."""
        return self.encoder(images)


class ControlNet(nn.Module):
    """ControlNet for guided generation."""
    
    def __init__(self, config: PipelineConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        
        # ControlNet architecture
        self.control_encoder = nn.Sequential(
            nn.Conv2d(config.control_net_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, config.model_config.hidden_size, 3, padding=1)
        )
        
        # Control signal processing
        self.control_projection = nn.Linear(config.model_config.hidden_size, config.model_config.hidden_size)
    
    def forward(self, control_image: torch.Tensor) -> torch.Tensor:
        """Process control image."""
        control_features = self.control_encoder(control_image)
        control_signal = self.control_projection(control_features.mean(dim=(2, 3)))
        return control_signal


class BaseDiffusionPipeline(ABC):
    """Base class for diffusion pipelines."""
    
    def __init__(self, config: PipelineConfig):
        
    """__init__ function."""
self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = self._setup_logging()
        
        # Initialize components
        self._initialize_components()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def _initialize_components(self) -> Any:
        """Initialize pipeline components."""
        pass
    
    @abstractmethod
    def generate(self, **kwargs) -> Dict[str, Any]:
        """Generate samples using the pipeline."""
        pass
    
    def to(self, device: torch.device):
        """Move pipeline to device."""
        self.device = device
        for component in self._get_components():
            if hasattr(component, 'to'):
                component.to(device)
        return self
    
    def _get_components(self) -> List[nn.Module]:
        """Get all pipeline components."""
        components = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, nn.Module):
                components.append(attr)
        return components


class StableDiffusionPipeline(BaseDiffusionPipeline):
    """Stable Diffusion Pipeline implementation."""
    
    def _initialize_components(self) -> Any:
        """Initialize Stable Diffusion components."""
        # Initialize model config if not provided
        if self.config.model_config is None:
            self.config.model_config = DiffusionConfig(
                image_size=self.config.image_size,
                in_channels=self.config.in_channels,
                hidden_size=768,
                num_layers=12,
                num_heads=12,
                num_timesteps=1000,
                scheduler_type=SchedulerType.COSINE
            )
        
        # Core diffusion model
        self.unet = DiffusionModel(self.config.model_config)
        
        # Text encoder
        if self.config.use_text_conditioning:
            self.text_encoder = TextEncoder(self.config)
        
        # Image encoder (for image-to-image)
        if self.config.use_image_conditioning:
            self.image_encoder = ImageEncoder(self.config)
        
        # ControlNet
        if self.config.use_control_net:
            self.control_net = ControlNet(self.config)
        
        # Move to device
        self.to(self.device)
        
        self.logger.info(f"Initialized {self.config.pipeline_type.value} pipeline")
    
    def generate(self, 
                 prompt: Optional[str] = None,
                 negative_prompt: Optional[str] = None,
                 image: Optional[torch.Tensor] = None,
                 control_image: Optional[torch.Tensor] = None,
                 num_images: int = 1,
                 **kwargs) -> Dict[str, Any]:
        """Generate images using Stable Diffusion."""
        
        # Prepare conditioning
        conditioning = self._prepare_conditioning(prompt, negative_prompt, image, control_image)
        
        # Generate samples
        with torch.no_grad():
            samples = self.unet.sample(
                batch_size=num_images,
                conditioning=conditioning,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                eta=self.config.eta
            )
        
        return {
            'images': samples,
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'guidance_scale': self.config.guidance_scale,
            'num_inference_steps': self.config.num_inference_steps
        }
    
    def _prepare_conditioning(self, 
                             prompt: Optional[str] = None,
                             negative_prompt: Optional[str] = None,
                             image: Optional[torch.Tensor] = None,
                             control_image: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Prepare conditioning for generation."""
        conditioning_features = []
        
        # Text conditioning
        if prompt and self.config.use_text_conditioning:
            # Simple tokenization (in practice, use proper tokenizer)
            prompt_tokens = self._tokenize_text(prompt)
            prompt_features = self.text_encoder(prompt_tokens)
            conditioning_features.append(prompt_features)
        
        # Negative text conditioning
        if negative_prompt and self.config.use_text_conditioning:
            neg_tokens = self._tokenize_text(negative_prompt)
            neg_features = self.text_encoder(neg_tokens)
            conditioning_features.append(neg_features)
        
        # Image conditioning
        if image is not None and self.config.use_image_conditioning:
            image_features = self.image_encoder(image)
            conditioning_features.append(image_features)
        
        # ControlNet conditioning
        if control_image is not None and self.config.use_control_net:
            control_features = self.control_net(control_image)
            conditioning_features.append(control_features)
        
        if conditioning_features:
            # Combine all conditioning features
            combined_conditioning = torch.cat(conditioning_features, dim=1)
            return combined_conditioning
        
        return None
    
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Simple text tokenization (in practice, use proper tokenizer)."""
        # Simple character-based tokenization for demonstration
        tokens = [ord(c) % self.config.max_text_length for c in text[:self.config.max_text_length]]
        tokens = tokens + [0] * (self.config.max_text_length - len(tokens))  # Pad
        return torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)


class StableDiffusionXLPipeline(BaseDiffusionPipeline):
    """Stable Diffusion XL Pipeline implementation."""
    
    def _initialize_components(self) -> Any:
        """Initialize Stable Diffusion XL components."""
        # XL uses larger model
        if self.config.model_config is None:
            self.config.model_config = DiffusionConfig(
                image_size=self.config.image_size,
                in_channels=self.config.in_channels,
                hidden_size=1024,  # Larger than SD
                num_layers=16,     # More layers
                num_heads=16,      # More heads
                num_timesteps=1000,
                scheduler_type=SchedulerType.COSINE
            )
        
        # Core diffusion model
        self.unet = DiffusionModel(self.config.model_config)
        
        # Dual text encoders (like SDXL)
        if self.config.use_text_conditioning:
            self.text_encoder_1 = TextEncoder(self.config)  # OpenCLIP
            self.text_encoder_2 = TextEncoder(self.config)  # CLIP
            
            # Update config for second encoder
            config_2 = PipelineConfig(
                text_encoder_dim=1280,  # Different dimension
                model_config=self.config.model_config
            )
            self.text_encoder_2 = TextEncoder(config_2)
        
        # Image encoder
        if self.config.use_image_conditioning:
            self.image_encoder = ImageEncoder(self.config)
        
        # ControlNet
        if self.config.use_control_net:
            self.control_net = ControlNet(self.config)
        
        # Move to device
        self.to(self.device)
        
        self.logger.info(f"Initialized {self.config.pipeline_type.value} pipeline")
    
    def generate(self, 
                 prompt: Optional[str] = None,
                 negative_prompt: Optional[str] = None,
                 image: Optional[torch.Tensor] = None,
                 control_image: Optional[torch.Tensor] = None,
                 num_images: int = 1,
                 **kwargs) -> Dict[str, Any]:
        """Generate images using Stable Diffusion XL."""
        
        # Prepare dual conditioning
        conditioning = self._prepare_dual_conditioning(prompt, negative_prompt, image, control_image)
        
        # Generate samples
        with torch.no_grad():
            samples = self.unet.sample(
                batch_size=num_images,
                conditioning=conditioning,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                eta=self.config.eta
            )
        
        return {
            'images': samples,
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'guidance_scale': self.config.guidance_scale,
            'num_inference_steps': self.config.num_inference_steps,
            'pipeline_type': 'stable_diffusion_xl'
        }
    
    def _prepare_dual_conditioning(self, 
                                  prompt: Optional[str] = None,
                                  negative_prompt: Optional[str] = None,
                                  image: Optional[torch.Tensor] = None,
                                  control_image: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Prepare dual conditioning for SDXL."""
        conditioning_features = []
        
        # Dual text conditioning
        if prompt and self.config.use_text_conditioning:
            prompt_tokens = self._tokenize_text(prompt)
            prompt_features_1 = self.text_encoder_1(prompt_tokens)
            prompt_features_2 = self.text_encoder_2(prompt_tokens)
            
            # Combine dual text features
            combined_text = torch.cat([prompt_features_1, prompt_features_2], dim=1)
            conditioning_features.append(combined_text)
        
        # Negative text conditioning
        if negative_prompt and self.config.use_text_conditioning:
            neg_tokens = self._tokenize_text(negative_prompt)
            neg_features_1 = self.text_encoder_1(neg_tokens)
            neg_features_2 = self.text_encoder_2(neg_tokens)
            
            combined_neg = torch.cat([neg_features_1, neg_features_2], dim=1)
            conditioning_features.append(combined_neg)
        
        # Image conditioning
        if image is not None and self.config.use_image_conditioning:
            image_features = self.image_encoder(image)
            conditioning_features.append(image_features)
        
        # ControlNet conditioning
        if control_image is not None and self.config.use_control_net:
            control_features = self.control_net(control_image)
            conditioning_features.append(control_features)
        
        if conditioning_features:
            combined_conditioning = torch.cat(conditioning_features, dim=1)
            return combined_conditioning
        
        return None
    
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Simple text tokenization."""
        tokens = [ord(c) % self.config.max_text_length for c in text[:self.config.max_text_length]]
        tokens = tokens + [0] * (self.config.max_text_length - len(tokens))
        return torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)


class TextToImagePipeline(BaseDiffusionPipeline):
    """Text-to-Image Pipeline implementation."""
    
    def _initialize_components(self) -> Any:
        """Initialize Text-to-Image components."""
        if self.config.model_config is None:
            self.config.model_config = DiffusionConfig(
                image_size=self.config.image_size,
                in_channels=self.config.in_channels,
                hidden_size=512,
                num_layers=8,
                num_heads=8,
                num_timesteps=1000,
                scheduler_type=SchedulerType.COSINE
            )
        
        # Core diffusion model
        self.unet = DiffusionModel(self.config.model_config)
        
        # Text encoder
        if self.config.use_text_conditioning:
            self.text_encoder = TextEncoder(self.config)
        
        self.to(self.device)
        self.logger.info(f"Initialized {self.config.pipeline_type.value} pipeline")
    
    def generate(self, 
                 prompt: str,
                 negative_prompt: Optional[str] = None,
                 num_images: int = 1,
                 **kwargs) -> Dict[str, Any]:
        """Generate images from text."""
        
        # Prepare text conditioning
        conditioning = self._prepare_text_conditioning(prompt, negative_prompt)
        
        # Generate samples
        with torch.no_grad():
            samples = self.unet.sample(
                batch_size=num_images,
                conditioning=conditioning,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                eta=self.config.eta
            )
        
        return {
            'images': samples,
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'guidance_scale': self.config.guidance_scale,
            'num_inference_steps': self.config.num_inference_steps,
            'pipeline_type': 'text_to_image'
        }
    
    def _prepare_text_conditioning(self, prompt: str, negative_prompt: Optional[str] = None) -> Optional[torch.Tensor]:
        """Prepare text conditioning."""
        conditioning_features = []
        
        # Positive prompt
        prompt_tokens = self._tokenize_text(prompt)
        prompt_features = self.text_encoder(prompt_tokens)
        conditioning_features.append(prompt_features)
        
        # Negative prompt
        if negative_prompt:
            neg_tokens = self._tokenize_text(negative_prompt)
            neg_features = self.text_encoder(neg_tokens)
            conditioning_features.append(neg_features)
        
        if conditioning_features:
            combined_conditioning = torch.cat(conditioning_features, dim=1)
            return combined_conditioning
        
        return None
    
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Simple text tokenization."""
        tokens = [ord(c) % self.config.max_text_length for c in text[:self.config.max_text_length]]
        tokens = tokens + [0] * (self.config.max_text_length - len(tokens))
        return torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)


class ImageToImagePipeline(BaseDiffusionPipeline):
    """Image-to-Image Pipeline implementation."""
    
    def _initialize_components(self) -> Any:
        """Initialize Image-to-Image components."""
        if self.config.model_config is None:
            self.config.model_config = DiffusionConfig(
                image_size=self.config.image_size,
                in_channels=self.config.in_channels,
                hidden_size=512,
                num_layers=8,
                num_heads=8,
                num_timesteps=1000,
                scheduler_type=SchedulerType.COSINE
            )
        
        # Core diffusion model
        self.unet = DiffusionModel(self.config.model_config)
        
        # Image encoder
        if self.config.use_image_conditioning:
            self.image_encoder = ImageEncoder(self.config)
        
        self.to(self.device)
        self.logger.info(f"Initialized {self.config.pipeline_type.value} pipeline")
    
    def generate(self, 
                 image: torch.Tensor,
                 prompt: Optional[str] = None,
                 strength: float = 0.8,
                 num_images: int = 1,
                 **kwargs) -> Dict[str, Any]:
        """Generate images from input image."""
        
        # Prepare image conditioning
        conditioning = self._prepare_image_conditioning(image, prompt)
        
        # Add noise to input image based on strength
        noisy_image = self._add_noise_to_image(image, strength)
        
        # Generate samples
        with torch.no_grad():
            samples = self.unet.sample(
                batch_size=num_images,
                conditioning=conditioning,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                eta=self.config.eta
            )
        
        return {
            'images': samples,
            'input_image': image,
            'strength': strength,
            'guidance_scale': self.config.guidance_scale,
            'num_inference_steps': self.config.num_inference_steps,
            'pipeline_type': 'image_to_image'
        }
    
    def _prepare_image_conditioning(self, image: torch.Tensor, prompt: Optional[str] = None) -> Optional[torch.Tensor]:
        """Prepare image conditioning."""
        conditioning_features = []
        
        # Image conditioning
        if self.config.use_image_conditioning:
            image_features = self.image_encoder(image)
            conditioning_features.append(image_features)
        
        # Text conditioning (optional)
        if prompt and self.config.use_text_conditioning:
            # Would need text encoder here
            pass
        
        if conditioning_features:
            combined_conditioning = torch.cat(conditioning_features, dim=1)
            return combined_conditioning
        
        return None
    
    def _add_noise_to_image(self, image: torch.Tensor, strength: float) -> torch.Tensor:
        """Add noise to input image based on strength."""
        noise = torch.randn_like(image)
        noisy_image = image + strength * noise
        return noisy_image


class PipelineFactory:
    """Factory for creating different diffusion pipelines."""
    
    @staticmethod
    def create_pipeline(pipeline_type: PipelineType, config: PipelineConfig) -> BaseDiffusionPipeline:
        """Create a pipeline based on type."""
        
        if pipeline_type == PipelineType.STABLE_DIFFUSION:
            return StableDiffusionPipeline(config)
        elif pipeline_type == PipelineType.STABLE_DIFFUSION_XL:
            return StableDiffusionXLPipeline(config)
        elif pipeline_type == PipelineType.TEXT_TO_IMAGE:
            return TextToImagePipeline(config)
        elif pipeline_type == PipelineType.IMAGE_TO_IMAGE:
            return ImageToImagePipeline(config)
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")


class PipelineAnalyzer:
    """Analyzer for diffusion pipelines."""
    
    def __init__(self) -> Any:
        self.analysis_results = {}
    
    def analyze_pipeline(self, pipeline: BaseDiffusionPipeline) -> Dict[str, Any]:
        """Analyze pipeline properties."""
        analysis = {
            'pipeline_type': pipeline.config.pipeline_type.value,
            'model_parameters': sum(p.numel() for p in pipeline.unet.parameters()),
            'text_encoder_parameters': sum(p.numel() for p in pipeline.text_encoder.parameters()) if hasattr(pipeline, 'text_encoder') else 0,
            'image_encoder_parameters': sum(p.numel() for p in pipeline.image_encoder.parameters()) if hasattr(pipeline, 'image_encoder') else 0,
            'total_parameters': 0,
            'config': pipeline.config.__dict__
        }
        
        # Calculate total parameters
        total_params = analysis['model_parameters']
        if hasattr(pipeline, 'text_encoder'):
            total_params += analysis['text_encoder_parameters']
        if hasattr(pipeline, 'image_encoder'):
            total_params += analysis['image_encoder_parameters']
        
        analysis['total_parameters'] = total_params
        analysis['model_size_mb'] = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        return analysis
    
    def benchmark_pipeline(self, pipeline: BaseDiffusionPipeline, 
                          num_samples: int = 1, **kwargs) -> Dict[str, Any]:
        """Benchmark pipeline performance."""
        
        # Warmup
        with torch.no_grad():
            _ = pipeline.generate(num_images=1, **kwargs)
        
        # Benchmark
        times = []
        memory_usage = []
        
        for _ in range(3):
            start_time = time.time()
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                results = pipeline.generate(num_images=num_samples, **kwargs)
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            if torch.cuda.is_available():
                memory_usage.append(torch.cuda.max_memory_allocated() / (1024 * 1024))
        
        return {
            'generation_time': {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times)
            },
            'memory_usage_mb': {
                'mean': np.mean(memory_usage) if memory_usage else 0,
                'max': np.max(memory_usage) if memory_usage else 0
            },
            'samples_per_second': num_samples / np.mean(times),
            'generated_images_shape': results['images'].shape
        }


def demonstrate_pipelines():
    """Demonstrate different diffusion pipelines."""
    print("Diffusion Pipelines Demonstration")
    print("=" * 50)
    
    # Test different pipeline types
    pipeline_configs = [
        (PipelineType.STABLE_DIFFUSION, "Stable Diffusion"),
        (PipelineType.STABLE_DIFFUSION_XL, "Stable Diffusion XL"),
        (PipelineType.TEXT_TO_IMAGE, "Text-to-Image"),
        (PipelineType.IMAGE_TO_IMAGE, "Image-to-Image")
    ]
    
    results = {}
    analyzer = PipelineAnalyzer()
    
    for pipeline_type, name in pipeline_configs:
        print(f"\nTesting {name}:")
        
        try:
            # Create configuration
            config = PipelineConfig(
                pipeline_type=pipeline_type,
                image_size=64,  # Smaller for demo
                use_text_conditioning=True,
                use_image_conditioning=True,
                guidance_scale=7.5,
                num_inference_steps=20
            )
            
            # Create pipeline
            pipeline = PipelineFactory.create_pipeline(pipeline_type, config)
            
            # Analyze pipeline
            analysis = analyzer.analyze_pipeline(pipeline)
            print(f"  Total parameters: {analysis['total_parameters']:,}")
            print(f"  Model size: {analysis['model_size_mb']:.2f} MB")
            
            # Benchmark pipeline
            benchmark_results = analyzer.benchmark_pipeline(
                pipeline, 
                num_samples=2,
                prompt="a beautiful landscape",
                negative_prompt="blurry, low quality"
            )
            
            print(f"  Generation time: {benchmark_results['generation_time']['mean']:.4f}s")
            print(f"  Samples per second: {benchmark_results['samples_per_second']:.2f}")
            print(f"  Generated shape: {benchmark_results['generated_images_shape']}")
            
            results[pipeline_type.value] = {
                'config': config,
                'analysis': analysis,
                'benchmark_results': benchmark_results,
                'success': True
            }
            
        except Exception as e:
            print(f"  Error: {e}")
            results[pipeline_type.value] = {
                'config': config,
                'error': str(e),
                'success': False
            }
    
    return results


if __name__ == "__main__":
    # Demonstrate pipelines
    results = demonstrate_pipelines()
    print("\nPipeline demonstration completed!") 