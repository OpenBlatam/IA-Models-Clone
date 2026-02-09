"""
Object-Oriented Programming for Model Architectures and Functional Programming for Data Processing Pipelines
Implements best practices for OOP model design and FP data processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, TypeVar, Generic
from dataclasses import dataclass, field
from functools import partial, reduce, wraps
from pathlib import Path
import logging
import numpy as np
from enum import Enum
import json
import yaml

# Type variables for generic programming
T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')

# =============================================================================
# OBJECT-ORIENTED PROGRAMMING FOR MODEL ARCHITECTURES
# =============================================================================

class ModelType(Enum):
    """Enumeration of model types"""
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"
    DIFFUSION = "diffusion"
    HYBRID = "hybrid"

class BaseModel(ABC, nn.Module):
    """Abstract base class for all model architectures"""
    
    def __init__(self, model_type: ModelType, config: Dict[str, Any]):
        super().__init__()
        self.model_type = model_type
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Model state tracking
        self.is_training = False
        self.current_epoch = 0
        self.total_steps = 0
        
        # Initialize model components
        self._initialize_components()
        self._setup_optimization()
    
    @abstractmethod
    def _initialize_components(self):
        """Initialize model components - must be implemented by subclasses"""
        pass
    
    def _setup_optimization(self):
        """Setup optimization-related configurations"""
        # Enable gradient checkpointing if specified
        if self.config.get('enable_gradient_checkpointing', False):
            self.gradient_checkpointing_enable()
            self.logger.info("Gradient checkpointing enabled")
        
        # Setup mixed precision if specified
        if self.config.get('enable_mixed_precision', False):
            self._setup_mixed_precision()
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training"""
        if hasattr(self, 'autocast_enabled'):
            self.autocast_enabled = True
            self.logger.info("Mixed precision enabled")
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass - must be implemented by subclasses"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": self.model_type.value,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
            "config": self.config,
            "current_epoch": self.current_epoch,
            "total_steps": self.total_steps
        }
    
    def save_checkpoint(self, filepath: str, additional_info: Dict[str, Any] = None):
        """Save model checkpoint with metadata"""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_type": self.model_type.value,
            "config": self.config,
            "current_epoch": self.current_epoch,
            "total_steps": self.total_steps,
            "additional_info": additional_info or {}
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        self.load_state_dict(checkpoint["model_state_dict"])
        self.current_epoch = checkpoint.get("current_epoch", 0)
        self.total_steps = checkpoint.get("total_steps", 0)
        
        self.logger.info(f"Checkpoint loaded from {filepath}")
        return checkpoint

class TransformerModel(BaseModel):
    """Object-oriented Transformer model architecture"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(ModelType.TRANSFORMER, config)
    
    def _initialize_components(self):
        """Initialize Transformer components"""
        # Extract configuration
        vocab_size = self.config.get('vocab_size', 30000)
        d_model = self.config.get('d_model', 512)
        n_heads = self.config.get('n_heads', 8)
        n_layers = self.config.get('n_layers', 6)
        d_ff = self.config.get('d_ff', 2048)
        max_seq_len = self.config.get('max_seq_len', 512)
        dropout = self.config.get('dropout', 0.1)
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.logger.info(f"Transformer initialized: {n_layers} layers, {n_heads} heads, {d_model} dims")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for Transformer model"""
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert attention mask to format expected by PyTorch
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits

class CNNModel(BaseModel):
    """Object-oriented CNN model architecture"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(ModelType.CNN, config)
    
    def _initialize_components(self):
        """Initialize CNN components"""
        # Extract configuration
        input_channels = self.config.get('input_channels', 3)
        num_classes = self.config.get('num_classes', 1000)
        base_channels = self.config.get('base_channels', 64)
        num_layers = self.config.get('num_layers', 4)
        
        # Build CNN layers dynamically
        layers = []
        in_channels = input_channels
        
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            in_channels = out_channels
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_channels, num_classes)
        )
        
        self.logger.info(f"CNN initialized: {num_layers} layers, {base_channels} base channels")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for CNN model"""
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Global pooling
        pooled = self.global_pool(features)
        
        # Flatten
        flattened = pooled.view(pooled.size(0), -1)
        
        # Classification
        logits = self.classifier(flattened)
        
        return logits

class DiffusionModel(BaseModel):
    """Object-oriented Diffusion model architecture"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(ModelType.DIFFUSION, config)
    
    def _initialize_components(self):
        """Initialize Diffusion model components"""
        # Extract configuration
        input_channels = self.config.get('input_channels', 3)
        base_channels = self.config.get('base_channels', 128)
        time_dim = self.config.get('time_dim', 256)
        
        # Time embedding
        self.time_embedding = TimeEmbedding(time_dim)
        
        # UNet backbone
        self.unet = UNet(
            input_channels=input_channels,
            base_channels=base_channels,
            time_dim=time_dim
        )
        
        # Noise scheduler
        self.noise_scheduler = NoiseScheduler(
            num_train_timesteps=self.config.get('num_train_timesteps', 1000),
            beta_start=self.config.get('beta_start', 0.0001),
            beta_end=self.config.get('beta_end', 0.02)
        )
        
        self.logger.info("Diffusion model initialized")
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass for Diffusion model"""
        # Time embedding
        time_emb = self.time_embedding(timesteps)
        
        # UNet prediction
        noise_pred = self.unet(x, time_emb)
        
        return noise_pred

class HybridModel(BaseModel):
    """Object-oriented Hybrid model architecture combining multiple approaches"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(ModelType.HYBRID, config)
    
    def _initialize_components(self):
        """Initialize Hybrid model components"""
        # Extract configuration
        self.use_transformer = self.config.get('use_transformer', True)
        self.use_cnn = self.config.get('use_cnn', True)
        self.use_rnn = self.config.get('use_rnn', False)
        
        # Initialize sub-models
        if self.use_transformer:
            self.transformer = TransformerModel(self.config.get('transformer_config', {}))
        
        if self.use_cnn:
            self.cnn = CNNModel(self.config.get('cnn_config', {}))
        
        if self.use_rnn:
            self.rnn = RNNModel(self.config.get('rnn_config', {}))
        
        # Fusion layer
        fusion_dim = self.config.get('fusion_dim', 512)
        self.fusion_layer = nn.Linear(self._get_total_features(), fusion_dim)
        
        # Output layer
        num_classes = self.config.get('num_classes', 1000)
        self.output_layer = nn.Linear(fusion_dim, num_classes)
        
        self.logger.info("Hybrid model initialized")
    
    def _get_total_features(self) -> int:
        """Calculate total feature dimension from all sub-models"""
        total_features = 0
        
        if self.use_transformer:
            total_features += self.config.get('transformer_config', {}).get('d_model', 512)
        
        if self.use_cnn:
            base_channels = self.config.get('cnn_config', {}).get('base_channels', 64)
            num_layers = self.config.get('cnn_config', {}).get('num_layers', 4)
            total_features += base_channels * (2 ** (num_layers - 1))
        
        if self.use_rnn:
            total_features += self.config.get('rnn_config', {}).get('hidden_size', 256)
        
        return total_features
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass for Hybrid model"""
        features = []
        
        # Extract features from each sub-model
        if self.use_transformer:
            transformer_features = self.transformer(*args, **kwargs)
            # Global pooling for transformer
            if len(transformer_features.shape) == 3:
                transformer_features = torch.mean(transformer_features, dim=1)
            features.append(transformer_features)
        
        if self.use_cnn:
            cnn_features = self.cnn(*args, **kwargs)
            features.append(cnn_features)
        
        if self.use_rnn:
            rnn_features = self.rnn(*args, **kwargs)
            features.append(rnn_features)
        
        # Concatenate features
        combined_features = torch.cat(features, dim=1)
        
        # Fusion
        fused = self.fusion_layer(combined_features)
        fused = F.relu(fused)
        
        # Output
        output = self.output_layer(fused)
        
        return output

# =============================================================================
# SUPPORTING COMPONENTS FOR MODEL ARCHITECTURES
# =============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TimeEmbedding(nn.Module):
    """Time embedding for Diffusion models"""
    
    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # Convert timesteps to float and reshape
        timesteps = timesteps.float().view(-1, 1)
        return self.time_mlp(timesteps)

class UNet(nn.Module):
    """UNet architecture for Diffusion models"""
    
    def __init__(self, input_channels: int, base_channels: int, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        
        # Initial convolution
        self.init_conv = nn.Conv2d(input_channels, base_channels, 3, padding=1)
        
        # Downsampling path
        self.down1 = self._make_down_block(base_channels, base_channels * 2)
        self.down2 = self._make_down_block(base_channels * 2, base_channels * 4)
        self.down3 = self._make_down_block(base_channels * 4, base_channels * 8)
        
        # Middle
        self.middle = self._make_middle_block(base_channels * 8)
        
        # Upsampling path
        self.up3 = self._make_up_block(base_channels * 8, base_channels * 4)
        self.up2 = self._make_up_block(base_channels * 4, base_channels * 2)
        self.up1 = self._make_up_block(base_channels * 2, base_channels)
        
        # Final convolution
        self.final_conv = nn.Conv2d(base_channels, input_channels, 3, padding=1)
        
        # Time embedding projection
        self.time_proj = nn.Linear(time_dim, base_channels)
    
    def _make_down_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    
    def _make_middle_block(self, channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_up_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        # Time embedding
        time_emb = self.time_proj(time_emb)
        time_emb = time_emb.view(time_emb.size(0), time_emb.size(1), 1, 1)
        
        # Initial convolution
        x = self.init_conv(x)
        x = x + time_emb
        
        # Downsampling
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        
        # Middle
        middle = self.middle(d3)
        
        # Upsampling with skip connections
        u3 = self.up3(middle)
        u2 = self.up2(u3)
        u1 = self.up1(u2)
        
        # Final convolution
        output = self.final_conv(u1)
        
        return output

class NoiseScheduler:
    """Noise scheduler for Diffusion models"""
    
    def __init__(self, num_train_timesteps: int, beta_start: float, beta_end: float):
        self.num_train_timesteps = num_train_timesteps
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def add_noise(self, original_samples: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to samples according to timestep"""
        sqrt_alpha_cumprod = self.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = (1 - self.alphas_cumprod[timesteps]).view(-1, 1, 1, 1)
        
        noise = torch.randn_like(original_samples)
        noisy_samples = sqrt_alpha_cumprod * original_samples + sqrt_one_minus_alpha_cumprod * noise
        
        return noisy_samples

class RNNModel(BaseModel):
    """Object-oriented RNN model architecture"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(ModelType.RNN, config)
    
    def _initialize_components(self):
        """Initialize RNN components"""
        # Extract configuration
        input_size = self.config.get('input_size', 100)
        hidden_size = self.config.get('hidden_size', 256)
        num_layers = self.config.get('num_layers', 2)
        num_classes = self.config.get('num_classes', 10)
        dropout = self.config.get('dropout', 0.1)
        
        # RNN layer
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, num_classes)
        
        self.logger.info(f"RNN initialized: {num_layers} layers, {hidden_size} hidden size")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for RNN model"""
        # RNN forward pass
        rnn_out, _ = self.rnn(x)
        
        # Take the last output
        last_output = rnn_out[:, -1, :]
        
        # Classification
        output = self.output_layer(last_output)
        
        return output

# =============================================================================
# FUNCTIONAL PROGRAMMING FOR DATA PROCESSING PIPELINES
# =============================================================================

@dataclass
class DataConfig:
    """Configuration for data processing"""
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True
    drop_last: bool = False
    prefetch_factor: int = 2

class DataProcessor:
    """Functional programming approach for data processing"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @staticmethod
    def compose(*functions: Callable) -> Callable:
        """Compose multiple functions: compose(f, g, h)(x) = f(g(h(x)))"""
        def compose_functions(x):
            for f in reversed(functions):
                x = f(x)
            return x
        return compose_functions
    
    @staticmethod
    def pipe(data: T, *functions: Callable) -> U:
        """Pipe data through multiple functions: pipe(data, f, g, h) = h(g(f(data)))"""
        return reduce(lambda x, f: f(x), functions, data)
    
    @staticmethod
    def curry(func: Callable, *args, **kwargs) -> Callable:
        """Curry a function with partial arguments"""
        return partial(func, *args, **kwargs)
    
    @staticmethod
    def map_batch(func: Callable, batch: List[T]) -> List[U]:
        """Apply function to each item in batch"""
        return [func(item) for item in batch]
    
    @staticmethod
    def filter_batch(predicate: Callable, batch: List[T]) -> List[T]:
        """Filter batch based on predicate"""
        return [item for item in batch if predicate(item)]
    
    @staticmethod
    def reduce_batch(func: Callable, batch: List[T], initial: U = None) -> U:
        """Reduce batch using function"""
        if initial is None:
            return reduce(func, batch)
        return reduce(func, batch, initial)

class FunctionalDataset(Dataset):
    """Functional programming approach to dataset creation"""
    
    def __init__(self, data: List[T], transform_pipeline: Optional[Callable] = None):
        self.data = data
        self.transform_pipeline = transform_pipeline or (lambda x: x)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> U:
        item = self.data[idx]
        return self.transform_pipeline(item)
    
    def map(self, func: Callable) -> 'FunctionalDataset':
        """Apply function to all data items"""
        new_data = [func(item) for item in self.data]
        return FunctionalDataset(new_data, self.transform_pipeline)
    
    def filter(self, predicate: Callable) -> 'FunctionalDataset':
        """Filter data based on predicate"""
        new_data = [item for item in self.data if predicate(item)]
        return FunctionalDataset(new_data, self.transform_pipeline)
    
    def batch(self, batch_size: int) -> List[List[T]]:
        """Create batches from data"""
        return [self.data[i:i + batch_size] for i in range(0, len(self.data), batch_size)]

class TransformPipeline:
    """Functional transform pipeline for data processing"""
    
    def __init__(self):
        self.transforms: List[Callable] = []
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_transform(self, transform: Callable) -> 'TransformPipeline':
        """Add transform to pipeline"""
        self.transforms.append(transform)
        return self
    
    def compose(self) -> Callable:
        """Compose all transforms into single function"""
        if not self.transforms:
            return lambda x: x
        
        def composed_transform(x):
            for transform in self.transforms:
                x = transform(x)
            return x
        
        return composed_transform
    
    def apply(self, data: T) -> U:
        """Apply pipeline to data"""
        return self.compose()(data)
    
    def apply_batch(self, batch: List[T]) -> List[U]:
        """Apply pipeline to batch of data"""
        return [self.apply(item) for item in batch]

# =============================================================================
# FUNCTIONAL DATA TRANSFORMS
# =============================================================================

def text_tokenize(text: str, tokenizer: Callable) -> List[int]:
    """Functional text tokenization"""
    return tokenizer(text)

def text_pad(tokens: List[int], max_length: int, pad_token: int = 0) -> List[int]:
    """Functional text padding"""
    if len(tokens) >= max_length:
        return tokens[:max_length]
    return tokens + [pad_token] * (max_length - len(tokens))

def text_truncate(tokens: List[int], max_length: int) -> List[int]:
    """Functional text truncation"""
    return tokens[:max_length]

def image_resize(image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Functional image resizing"""
    return F.interpolate(image.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)

def image_normalize(image: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """Functional image normalization"""
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)
    return (image - mean_tensor) / std_tensor

def image_augment(image: torch.Tensor, augmentation_type: str) -> torch.Tensor:
    """Functional image augmentation"""
    if augmentation_type == "horizontal_flip" and torch.rand(1) > 0.5:
        return torch.flip(image, [-1])
    elif augmentation_type == "vertical_flip" and torch.rand(1) > 0.5:
        return torch.flip(image, [-2])
    elif augmentation_type == "rotation" and torch.rand(1) > 0.5:
        angle = torch.rand(1) * 30 - 15  # Random rotation between -15 and 15 degrees
        return F.rotate(image, angle.item())
    
    return image

# =============================================================================
# FACTORY PATTERN FOR MODEL CREATION
# =============================================================================

class ModelFactory:
    """Factory for creating model instances"""
    
    @staticmethod
    def create_model(model_type: ModelType, config: Dict[str, Any]) -> BaseModel:
        """Create model instance based on type"""
        if model_type == ModelType.TRANSFORMER:
            return TransformerModel(config)
        elif model_type == ModelType.CNN:
            return CNNModel(config)
        elif model_type == ModelType.RNN:
            return RNNModel(config)
        elif model_type == ModelType.DIFFUSION:
            return DiffusionModel(config)
        elif model_type == ModelType.HYBRID:
            return HybridModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

# =============================================================================
# BUILDER PATTERN FOR COMPLEX CONFIGURATIONS
# =============================================================================

class ModelConfigBuilder:
    """Builder pattern for complex model configurations"""
    
    def __init__(self):
        self.config = {}
    
    def set_model_type(self, model_type: ModelType) -> 'ModelConfigBuilder':
        """Set model type"""
        self.config['model_type'] = model_type
        return self
    
    def set_transformer_config(self, **kwargs) -> 'ModelConfigBuilder':
        """Set Transformer-specific configuration"""
        if 'transformer_config' not in self.config:
            self.config['transformer_config'] = {}
        self.config['transformer_config'].update(kwargs)
        return self
    
    def set_cnn_config(self, **kwargs) -> 'ModelConfigBuilder':
        """Set CNN-specific configuration"""
        if 'cnn_config' not in self.config:
            self.config['cnn_config'] = {}
        self.config['cnn_config'].update(kwargs)
        return self
    
    def set_rnn_config(self, **kwargs) -> 'ModelConfigBuilder':
        """Set RNN-specific configuration"""
        if 'rnn_config' not in self.config:
            self.config['rnn_config'] = {}
        self.config['rnn_config'].update(kwargs)
        return self
    
    def set_optimization_config(self, **kwargs) -> 'ModelConfigBuilder':
        """Set optimization configuration"""
        self.config.update(kwargs)
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build final configuration"""
        return self.config.copy()

# =============================================================================
# EXAMPLE USAGE AND DEMONSTRATION
# =============================================================================

def create_text_processing_pipeline() -> TransformPipeline:
    """Create functional text processing pipeline"""
    pipeline = TransformPipeline()
    
    # Add transforms using functional composition
    pipeline.add_transform(lambda x: x.lower())  # Lowercase
    pipeline.add_transform(lambda x: x.strip())  # Strip whitespace
    pipeline.add_transform(lambda x: ' '.join(x.split()))  # Normalize whitespace
    
    return pipeline

def create_image_processing_pipeline() -> TransformPipeline:
    """Create functional image processing pipeline"""
    pipeline = TransformPipeline()
    
    # Add transforms
    pipeline.add_transform(lambda x: image_resize(x, (224, 224)))
    pipeline.add_transform(lambda x: image_normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    pipeline.add_transform(lambda x: image_augment(x, "horizontal_flip"))
    
    return pipeline

def main():
    """Example usage of OOP and FP systems"""
    
    # =============================================================================
    # OBJECT-ORIENTED MODEL CREATION
    # =============================================================================
    
    print("=== Object-Oriented Model Creation ===")
    
    # Create Transformer model using factory
    transformer_config = {
        'vocab_size': 30000,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 2048,
        'max_seq_len': 512,
        'dropout': 0.1,
        'enable_gradient_checkpointing': True,
        'enable_mixed_precision': True
    }
    
    transformer_model = ModelFactory.create_model(ModelType.TRANSFORMER, transformer_config)
    print(f"Transformer model created: {transformer_model.get_model_info()}")
    
    # Create CNN model
    cnn_config = {
        'input_channels': 3,
        'num_classes': 1000,
        'base_channels': 64,
        'num_layers': 4
    }
    
    cnn_model = ModelFactory.create_model(ModelType.CNN, cnn_config)
    print(f"CNN model created: {cnn_model.get_model_info()}")
    
    # Create Hybrid model using builder pattern
    hybrid_config = (ModelConfigBuilder()
                    .set_model_type(ModelType.HYBRID)
                    .set_transformer_config(vocab_size=30000, d_model=256, n_heads=4, n_layers=3)
                    .set_cnn_config(input_channels=3, num_classes=1000, base_channels=32, num_layers=3)
                    .set_optimization_config(enable_gradient_checkpointing=True)
                    .build())
    
    hybrid_model = ModelFactory.create_model(ModelType.HYBRID, hybrid_config)
    print(f"Hybrid model created: {hybrid_model.get_model_info()}")
    
    # =============================================================================
    # FUNCTIONAL PROGRAMMING DATA PROCESSING
    # =============================================================================
    
    print("\n=== Functional Programming Data Processing ===")
    
    # Create data processor
    data_config = DataConfig(batch_size=16, num_workers=2)
    data_processor = DataProcessor(data_config)
    
    # Example data
    sample_texts = ["Hello World", "  Python Programming  ", "Deep Learning"]
    sample_images = [torch.randn(3, 256, 256) for _ in range(3)]
    
    # Functional composition example
    text_pipeline = create_text_processing_pipeline()
    image_pipeline = create_image_processing_pipeline()
    
    # Process text data functionally
    processed_texts = data_processor.pipe(
        sample_texts,
        lambda x: data_processor.map_batch(str.lower, x),
        lambda x: data_processor.map_batch(str.strip, x),
        lambda x: data_processor.map_batch(lambda s: ' '.join(s.split()), x)
    )
    
    print(f"Original texts: {sample_texts}")
    print(f"Processed texts: {processed_texts}")
    
    # Process image data functionally
    processed_images = data_processor.pipe(
        sample_images,
        lambda x: data_processor.map_batch(lambda img: image_resize(img, (224, 224)), x),
        lambda x: data_processor.map_batch(lambda img: image_normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), x)
    )
    
    print(f"Image processing completed: {len(processed_images)} images")
    
    # =============================================================================
    # FUNCTIONAL DATASET CREATION
    # =============================================================================
    
    print("\n=== Functional Dataset Creation ===")
    
    # Create functional dataset
    dataset = FunctionalDataset(sample_texts, text_pipeline.compose())
    
    # Apply functional operations
    filtered_dataset = dataset.filter(lambda x: len(x) > 5)
    mapped_dataset = dataset.map(lambda x: f"Processed: {x}")
    
    print(f"Original dataset size: {len(dataset)}")
    print(f"Filtered dataset size: {len(filtered_dataset)}")
    print(f"Mapped dataset sample: {mapped_dataset[0]}")
    
    # =============================================================================
    # MODEL CHECKPOINTING
    # =============================================================================
    
    print("\n=== Model Checkpointing ===")
    
    # Save checkpoint
    checkpoint_path = "transformer_checkpoint.pt"
    transformer_model.save_checkpoint(checkpoint_path, {"example": "data"})
    
    # Load checkpoint
    loaded_model = ModelFactory.create_model(ModelType.TRANSFORMER, transformer_config)
    checkpoint_info = loaded_model.load_checkpoint(checkpoint_path)
    
    print(f"Checkpoint loaded: {checkpoint_info}")

if __name__ == "__main__":
    main()


