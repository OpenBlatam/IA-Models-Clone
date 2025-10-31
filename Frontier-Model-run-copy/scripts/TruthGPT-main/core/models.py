"""
Model Management System
Unified interface for model loading, configuration, and management
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union, Type
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Supported model types"""
    TRANSFORMER = "transformer"
    CONVOLUTIONAL = "convolutional"
    RECURRENT = "recurrent"
    HYBRID = "hybrid"
    CUSTOM = "custom"

@dataclass
class ModelConfig:
    """Configuration for model management"""
    model_type: ModelType = ModelType.TRANSFORMER
    model_name: str = "default"
    model_path: Optional[str] = None
    
    # Model architecture
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    vocab_size: int = 50000
    
    # Loading options
    load_pretrained: bool = True
    freeze_embeddings: bool = False
    custom_architecture: bool = False
    
    # Device and precision
    device: str = "auto"
    precision: str = "float32"
    
    def __post_init__(self):
        """Set device automatically if not specified"""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

class ModelManager:
    """Unified model management system"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.model_registry = {}
        
        logger.info(f"Initialized ModelManager for {config.model_type.value} model")
    
    def load_model(self, model_path: Optional[str] = None) -> nn.Module:
        """Load a model from path or create new one"""
        if model_path:
            self.config.model_path = model_path
            
        if self.config.model_path and Path(self.config.model_path).exists():
            return self._load_pretrained_model()
        else:
            return self._create_new_model()
    
    def _load_pretrained_model(self) -> nn.Module:
        """Load a pretrained model"""
        try:
            logger.info(f"Loading pretrained model from {self.config.model_path}")
            
            # Load model state dict
            checkpoint = torch.load(self.config.model_path, map_location=self.config.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            # Create model architecture
            model = self._create_model_architecture()
            model.load_state_dict(state_dict)
            model.to(self.config.device)
            
            self.model = model
            logger.info("Model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
            return self._create_new_model()
    
    def _create_new_model(self) -> nn.Module:
        """Create a new model with specified architecture"""
        logger.info("Creating new model with specified architecture")
        
        model = self._create_model_architecture()
        model.to(self.config.device)
        
        self.model = model
        return model
    
    def _create_model_architecture(self) -> nn.Module:
        """Create model architecture based on type"""
        if self.config.model_type == ModelType.TRANSFORMER:
            return self._create_transformer_model()
        elif self.config.model_type == ModelType.CONVOLUTIONAL:
            return self._create_convolutional_model()
        elif self.config.model_type == ModelType.RECURRENT:
            return self._create_recurrent_model()
        elif self.config.model_type == ModelType.HYBRID:
            return self._create_hybrid_model()
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def _create_transformer_model(self) -> nn.Module:
        """Create a transformer model"""
        from .architectures import TransformerModel
        
        return TransformerModel(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads
        )
    
    def _create_convolutional_model(self) -> nn.Module:
        """Create a convolutional model"""
        from .architectures import ConvolutionalModel
        
        return ConvolutionalModel(
            input_size=self.config.hidden_size,
            num_classes=self.config.vocab_size
        )
    
    def _create_recurrent_model(self) -> nn.Module:
        """Create a recurrent model"""
        from .architectures import RecurrentModel
        
        return RecurrentModel(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            vocab_size=self.config.vocab_size
        )
    
    def _create_hybrid_model(self) -> nn.Module:
        """Create a hybrid model"""
        from .architectures import HybridModel
        
        return HybridModel(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers
        )
    
    def save_model(self, save_path: str, include_optimizer: bool = False, optimizer_state: Optional[Dict] = None) -> None:
        """Save model to file"""
        try:
            save_data = {
                'model_state_dict': self.model.state_dict(),
                'config': self.config.__dict__,
                'model_type': self.config.model_type.value
            }
            
            if include_optimizer and optimizer_state:
                save_data['optimizer_state_dict'] = optimizer_state
                
            torch.save(save_data, save_path)
            logger.info(f"Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if self.model is None:
            return {"status": "No model loaded"}
            
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_type": self.config.model_type.value,
            "model_name": self.config.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(next(self.model.parameters()).device),
            "dtype": str(next(self.model.parameters()).dtype)
        }
    
    def register_model(self, name: str, model_class: Type[nn.Module], config: Dict[str, Any]) -> None:
        """Register a custom model class"""
        self.model_registry[name] = {
            'class': model_class,
            'config': config
        }
        logger.info(f"Registered custom model: {name}")
    
    def create_custom_model(self, name: str, **kwargs) -> nn.Module:
        """Create a custom registered model"""
        if name not in self.model_registry:
            raise ValueError(f"Model '{name}' not found in registry")
            
        model_info = self.model_registry[name]
        model_class = model_info['class']
        config = {**model_info['config'], **kwargs}
        
        model = model_class(**config)
        model.to(self.config.device)
        
        return model

