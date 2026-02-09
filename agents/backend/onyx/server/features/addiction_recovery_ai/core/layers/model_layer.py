"""
Model Layer - Ultra Modular Model Architecture
Separates model definition, configuration, and building
"""

from typing import Optional, Dict, Any, List, Type, Callable
import torch
import torch.nn as nn
import logging
from abc import ABC, abstractmethod

from .interfaces import IModel, IModelBuilder, BaseModel

logger = logging.getLogger(__name__)


# ============================================================================
# Model Configuration - Separate config from implementation
# ============================================================================

class ModelConfig:
    """Model configuration container"""
    
    def __init__(self, **kwargs):
        self.config = kwargs
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value"""
        return self.config.get(key, default)
    
    def update(self, **kwargs):
        """Update config"""
        self.config.update(kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.config.copy()
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary"""
        return cls(**config)


# ============================================================================
# Model Registry - Centralized model management
# ============================================================================

class ModelRegistry:
    """Registry for model classes and factories"""
    
    _models: Dict[str, Type] = {}
    _builders: Dict[str, Callable] = {}
    
    @classmethod
    def register_model(cls, name: str, model_class: Type):
        """Register a model class"""
        cls._models[name] = model_class
        logger.info(f"Registered model: {name}")
    
    @classmethod
    def register_builder(cls, name: str, builder_func: Callable):
        """Register a model builder function"""
        cls._builders[name] = builder_func
        logger.info(f"Registered builder: {name}")
    
    @classmethod
    def get_model_class(cls, name: str) -> Optional[Type]:
        """Get model class by name"""
        return cls._models.get(name)
    
    @classmethod
    def get_builder(cls, name: str) -> Optional[Callable]:
        """Get builder function by name"""
        return cls._builders.get(name)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered models"""
        return list(cls._models.keys())
    
    @classmethod
    def list_builders(cls) -> List[str]:
        """List all registered builders"""
        return list(cls._builders.keys())


# ============================================================================
# Model Builder - Fluent interface for building models
# ============================================================================

class ModelBuilder:
    """
    Fluent builder pattern for model construction
    Separates configuration from instantiation
    """
    
    def __init__(self):
        self.config = ModelConfig()
        self.device: Optional[torch.device] = None
        self.use_mixed_precision = False
    
    def with_config(self, **kwargs) -> 'ModelBuilder':
        """Add configuration parameters"""
        self.config.update(**kwargs)
        return self
    
    def with_device(self, device: torch.device) -> 'ModelBuilder':
        """Set device"""
        self.device = device
        return self
    
    def with_mixed_precision(self, enabled: bool = True) -> 'ModelBuilder':
        """Enable mixed precision"""
        self.use_mixed_precision = enabled
        return self
    
    def build(self, model_type: str) -> nn.Module:
        """Build model from type and config"""
        # Get model class or builder
        model_class = ModelRegistry.get_model_class(model_type)
        builder_func = ModelRegistry.get_builder(model_type)
        
        if model_class:
            model = model_class(**self.config.to_dict())
        elif builder_func:
            model = builder_func(**self.config.to_dict())
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Move to device
        if self.device:
            model = model.to(self.device)
        
        # Apply optimizations
        if self.use_mixed_precision and self.device and self.device.type == "cuda":
            model = model.half()
        
        return model
    
    def reset(self) -> 'ModelBuilder':
        """Reset builder state"""
        self.config = ModelConfig()
        self.device = None
        self.use_mixed_precision = False
        return self


# ============================================================================
# Model Factory - Simple factory for model creation
# ============================================================================

class ModelFactory:
    """Simple factory for creating models"""
    
    @staticmethod
    def create(
        model_type: str,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None
    ) -> nn.Module:
        """Create model using builder pattern"""
        builder = ModelBuilder()
        
        if config:
            builder.with_config(**config)
        if device:
            builder.with_device(device)
        
        return builder.build(model_type)
    
    @staticmethod
    def create_from_config(config: ModelConfig, device: Optional[torch.device] = None) -> nn.Module:
        """Create model from config object"""
        model_type = config.get('type') or config.get('model_type')
        if not model_type:
            raise ValueError("Model type not specified in config")
        
        builder = ModelBuilder().with_config(**config.to_dict())
        if device:
            builder.with_device(device)
        
        return builder.build(model_type)


# ============================================================================
# Model Loader - Load models from checkpoints
# ============================================================================

class ModelLoader:
    """Load models from various sources"""
    
    @staticmethod
    def load_from_checkpoint(
        model: nn.Module,
        checkpoint_path: str,
        strict: bool = True
    ) -> nn.Module:
        """Load model weights from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=strict)
        logger.info(f"Loaded model from {checkpoint_path}")
        return model
    
    @staticmethod
    def load_from_huggingface(
        model_type: str,
        model_name: str,
        device: Optional[torch.device] = None
    ) -> nn.Module:
        """Load model from HuggingFace"""
        try:
            from transformers import AutoModel
            
            model = AutoModel.from_pretrained(model_name)
            if device:
                model = model.to(device)
            
            logger.info(f"Loaded {model_type} from HuggingFace: {model_name}")
            return model
        except ImportError:
            raise ImportError("transformers library required for HuggingFace models")


# Export main components
__all__ = [
    "ModelConfig",
    "ModelRegistry",
    "ModelBuilder",
    "ModelFactory",
    "ModelLoader",
]



