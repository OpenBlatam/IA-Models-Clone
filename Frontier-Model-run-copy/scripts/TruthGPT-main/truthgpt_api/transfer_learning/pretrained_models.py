"""
Pretrained Models for TruthGPT API
==================================

TensorFlow-like pretrained model loading implementation.
"""

import torch
import torchvision.models as tv_models
from typing import Optional, Dict, Any, List


class PretrainedModelLoader:
    """
    Load pretrained models.
    
    Similar to tf.keras.applications, this class
    provides pretrained models for transfer learning.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize PretrainedModelLoader.
        
        Args:
            name: Optional name for the loader
        """
        self.name = name or "pretrained_model_loader"
        self.available_models = {
            'resnet18': tv_models.resnet18,
            'resnet34': tv_models.resnet34,
            'resnet50': tv_models.resnet50,
            'resnet101': tv_models.resnet101,
            'resnet152': tv_models.resnet152,
            'densenet121': tv_models.densenet121,
            'densenet169': tv_models.densenet169,
            'densenet201': tv_models.densenet201,
            'efficientnet_b0': tv_models.efficientnet_b0,
            'efficientnet_b1': tv_models.efficientnet_b1,
            'efficientnet_b2': tv_models.efficientnet_b2,
            'vgg11': tv_models.vgg11,
            'vgg13': tv_models.vgg13,
            'vgg16': tv_models.vgg16,
            'vgg19': tv_models.vgg19,
            'alexnet': tv_models.alexnet,
            'mobilenet_v2': tv_models.mobilenet_v2,
            'mobilenet_v3_small': tv_models.mobilenet_v3_small,
            'mobilenet_v3_large': tv_models.mobilenet_v3_large
        }
    
    def load(self, model_name: str, pretrained: bool = True, num_classes: int = 1000) -> Any:
        """
        Load a pretrained model.
        
        Args:
            model_name: Name of the model to load
            pretrained: Whether to load pretrained weights
            num_classes: Number of output classes
            
        Returns:
            Pretrained model
        """
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available. Available models: {list(self.available_models.keys())}")
        
        print(f"📥 Loading model: {model_name}")
        
        # Load model
        model_fn = self.available_models[model_name]
        
        if pretrained:
            print(f"   Loading pretrained weights...")
            model = model_fn(weights='DEFAULT', num_classes=num_classes)
        else:
            model = model_fn(pretrained=False, num_classes=num_classes)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Model: {model_name}")
        print(f"   Pretrained: {pretrained}")
        print(f"   Num classes: {num_classes}")
        
        return model
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.available_models.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information
        """
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available")
        
        # Get model architecture info
        model = self.available_models[model_name](pretrained=False)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        info = {
            'name': model_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'size_mb': total_params * 4 / (1024 ** 2)  # Assuming float32
        }
        
        return info
    
    def __repr__(self):
        return f"PretrainedModelLoader(available_models={len(self.available_models)})"
