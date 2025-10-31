"""
Model Manager for Enhanced Transformer Models

This module provides comprehensive model management
for the refactored transformer architecture.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
import time
from ..base import BaseModelManager
from ..factories import ModelFactoryRegistry
from ...transformer_config import TransformerConfig


class EnhancedModelManager(BaseModelManager):
    """Enhanced model manager with advanced features."""
    
    def __init__(self, factory_registry: ModelFactoryRegistry):
        super().__init__(factory_registry.get_factory("enhanced"))
        self.factory_registry = factory_registry
        self.model_cache = {}
        self.model_metadata = {}
        self.performance_history = {}
    
    def create_model(self, config: TransformerConfig, model_type: str, factory_name: str = "enhanced") -> nn.Module:
        """Create a new model with caching."""
        cache_key = f"{factory_name}_{model_type}_{hash(str(config.__dict__))}"
        
        # Check cache first
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
        
        # Create model
        factory = self.factory_registry.get_factory(factory_name)
        model = factory.create_model(config, model_type)
        
        # Cache the model
        self.model_cache[cache_key] = model
        
        # Store metadata
        self.model_metadata[cache_key] = {
            "model_type": model_type,
            "factory_name": factory_name,
            "config": config.__dict__,
            "created_at": time.time(),
            "parameters": self.get_model_info(model)
        }
        
        return model
    
    def load_model(self, model_path: str, config: Optional[TransformerConfig] = None) -> nn.Module:
        """Load a model from file with metadata."""
        model_path = Path(model_path)
        
        # Load model state
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Load metadata if available
        metadata_path = model_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Recreate model if config is available
            if config is None and 'config' in metadata:
                config_dict = metadata['config']
                config = TransformerConfig(**config_dict)
            
            if config is not None:
                model = self.create_model(config, metadata.get('model_type', 'standard'))
                model.load_state_dict(state_dict)
                return model
        
        # Fallback to direct loading
        return super().load_model(model_path)
    
    def save_model(self, model: nn.Module, model_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save a model to file with metadata."""
        model_path = Path(model_path)
        
        # Save model state
        torch.save(model.state_dict(), model_path)
        
        # Save metadata
        if metadata is None:
            metadata = self._extract_model_metadata(model)
        
        metadata_path = model_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _extract_model_metadata(self, model: nn.Module) -> Dict[str, Any]:
        """Extract metadata from model."""
        return {
            "model_type": type(model).__name__,
            "parameters": self.get_model_info(model),
            "saved_at": time.time()
        }
    
    def get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = super().get_model_info(model)
        
        # Add additional information
        info.update({
            "model_class": type(model).__name__,
            "device": next(model.parameters()).device.type if list(model.parameters()) else 'cpu',
            "dtype": next(model.parameters()).dtype if list(model.parameters()) else None,
            "is_training": model.training,
            "num_layers": len([m for m in model.modules() if isinstance(m, nn.Module) and len(list(m.children())) == 0])
        })
        
        return info
    
    def compare_models(self, model1: nn.Module, model2: nn.Module) -> Dict[str, Any]:
        """Compare two models."""
        info1 = self.get_model_info(model1)
        info2 = self.get_model_info(model2)
        
        comparison = {
            "parameter_difference": info1['total_parameters'] - info2['total_parameters'],
            "memory_difference": info1['memory_mb'] - info2['memory_mb'],
            "model_classes": {
                "model1": info1['model_class'],
                "model2": info2['model_class']
            },
            "devices": {
                "model1": info1['device'],
                "model2": info2['device']
            }
        }
        
        return comparison
    
    def optimize_model(self, model: nn.Module, optimization_type: str = "memory") -> nn.Module:
        """Optimize model for different criteria."""
        if optimization_type == "memory":
            return self._optimize_memory(model)
        elif optimization_type == "speed":
            return self._optimize_speed(model)
        elif optimization_type == "accuracy":
            return self._optimize_accuracy(model)
        else:
            raise ValueError(f"Unsupported optimization type: {optimization_type}")
    
    def _optimize_memory(self, model: nn.Module) -> nn.Module:
        """Optimize model for memory usage."""
        # Convert to half precision
        model = model.half()
        
        # Apply gradient checkpointing if available
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        return model
    
    def _optimize_speed(self, model: nn.Module) -> nn.Module:
        """Optimize model for speed."""
        # Compile model if PyTorch 2.0+
        if hasattr(torch, 'compile'):
            model = torch.compile(model)
        
        return model
    
    def _optimize_accuracy(self, model: nn.Module) -> nn.Module:
        """Optimize model for accuracy."""
        # Use double precision
        model = model.double()
        
        return model
    
    def benchmark_model(self, model: nn.Module, input_shape: tuple, num_runs: int = 10) -> Dict[str, float]:
        """Benchmark model performance."""
        model.eval()
        device = next(model.parameters()).device
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_input)
        
        # Benchmark
        times = []
        memory_usage = []
        
        for _ in range(num_runs):
            start_time = time.time()
            
            with torch.no_grad():
                _ = model(dummy_input)
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            # Memory usage (simplified)
            if torch.cuda.is_available():
                memory_usage.append(torch.cuda.max_memory_allocated() / 1024 / 1024)
        
        return {
            "avg_inference_time": sum(times) / len(times),
            "min_inference_time": min(times),
            "max_inference_time": max(times),
            "avg_memory_mb": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
            "max_memory_mb": max(memory_usage) if memory_usage else 0
        }
    
    def clear_cache(self):
        """Clear model cache."""
        self.model_cache.clear()
        self.model_metadata.clear()
    
    def get_cached_models(self) -> List[str]:
        """Get list of cached model keys."""
        return list(self.model_cache.keys())
    
    def get_model_metadata(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a cached model."""
        return self.model_metadata.get(cache_key)


class ModelRegistry:
    """Registry for managing model instances."""
    
    def __init__(self):
        self.models = {}
        self.model_configs = {}
    
    def register_model(self, name: str, model: nn.Module, config: TransformerConfig):
        """Register a model with name and config."""
        self.models[name] = model
        self.model_configs[name] = config
    
    def get_model(self, name: str) -> Optional[nn.Module]:
        """Get a model by name."""
        return self.models.get(name)
    
    def get_config(self, name: str) -> Optional[TransformerConfig]:
        """Get config for a model by name."""
        return self.model_configs.get(name)
    
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self.models.keys())
    
    def unregister_model(self, name: str):
        """Unregister a model."""
        if name in self.models:
            del self.models[name]
        if name in self.model_configs:
            del self.model_configs[name]
    
    def clear(self):
        """Clear all registered models."""
        self.models.clear()
        self.model_configs.clear()

