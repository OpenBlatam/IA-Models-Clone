"""
Base Classes for Enhanced Transformer Models

This module provides concrete implementations of base classes
and common functionality for transformer components.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List
from .interfaces import (
    TransformerComponent, 
    AttentionMechanism, 
    TransformerBlock, 
    FeatureModule,
    Coordinator,
    ModelFactory,
    AttentionFactory,
    ConfigManager,
    ModelManager,
    PerformanceMonitor,
    PluginInterface
)
from ..transformer_config import TransformerConfig


class BaseTransformerComponent(TransformerComponent):
    """Base implementation of transformer component."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training = True
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to appropriate device."""
        return tensor.to(self.device)
    
    def get_parameters_count(self) -> int:
        """Get the number of parameters in this component."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_memory_usage(self) -> float:
        """Get estimated memory usage in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / 1024 / 1024
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Base forward pass - to be overridden by subclasses."""
        return x


class BaseAttentionMechanism(BaseTransformerComponent, AttentionMechanism):
    """Base implementation of attention mechanism."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Base attention forward pass."""
        batch_size, seq_len, _ = query.size()
        
        # Reshape for multi-head attention
        q = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        return context, attn_weights


class BaseTransformerBlock(BaseTransformerComponent, TransformerBlock):
    """Base implementation of transformer block."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.attention = self._create_attention(config)
        self.ffn = self._create_ffn(config)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
    
    def _create_attention(self, config: TransformerConfig) -> AttentionMechanism:
        """Create attention mechanism - to be overridden by subclasses."""
        from ..attention_mechanisms import MultiHeadAttention
        return MultiHeadAttention(config)
    
    def _create_ffn(self, config: TransformerConfig) -> nn.Module:
        """Create feed-forward network."""
        return nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, 
                x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Base transformer block forward pass."""
        # Self-attention
        attn_output, attn_weights = self.attention(x, x, x, attention_mask)
        x = self.attention_norm(x + self.dropout(attn_output))
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.ffn_norm(x + ffn_output)
        
        return x, attn_weights


class BaseFeatureModule(FeatureModule):
    """Base implementation of feature module."""
    
    def __init__(self, 
                 hidden_size: int, 
                 feature_dim: int = None,
                 feature_level: float = 1.0):
        super().__init__(hidden_size, feature_dim, feature_level)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training = True
    
    def get_feature_level(self) -> float:
        """Get the current feature level."""
        return self.feature_level
    
    def update_feature_level(self, new_level: float) -> None:
        """Update the feature level."""
        self.feature_level = max(0.0, min(1.0, new_level))
    
    def get_parameters_count(self) -> int:
        """Get the number of parameters in this component."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_memory_usage(self) -> float:
        """Get estimated memory usage in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / 1024 / 1024


class BaseCoordinator(BaseFeatureModule, Coordinator):
    """Base implementation of coordinator."""
    
    def __init__(self, 
                 hidden_size: int, 
                 feature_level: float = 1.0):
        super().__init__(hidden_size, feature_level=feature_level)
        self.feature_modules = nn.ModuleList()
        self.integration_network = self._create_integration_network()
    
    def _create_integration_network(self) -> nn.Module:
        """Create integration network - to be overridden by subclasses."""
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh()
        )
    
    def add_feature_module(self, module: FeatureModule) -> None:
        """Add a feature module to the coordinator."""
        self.feature_modules.append(module)
    
    def integrate_features(self, x: torch.Tensor) -> torch.Tensor:
        """Integrate all feature modules."""
        if not self.feature_modules:
            return x
        
        outputs = []
        for module in self.feature_modules:
            outputs.append(module(x))
        
        # Combine outputs
        if len(outputs) == 1:
            combined = outputs[0]
        else:
            combined = torch.cat(outputs, dim=-1)
        
        # Integrate
        integrated = self.integration_network(combined)
        return integrated
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of coordinator."""
        return self.integrate_features(x)


class BaseModelFactory(ModelFactory):
    """Base implementation of model factory."""
    
    def __init__(self):
        self.supported_types = []
    
    def create_model(self, config: TransformerConfig, model_type: str) -> nn.Module:
        """Create a model based on configuration and type."""
        if model_type not in self.supported_types:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return self._create_specific_model(config, model_type)
    
    def _create_specific_model(self, config: TransformerConfig, model_type: str) -> nn.Module:
        """Create specific model - to be overridden by subclasses."""
        raise NotImplementedError
    
    def get_supported_types(self) -> List[str]:
        """Get list of supported model types."""
        return self.supported_types.copy()


class BaseAttentionFactory(AttentionFactory):
    """Base implementation of attention factory."""
    
    def __init__(self):
        self.supported_types = []
    
    def create_attention(self, attention_type: str, config: TransformerConfig) -> AttentionMechanism:
        """Create an attention mechanism based on type and configuration."""
        if attention_type not in self.supported_types:
            raise ValueError(f"Unsupported attention type: {attention_type}")
        
        return self._create_specific_attention(attention_type, config)
    
    def _create_specific_attention(self, attention_type: str, config: TransformerConfig) -> AttentionMechanism:
        """Create specific attention - to be overridden by subclasses."""
        raise NotImplementedError
    
    def get_supported_types(self) -> List[str]:
        """Get list of supported attention types."""
        return self.supported_types.copy()


class BaseConfigManager(ConfigManager):
    """Base implementation of configuration manager."""
    
    def load_config(self, config_path: str) -> TransformerConfig:
        """Load configuration from file."""
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return TransformerConfig(**config_dict)
    
    def save_config(self, config: TransformerConfig, config_path: str) -> None:
        """Save configuration to file."""
        import json
        config_dict = config.__dict__.copy()
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def validate_config(self, config: TransformerConfig) -> bool:
        """Validate configuration."""
        try:
            # Basic validation
            assert config.hidden_size > 0
            assert config.num_layers > 0
            assert config.num_attention_heads > 0
            assert config.intermediate_size > 0
            assert 0.0 <= config.dropout <= 1.0
            assert 0.0 <= config.layer_norm_eps <= 1.0
            return True
        except (AssertionError, AttributeError):
            return False


class BaseModelManager(ModelManager):
    """Base implementation of model manager."""
    
    def __init__(self, factory: ModelFactory):
        self.factory = factory
    
    def create_model(self, config: TransformerConfig, model_type: str) -> nn.Module:
        """Create a new model."""
        return self.factory.create_model(config, model_type)
    
    def load_model(self, model_path: str) -> nn.Module:
        """Load a model from file."""
        return torch.load(model_path, map_location='cpu')
    
    def save_model(self, model: nn.Module, model_path: str) -> None:
        """Save a model to file."""
        torch.save(model.state_dict(), model_path)
    
    def get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """Get information about a model."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'memory_mb': self._estimate_memory_usage(model),
            'device': next(model.parameters()).device.type if list(model.parameters()) else 'cpu'
        }
    
    def _estimate_memory_usage(self, model: nn.Module) -> float:
        """Estimate memory usage of model."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / 1024 / 1024


class BasePerformanceMonitor(PerformanceMonitor):
    """Base implementation of performance monitor."""
    
    def __init__(self):
        self.metrics = {}
        self.monitoring = False
    
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        self.monitoring = True
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.monitoring = False
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return self.metrics.copy()
    
    def log_metric(self, name: str, value: float) -> None:
        """Log a performance metric."""
        if self.monitoring:
            self.metrics[name] = value


class BasePlugin(PluginInterface):
    """Base implementation of plugin."""
    
    def __init__(self, name: str, version: str):
        self._name = name
        self._version = version
        self.initialized = False
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin."""
        self.initialized = True
    
    def process(self, input_data: Any) -> Any:
        """Process input data."""
        if not self.initialized:
            raise RuntimeError("Plugin not initialized")
        return input_data
    
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        self.initialized = False
    
    def get_name(self) -> str:
        """Get plugin name."""
        return self._name
    
    def get_version(self) -> str:
        """Get plugin version."""
        return self._version

