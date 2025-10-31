"""
Base Interfaces for Enhanced Transformer Models

This module defines the core interfaces and abstract base classes
that all transformer components must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
from ..transformer_config import TransformerConfig


class TransformerComponent(ABC):
    """Abstract base class for all transformer components."""
    
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.hidden_size = config.hidden_size
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass of the component."""
        pass
    
    @abstractmethod
    def get_parameters_count(self) -> int:
        """Get the number of parameters in this component."""
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> float:
        """Get estimated memory usage in MB."""
        pass


class AttentionMechanism(TransformerComponent):
    """Abstract base class for attention mechanisms."""
    
    @abstractmethod
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of attention mechanism."""
        pass


class TransformerBlock(TransformerComponent):
    """Abstract base class for transformer blocks."""
    
    @abstractmethod
    def forward(self, 
                x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of transformer block."""
        pass


class FeatureModule(TransformerComponent):
    """Abstract base class for feature modules."""
    
    def __init__(self, 
                 hidden_size: int, 
                 feature_dim: int = None,
                 feature_level: float = 1.0):
        self.hidden_size = hidden_size
        self.feature_dim = feature_dim or hidden_size
        self.feature_level = feature_level
        super().__init__(None)  # We'll set config later if needed
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of feature module."""
        pass
    
    @abstractmethod
    def get_feature_level(self) -> float:
        """Get the current feature level."""
        pass
    
    @abstractmethod
    def update_feature_level(self, new_level: float) -> None:
        """Update the feature level."""
        pass


class Coordinator(FeatureModule):
    """Abstract base class for coordinators."""
    
    @abstractmethod
    def integrate_features(self, x: torch.Tensor) -> torch.Tensor:
        """Integrate all feature modules."""
        pass


class ModelFactory(ABC):
    """Abstract base class for model factories."""
    
    @abstractmethod
    def create_model(self, config: TransformerConfig, model_type: str) -> nn.Module:
        """Create a model based on configuration and type."""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[str]:
        """Get list of supported model types."""
        pass


class AttentionFactory(ABC):
    """Abstract base class for attention factories."""
    
    @abstractmethod
    def create_attention(self, attention_type: str, config: TransformerConfig) -> AttentionMechanism:
        """Create an attention mechanism based on type and configuration."""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[str]:
        """Get list of supported attention types."""
        pass


class ConfigManager(ABC):
    """Abstract base class for configuration managers."""
    
    @abstractmethod
    def load_config(self, config_path: str) -> TransformerConfig:
        """Load configuration from file."""
        pass
    
    @abstractmethod
    def save_config(self, config: TransformerConfig, config_path: str) -> None:
        """Save configuration to file."""
        pass
    
    @abstractmethod
    def validate_config(self, config: TransformerConfig) -> bool:
        """Validate configuration."""
        pass


class ModelManager(ABC):
    """Abstract base class for model managers."""
    
    @abstractmethod
    def create_model(self, config: TransformerConfig, model_type: str) -> nn.Module:
        """Create a new model."""
        pass
    
    @abstractmethod
    def load_model(self, model_path: str) -> nn.Module:
        """Load a model from file."""
        pass
    
    @abstractmethod
    def save_model(self, model: nn.Module, model_path: str) -> None:
        """Save a model to file."""
        pass
    
    @abstractmethod
    def get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """Get information about a model."""
        pass


class PerformanceMonitor(ABC):
    """Abstract base class for performance monitors."""
    
    @abstractmethod
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        pass
    
    @abstractmethod
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        pass
    
    @abstractmethod
    def log_metric(self, name: str, value: float) -> None:
        """Log a performance metric."""
        pass


class PluginInterface(ABC):
    """Abstract base class for plugins."""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin."""
        pass
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input data."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get plugin name."""
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Get plugin version."""
        pass

