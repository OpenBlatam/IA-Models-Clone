"""
Refactored Enhanced Transformer Models

This module provides a completely refactored and modular
architecture for the Enhanced Transformer Models with
advanced features and extensibility.
"""

# Base components
from .base import (
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
    PluginInterface,
    BaseTransformerComponent,
    BaseAttentionMechanism,
    BaseTransformerBlock,
    BaseFeatureModule,
    BaseCoordinator,
    BaseModelFactory,
    BaseAttentionFactory,
    BaseConfigManager,
    BaseModelManager,
    BasePerformanceMonitor,
    BasePlugin
)

# Core components
from .core import (
    EnhancedMultiHeadAttention,
    EnhancedFeedForwardNetwork,
    EnhancedTransformerBlock,
    EnhancedTransformerModel,
    SparseAttention,
    LinearAttention,
    AdaptiveAttention,
    CausalAttention
)

# Feature modules
from .features import (
    QuantumGate,
    HadamardGate,
    PauliXGate,
    PauliYGate,
    PauliZGate,
    CNOTGate,
    QuantumEntanglement,
    QuantumSuperposition,
    QuantumMeasurement,
    QuantumNeuralNetwork,
    QuantumAttention,
    QuantumTransformerBlock,
    QuantumCoordinator
)

# Factories
from .factories import (
    EnhancedModelFactory,
    EnhancedAttentionFactory,
    HybridModelFactory,
    ModelFactoryRegistry
)

# Management
from .management import (
    EnhancedConfigManager,
    ConfigBuilder,
    EnhancedModelManager,
    ModelRegistry
)

# Main API
from .api import (
    create_transformer_model,
    create_attention_mechanism,
    get_model_info,
    get_supported_types,
    EnhancedTransformerAPI
)

__version__ = "2.0.0"
__author__ = "Enhanced Transformer Team"
__email__ = "enhanced-transformer@example.com"

__all__ = [
    # Base Components
    "TransformerComponent",
    "AttentionMechanism",
    "TransformerBlock",
    "FeatureModule",
    "Coordinator",
    "ModelFactory",
    "AttentionFactory",
    "ConfigManager",
    "ModelManager",
    "PerformanceMonitor",
    "PluginInterface",
    "BaseTransformerComponent",
    "BaseAttentionMechanism",
    "BaseTransformerBlock",
    "BaseFeatureModule",
    "BaseCoordinator",
    "BaseModelFactory",
    "BaseAttentionFactory",
    "BaseConfigManager",
    "BaseModelManager",
    "BasePerformanceMonitor",
    "BasePlugin",
    
    # Core Components
    "EnhancedMultiHeadAttention",
    "EnhancedFeedForwardNetwork",
    "EnhancedTransformerBlock",
    "EnhancedTransformerModel",
    "SparseAttention",
    "LinearAttention",
    "AdaptiveAttention",
    "CausalAttention",
    
    # Feature Modules
    "QuantumGate",
    "HadamardGate",
    "PauliXGate",
    "PauliYGate",
    "PauliZGate",
    "CNOTGate",
    "QuantumEntanglement",
    "QuantumSuperposition",
    "QuantumMeasurement",
    "QuantumNeuralNetwork",
    "QuantumAttention",
    "QuantumTransformerBlock",
    "QuantumCoordinator",
    
    # Factories
    "EnhancedModelFactory",
    "EnhancedAttentionFactory",
    "HybridModelFactory",
    "ModelFactoryRegistry",
    
    # Management
    "EnhancedConfigManager",
    "ConfigBuilder",
    "EnhancedModelManager",
    "ModelRegistry",
    
    # Main API
    "create_transformer_model",
    "create_attention_mechanism",
    "get_model_info",
    "get_supported_types",
    "EnhancedTransformerAPI"
]

