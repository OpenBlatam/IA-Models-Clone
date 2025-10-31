"""
Base Module for Enhanced Transformer Models

This module provides the foundational interfaces and base classes
for the refactored transformer architecture.
"""

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

from .base_classes import (
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

__all__ = [
    # Interfaces
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
    
    # Base Classes
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
    "BasePlugin"
]

