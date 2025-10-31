"""
HeyGen AI Plugins Package

This package contains various plugins for the HeyGen AI system:
- Model plugins for different AI architectures
- Optimization plugins for performance enhancement
- Feature plugins for extended functionality
- Custom plugins for specific use cases
"""

__version__ = "1.0.0"
__author__ = "HeyGen AI Team"
__description__ = "Plugin system for HeyGen AI"

# Import available plugins
try:
    from .transformer_plugin import TransformerPlugin
    from .diffusion_plugin import DiffusionPlugin
    from .optimization_plugin import OptimizationPlugin
    from .feature_plugin import FeaturePlugin
    
    __all__ = [
        "TransformerPlugin",
        "DiffusionPlugin", 
        "OptimizationPlugin",
        "FeaturePlugin"
    ]
    
except ImportError as e:
    # Some plugins may not be available
    __all__ = []
    print(f"Warning: Some plugins not available: {e}")
