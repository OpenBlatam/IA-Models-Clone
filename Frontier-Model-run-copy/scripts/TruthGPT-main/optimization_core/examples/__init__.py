"""
Examples for TruthGPT Optimization Core
Provides comprehensive examples demonstrating usage of the refactored architecture
"""

from .basic_usage import (
    basic_transformer_example,
    configuration_example,
    attention_example
)

from .advanced_usage import (
    flash_attention_example,
    mixed_precision_example,
    gradient_checkpointing_example
)

from .production_example import (
    production_model_example,
    monitoring_example,
    deployment_example
)

__all__ = [
    # Basic Usage
    'basic_transformer_example',
    'configuration_example',
    'attention_example',
    
    # Advanced Usage
    'flash_attention_example',
    'mixed_precision_example',
    'gradient_checkpointing_example',
    
    # Production Examples
    'production_model_example',
    'monitoring_example',
    'deployment_example'
]


