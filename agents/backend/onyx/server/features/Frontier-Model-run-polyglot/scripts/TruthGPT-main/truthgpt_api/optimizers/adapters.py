"""
Optimization Core Adapters for TruthGPT API
===========================================

Main entry point for optimizer adapters. This module provides a unified
interface to all optimizer functionality.

This module has been modularized:
- optimizer_factories.py: Factory functions for creating optimizers
- optimizer_adapter.py: OptimizationCoreAdapter class
- paper_optimizer_utils.py: Paper-enhanced optimizer utilities
- amsgrad_utils.py: AMSGrad-specific utilities

Refactored to:
- Separate core detection logic (moved to core_detector.py)
- Eliminate code duplication in optimizer creation
- Improve separation of concerns
- Consolidate parameter mapping logic
- Modularize into specialized modules
"""

from __future__ import annotations

# Import optimizer factories
from .optimizer_factories import (
    create_tensorflow_optimizer,
    create_core_optimizer,
    create_optimizer_from_core,
    create_pytorch_optimizer,
    _map_tensorflow_to_pytorch_params
)

# Import main adapter class
from .optimizer_adapter import OptimizationCoreAdapter

# Import paper-enhanced utilities
from .paper_optimizer_utils import (
    create_paper_enhanced_optimizer,
    get_optimizer_with_paper_recommendations,
    get_paper_integration_summary
)

# Import AMSGrad utilities
from .amsgrad_utils import (
    create_amsgrad_optimizer,
    create_amsgrad_from_config,
    migrate_to_amsgrad,
    is_amsgrad_enabled,
    toggle_amsgrad,
    validate_amsgrad_params,
    get_amsgrad_performance_analysis,
    compare_amsgrad_vs_standard,
    compare_adam_variants,
    get_amsgrad_recommendations,
    get_amsgrad_statistics,
    get_optimal_amsgrad_params,
    batch_create_amsgrad_optimizers,
    get_amsgrad_summary
)

# Re-export everything for backward compatibility
__all__ = [
    # Factory functions
    'create_tensorflow_optimizer',
    'create_core_optimizer',
    'create_optimizer_from_core',
    'create_pytorch_optimizer',
    '_map_tensorflow_to_pytorch_params',
    
    # Main adapter class
    'OptimizationCoreAdapter',
    
    # Paper-enhanced utilities
    'create_paper_enhanced_optimizer',
    'get_optimizer_with_paper_recommendations',
    'get_paper_integration_summary',
    
    # AMSGrad utilities
    'create_amsgrad_optimizer',
    'create_amsgrad_from_config',
    'migrate_to_amsgrad',
    'is_amsgrad_enabled',
    'toggle_amsgrad',
    'validate_amsgrad_params',
    'get_amsgrad_performance_analysis',
    'compare_amsgrad_vs_standard',
    'compare_adam_variants',
    'get_amsgrad_recommendations',
    'get_amsgrad_statistics',
    'get_optimal_amsgrad_params',
    'batch_create_amsgrad_optimizers',
    'get_amsgrad_summary',
]
