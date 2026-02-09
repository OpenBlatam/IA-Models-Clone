"""
Unified TruthGPT Optimizers
============================
Consolidated optimizer system to replace duplicate optimizer files.

This module provides organized access to optimizer components:
- core: Core optimizer classes and utilities
- truthgpt: TruthGPT-specific optimizers
- specialized: Specialized optimizers for specific use cases
- optimization_cores: Various optimization core implementations
- techniques: Optimization techniques and computational optimizations
- compatibility: Compatibility layers
- registries: Optimization registry systems
- kv_cache: KV cache optimizers
- tensorflow: TensorFlow optimizers
- quantum: Quantum optimizers
- production: Production optimizers
"""

from __future__ import annotations

from typing import Dict, Any

# Direct imports for backward compatibility
from optimizers.core.base_truthgpt_optimizer import (
    BaseTruthGPTOptimizer,
    UnifiedTruthGPTOptimizer,
    OptimizationLevel,
    OptimizationResult,
)

# New unified optimizer system
from optimizers.core.unified_optimizer import UnifiedOptimizer

# Compatibility shims for deprecated optimization cores
from optimizers.compatibility.shims import (
    EnhancedOptimizationCore,
    HybridOptimizationCore,
)
from optimizers.component_optimizers import (
    ComponentOptimizer,
    get_component_optimizer,
)

# Lazy imports for organized submodules
_LAZY_IMPORTS = {
    'core': '.core',
    'truthgpt': '.truthgpt',
    'specialized': '.specialized',
    'optimization_cores': '.optimization_cores',
    'techniques': '.techniques',
    'compatibility': '.compatibility',
    'registries': '.registries',
    'kv_cache': '.kv_cache',
    'tensorflow': '.tensorflow',
    'quantum': '.quantum',
    'production': '.production',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for optimizer submodules."""
    if name.startswith('_'):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name in _import_cache:
        return _import_cache[name]
    
    module_path = _LAZY_IMPORTS[name]
    try:
        module = __import__(module_path, fromlist=[name], level=1)
        _import_cache[name] = module
        return module
    except (ImportError, AttributeError) as e:
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'. "
            f"Failed to import: {e}"
        ) from e


def list_available_optimizer_submodules() -> list[str]:
    """List all available optimizer submodules."""
    return list(_LAZY_IMPORTS.keys())

# Factory function for backward compatibility
def create_truthgpt_optimizer(level: str = "basic", config: Dict[str, Any] = None):
    """
    Create a TruthGPT optimizer with the specified level.
    
    Args:
        level: Optimization level (basic, advanced, expert, etc.)
        config: Optional configuration dictionary
    
    Returns:
        UnifiedTruthGPTOptimizer instance
    """
    from typing import Dict, Any
    
    # Map string level to enum
    level_map = {
        'basic': OptimizationLevel.BASIC,
        'advanced': OptimizationLevel.ADVANCED,
        'expert': OptimizationLevel.EXPERT,
        'master': OptimizationLevel.MASTER,
        'legendary': OptimizationLevel.LEGENDARY,
        'transcendent': OptimizationLevel.TRANSCENDENT,
        'divine': OptimizationLevel.DIVINE,
        'omnipotent': OptimizationLevel.OMNIPOTENT,
        'infinite': OptimizationLevel.INFINITE,
        'ultimate': OptimizationLevel.ULTIMATE,
        'supreme': OptimizationLevel.SUPREME,
        'enterprise': OptimizationLevel.ENTERPRISE,
        'ultra_fast': OptimizationLevel.ULTRA_FAST,
        'ultra_speed': OptimizationLevel.ULTRA_SPEED,
        'hyper_speed': OptimizationLevel.HYPER_SPEED,
        'lightning_speed': OptimizationLevel.LIGHTNING_SPEED,
    }
    
    opt_level = level_map.get(level.lower(), OptimizationLevel.BASIC)
    return UnifiedTruthGPTOptimizer(config=config or {}, level=opt_level)


# Backward compatibility aliases
def create_advanced_truthgpt_optimizer(config: Dict[str, Any] = None):
    """Create an advanced TruthGPT optimizer (backward compatibility)."""
    return create_truthgpt_optimizer('advanced', config)


def create_expert_truthgpt_optimizer(config: Dict[str, Any] = None):
    """Create an expert TruthGPT optimizer (backward compatibility)."""
    return create_truthgpt_optimizer('expert', config)


def create_ultimate_truthgpt_optimizer(config: Dict[str, Any] = None):
    """Create an ultimate TruthGPT optimizer (backward compatibility)."""
    return create_truthgpt_optimizer('ultimate', config)


def create_supreme_truthgpt_optimizer(config: Dict[str, Any] = None):
    """Create a supreme TruthGPT optimizer (backward compatibility)."""
    return create_truthgpt_optimizer('supreme', config)


def create_enterprise_truthgpt_optimizer(config: Dict[str, Any] = None):
    """Create an enterprise TruthGPT optimizer (backward compatibility)."""
    return create_truthgpt_optimizer('enterprise', config)


def create_ultra_fast_truthgpt_optimizer(config: Dict[str, Any] = None):
    """Create an ultra-fast TruthGPT optimizer (backward compatibility)."""
    return create_truthgpt_optimizer('ultra_fast', config)


def create_ultra_speed_truthgpt_optimizer(config: Dict[str, Any] = None):
    """Create an ultra-speed TruthGPT optimizer (backward compatibility)."""
    return create_truthgpt_optimizer('ultra_speed', config)


def create_hyper_speed_truthgpt_optimizer(config: Dict[str, Any] = None):
    """Create a hyper-speed TruthGPT optimizer (backward compatibility)."""
    return create_truthgpt_optimizer('hyper_speed', config)


def create_lightning_speed_truthgpt_optimizer(config: Dict[str, Any] = None):
    """Create a lightning-speed TruthGPT optimizer (backward compatibility)."""
    return create_truthgpt_optimizer('lightning_speed', config)


# Generic optimizers
from optimizers.generic_optimizer import (
    UnifiedGenericOptimizer,
    GenericOptimizationLevel,
    GenericOptimizationResult,
    create_generic_optimizer,
)

# Optimization techniques and pipeline
from optimizers.techniques import (
    OptimizationTechnique,
    get_technique,
    register_technique,
    TechniqueRegistry,
)

from optimizers.optimization_pipeline import (
    OptimizationPipeline,
    OptimizationStep,
    build_optimization_pipeline,
    LevelBasedPipelineBuilder,
)

from optimizers.metrics import (
    MetricsCalculator,
    calculate_optimization_metrics,
)

__all__ = [
    # TruthGPT optimizers (backward compatible)
    'BaseTruthGPTOptimizer',
    'UnifiedTruthGPTOptimizer',
    'OptimizationLevel',
    'OptimizationResult',
    # New unified optimizer system
    'UnifiedOptimizer',
    # Compatibility shims (deprecated)
    'EnhancedOptimizationCore',
    'HybridOptimizationCore',
    'ComponentOptimizer',
    'get_component_optimizer',
    'create_truthgpt_optimizer',
    'create_advanced_truthgpt_optimizer',
    'create_expert_truthgpt_optimizer',
    'create_ultimate_truthgpt_optimizer',
    'create_supreme_truthgpt_optimizer',
    'create_enterprise_truthgpt_optimizer',
    'create_ultra_fast_truthgpt_optimizer',
    'create_ultra_speed_truthgpt_optimizer',
    'create_hyper_speed_truthgpt_optimizer',
    'create_lightning_speed_truthgpt_optimizer',
    # Generic optimizers
    'UnifiedGenericOptimizer',
    'GenericOptimizationLevel',
    'GenericOptimizationResult',
    'create_generic_optimizer',
    # Optimization techniques
    'OptimizationTechnique',
    'get_technique',
    'register_technique',
    'TechniqueRegistry',
    # Optimization pipeline
    'OptimizationPipeline',
    'OptimizationStep',
    'build_optimization_pipeline',
    'LevelBasedPipelineBuilder',
    # Metrics
    'MetricsCalculator',
    'calculate_optimization_metrics',
    # Submodules
    'core',
    'truthgpt',
    'specialized',
    'optimization_cores',
    'techniques',
    'compatibility',
    'registries',
    'kv_cache',
    'tensorflow',
    'quantum',
    'production',
    'list_available_optimizer_submodules',
]
