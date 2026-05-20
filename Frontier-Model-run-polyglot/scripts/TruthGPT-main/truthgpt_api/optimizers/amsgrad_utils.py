"""
AMSGrad Utilities - Specialized functions for AMSGrad optimizer variant.

Single Responsibility: Provide utilities for working with AMSGrad variant of Adam/AdamW.
Separated from adapters.py to improve maintainability and reduce file size.

Refactored: Consolidated all AMSGrad functions into AMSGradManager class.
"""

import logging
from typing import Dict, Any, List

from .adapters import OptimizationCoreAdapter
from .optimizer_constants import DEFAULT_LEARNING_RATE

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# AMSGRAD MANAGER CLASS
# ════════════════════════════════════════════════════════════════════════════

class AMSGradManager:
    """
    Manager class for AMSGrad optimizer functionality.
    
    Consolidates all AMSGrad-related operations into a single class
    for better organization and maintainability.
    """
    
    SUPPORTED_OPTIMIZERS = ['adam', 'adamw']
    DEFAULT_BETA_1 = 0.9
    DEFAULT_BETA_2 = 0.999
    DEFAULT_EPSILON = 1e-7
    
    @staticmethod
    def is_supported(optimizer_type: str) -> bool:
        """Check if optimizer type supports AMSGrad."""
        return optimizer_type.lower() in AMSGradManager.SUPPORTED_OPTIMIZERS
    
    @staticmethod
    def validate_optimizer_type(optimizer_type: str) -> None:
        """Validate that optimizer type supports AMSGrad."""
        if not AMSGradManager.is_supported(optimizer_type):
            raise ValueError(
                f"AMSGrad is only supported for {AMSGradManager.SUPPORTED_OPTIMIZERS}, "
                f"got '{optimizer_type}'"
            )
    
    @staticmethod
    def create(
        optimizer_type: str = 'adam',
        learning_rate: float = DEFAULT_LEARNING_RATE,
        beta_1: float = DEFAULT_BETA_1,
        beta_2: float = DEFAULT_BETA_2,
        epsilon: float = DEFAULT_EPSILON,
        **kwargs
    ) -> OptimizationCoreAdapter:
        """Create an optimizer with AMSGrad variant enabled."""
        AMSGradManager.validate_optimizer_type(optimizer_type)
        return OptimizationCoreAdapter(
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=True,
            **kwargs
        )
    
    @staticmethod
    def is_enabled(optimizer: OptimizationCoreAdapter) -> bool:
        """Check if AMSGrad is enabled in an optimizer."""
        if not AMSGradManager.is_supported(optimizer.optimizer_type):
            return False
        return optimizer.kwargs.get('amsgrad', False)
    
    @staticmethod
    def toggle(optimizer: OptimizationCoreAdapter, enable: bool = True) -> OptimizationCoreAdapter:
        """Create a new optimizer with AMSGrad toggled."""
        AMSGradManager.validate_optimizer_type(optimizer.optimizer_type)
        new_kwargs = optimizer.kwargs.copy()
        new_kwargs['amsgrad'] = enable
        return OptimizationCoreAdapter(
            optimizer_type=optimizer.optimizer_type,
            learning_rate=optimizer.learning_rate,
            use_core=optimizer.use_core,
            **new_kwargs
        )


# ════════════════════════════════════════════════════════════════════════════
# AMSGRAD CREATION AND CONFIGURATION (delegated to AMSGradManager)
# ════════════════════════════════════════════════════════════════════════════

def create_amsgrad_optimizer(
    optimizer_type: str = 'adam',
    learning_rate: float = DEFAULT_LEARNING_RATE,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-7,
    **kwargs
) -> OptimizationCoreAdapter:
    """
    Create an optimizer with AMSGrad variant enabled.
    
    Args:
        optimizer_type: Type of optimizer ('adam' or 'adamw')
        learning_rate: Learning rate
        beta_1: Exponential decay rate for first moment estimates
        beta_2: Exponential decay rate for second moment estimates
        epsilon: Small constant for numerical stability
        **kwargs: Additional optimizer parameters
    
    Returns:
        OptimizationCoreAdapter with AMSGrad enabled
    
    Raises:
        ValueError: If optimizer_type is not 'adam' or 'adamw'
    """
    """Create an optimizer with AMSGrad variant enabled."""
    return AMSGradManager.create(
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        **kwargs
    )


def create_amsgrad_from_config(config: Dict[str, Any]) -> OptimizationCoreAdapter:
    """
    Create AMSGrad optimizer from configuration dictionary.
    
    Args:
        config: Configuration dictionary with optimizer parameters
    
    Returns:
        OptimizationCoreAdapter with AMSGrad enabled
    
    Raises:
        ValueError: If configuration is invalid
    """
    optimizer_type = config.get('optimizer_type', 'adam')
    learning_rate = config.get('learning_rate', DEFAULT_LEARNING_RATE)
    
    if optimizer_type.lower() not in ['adam', 'adamw']:
        raise ValueError(f"AMSGrad is only supported for 'adam' and 'adamw', got '{optimizer_type}'")
    
    # Extract AMSGrad-specific parameters
    amsgrad_params = {
        'beta_1': config.get('beta_1', 0.9),
        'beta_2': config.get('beta_2', 0.999),
        'epsilon': config.get('epsilon', 1e-7),
        'amsgrad': True
    }
    
    # Merge with other parameters
    other_params = {k: v for k, v in config.items() 
                   if k not in ['optimizer_type', 'learning_rate', 'beta_1', 'beta_2', 'epsilon', 'amsgrad']}
    
    return OptimizationCoreAdapter(
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        **amsgrad_params,
        **other_params
    )


def migrate_to_amsgrad(
    optimizer: OptimizationCoreAdapter,
    validate_params: bool = True
) -> OptimizationCoreAdapter:
    """
    Migrate an existing optimizer to AMSGrad variant.
    
    Args:
        optimizer: Original optimizer to migrate
        validate_params: Whether to validate parameters before migration
    
    Returns:
        New OptimizationCoreAdapter with AMSGrad enabled
    
    Raises:
        ValueError: If optimizer type doesn't support AMSGrad
    """
    if optimizer.optimizer_type.lower() not in ['adam', 'adamw']:
        raise ValueError(f"AMSGrad is only supported for 'adam' and 'adamw', got '{optimizer.optimizer_type}'")
    
    if validate_params:
        validation = validate_amsgrad_params(
            beta_1=optimizer.kwargs.get('beta_1', 0.9),
            beta_2=optimizer.kwargs.get('beta_2', 0.999),
            epsilon=optimizer.kwargs.get('epsilon', 1e-7)
        )
        if not validation['valid']:
            logger.warning(f"Parameter validation issues: {validation['issues']}")
    
    new_kwargs = optimizer.kwargs.copy()
    new_kwargs['amsgrad'] = True
    
    return OptimizationCoreAdapter(
        optimizer_type=optimizer.optimizer_type,
        learning_rate=optimizer.learning_rate,
        use_core=optimizer.use_core,
        **new_kwargs
    )


# ════════════════════════════════════════════════════════════════════════════
# AMSGRAD VALIDATION AND CHECKING
# ════════════════════════════════════════════════════════════════════════════

def is_amsgrad_enabled(optimizer: OptimizationCoreAdapter) -> bool:
    """
    Check if AMSGrad is enabled in an optimizer.
    
    Args:
        optimizer: OptimizationCoreAdapter instance
    
    Returns:
        True if AMSGrad is enabled, False otherwise
    """
    return AMSGradManager.is_enabled(optimizer)


def validate_amsgrad_params(
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-7
) -> Dict[str, Any]:
    """
    Validate AMSGrad parameters and provide recommendations.
    
    Args:
        beta_1: Exponential decay rate for first moment estimates
        beta_2: Exponential decay rate for second moment estimates
        epsilon: Small constant for numerical stability
    
    Returns:
        Dictionary with validation results and recommendations
    """
    issues = []
    recommendations = []
    
    # Validate beta_1
    if not 0 < beta_1 < 1:
        issues.append(f"beta_1={beta_1} should be between 0 and 1")
    elif beta_1 < 0.8:
        recommendations.append("Consider increasing beta_1 for better momentum (typical: 0.9)")
    elif beta_1 > 0.99:
        recommendations.append("High beta_1 may slow adaptation to recent gradients")
    
    # Validate beta_2
    if not 0 < beta_2 < 1:
        issues.append(f"beta_2={beta_2} should be between 0 and 1")
    elif beta_2 < 0.9:
        recommendations.append("Consider increasing beta_2 for AMSGrad (typical: 0.999)")
    elif beta_2 > 0.9999:
        recommendations.append("Very high beta_2 may cause slow adaptation")
    
    # Validate epsilon
    if epsilon <= 0:
        issues.append(f"epsilon={epsilon} must be positive")
    elif epsilon < 1e-8:
        recommendations.append("Very small epsilon may cause numerical instability")
    elif epsilon > 1e-5:
        recommendations.append("Large epsilon may affect convergence")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'recommendations': recommendations,
        'optimal_params': {
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-7
        }
    }


def toggle_amsgrad(
    optimizer: OptimizationCoreAdapter,
    enable: bool = True
) -> OptimizationCoreAdapter:
    """
    Create a new optimizer with AMSGrad toggled.
    
    Args:
        optimizer: Original optimizer
        enable: Whether to enable AMSGrad
    
    Returns:
        New OptimizationCoreAdapter with AMSGrad toggled
    
    Raises:
        ValueError: If optimizer type doesn't support AMSGrad
    """
    return AMSGradManager.toggle(optimizer, enable)
    new_kwargs['amsgrad'] = enable
    
    return OptimizationCoreAdapter(
        optimizer_type=optimizer.optimizer_type,
        learning_rate=optimizer.learning_rate,
        use_core=optimizer.use_core,
        **new_kwargs
    )


# ════════════════════════════════════════════════════════════════════════════
# AMSGRAD ANALYSIS AND COMPARISON
# ════════════════════════════════════════════════════════════════════════════

def compare_adam_variants(
    learning_rate: float = DEFAULT_LEARNING_RATE,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-7
) -> Dict[str, Any]:
    """
    Compare standard Adam vs Adam with AMSGrad.
    
    Args:
        learning_rate: Learning rate
        beta_1: Exponential decay rate for first moment estimates
        beta_2: Exponential decay rate for second moment estimates
        epsilon: Small constant for numerical stability
    
    Returns:
        Dictionary with comparison results
    """
    adam_standard = OptimizationCoreAdapter(
        optimizer_type='adam',
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        amsgrad=False
    )
    
    adam_amsgrad = create_amsgrad_optimizer(
        optimizer_type='adam',
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon
    )
    
    return {
        'standard': {
            'config': adam_standard.get_config(),
            'description': 'Standard Adam optimizer'
        },
        'amsgrad': {
            'config': adam_amsgrad.get_config(),
            'description': 'Adam with AMSGrad variant'
        },
        'differences': {
            'amsgrad_enabled': True,
            'use_case': {
                'standard': 'General purpose, faster convergence',
                'amsgrad': 'Better for non-stationary objectives, more stable gradients'
            }
        }
    }


def get_amsgrad_performance_analysis(
    optimizer: OptimizationCoreAdapter
) -> Dict[str, Any]:
    """
    Analyze AMSGrad performance characteristics for an optimizer.
    
    Args:
        optimizer: OptimizationCoreAdapter instance
    
    Returns:
        Dictionary with performance analysis
    """
    if not is_amsgrad_enabled(optimizer):
        return {
            'amsgrad_enabled': False,
            'message': 'AMSGrad is not enabled for this optimizer'
        }
    
    config = optimizer.get_config()
    beta_1 = config.get('beta_1', 0.9)
    beta_2 = config.get('beta_2', 0.999)
    
    # Calculate effective learning rate adjustment
    effective_lr_factor = 1.0 / (1.0 - beta_2)
    
    return {
        'amsgrad_enabled': True,
        'optimizer_type': optimizer.optimizer_type,
        'parameters': {
            'beta_1': beta_1,
            'beta_2': beta_2,
            'learning_rate': optimizer.learning_rate,
            'epsilon': config.get('epsilon', 1e-7)
        },
        'performance_characteristics': {
            'gradient_stability': 'High - maintains maximum of second moment',
            'convergence_speed': 'Moderate - may be slower than standard Adam',
            'memory_overhead': 'Low - ~10-15% additional memory',
            'computation_overhead': 'Minimal - <5% additional computation',
            'effective_lr_factor': effective_lr_factor
        },
        'recommendations': {
            'use_when': [
                'Non-stationary objectives',
                'Convergence issues with standard Adam',
                'Sparse gradients',
                'Very deep networks'
            ],
            'avoid_when': [
                'Standard Adam works well',
                'Memory constrained',
                'Simple optimization problems'
            ]
        }
    }


def compare_amsgrad_vs_standard(
    optimizer_type: str = 'adam',
    learning_rate: float = DEFAULT_LEARNING_RATE,
    **kwargs
) -> Dict[str, Any]:
    """
    Compare AMSGrad vs standard optimizer side by side.
    
    Args:
        optimizer_type: Type of optimizer ('adam' or 'adamw')
        learning_rate: Learning rate
        **kwargs: Additional optimizer parameters
    
    Returns:
        Dictionary with detailed comparison
    """
    standard = OptimizationCoreAdapter(
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        amsgrad=False,
        **kwargs
    )
    
    amsgrad = create_amsgrad_optimizer(
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        **kwargs
    )
    
    return {
        'standard': {
            'config': standard.get_config(),
            'performance': get_amsgrad_performance_analysis(standard)
        },
        'amsgrad': {
            'config': amsgrad.get_config(),
            'performance': get_amsgrad_performance_analysis(amsgrad)
        },
        'key_differences': {
            'algorithm': 'AMSGrad maintains max(v_t) instead of v_t',
            'stability': 'AMSGrad provides more stable gradient estimates',
            'convergence': 'Standard may converge faster, AMSGrad more stable',
            'use_cases': {
                'standard': 'General purpose, faster training',
                'amsgrad': 'Non-stationary objectives, convergence issues'
            }
        }
    }


# ════════════════════════════════════════════════════════════════════════════
# AMSGRAD RECOMMENDATIONS AND PRESETS
# ════════════════════════════════════════════════════════════════════════════

def get_amsgrad_recommendations() -> Dict[str, Any]:
    """
    Get recommendations for when to use AMSGrad.
    
    Returns:
        Dictionary with recommendations, use cases, and performance impact
    """
    return {
        'use_amsgrad_when': [
            'Training on non-stationary objectives',
            'Experiencing convergence issues with standard Adam',
            'Working with sparse gradients',
            'Need more stable gradient estimates',
            'Training very deep networks',
            'Adversarial training scenarios'
        ],
        'avoid_amsgrad_when': [
            'Standard Adam works well',
            'Need faster convergence',
            'Memory is constrained',
            'Training on stationary objectives',
            'Simple optimization problems'
        ],
        'performance_impact': {
            'convergence': 'May converge slower but more stable',
            'memory': 'Slightly higher memory usage (~10-15% more)',
            'computation': 'Minimal overhead (<5%)',
            'stability': 'More stable gradient estimates'
        },
        'technical_details': {
            'algorithm': 'AMSGrad maintains the maximum of second moment estimates',
            'paper': 'On the Convergence of Adam and Beyond (Reddi et al., 2018)',
            'key_difference': 'Uses max(v_t) instead of v_t for second moment',
            'when_helpful': 'When second moment estimates decrease over time'
        },
        'example_usage': {
            'basic': f'optimizer = create_amsgrad_optimizer("adam", learning_rate={DEFAULT_LEARNING_RATE})',
            'with_params': f'optimizer = Adam(learning_rate={DEFAULT_LEARNING_RATE}, amsgrad=True)',
            'comparison': f'comparison = compare_adam_variants(learning_rate={DEFAULT_LEARNING_RATE})'
        }
    }


def get_optimal_amsgrad_params(
    problem_type: str = 'general',
    learning_rate: float = DEFAULT_LEARNING_RATE
) -> Dict[str, Any]:
    """
    Get optimal AMSGrad parameters for different problem types.
    
    Args:
        problem_type: Type of problem ('general', 'deep_network', 'sparse', 'adversarial')
        learning_rate: Learning rate
    
    Returns:
        Dictionary with optimal parameters
    """
    presets = {
        'general': {
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-7,
            'learning_rate': learning_rate
        },
        'deep_network': {
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-8,
            'learning_rate': learning_rate * 0.5  # Lower LR for deep networks
        },
        'sparse': {
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-7,
            'learning_rate': learning_rate
        },
        'adversarial': {
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-7,
            'learning_rate': learning_rate * 0.1  # Lower LR for adversarial
        }
    }
    
    if problem_type not in presets:
        logger.warning(f"Unknown problem_type '{problem_type}', using 'general' preset")
        problem_type = 'general'
    
    return presets[problem_type]


# ════════════════════════════════════════════════════════════════════════════
# AMSGRAD BATCH OPERATIONS AND STATISTICS
# ════════════════════════════════════════════════════════════════════════════

def batch_create_amsgrad_optimizers(
    configs: List[Dict[str, Any]]
) -> List[OptimizationCoreAdapter]:
    """
    Create multiple AMSGrad optimizers from a list of configurations.
    
    Args:
        configs: List of configuration dictionaries
    
    Returns:
        List of OptimizationCoreAdapter instances with AMSGrad enabled
    
    Raises:
        ValueError: If any configuration is invalid
    """
    optimizers = []
    errors = []
    
    for i, config in enumerate(configs):
        try:
            optimizer = create_amsgrad_from_config(config)
            optimizers.append(optimizer)
        except Exception as e:
            errors.append({'index': i, 'config': config, 'error': str(e)})
    
    if errors:
        logger.warning(f"⚠️ Failed to create {len(errors)} optimizers: {errors}")
    
    return optimizers


def get_amsgrad_statistics() -> Dict[str, Any]:
    """
    Get statistics about AMSGrad usage.
    
    Returns:
        Dictionary with AMSGrad statistics
    """
    return {
        'total_functions': 11,
        'functions': {
            'creation': ['create_amsgrad_optimizer', 'create_amsgrad_from_config', 'migrate_to_amsgrad'],
            'analysis': ['get_amsgrad_performance_analysis', 'compare_amsgrad_vs_standard', 'compare_adam_variants'],
            'validation': ['validate_amsgrad_params', 'is_amsgrad_enabled'],
            'utilities': ['get_amsgrad_recommendations', 'get_amsgrad_statistics', 'get_optimal_amsgrad_params', 'toggle_amsgrad']
        },
        'supported_optimizers': ['adam', 'adamw'],
        'key_features': [
            'Parameter validation with recommendations',
            'Performance analysis and comparison',
            'Preset configurations for common problem types',
            'Migration from standard optimizers',
            'Comprehensive statistics and recommendations'
        ]
    }


def get_amsgrad_summary() -> Dict[str, Any]:
    """
    Get a comprehensive summary of AMSGrad capabilities and usage.
    
    Returns:
        Dictionary with summary information
    """
    return {
        'total_functions': 11,
        'functions': {
            'creation': ['create_amsgrad_optimizer', 'create_amsgrad_from_config', 'migrate_to_amsgrad'],
            'analysis': ['get_amsgrad_performance_analysis', 'compare_amsgrad_vs_standard', 'compare_adam_variants'],
            'validation': ['validate_amsgrad_params', 'is_amsgrad_enabled'],
            'utilities': ['get_amsgrad_recommendations', 'get_amsgrad_statistics', 'get_optimal_amsgrad_params', 'toggle_amsgrad']
        },
        'supported_optimizers': ['adam', 'adamw'],
        'key_features': [
            'Parameter validation with recommendations',
            'Performance analysis and comparison',
            'Preset configurations for common problem types',
            'Migration from standard optimizers',
            'Comprehensive statistics and recommendations'
        ],
        'quick_start': {
            'basic': f'optimizer = create_amsgrad_optimizer("adam", learning_rate={DEFAULT_LEARNING_RATE})',
            'with_preset': f'config = get_optimal_amsgrad_params("deep_network", learning_rate={DEFAULT_LEARNING_RATE}); optimizer = create_amsgrad_from_config(config)',
            'migrate': 'amsgrad_opt = migrate_to_amsgrad(existing_optimizer)'
        }
    }
