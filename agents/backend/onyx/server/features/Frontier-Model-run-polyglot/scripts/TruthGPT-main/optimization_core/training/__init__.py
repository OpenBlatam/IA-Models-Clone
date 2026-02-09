"""
Training Module - Professional Training Components.

This module provides organized access to training components:
- Training loop: Core training loop with AMP, gradient accumulation
- Checkpoint manager: Save/load training state
- EMA manager: Exponential Moving Average for model weights
- Evaluator: Model evaluation and metrics
- Experiment tracker: WandB and TensorBoard integration
"""

from __future__ import annotations

# Direct imports for backward compatibility
from .training_loop import TrainingLoop
from .checkpoint_manager import CheckpointManager
from .ema_manager import EMAManager
from .evaluator import Evaluator
from .experiment_tracker import ExperimentTracker

# ════════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def create_training_component(
    component_type: str,
    config: dict = None
):
    """
    Unified factory function to create training components.
    
    Args:
        component_type: Type of component. Options:
            - "training_loop": TrainingLoop
            - "checkpoint_manager": CheckpointManager
            - "ema_manager": EMAManager
            - "evaluator": Evaluator
            - "experiment_tracker": ExperimentTracker
        config: Optional configuration dictionary
    
    Returns:
        The requested component instance
    
    Examples:
        >>> # Create training loop
        >>> loop = create_training_component(
        ...     "training_loop",
        ...     {"use_amp": True, "max_grad_norm": 1.0}
        ... )
        
        >>> # Create checkpoint manager
        >>> checkpoint = create_training_component(
        ...     "checkpoint_manager",
        ...     {"output_dir": "./checkpoints"}
        ... )
        
        >>> # Create EMA manager
        >>> ema = create_training_component(
        ...     "ema_manager",
        ...     {"decay": 0.999}
        ... )
    """
    if config is None:
        config = {}
    
    component_type = component_type.lower()
    
    factory_map = {
        "training_loop": lambda cfg: TrainingLoop(**cfg),
        "checkpoint_manager": lambda cfg: CheckpointManager(**cfg),
        "ema_manager": lambda cfg: EMAManager(**cfg),
        "evaluator": lambda cfg: Evaluator(**cfg),
        "experiment_tracker": lambda cfg: ExperimentTracker(**cfg),
    }
    
    if component_type not in factory_map:
        available = ", ".join(factory_map.keys())
        raise ValueError(
            f"Unknown training component type: '{component_type}'. "
            f"Available types: {available}"
        )
    
    factory = factory_map[component_type]
    return factory(config)


# ════════════════════════════════════════════════════════════════════════════════
# REGISTRY SYSTEM
# ════════════════════════════════════════════════════════════════════════════════

TRAINING_COMPONENT_REGISTRY = {
    "training_loop": {
        "class": TrainingLoop,
        "module": "training.training_loop",
        "description": "Core training loop with AMP, gradient accumulation, and gradient clipping",
        "default_config": {
            "use_amp": False,
            "max_grad_norm": 1.0,
            "grad_accum_steps": 1,
        },
    },
    "checkpoint_manager": {
        "class": CheckpointManager,
        "module": "training.checkpoint_manager",
        "description": "Manages training checkpoints with SafeTensors support",
        "default_config": {
            "output_dir": "./checkpoints",
        },
    },
    "ema_manager": {
        "class": EMAManager,
        "module": "training.ema_manager",
        "description": "Exponential Moving Average manager for model weights",
        "default_config": {
            "decay": 0.999,
        },
    },
    "evaluator": {
        "class": Evaluator,
        "module": "training.evaluator",
        "description": "Model evaluator with AMP support and comprehensive metrics",
        "default_config": {
            "use_amp": False,
        },
    },
    "experiment_tracker": {
        "class": ExperimentTracker,
        "module": "training.experiment_tracker",
        "description": "Experiment tracking with WandB and TensorBoard support",
        "default_config": {
            "trackers": ["tensorboard"],
        },
    },
}


def list_available_training_components() -> list[str]:
    """
    List all available training component types.
    
    Returns:
        List of component type names
    
    Examples:
        >>> components = list_available_training_components()
        >>> print(components)
        ['training_loop', 'checkpoint_manager', 'ema_manager', 'evaluator', 'experiment_tracker']
    """
    return list(TRAINING_COMPONENT_REGISTRY.keys())


def get_training_component_info(component_type: str) -> dict[str, any]:
    """
    Get information about a training component.
    
    Args:
        component_type: Component type name
    
    Returns:
        Dictionary with component information
    
    Examples:
        >>> info = get_training_component_info("training_loop")
        >>> print(info['description'])
        'Core training loop with AMP, gradient accumulation, and gradient clipping'
    """
    if component_type not in TRAINING_COMPONENT_REGISTRY:
        raise ValueError(
            f"Unknown training component: {component_type}. "
            f"Available: {list_available_training_components()}"
        )
    
    registry_entry = TRAINING_COMPONENT_REGISTRY[component_type]
    return {
        'name': component_type,
        'class': registry_entry['class'].__name__,
        'module': registry_entry['module'],
        'description': registry_entry['description'],
        'default_config': registry_entry.get('default_config', {}),
    }


# ════════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def create_training_loop(
    use_amp: bool = False,
    max_grad_norm: float = 1.0,
    grad_accum_steps: int = 1,
    **kwargs
) -> TrainingLoop:
    """
    Create a training loop with common configuration.
    
    Args:
        use_amp: Use automatic mixed precision
        max_grad_norm: Maximum gradient norm for clipping
        grad_accum_steps: Gradient accumulation steps
        **kwargs: Additional TrainingLoop arguments
    
    Returns:
        TrainingLoop instance
    """
    return TrainingLoop(
        use_amp=use_amp,
        max_grad_norm=max_grad_norm,
        grad_accum_steps=grad_accum_steps,
        **kwargs
    )


def create_checkpoint_manager(output_dir: str) -> CheckpointManager:
    """
    Create a checkpoint manager.
    
    Args:
        output_dir: Output directory for checkpoints
    
    Returns:
        CheckpointManager instance
    """
    return CheckpointManager(output_dir=output_dir)


def create_ema_manager(decay: float = 0.999, model=None) -> EMAManager:
    """
    Create an EMA manager.
    
    Args:
        decay: EMA decay factor
        model: Optional model to initialize from
    
    Returns:
        EMAManager instance
    """
    return EMAManager(decay=decay, model=model)


def create_evaluator(
    use_amp: bool = False,
    amp_dtype=None,
    device=None,
    **kwargs
) -> Evaluator:
    """
    Create an evaluator.
    
    Args:
        use_amp: Use automatic mixed precision
        amp_dtype: AMP dtype (bf16/fp16)
        device: Device for evaluation
        **kwargs: Additional Evaluator arguments
    
    Returns:
        Evaluator instance
    """
    return Evaluator(
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        device=device,
        **kwargs
    )


def create_experiment_tracker(
    trackers: list[str] = None,
    project: str = None,
    run_name: str = None,
    log_dir: str = None,
    **kwargs
) -> ExperimentTracker:
    """
    Create an experiment tracker.
    
    Args:
        trackers: List of trackers (wandb|tensorboard)
        project: Project name (for WandB)
        run_name: Run name
        log_dir: Log directory (for TensorBoard)
        **kwargs: Additional ExperimentTracker arguments
    
    Returns:
        ExperimentTracker instance
    """
    return ExperimentTracker(
        trackers=trackers,
        project=project,
        run_name=run_name,
        log_dir=log_dir,
        **kwargs
    )


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Core components (backward compatible)
    "TrainingLoop",
    "CheckpointManager",
    "EMAManager",
    "Evaluator",
    "ExperimentTracker",
    # Factory functions
    "create_training_component",
    "create_training_loop",
    "create_checkpoint_manager",
    "create_ema_manager",
    "create_evaluator",
    "create_experiment_tracker",
    # Registry
    "TRAINING_COMPONENT_REGISTRY",
    "list_available_training_components",
    "get_training_component_info",
]
