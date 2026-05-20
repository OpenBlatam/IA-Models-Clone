"""
Common Module
Shared base classes, utilities, and validators
"""

from .base_classes import (
    BaseComponent,
    BaseProcessor,
    BaseEvaluator,
    ConfigurableMixin,
    SaveableMixin,
    DeviceAwareMixin
)

from .validators import (
    InputValidator,
    ModelValidator,
    ConfigValidator
)

from .errors import (
    MLException,
    ModelError,
    TrainingError,
    DataError,
    ValidationError,
    ConfigurationError,
    InferenceError,
    OptimizationError
)

from .utils import (
    ensure_tensor,
    ensure_numpy,
    get_device,
    count_parameters,
    get_model_size,
    set_seed,
    format_time,
    format_size,
    safe_divide,
    clip_values,
    normalize_tensor,
    denormalize_tensor
)

from .decorators import (
    timing_decorator,
    validate_inputs,
    handle_errors,
    ensure_device,
    cache_result,
    retry_on_failure,
    log_execution,
    profile_memory
)

from .logging_utils import (
    setup_logging,
    TrainingLogger,
    log_model_info,
    log_training_config,
    MLFormatter
)

from .config_utils import (
    ConfigManager,
    load_config,
    save_config,
    deep_merge,
    TrainingConfig,
    ModelConfig
)

from .checkpoint_utils import (
    CheckpointManager,
    save_checkpoint,
    load_checkpoint
)

try:
    from .visualization import (
        plot_training_history,
        plot_metrics,
        plot_confusion_matrix,
        plot_predictions
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

from .performance import (
    PerformanceMonitor,
    performance_context,
    benchmark_function,
    get_system_info,
    optimize_memory,
    clear_cache
)

__all__ = [
    # Base classes
    'BaseComponent',
    'BaseProcessor',
    'BaseEvaluator',
    'ConfigurableMixin',
    'SaveableMixin',
    'DeviceAwareMixin',
    # Validators
    'InputValidator',
    'ModelValidator',
    'ConfigValidator',
    # Errors
    'MLException',
    'ModelError',
    'TrainingError',
    'DataError',
    'ValidationError',
    'ConfigurationError',
    'InferenceError',
    'OptimizationError',
    # Utils
    'ensure_tensor',
    'ensure_numpy',
    'get_device',
    'count_parameters',
    'get_model_size',
    'set_seed',
    'format_time',
    'format_size',
    'safe_divide',
    'clip_values',
    'normalize_tensor',
    'denormalize_tensor',
    # Decorators
    'timing_decorator',
    'validate_inputs',
    'handle_errors',
    'ensure_device',
    'cache_result',
    'retry_on_failure',
    'log_execution',
    'profile_memory',
    # Logging
    'setup_logging',
    'TrainingLogger',
    'log_model_info',
    'log_training_config',
    'MLFormatter',
    # Performance
    'PerformanceMonitor',
    'performance_context',
    'benchmark_function',
    'get_system_info',
    'optimize_memory',
    'clear_cache',
    # Config
    'ConfigManager',
    'load_config',
    'save_config',
    'deep_merge',
    'TrainingConfig',
    'ModelConfig',
    # Checkpoint
    'CheckpointManager',
    'save_checkpoint',
    'load_checkpoint'
]

# Add visualization if available
if VISUALIZATION_AVAILABLE:
    __all__.extend([
        'plot_training_history',
        'plot_metrics',
        'plot_confusion_matrix',
        'plot_predictions'
    ])

