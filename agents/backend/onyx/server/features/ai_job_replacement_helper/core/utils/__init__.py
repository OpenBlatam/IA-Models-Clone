"""
Utilities Module - Módulo de utilidades
========================================

Utilidades comunes para servicios de deep learning.
"""

from .model_utils import (
    initialize_weights,
    count_parameters,
    get_model_size,
    freeze_model,
    freeze_layers,
    get_gradient_norm,
    clip_gradients,
    set_dropout,
    set_batch_norm_momentum,
    get_layer_output_shape,
    check_for_nan_inf,
)

from .training_utils import (
    TrainingMetrics,
    EarlyStopping,
    create_optimizer,
    create_scheduler,
    train_one_epoch,
    validate_one_epoch,
)

from .data_utils import (
    TensorDataset,
    create_data_splits,
    create_dataloader,
    normalize_tensor,
    one_hot_encode,
    balance_dataset,
    get_class_weights,
    collate_fn_pad,
)

from .validation_utils import (
    validate_model_config,
    validate_training_config,
    validate_data_shape,
    validate_model_output,
    validate_gradients,
    check_device_compatibility,
)

from .performance_utils import (
    timer,
    profile_model,
    benchmark_dataloader,
    get_memory_usage,
    clear_cache,
    optimize_model_for_inference,
    count_flops,
)

from .visualization_utils import (
    plot_training_history,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_learning_curve,
)

from .checkpoint_utils import (
    save_checkpoint,
    load_checkpoint,
    list_checkpoints,
    cleanup_old_checkpoints,
)

from .debugging_utils import (
    check_model_health,
    diagnose_training_issue,
    compare_models,
    trace_model_forward,
)

from .export_utils import (
    export_to_onnx,
    export_to_torchscript,
    export_model_summary,
)

__all__ = [
    # Model utilities
    "initialize_weights",
    "count_parameters",
    "get_model_size",
    "freeze_model",
    "freeze_layers",
    "get_gradient_norm",
    "clip_gradients",
    "set_dropout",
    "set_batch_norm_momentum",
    "get_layer_output_shape",
    "check_for_nan_inf",
    # Training utilities
    "TrainingMetrics",
    "EarlyStopping",
    "create_optimizer",
    "create_scheduler",
    "train_one_epoch",
    "validate_one_epoch",
    # Data utilities
    "TensorDataset",
    "create_data_splits",
    "create_dataloader",
    "normalize_tensor",
    "one_hot_encode",
    "balance_dataset",
    "get_class_weights",
    "collate_fn_pad",
    # Validation utilities
    "validate_model_config",
    "validate_training_config",
    "validate_data_shape",
    "validate_model_output",
    "validate_gradients",
    "check_device_compatibility",
    # Performance utilities
    "timer",
    "profile_model",
    "benchmark_dataloader",
    "get_memory_usage",
    "clear_cache",
    "optimize_model_for_inference",
    "count_flops",
    # Visualization utilities
    "plot_training_history",
    "plot_confusion_matrix",
    "plot_feature_importance",
    "plot_learning_curve",
    # Checkpoint utilities
    "save_checkpoint",
    "load_checkpoint",
    "list_checkpoints",
    "cleanup_old_checkpoints",
    # Debugging utilities
    "check_model_health",
    "diagnose_training_issue",
    "compare_models",
    "trace_model_forward",
    # Export utilities
    "export_to_onnx",
    "export_to_torchscript",
    "export_model_summary",
]

