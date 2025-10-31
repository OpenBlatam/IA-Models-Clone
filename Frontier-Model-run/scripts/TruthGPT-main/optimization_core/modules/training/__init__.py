"""
Ultra-fast modular training components
Following deep learning best practices
"""

from .trainer import FastTrainer, TrainerConfig, TrainingStep
from .data_loader import FastDataLoader, DataLoaderConfig, DataProcessor
from .optimizer import FastOptimizer, OptimizerConfig, SchedulerConfig
from .loss import LossFunction, LossConfig, compute_loss
from .metrics import MetricsTracker, MetricConfig, compute_metrics
from .checkpoint import CheckpointManager, CheckpointConfig, save_checkpoint, load_checkpoint
from .validation import Validator, ValidationConfig, validate_model
from .profiler import TrainingProfiler, ProfilerConfig, profile_training

__all__ = [
    # Training
    'FastTrainer', 'TrainerConfig', 'TrainingStep',
    
    # Data loading
    'FastDataLoader', 'DataLoaderConfig', 'DataProcessor',
    
    # Optimization
    'FastOptimizer', 'OptimizerConfig', 'SchedulerConfig',
    
    # Loss functions
    'LossFunction', 'LossConfig', 'compute_loss',
    
    # Metrics
    'MetricsTracker', 'MetricConfig', 'compute_metrics',
    
    # Checkpointing
    'CheckpointManager', 'CheckpointConfig', 'save_checkpoint', 'load_checkpoint',
    
    # Validation
    'Validator', 'ValidationConfig', 'validate_model',
    
    # Profiling
    'TrainingProfiler', 'ProfilerConfig', 'profile_training'
]


