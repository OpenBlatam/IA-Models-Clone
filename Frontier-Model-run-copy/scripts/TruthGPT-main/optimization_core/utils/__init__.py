"""
Utility functions for TruthGPT Optimization Core
Provides common utilities for logging, monitoring, and performance optimization
"""

from .logging_utils import (
    setup_logging,
    get_logger,
    log_performance_metrics,
    log_model_info
)

from .monitoring_utils import (
    PerformanceMonitor,
    MemoryMonitor,
    create_performance_monitor,
    create_memory_monitor
)

from .performance_utils import (
    benchmark_model,
    profile_model,
    optimize_model,
    create_optimization_report
)

from .validation_utils import (
    validate_model_config,
    validate_training_config,
    validate_optimization_config,
    create_validation_report
)

# Import TruthGPT specific utilities
from .truthgpt_adapters import (
    TruthGPTAdapter, TruthGPTPerformanceAdapter, TruthGPTMemoryAdapter,
    TruthGPTGPUAdapter, TruthGPTValidationAdapter, TruthGPTIntegratedAdapter,
    TruthGPTConfig, create_truthgpt_adapter, quick_truthgpt_setup
)

from .truthgpt_optimization_utils import (
    TruthGPTQuantizer, TruthGPTPruner, TruthGPTDistiller,
    TruthGPTParallelProcessor, TruthGPTMemoryOptimizer, TruthGPTPerformanceOptimizer,
    TruthGPTIntegratedOptimizer, TruthGPTOptimizationConfig,
    create_truthgpt_optimizer, quick_truthgpt_optimization
)

from .truthgpt_monitoring import (
    TruthGPTMonitor, TruthGPTAnalytics, TruthGPTDashboard, TruthGPTMetrics,
    create_truthgpt_monitoring_suite, quick_truthgpt_monitoring_setup
)

from .truthgpt_integration import (
    TruthGPTIntegrationManager, TruthGPTIntegrationConfig, TruthGPTQuickSetup,
    create_truthgpt_integration, quick_truthgpt_integration,
    truthgpt_monitoring_context, truthgpt_optimization_context
)

# Import modular TruthGPT components
from .modules import (
    # Training
    TruthGPTTrainer, TruthGPTTrainingConfig, TruthGPTTrainingMetrics,
    create_truthgpt_trainer, quick_truthgpt_training,
    
    # Data
    TruthGPTDataLoader, TruthGPTDataset, TruthGPTDataConfig,
    create_truthgpt_dataloader, create_truthgpt_dataset,
    
    # Models
    TruthGPTModel, TruthGPTConfig, TruthGPTModelConfig,
    create_truthgpt_model, load_truthgpt_model, save_truthgpt_model,
    
    # Optimizers
    TruthGPTOptimizer, TruthGPTScheduler, TruthGPTOptimizerConfig,
    create_truthgpt_optimizer, create_truthgpt_scheduler,
    
    # Evaluation
    TruthGPTEvaluator, TruthGPTMetrics, TruthGPTEvaluationConfig,
    create_truthgpt_evaluator, evaluate_truthgpt_model,
    
    # Inference
    TruthGPTInference, TruthGPTInferenceConfig, TruthGPTInferenceMetrics,
    create_truthgpt_inference, quick_truthgpt_inference,
    
    # Monitoring
    TruthGPTMonitor, TruthGPTProfiler, TruthGPTLogger,
    create_truthgpt_monitor, create_truthgpt_profiler, create_truthgpt_logger
)

__all__ = [
    # Logging Utils
    'setup_logging',
    'get_logger',
    'log_performance_metrics',
    'log_model_info',
    
    # Monitoring Utils
    'PerformanceMonitor',
    'MemoryMonitor',
    'create_performance_monitor',
    'create_memory_monitor',
    
    # Performance Utils
    'benchmark_model',
    'profile_model',
    'optimize_model',
    'create_optimization_report',
    
    # Validation Utils
    'validate_model_config',
    'validate_training_config',
    'validate_optimization_config',
    'create_validation_report',
    
    # TruthGPT Adapters
    'TruthGPTAdapter', 'TruthGPTPerformanceAdapter', 'TruthGPTMemoryAdapter',
    'TruthGPTGPUAdapter', 'TruthGPTValidationAdapter', 'TruthGPTIntegratedAdapter',
    'TruthGPTConfig', 'create_truthgpt_adapter', 'quick_truthgpt_setup',
    
    # TruthGPT Optimization Utils
    'TruthGPTQuantizer', 'TruthGPTPruner', 'TruthGPTDistiller',
    'TruthGPTParallelProcessor', 'TruthGPTMemoryOptimizer', 'TruthGPTPerformanceOptimizer',
    'TruthGPTIntegratedOptimizer', 'TruthGPTOptimizationConfig',
    'create_truthgpt_optimizer', 'quick_truthgpt_optimization',
    
    # TruthGPT Monitoring
    'TruthGPTMonitor', 'TruthGPTAnalytics', 'TruthGPTDashboard', 'TruthGPTMetrics',
    'create_truthgpt_monitoring_suite', 'quick_truthgpt_monitoring_setup',
    
    # TruthGPT Integration
    'TruthGPTIntegrationManager', 'TruthGPTIntegrationConfig', 'TruthGPTQuickSetup',
    'create_truthgpt_integration', 'quick_truthgpt_integration',
    'truthgpt_monitoring_context', 'truthgpt_optimization_context',
    
    # Modular TruthGPT Components
    # Training
    'TruthGPTTrainer', 'TruthGPTTrainingConfig', 'TruthGPTTrainingMetrics',
    'create_truthgpt_trainer', 'quick_truthgpt_training',
    
    # Data
    'TruthGPTDataLoader', 'TruthGPTDataset', 'TruthGPTDataConfig',
    'create_truthgpt_dataloader', 'create_truthgpt_dataset',
    
    # Models
    'TruthGPTModel', 'TruthGPTConfig', 'TruthGPTModelConfig',
    'create_truthgpt_model', 'load_truthgpt_model', 'save_truthgpt_model',
    
    # Optimizers
    'TruthGPTOptimizer', 'TruthGPTScheduler', 'TruthGPTOptimizerConfig',
    'create_truthgpt_optimizer', 'create_truthgpt_scheduler',
    
    # Evaluation
    'TruthGPTEvaluator', 'TruthGPTMetrics', 'TruthGPTEvaluationConfig',
    'create_truthgpt_evaluator', 'evaluate_truthgpt_model',
    
    # Inference
    'TruthGPTInference', 'TruthGPTInferenceConfig', 'TruthGPTInferenceMetrics',
    'create_truthgpt_inference', 'quick_truthgpt_inference',
    
    # Monitoring
    'TruthGPTMonitor', 'TruthGPTProfiler', 'TruthGPTLogger',
    'create_truthgpt_monitor', 'create_truthgpt_profiler', 'create_truthgpt_logger',
    
    # Advanced Modules
    # Configuration
    'TruthGPTBaseConfig', 'TruthGPTModelConfig', 'TruthGPTTrainingConfig', 'TruthGPTDataConfig', 'TruthGPTInferenceConfig',
    'TruthGPTConfigManager', 'TruthGPTConfigValidator',
    'create_truthgpt_config_manager', 'create_truthgpt_config_validator',
    
    # Distributed Training
    'TruthGPTDistributedConfig', 'TruthGPTDistributedManager', 'TruthGPTDistributedTrainer',
    'create_truthgpt_distributed_manager', 'create_truthgpt_distributed_trainer',
    
    # Model Compression
    'TruthGPTCompressionConfig', 'TruthGPTCompressionManager',
    'create_truthgpt_compression_manager', 'compress_truthgpt_model',
    
    # Advanced Attention
    'TruthGPTAttentionConfig', 'TruthGPTRotaryEmbedding', 'TruthGPTAttentionFactory',
    'create_truthgpt_attention', 'create_truthgpt_rotary_embedding',
    
    # Data Augmentation
    'TruthGPTAugmentationConfig', 'TruthGPTAugmentationManager',
    'create_truthgpt_augmentation_manager', 'augment_truthgpt_data',
    
    # Analytics
    'TruthGPTAnalyticsConfig', 'TruthGPTAnalyticsManager',
    'create_truthgpt_analytics_manager', 'analyze_truthgpt_model',
    
    # Deployment
    'TruthGPTDeploymentConfig', 'TruthGPTDeploymentManager', 'TruthGPTDeploymentMonitor',
    'create_truthgpt_deployment_manager', 'deploy_truthgpt_model',
    
    # Integration
    'TruthGPTIntegrationConfig', 'TruthGPTIntegrationManager',
    'create_truthgpt_integration_manager', 'integrate_truthgpt',
    
    # Security
    'TruthGPTSecurityConfig', 'TruthGPTSecurityManager',
    'create_truthgpt_security_manager'
]