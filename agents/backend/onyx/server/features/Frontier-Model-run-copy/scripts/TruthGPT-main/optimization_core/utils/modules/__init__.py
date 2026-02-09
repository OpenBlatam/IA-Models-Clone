"""
TruthGPT Modules Package
Modular components for TruthGPT optimization
"""

from .training import (
    TruthGPTTrainer, TruthGPTTrainingConfig, TruthGPTTrainingMetrics,
    create_truthgpt_trainer, quick_truthgpt_training
)

from .data import (
    TruthGPTDataLoader, TruthGPTDataset, TruthGPTDataConfig,
    create_truthgpt_dataloader, create_truthgpt_dataset
)

from .models import (
    TruthGPTModel, TruthGPTConfig, TruthGPTModelConfig,
    create_truthgpt_model, load_truthgpt_model, save_truthgpt_model
)

from .optimizers import (
    TruthGPTOptimizer, TruthGPTScheduler, TruthGPTOptimizerConfig,
    create_truthgpt_optimizer, create_truthgpt_scheduler
)

from .evaluation import (
    TruthGPTEvaluator, TruthGPTMetrics, TruthGPTEvaluationConfig,
    create_truthgpt_evaluator, evaluate_truthgpt_model
)

from .inference import (
    TruthGPTInference, TruthGPTInferenceConfig, TruthGPTInferenceMetrics,
    create_truthgpt_inference, quick_truthgpt_inference
)

from .monitoring import (
    TruthGPTMonitor, TruthGPTProfiler, TruthGPTLogger,
    create_truthgpt_monitor, create_truthgpt_profiler, create_truthgpt_logger
)

# Advanced modules
from .config import (
    TruthGPTBaseConfig, TruthGPTModelConfig, TruthGPTTrainingConfig, TruthGPTDataConfig, TruthGPTInferenceConfig,
    TruthGPTConfigManager, TruthGPTConfigValidator,
    create_truthgpt_config_manager, create_truthgpt_config_validator
)

from .distributed import (
    TruthGPTDistributedConfig, TruthGPTDistributedManager, TruthGPTDistributedTrainer,
    create_truthgpt_distributed_manager, create_truthgpt_distributed_trainer
)

from .compression import (
    TruthGPTCompressionConfig, TruthGPTCompressionManager,
    create_truthgpt_compression_manager, compress_truthgpt_model
)

from .attention import (
    TruthGPTAttentionConfig, TruthGPTRotaryEmbedding, TruthGPTAttentionFactory,
    create_truthgpt_attention, create_truthgpt_rotary_embedding
)

from .augmentation import (
    TruthGPTAugmentationConfig, TruthGPTAugmentationManager,
    create_truthgpt_augmentation_manager, augment_truthgpt_data
)

from .analytics import (
    TruthGPTAnalyticsConfig, TruthGPTAnalyticsManager,
    create_truthgpt_analytics_manager, analyze_truthgpt_model
)

from .deployment import (
    TruthGPTDeploymentConfig, TruthGPTDeploymentManager, TruthGPTDeploymentMonitor,
    create_truthgpt_deployment_manager, deploy_truthgpt_model
)

from .integration import (
    TruthGPTIntegrationConfig, TruthGPTIntegrationManager,
    create_truthgpt_integration_manager, integrate_truthgpt
)

from .security import (
    TruthGPTSecurityConfig, TruthGPTSecurityManager,
    create_truthgpt_security_manager
)

# Version information
__version__ = "1.0.0"
__author__ = "TruthGPT Optimization Core Team"

# Package exports
__all__ = [
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
    
    # Advanced modules
    # Configuration
    'TruthGPTBaseConfig', 'TruthGPTModelConfig', 'TruthGPTTrainingConfig', 'TruthGPTDataConfig', 'TruthGPTInferenceConfig',
    'TruthGPTConfigManager', 'TruthGPTConfigValidator',
    'create_truthgpt_config_manager', 'create_truthgpt_config_validator',
    
    # Distributed training
    'TruthGPTDistributedConfig', 'TruthGPTDistributedManager', 'TruthGPTDistributedTrainer',
    'create_truthgpt_distributed_manager', 'create_truthgpt_distributed_trainer',
    
    # Model compression
    'TruthGPTCompressionConfig', 'TruthGPTCompressionManager',
    'create_truthgpt_compression_manager', 'compress_truthgpt_model',
    
    # Advanced attention
    'TruthGPTAttentionConfig', 'TruthGPTRotaryEmbedding', 'TruthGPTAttentionFactory',
    'create_truthgpt_attention', 'create_truthgpt_rotary_embedding',
    
    # Data augmentation
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
    'create_truthgpt_security_manager',
    
    # Package info
    '__version__', '__author__'
]
