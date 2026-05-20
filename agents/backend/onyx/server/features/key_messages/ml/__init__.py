from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

from .models import (
from .data_loader import (
from .training import (
from .evaluation import (
from .config import (
from .experiment_tracking import (
from .version_control import (
import os
import structlog
from typing import Dict, Any
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Key Messages ML Pipeline - Modular Architecture
Optimized with guard clauses and early validation
"""

# Core ML components
    BaseModel, GPT2MessageModel, BERTClassifierModel, CustomTransformerModel,
    ModelFactory, ModelEnsemble, ModelConfig, create_model, create_ensemble
)

    DataManager, MessageDataset, DataPreprocessor, 
    load_data, validate_data_quality
)

    Trainer, TrainingConfig, TrainingManager,
    train_model, prepare_training
)

    ModelEvaluator, EvaluationMetrics, 
    evaluate_model, calculate_metrics
)

# Configuration and utilities
    ConfigManager, get_model_config, 
    load_config, validate_config
)

# Experiment tracking
    ExperimentTracker, TensorBoardTracker, WandBTracker, MLflowTracker,
    CompositeTracker, setup_tracking
)

# Version control
    ModelVersionControl, save_model_version, load_model_version,
    list_model_versions, compare_models
)

# Main pipeline functions
def create_ml_pipeline(config_path: str = None) -> Dict[str, Any]:
    """Create complete ML pipeline with all components."""
    # Guard clauses for early validation
    if config_path and not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        # Load configuration
        config = load_config(config_path) if config_path else get_default_config()
        
        # Validate configuration
        validate_config(config)
        
        # Initialize components
        data_manager = DataManager(config.get('data', {}))
        model_factory = ModelFactory()
        evaluator = ModelEvaluator()
        tracker = setup_tracking(config.get('experiment_tracking', {}))
        
        pipeline = {
            'config': config,
            'data_manager': data_manager,
            'model_factory': model_factory,
            'evaluator': evaluator,
            'tracker': tracker
        }
        
        logger.info("ML pipeline created successfully", 
                   config_path=config_path,
                   components=list(pipeline.keys()))
        
        return pipeline
        
    except Exception as e:
        logger.error("Failed to create ML pipeline", error=str(e))
        raise

def get_default_config() -> Dict[str, Any]:
    """Get default configuration for the ML pipeline."""
    return {
        'app': {
            'name': 'key_messages_ml',
            'version': '2.0.0'
        },
        'models': {
            'gpt2': {
                'type': 'gpt2',
                'model_name': 'gpt2',
                'max_length': 512,
                'temperature': 0.7,
                'top_p': 0.9
            },
            'bert': {
                'type': 'bert',
                'model_name': 'bert-base-uncased',
                'max_length': 512,
                'num_labels': 5
            }
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'num_epochs': 10,
            'warmup_steps': 1000,
            'max_grad_norm': 1.0,
            'use_mixed_precision': True
        },
        'data': {
            'max_file_size': 100 * 1024 * 1024,  # 100MB
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'max_length': 512
        },
        'experiment_tracking': {
            'use_tensorboard': True,
            'use_wandb': False,
            'use_mlflow': False,
            'log_dir': 'logs/key_messages'
        },
        'evaluation': {
            'metrics': ['accuracy', 'precision', 'recall', 'f1'],
            'cross_validation_folds': 5
        }
    }

# Export main components
__all__ = [
    # Models
    'BaseModel', 'GPT2MessageModel', 'BERTClassifierModel', 'CustomTransformerModel',
    'ModelFactory', 'ModelEnsemble', 'ModelConfig', 'create_model', 'create_ensemble',
    
    # Data
    'DataManager', 'MessageDataset', 'DataPreprocessor', 'load_data', 'validate_data_quality',
    
    # Training
    'Trainer', 'TrainingConfig', 'TrainingManager', 'train_model', 'prepare_training',
    
    # Evaluation
    'ModelEvaluator', 'EvaluationMetrics', 'evaluate_model', 'calculate_metrics',
    
    # Configuration
    'ConfigManager', 'get_model_config', 'load_config', 'validate_config',
    
    # Experiment tracking
    'ExperimentTracker', 'TensorBoardTracker', 'WandBTracker', 'MLflowTracker',
    'CompositeTracker', 'setup_tracking',
    
    # Version control
    'ModelVersionControl', 'save_model_version', 'load_model_version',
    'list_model_versions', 'compare_models',
    
    # Pipeline
    'create_ml_pipeline', 'get_default_config'
]

# Import required modules

logger = structlog.get_logger(__name__) 