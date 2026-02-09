from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
import subprocess
from datetime import datetime
from functools import partial, reduce
from operator import itemgetter
import hashlib
from functional_training import TrainingConfig, ModelType, TrainingMode
    import platform
    import torch
    import sys
    import psutil
    import torch
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🚀 Functional Configuration Loader
=================================

Pure functional, declarative approach to configuration management.
Uses data transformations, pure functions, and functional patterns instead of classes.

Key Principles:
- Pure functions with no side effects
- Data transformations over mutable state
- Composition over inheritance
- Immutable data structures
- Declarative configuration
"""



# ============================================================================
# Pure Data Structures
# ============================================================================

@dataclass(frozen=True)
class ExperimentMetadata:
    """Immutable experiment metadata."""
    experiment_id: str
    description: str
    git_commit: str
    git_diff: str
    timestamp: str
    environment_info: Dict[str, Any]

@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable experiment configuration."""
    experiment_id: str
    description: str
    config: TrainingConfig
    metadata: ExperimentMetadata

@dataclass(frozen=True)
class ConfigValidationResult:
    """Immutable configuration validation result."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]

# ============================================================================
# Pure Functions - Git Operations
# ============================================================================

def get_git_commit_hash() -> str:
    """Get current git commit hash in a pure functional way."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

def get_git_diff() -> str:
    """Get current git diff in a pure functional way."""
    try:
        result = subprocess.run(
            ['git', 'diff'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""

def get_git_branch() -> str:
    """Get current git branch in a pure functional way."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

# ============================================================================
# Pure Functions - Environment Information
# ============================================================================

def get_environment_info() -> Dict[str, Any]:
    """Get environment information in a pure functional way."""
    
    return {
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'platform': platform.platform(),
        'architecture': platform.architecture()[0],
        'processor': platform.processor(),
        'timestamp': datetime.now().isoformat(),
        'git_commit': get_git_commit_hash(),
        'git_branch': get_git_branch(),
        'sys_path': sys.path[:5]  # First 5 entries for brevity
    }

def get_system_resources() -> Dict[str, Any]:
    """Get system resource information in a pure functional way."""
    
    cpu_info = {
        'cpu_count': psutil.cpu_count(),
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'memory_percent': psutil.virtual_memory().percent
    }
    
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            'gpu_count': torch.cuda.device_count(),
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'gpu_memory_total': [torch.cuda.get_device_properties(i).total_memory 
                               for i in range(torch.cuda.device_count())],
            'gpu_memory_allocated': [torch.cuda.memory_allocated(i) 
                                   for i in range(torch.cuda.device_count())]
        }
    
    return {**cpu_info, **gpu_info}

# ============================================================================
# Pure Functions - Configuration Loading
# ============================================================================

def load_yaml_file(yaml_path: str) -> Dict[str, Any]:
    """Load YAML file in a pure functional way."""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise ValueError(f"Config file not found: {yaml_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format: {e}")

def save_yaml_file(data: Dict[str, Any], yaml_path: str) -> None:
    """Save data to YAML file in a pure functional way."""
    try:
        with open(yaml_path, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(data, f, default_flow_style=False, indent=2, allow_unicode=True)
    except Exception as e:
        raise ValueError(f"Failed to save YAML file: {e}")

def convert_string_enums(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert string enum values to actual enum objects in a pure functional way."""
    converted = config_dict.copy()
    
    if 'model_type' in converted:
        try:
            converted['model_type'] = ModelType(converted['model_type'])
        except ValueError:
            raise ValueError(f"Invalid model_type: {converted['model_type']}")
    
    if 'training_mode' in converted:
        try:
            converted['training_mode'] = TrainingMode(converted['training_mode'])
        except ValueError:
            raise ValueError(f"Invalid training_mode: {converted['training_mode']}")
    
    return converted

def convert_enums_to_strings(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert enum objects to strings for serialization in a pure functional way."""
    converted = config_dict.copy()
    
    if 'model_type' in converted and isinstance(converted['model_type'], ModelType):
        converted['model_type'] = converted['model_type'].value
    
    if 'training_mode' in converted and isinstance(converted['training_mode'], TrainingMode):
        converted['training_mode'] = converted['training_mode'].value
    
    return converted

def load_config_from_yaml(yaml_path: str) -> TrainingConfig:
    """Load TrainingConfig from YAML file in a pure functional way."""
    config_dict = load_yaml_file(yaml_path)
    config_dict = convert_string_enums(config_dict)
    
    try:
        return TrainingConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid configuration parameters: {e}")

def save_config_to_yaml(config: TrainingConfig, yaml_path: str) -> None:
    """Save TrainingConfig to YAML file in a pure functional way."""
    config_dict = asdict(config)
    config_dict = convert_enums_to_strings(config_dict)
    save_yaml_file(config_dict, yaml_path)

# ============================================================================
# Pure Functions - Experiment Configuration
# ============================================================================

def create_experiment_metadata(
    experiment_id: str,
    description: str
) -> ExperimentMetadata:
    """Create experiment metadata in a pure functional way."""
    return ExperimentMetadata(
        experiment_id=experiment_id,
        description=description,
        git_commit=get_git_commit_hash(),
        git_diff=get_git_diff(),
        timestamp=datetime.now().isoformat(),
        environment_info=get_environment_info()
    )

def create_experiment_config(
    experiment_id: str,
    description: str,
    config: TrainingConfig
) -> ExperimentConfig:
    """Create experiment configuration in a pure functional way."""
    metadata = create_experiment_metadata(experiment_id, description)
    
    return ExperimentConfig(
        experiment_id=experiment_id,
        description=description,
        config=config,
        metadata=metadata
    )

def save_experiment_config(exp_config: ExperimentConfig, output_dir: str) -> Dict[str, str]:
    """Save experiment configuration to files in a pure functional way."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save main config
    config_path = output_path / f"{exp_config.experiment_id}_config.yaml"
    save_config_to_yaml(exp_config.config, str(config_path))
    
    # Save metadata
    metadata_path = output_path / f"{exp_config.experiment_id}_metadata.json"
    metadata_dict = {
        'experiment_id': exp_config.metadata.experiment_id,
        'description': exp_config.metadata.description,
        'git_commit': exp_config.metadata.git_commit,
        'timestamp': exp_config.metadata.timestamp,
        'environment_info': exp_config.metadata.environment_info
    }
    
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise ValueError(f"Failed to save metadata: {e}")
    
    return {
        'config_path': str(config_path),
        'metadata_path': str(metadata_path)
    }

def load_experiment_config(experiment_id: str, config_dir: str) -> ExperimentConfig:
    """Load experiment configuration from files in a pure functional way."""
    config_path = Path(config_dir) / f"{experiment_id}_config.yaml"
    metadata_path = Path(config_dir) / f"{experiment_id}_metadata.json"
    
    # Load config
    config = load_config_from_yaml(str(config_path))
    
    # Load metadata
    metadata_dict = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                metadata_dict = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load metadata: {e}")
    
    # Create metadata object
    metadata = ExperimentMetadata(
        experiment_id=metadata_dict.get('experiment_id', experiment_id),
        description=metadata_dict.get('description', ''),
        git_commit=metadata_dict.get('git_commit', ''),
        git_diff='',  # Not stored in metadata file
        timestamp=metadata_dict.get('timestamp', ''),
        environment_info=metadata_dict.get('environment_info', {})
    )
    
    return ExperimentConfig(
        experiment_id=experiment_id,
        description=metadata.description,
        config=config,
        metadata=metadata
    )

# ============================================================================
# Pure Functions - Configuration Templates
# ============================================================================

def create_config_template() -> Dict[str, Any]:
    """Create a template configuration dictionary in a pure functional way."""
    return {
        'model': {
            'type': 'transformer',
            'name': 'distilbert-base-uncased',
            'training_mode': 'fine_tune'
        },
        'training': {
            'batch_size': 16,
            'learning_rate': 2e-5,
            'num_epochs': 10,
            'warmup_steps': 100,
            'max_grad_norm': 1.0,
            'weight_decay': 0.01,
            'gradient_accumulation_steps': 1,
            'mixed_precision': True,
            'early_stopping_patience': 5
        },
        'data': {
            'dataset_path': 'data/dataset.csv',
            'eval_split': 0.2,
            'test_split': 0.1,
            'cross_validation_folds': 5
        },
        'optimization': {
            'enable_gpu_optimization': True,
            'enable_memory_optimization': True,
            'enable_batch_optimization': True,
            'enable_compilation': False,
            'enable_amp': True
        },
        'logging': {
            'log_to_tensorboard': True,
            'log_to_wandb': False,
            'log_to_mlflow': False
        },
        'output': {
            'output_dir': 'models',
            'save_steps': 500,
            'eval_steps': 500
        }
    }

def create_quick_config_template() -> Dict[str, Any]:
    """Create a quick training configuration template in a pure functional way."""
    return {
        'model_type': 'transformer',
        'training_mode': 'fine_tune',
        'model_name': 'distilbert-base-uncased',
        'dataset_path': 'data/dataset.csv',
        'batch_size': 16,
        'learning_rate': 2e-5,
        'num_epochs': 5,
        'mixed_precision': True,
        'enable_gpu_optimization': True
    }

def create_advanced_config_template() -> Dict[str, Any]:
    """Create an advanced training configuration template in a pure functional way."""
    return {
        'model_type': 'transformer',
        'training_mode': 'fine_tune',
        'model_name': 'bert-base-uncased',
        'dataset_path': 'data/dataset.csv',
        'batch_size': 8,
        'learning_rate': 1e-5,
        'num_epochs': 15,
        'warmup_steps': 200,
        'max_grad_norm': 1.0,
        'weight_decay': 0.01,
        'gradient_accumulation_steps': 4,
        'mixed_precision': True,
        'early_stopping_patience': 7,
        'enable_gpu_optimization': True,
        'enable_memory_optimization': True,
        'enable_batch_optimization': True,
        'enable_compilation': True,
        'enable_amp': True,
        'cross_validation_folds': 5
    }

def save_config_template(
    template: Dict[str, Any],
    template_path: str = "config_template.yaml"
) -> str:
    """Save configuration template to file in a pure functional way."""
    try:
        save_yaml_file(template, template_path)
        return template_path
    except Exception as e:
        raise ValueError(f"Failed to save template: {e}")

# ============================================================================
# Pure Functions - Configuration Validation
# ============================================================================

def validate_config_parameters(config: TrainingConfig) -> ConfigValidationResult:
    """Validate configuration parameters in a pure functional way."""
    errors = []
    warnings = []
    
    # Required parameters
    if config.batch_size <= 0:
        errors.append("batch_size must be positive")
    elif config.batch_size > 128:
        warnings.append("batch_size > 128 may cause memory issues")
    
    if config.learning_rate <= 0:
        errors.append("learning_rate must be positive")
    elif config.learning_rate > 1e-2:
        warnings.append("learning_rate > 1e-2 may cause training instability")
    
    if config.num_epochs <= 0:
        errors.append("num_epochs must be positive")
    elif config.num_epochs > 100:
        warnings.append("num_epochs > 100 may take a very long time")
    
    if config.gradient_accumulation_steps < 1:
        errors.append("gradient_accumulation_steps must be >= 1")
    elif config.gradient_accumulation_steps > 32:
        warnings.append("gradient_accumulation_steps > 32 may cause memory issues")
    
    # Split validation
    if config.eval_split <= 0 or config.eval_split >= 1:
        errors.append("eval_split must be between 0 and 1")
    
    if config.test_split <= 0 or config.test_split >= 1:
        errors.append("test_split must be between 0 and 1")
    
    if config.eval_split + config.test_split >= 1:
        errors.append("eval_split + test_split must be < 1")
    
    # Performance validation
    if config.num_workers > 16:
        warnings.append("num_workers > 16 may cause system overload")
    
    if config.enable_compilation and not hasattr(torch, 'compile'):
        warnings.append("torch.compile not available in this PyTorch version")
    
    return ConfigValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

def validate_config_file(config_path: str) -> ConfigValidationResult:
    """Validate configuration file in a pure functional way."""
    try:
        config = load_config_from_yaml(config_path)
        return validate_config_parameters(config)
    except Exception as e:
        return ConfigValidationResult(
            is_valid=False,
            errors=[f"Failed to load config file: {e}"],
            warnings=[]
        )

# ============================================================================
# Pure Functions - Configuration Utilities
# ============================================================================

def merge_configs(base_config: TrainingConfig, override_config: Dict[str, Any]) -> TrainingConfig:
    """Merge configurations in a pure functional way."""
    base_dict = asdict(base_config)
    merged_dict = {**base_dict, **override_config}
    
    # Convert string enums
    merged_dict = convert_string_enums(merged_dict)
    
    return TrainingConfig(**merged_dict)

def create_config_from_template(
    template_name: str,
    model_name: str,
    dataset_path: str,
    **overrides
) -> TrainingConfig:
    """Create configuration from template in a pure functional way."""
    templates = {
        'quick': create_quick_config_template,
        'advanced': create_advanced_config_template,
        'default': create_config_template
    }
    
    if template_name not in templates:
        raise ValueError(f"Unknown template: {template_name}")
    
    template = templates[template_name]()
    template['model_name'] = model_name
    template['dataset_path'] = dataset_path
    
    # Apply overrides
    template.update(overrides)
    
    # Convert to TrainingConfig
    template = convert_string_enums(template)
    return TrainingConfig(**template)

def quick_config_setup(
    model_name: str,
    dataset_path: str,
    experiment_id: str,
    description: str = ""
) -> ExperimentConfig:
    """Quick configuration setup in a pure functional way."""
    config = create_config_from_template('quick', model_name, dataset_path)
    return create_experiment_config(experiment_id, description, config)

def load_and_validate_config(yaml_path: str) -> Tuple[TrainingConfig, ConfigValidationResult]:
    """Load and validate configuration in a pure functional way."""
    config = load_config_from_yaml(yaml_path)
    validation = validate_config_parameters(config)
    return config, validation

# ============================================================================
# Pure Functions - Configuration Analysis
# ============================================================================

def analyze_config_complexity(config: TrainingConfig) -> Dict[str, Any]:
    """Analyze configuration complexity in a pure functional way."""
    complexity_score = 0
    complexity_factors = []
    
    # Model complexity
    if config.model_type == ModelType.TRANSFORMER:
        if 'bert' in config.model_name.lower():
            complexity_score += 3
            complexity_factors.append("Large transformer model")
        elif 'distilbert' in config.model_name.lower():
            complexity_score += 2
            complexity_factors.append("Distilled transformer model")
        else:
            complexity_score += 1
            complexity_factors.append("Standard transformer model")
    
    # Training complexity
    if config.gradient_accumulation_steps > 1:
        complexity_score += 1
        complexity_factors.append("Gradient accumulation")
    
    if config.mixed_precision:
        complexity_score += 1
        complexity_factors.append("Mixed precision training")
    
    if config.enable_compilation:
        complexity_score += 2
        complexity_factors.append("Model compilation")
    
    if config.num_gpus > 1:
        complexity_score += 2
        complexity_factors.append("Multi-GPU training")
    
    # Data complexity
    if config.cross_validation_folds > 1:
        complexity_score += 1
        complexity_factors.append("Cross-validation")
    
    return {
        'complexity_score': complexity_score,
        'complexity_level': 'low' if complexity_score <= 2 else 'medium' if complexity_score <= 4 else 'high',
        'complexity_factors': complexity_factors,
        'estimated_training_time_hours': complexity_score * 0.5,
        'memory_requirements_gb': complexity_score * 2
    }

def generate_config_hash(config: TrainingConfig) -> str:
    """Generate configuration hash in a pure functional way."""
    config_dict = asdict(config)
    config_str = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()

# ============================================================================
# Demo Functions
# ============================================================================

def demo_config_loader():
    """Demo the functional configuration loader."""
    print("🚀 Functional Configuration Loader Demo")
    print("=" * 50)
    
    # Create quick config
    config = create_config_from_template('quick', 'distilbert-base-uncased', 'data/dataset.csv')
    print(f"Quick config created: {config.model_name}")
    
    # Validate config
    validation = validate_config_parameters(config)
    print(f"Config valid: {validation.is_valid}")
    if validation.errors:
        print(f"Errors: {validation.errors}")
    if validation.warnings:
        print(f"Warnings: {validation.warnings}")
    
    # Analyze complexity
    complexity = analyze_config_complexity(config)
    print(f"Complexity: {complexity['complexity_level']} (score: {complexity['complexity_score']})")
    
    # Generate hash
    config_hash = generate_config_hash(config)
    print(f"Config hash: {config_hash}")
    
    # Create experiment config
    exp_config = create_experiment_config("demo_exp", "Demo experiment", config)
    print(f"Experiment config created: {exp_config.experiment_id}")
    
    # Get environment info
    env_info = get_environment_info()
    print(f"Environment: Python {env_info['python_version']}, PyTorch {env_info['pytorch_version']}")

match __name__:
    case "__main__":
    demo_config_loader() 