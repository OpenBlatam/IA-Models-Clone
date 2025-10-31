from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
    from config import ConfigManager, quick_load_config, create_custom_config
from .config_manager import (
from .config_loader import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Configuration Management Module
==============================

This module provides comprehensive configuration management for the AI video generation system.

Features:
- YAML-based configuration files
- Configuration validation and type checking
- Environment-specific configurations
- Configuration inheritance and merging
- Default configurations and overrides
- Configuration templates and presets
- Command-line interface for configuration management

Usage:
    
    # Load default configuration
    config = quick_load_config("diffusion_default")
    
    # Create custom configuration
    custom_config = create_custom_config(
        "diffusion_default",
        {"model.frame_size": [512, 512], "training.num_epochs": 50},
        "my_experiment"
    )
    
    # Use configuration manager
    manager = ConfigManager()
    config = manager.create_config("my_experiment", "diffusion")
    manager.save_config(config, "my_config.yaml")
"""

    BaseConfig,
    ModelConfig,
    DataConfig,
    TrainingConfig,
    EvaluationConfig,
    SystemConfig,
    CompleteConfig,
    ConfigManager,
    create_config_manager,
    load_config_from_file,
    save_config_to_file,
    create_default_config
)

    ConfigLoader,
    CommandLineConfigLoader,
    get_env_overrides,
    load_config_with_env,
    quick_load_config,
    create_custom_config
)

# Convenience imports
__all__ = [
    # Configuration classes
    "BaseConfig",
    "ModelConfig", 
    "DataConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "SystemConfig",
    "CompleteConfig",
    
    # Managers and loaders
    "ConfigManager",
    "ConfigLoader",
    "CommandLineConfigLoader",
    
    # Utility functions
    "create_config_manager",
    "load_config_from_file",
    "save_config_to_file",
    "create_default_config",
    "get_env_overrides",
    "load_config_with_env",
    "quick_load_config",
    "create_custom_config"
]

# Version information
__version__ = "1.0.0"
__author__ = "AI Video Team"
__description__ = "Configuration management for AI video generation system"

# Quick start examples
def get_example_configs():
    """Get example configurations for different use cases."""
    return {
        "diffusion": "diffusion_default",
        "gan": "gan_default", 
        "transformer": "transformer_default",
        "high_res": "high_res_config",
        "fast_training": "fast_training_config"
    }

def create_experiment_config(
    experiment_name: str,
    model_type: str = "diffusion",
    overrides: dict = None
) -> CompleteConfig:
    """
    Create a configuration for a new experiment.
    
    Args:
        experiment_name: Name of the experiment
        model_type: Type of model to use
        overrides: Dictionary of configuration overrides
    
    Returns:
        CompleteConfig: Configuration for the experiment
    """
    manager = ConfigManager()
    config = manager.create_config(experiment_name, model_type)
    
    if overrides:
        config = manager.merge_configs(config, overrides)
    
    return config

def validate_and_save_config(config: CompleteConfig, filepath: str) -> bool:
    """
    Validate and save a configuration.
    
    Args:
        config: Configuration to validate and save
        filepath: Path to save the configuration
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not config.validate():
            errors = config.get_validation_errors()
            print(f"Configuration validation failed:\n" + "\n".join(errors))
            return False
        
        config.save(filepath)
        print(f"Configuration saved to {filepath}")
        return True
    
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False

# Example usage
if __name__ == "__main__":
    print("🔧 Configuration Management System")
    print("=" * 40)
    
    # Example 1: Load default configuration
    print("\n1. Loading default diffusion configuration:")
    config = quick_load_config("diffusion_default")
    print(f"   Model: {config.model.model_name}")
    print(f"   Frame size: {config.model.frame_size}")
    print(f"   Training epochs: {config.training.num_epochs}")
    
    # Example 2: Create custom configuration
    print("\n2. Creating custom configuration:")
    custom_config = create_custom_config(
        "diffusion_default",
        {
            "model.frame_size": [512, 512],
            "training.num_epochs": 50,
            "training.learning_rate": 5e-5,
            "system.environment": "production"
        },
        "high_res_experiment"
    )
    print(f"   Custom config: {custom_config.name}")
    print(f"   Frame size: {custom_config.model.frame_size}")
    print(f"   Environment: {custom_config.system.environment}")
    
    # Example 3: Use configuration manager
    print("\n3. Using configuration manager:")
    manager = ConfigManager()
    
    # List available configs
    configs = manager.list_configs()
    print(f"   Available configs: {len(configs)} files")
    
    # Create and save new config
    new_config = manager.create_config("test_experiment", "gan")
    new_config.training.batch_size = 16
    filepath = manager.save_config(new_config)
    print(f"   Saved new config to: {filepath}")
    
    # Example 4: Environment overrides
    print("\n4. Environment variable overrides:")
    print("   Set AI_VIDEO_MODEL_FRAME_SIZE=512,512")
    print("   Set AI_VIDEO_TRAINING_BATCH_SIZE=16")
    print("   Then use: load_config_with_env('default_configs.yaml', 'diffusion_default')")
    
    print("\n✅ Configuration system ready!")
    print("\n📚 Available configurations:")
    example_configs = get_example_configs()
    for name, config_name in example_configs.items():
        print(f"   - {name}: {config_name}") 