from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import structlog
from configuration_management import (
        from configuration_management import ConfigurationTemplates
        from configuration_management import ConfigurationTemplates
        from configuration_management import ConfigurationTemplates
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Configuration Loader Utility
============================

This module provides utilities for loading and managing YAML configuration files
in the modular architecture system. It integrates with the configuration management
system and provides easy access to experiment configurations.
"""

    ExperimentConfig, ModelConfig, DataConfig, TrainingConfig, EvaluationConfig,
    ConfigurationManager, EnvironmentConfigManager, ConfigurationValidator
)

logger = structlog.get_logger()


class ConfigLoader:
    """Utility class for loading and managing configuration files."""
    
    def __init__(self, config_dir: str = "configs"):
        
    """__init__ function."""
self.config_dir = Path(config_dir)
        self.config_manager = ConfigurationManager(config_dir)
        self.env_manager = EnvironmentConfigManager(config_dir)
        self.validator = ConfigurationValidator()
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def load_experiment_config(self, config_name: str, environment: str = "development") -> ExperimentConfig:
        """
        Load experiment configuration from YAML file.
        
        Args:
            config_name: Name of the configuration file (without .yaml extension)
            environment: Environment to load (development, staging, production)
            
        Returns:
            Loaded experiment configuration
        """
        # Try to load environment-specific config first
        env_config_path = self.config_dir / f"{config_name}_{environment}.yaml"
        
        if env_config_path.exists():
            logger.info("Loading environment-specific configuration", 
                       config=config_name, environment=environment)
            config = ExperimentConfig.load(str(env_config_path))
        else:
            # Load base configuration
            base_config_path = self.config_dir / f"{config_name}.yaml"
            
            if not base_config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {base_config_path}")
            
            logger.info("Loading base configuration", config=config_name)
            config = ExperimentConfig.load(str(base_config_path))
            config.environment = environment
        
        # Validate configuration
        self._validate_config(config)
        
        return config
    
    def load_config_from_dict(self, config_dict: Dict[str, Any]) -> ExperimentConfig:
        """
        Load experiment configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Loaded experiment configuration
        """
        try:
            config = ExperimentConfig(**config_dict)
            self._validate_config(config)
            return config
        except Exception as e:
            logger.error("Failed to load configuration from dictionary", error=str(e))
            raise
    
    def load_config_from_yaml_string(self, yaml_string: str) -> ExperimentConfig:
        """
        Load experiment configuration from YAML string.
        
        Args:
            yaml_string: YAML configuration string
            
        Returns:
            Loaded experiment configuration
        """
        try:
            config_dict = yaml.safe_load(yaml_string)
            return self.load_config_from_dict(config_dict)
        except Exception as e:
            logger.error("Failed to load configuration from YAML string", error=str(e))
            raise
    
    def save_config(self, config: ExperimentConfig, filename: Optional[str] = None) -> Path:
        """
        Save experiment configuration to YAML file.
        
        Args:
            config: Experiment configuration to save
            filename: Optional filename (defaults to experiment_name.yaml)
            
        Returns:
            Path to saved configuration file
        """
        if filename is None:
            filename = f"{config.experiment_name}.yaml"
        
        filepath = self.config_dir / filename
        config.save(str(filepath))
        
        logger.info("Configuration saved", filepath=str(filepath))
        return filepath
    
    def list_available_configs(self) -> List[str]:
        """
        List all available configuration files.
        
        Returns:
            List of configuration file names
        """
        config_files = []
        
        for file_path in self.config_dir.glob("*.yaml"):
            # Remove .yaml extension and environment suffix
            name = file_path.stem
            if "_" in name and name.split("_")[-1] in ["development", "staging", "production"]:
                name = "_".join(name.split("_")[:-1])
            config_files.append(name)
        
        # Remove duplicates and sort
        config_files = sorted(list(set(config_files)))
        
        logger.info("Available configurations", configs=config_files)
        return config_files
    
    def create_environment_config(self, base_config_name: str, environment: str, 
                                overrides: Dict[str, Any]) -> ExperimentConfig:
        """
        Create environment-specific configuration with overrides.
        
        Args:
            base_config_name: Name of base configuration
            environment: Environment name
            overrides: Configuration overrides
            
        Returns:
            Environment-specific configuration
        """
        # Load base configuration
        base_config = self.load_experiment_config(base_config_name, "development")
        
        # Create environment-specific configuration
        env_config = self.env_manager.create_environment_config(
            base_config, environment, overrides
        )
        
        logger.info("Environment configuration created", 
                   base=base_config_name, environment=environment)
        
        return env_config
    
    def validate_config_file(self, config_name: str) -> bool:
        """
        Validate a configuration file.
        
        Args:
            config_name: Name of the configuration file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            config = self.load_experiment_config(config_name)
            return self._validate_config(config)
        except Exception as e:
            logger.error("Configuration validation failed", config=config_name, error=str(e))
            return False
    
    def _validate_config(self, config: ExperimentConfig) -> bool:
        """
        Validate experiment configuration.
        
        Args:
            config: Experiment configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Validate each component
        model_issues = self.validator.validate_model_config(config.model)
        training_issues = self.validator.validate_training_config(config.training)
        data_issues = self.validator.validate_data_config(config.data)
        
        all_issues = model_issues + training_issues + data_issues
        
        if all_issues:
            logger.warning("Configuration validation issues found", issues=all_issues)
            return False
        
        logger.info("Configuration validation passed", experiment=config.experiment_name)
        return True
    
    def get_config_summary(self, config: ExperimentConfig) -> Dict[str, Any]:
        """
        Get a summary of the configuration.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Configuration summary dictionary
        """
        summary = {
            "experiment_name": config.experiment_name,
            "experiment_description": config.experiment_description,
            "version": config.version,
            "environment": config.environment,
            "tags": config.tags,
            "model": {
                "type": config.model.model_type,
                "name": config.model.model_name,
                "num_classes": config.model.num_classes,
                "hidden_dim": config.model.hidden_dim
            },
            "data": {
                "data_path": config.data.data_path,
                "batch_size": config.data.batch_size,
                "target_column": config.data.target_column
            },
            "training": {
                "epochs": config.training.epochs,
                "learning_rate": config.training.learning_rate,
                "optimizer": config.training.optimizer,
                "scheduler": config.training.scheduler,
                "device": config.training.device
            },
            "evaluation": {
                "metrics": config.evaluation.metrics,
                "save_predictions": config.evaluation.save_predictions,
                "save_plots": config.evaluation.save_plots
            }
        }
        
        return summary
    
    def print_config_summary(self, config: ExperimentConfig):
        """
        Print a formatted summary of the configuration.
        
        Args:
            config: Experiment configuration
        """
        summary = self.get_config_summary(config)
        
        print("\n" + "="*60)
        print(f"EXPERIMENT CONFIGURATION SUMMARY")
        print("="*60)
        print(f"Name: {summary['experiment_name']}")
        print(f"Description: {summary['experiment_description']}")
        print(f"Version: {summary['version']}")
        print(f"Environment: {summary['environment']}")
        print(f"Tags: {', '.join(summary['tags'])}")
        
        print(f"\nMODEL:")
        print(f"  Type: {summary['model']['type']}")
        print(f"  Name: {summary['model']['name']}")
        print(f"  Classes: {summary['model']['num_classes']}")
        print(f"  Hidden Dim: {summary['model']['hidden_dim']}")
        
        print(f"\nDATA:")
        print(f"  Path: {summary['data']['data_path']}")
        print(f"  Batch Size: {summary['data']['batch_size']}")
        print(f"  Target Column: {summary['data']['target_column']}")
        
        print(f"\nTRAINING:")
        print(f"  Epochs: {summary['training']['epochs']}")
        print(f"  Learning Rate: {summary['training']['learning_rate']}")
        print(f"  Optimizer: {summary['training']['optimizer']}")
        print(f"  Scheduler: {summary['training']['scheduler']}")
        print(f"  Device: {summary['training']['device']}")
        
        print(f"\nEVALUATION:")
        print(f"  Metrics: {', '.join(summary['evaluation']['metrics'])}")
        print(f"  Save Predictions: {summary['evaluation']['save_predictions']}")
        print(f"  Save Plots: {summary['evaluation']['save_plots']}")
        print("="*60 + "\n")


class ConfigTemplateGenerator:
    """Generator for configuration templates."""
    
    def __init__(self, config_dir: str = "configs"):
        
    """__init__ function."""
self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.logger = structlog.get_logger(__name__)
    
    def generate_transformer_template(self, experiment_name: str, **kwargs) -> ExperimentConfig:
        """Generate transformer classification template."""
        
        config = ConfigurationTemplates.get_transformer_classification_template()
        config.experiment_name = experiment_name
        
        # Apply custom parameters
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif hasattr(config.model, key):
                setattr(config.model, key, value)
            elif hasattr(config.training, key):
                setattr(config.training, key, value)
            elif hasattr(config.data, key):
                setattr(config.data, key, value)
        
        # Save template
        filepath = self.config_dir / f"{experiment_name}.yaml"
        config.save(str(filepath))
        
        self.logger.info("Transformer template generated", 
                        experiment=experiment_name, filepath=str(filepath))
        
        return config
    
    def generate_cnn_template(self, experiment_name: str, **kwargs) -> ExperimentConfig:
        """Generate CNN image classification template."""
        
        config = ConfigurationTemplates.get_cnn_image_classification_template()
        config.experiment_name = experiment_name
        
        # Apply custom parameters
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif hasattr(config.model, key):
                setattr(config.model, key, value)
            elif hasattr(config.training, key):
                setattr(config.training, key, value)
            elif hasattr(config.data, key):
                setattr(config.data, key, value)
        
        # Save template
        filepath = self.config_dir / f"{experiment_name}.yaml"
        config.save(str(filepath))
        
        self.logger.info("CNN template generated", 
                        experiment=experiment_name, filepath=str(filepath))
        
        return config
    
    def generate_diffusion_template(self, experiment_name: str, **kwargs) -> ExperimentConfig:
        """Generate diffusion model template."""
        
        config = ConfigurationTemplates.get_diffusion_generation_template()
        config.experiment_name = experiment_name
        
        # Apply custom parameters
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif hasattr(config.model, key):
                setattr(config.model, key, value)
            elif hasattr(config.training, key):
                setattr(config.training, key, value)
            elif hasattr(config.data, key):
                setattr(config.data, key, value)
        
        # Save template
        filepath = self.config_dir / f"{experiment_name}.yaml"
        config.save(str(filepath))
        
        self.logger.info("Diffusion template generated", 
                        experiment=experiment_name, filepath=str(filepath))
        
        return config


def load_config(config_name: str, environment: str = "development") -> ExperimentConfig:
    """
    Convenience function to load configuration.
    
    Args:
        config_name: Name of the configuration file
        environment: Environment to load
        
    Returns:
        Loaded experiment configuration
    """
    loader = ConfigLoader()
    return loader.load_experiment_config(config_name, environment)


def save_config(config: ExperimentConfig, filename: Optional[str] = None) -> Path:
    """
    Convenience function to save configuration.
    
    Args:
        config: Experiment configuration to save
        filename: Optional filename
        
    Returns:
        Path to saved configuration file
    """
    loader = ConfigLoader()
    return loader.save_config(config, filename)


def validate_config(config_name: str) -> bool:
    """
    Convenience function to validate configuration.
    
    Args:
        config_name: Name of the configuration file
        
    Returns:
        True if valid, False otherwise
    """
    loader = ConfigLoader()
    return loader.validate_config_file(config_name)


def main():
    """Example usage of the configuration loader."""
    
    # Create configuration loader
    loader = ConfigLoader()
    
    # List available configurations
    configs = loader.list_available_configs()
    print(f"Available configurations: {configs}")
    
    # Load a configuration
    try:
        config = loader.load_experiment_config("transformer_classification")
        loader.print_config_summary(config)
        
        # Validate configuration
        is_valid = loader.validate_config_file("transformer_classification")
        print(f"Configuration is valid: {is_valid}")
        
        # Create environment-specific configuration
        prod_config = loader.create_environment_config(
            "transformer_classification",
            "production",
            {
                "training.epochs": 50,
                "training.learning_rate": 1e-5,
                "data.batch_size": 8
            }
        )
        
        print(f"Production configuration created: {prod_config.experiment_name}")
        
    except FileNotFoundError as e:
        print(f"Configuration file not found: {e}")
        print("Creating example configurations...")
        
        # Create example configurations
        generator = ConfigTemplateGenerator()
        
        # Generate templates
        generator.generate_transformer_template(
            "transformer_classification",
            experiment_description="BERT-based text classification example"
        )
        
        generator.generate_cnn_template(
            "cnn_image_classification",
            experiment_description="ResNet-based image classification example"
        )
        
        generator.generate_diffusion_template(
            "diffusion_generation",
            experiment_description="Stable Diffusion example"
        )
        
        print("Example configurations created successfully!")


match __name__:
    case "__main__":
    main() 