from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
import structlog
from copy import deepcopy
import torch
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Configuration Manager for Key Messages ML Pipeline
Handles loading, merging, and validating YAML configuration files
"""

logger = structlog.get_logger(__name__)

@dataclass
class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    message: str
    field: str
    value: Any

class ConfigManager:
    """Manages configuration loading, merging, and validation."""
    
    def __init__(self, config_dir: str = "config", environment: str = None):
        
    """__init__ function."""
self.config_dir = Path(config_dir)
        self.environment = environment or os.getenv("ML_ENVIRONMENT", "development")
        self.config_cache = {}
        
        # Validate config directory exists
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Configuration directory not found: {self.config_dir}")
        
        logger.info("ConfigManager initialized", 
                   config_dir=str(self.config_dir),
                   environment=self.environment)
    
    def load_config(self, config_name: str = "config.yaml", 
                   environment_override: bool = True) -> Dict[str, Any]:
        """
        Load configuration with optional environment overrides.
        
        Args:
            config_name: Name of the main configuration file
            environment_override: Whether to apply environment-specific overrides
            
        Returns:
            Merged configuration dictionary
        """
        # Load main configuration
        main_config_path = self.config_dir / config_name
        if not main_config_path.exists():
            raise FileNotFoundError(f"Main configuration file not found: {main_config_path}")
        
        with open(main_config_path, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            config = yaml.safe_load(f)
        
        logger.info("Main configuration loaded", config_file=str(main_config_path))
        
        # Apply environment-specific overrides
        if environment_override:
            config = self._apply_environment_overrides(config)
        
        # Validate configuration
        self._validate_config(config)
        
        # Cache configuration
        cache_key = f"{config_name}_{self.environment}"
        self.config_cache[cache_key] = deepcopy(config)
        
        return config
    
    def _apply_environment_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment-specific configuration overrides."""
        # Load environment-specific configuration
        env_config_path = self.config_dir / "environments" / f"{self.environment}.yaml"
        
        if env_config_path.exists():
            with open(env_config_path, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                env_config = yaml.safe_load(f)
            
            # Merge environment config with main config
            config = self._deep_merge(config, env_config)
            logger.info("Environment overrides applied", 
                       environment=self.environment,
                       env_config_file=str(env_config_path))
        else:
            logger.warning("Environment configuration file not found", 
                          environment=self.environment,
                          env_config_file=str(env_config_path))
        
        # Apply environment-specific overrides from main config
        if "environments" in config and self.environment in config["environments"]:
            env_overrides = config["environments"][self.environment]
            config = self._deep_merge(config, env_overrides)
            logger.info("Environment overrides from main config applied", 
                       environment=self.environment)
        
        return config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence."""
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration structure and values."""
        try:
            # Validate required sections
            required_sections = ["app", "models", "training", "data", "evaluation"]
            for section in required_sections:
                if section not in config:
                    raise ConfigValidationError(
                        f"Required configuration section '{section}' is missing",
                        section, None
                    )
            
            # Validate app configuration
            self._validate_app_config(config.get("app", {}))
            
            # Validate model configurations
            self._validate_model_configs(config.get("models", {}))
            
            # Validate training configurations
            self._validate_training_configs(config.get("training", {}))
            
            # Validate data configurations
            self._validate_data_configs(config.get("data", {}))
            
            # Validate evaluation configurations
            self._validate_evaluation_configs(config.get("evaluation", {}))
            
            logger.info("Configuration validation passed")
            
        except ConfigValidationError as e:
            logger.error("Configuration validation failed", 
                        field=e.field, value=e.value, message=e.message)
            raise
    
    def _validate_app_config(self, app_config: Dict[str, Any]):
        """Validate application configuration."""
        required_fields = ["name", "version", "environment"]
        for field in required_fields:
            if field not in app_config:
                raise ConfigValidationError(
                    f"Required app field '{field}' is missing",
                    f"app.{field}", None
                )
        
        # Validate environment
        valid_environments = ["development", "staging", "production"]
        if app_config.get("environment") not in valid_environments:
            raise ConfigValidationError(
                f"Invalid environment '{app_config.get('environment')}'. "
                f"Must be one of: {valid_environments}",
                "app.environment", app_config.get("environment")
            )
    
    def _validate_model_configs(self, models_config: Dict[str, Any]):
        """Validate model configurations."""
        if not models_config:
            raise ConfigValidationError(
                "At least one model configuration is required",
                "models", None
            )
        
        for model_name, model_config in models_config.items():
            if not isinstance(model_config, dict):
                raise ConfigValidationError(
                    f"Model configuration '{model_name}' must be a dictionary",
                    f"models.{model_name}", model_config
                )
            
            # Validate required model fields
            required_fields = ["model_name", "max_length", "temperature"]
            for field in required_fields:
                if field not in model_config:
                    raise ConfigValidationError(
                        f"Required model field '{field}' is missing for '{model_name}'",
                        f"models.{model_name}.{field}", None
                    )
            
            # Validate numeric fields
            numeric_fields = ["max_length", "temperature", "top_p"]
            for field in numeric_fields:
                if field in model_config:
                    value = model_config[field]
                    if not isinstance(value, (int, float)):
                        raise ConfigValidationError(
                            f"Model field '{field}' must be numeric for '{model_name}'",
                            f"models.{model_name}.{field}", value
                        )
            
            # Validate device
            if "device" in model_config:
                device = model_config["device"]
                valid_devices = ["auto", "cuda", "cpu"]
                if device not in valid_devices:
                    raise ConfigValidationError(
                        f"Invalid device '{device}' for model '{model_name}'. "
                        f"Must be one of: {valid_devices}",
                        f"models.{model_name}.device", device
                    )
    
    def _validate_training_configs(self, training_config: Dict[str, Any]):
        """Validate training configurations."""
        if not training_config:
            raise ConfigValidationError(
                "At least one training configuration is required",
                "training", None
            )
        
        for config_name, config in training_config.items():
            if not isinstance(config, dict):
                raise ConfigValidationError(
                    f"Training configuration '{config_name}' must be a dictionary",
                    f"training.{config_name}", config
                )
            
            # Validate required fields
            required_fields = ["model_type", "batch_size", "learning_rate", "num_epochs"]
            for field in required_fields:
                if field not in config:
                    raise ConfigValidationError(
                        f"Required training field '{field}' is missing for '{config_name}'",
                        f"training.{config_name}.{field}", None
                    )
            
            # Validate numeric fields
            numeric_fields = ["batch_size", "learning_rate", "num_epochs", "warmup_steps"]
            for field in numeric_fields:
                if field in config:
                    value = config[field]
                    if not isinstance(value, (int, float)):
                        raise ConfigValidationError(
                            f"Training field '{field}' must be numeric for '{config_name}'",
                            f"training.{config_name}.{field}", value
                        )
            
            # Validate boolean fields
            boolean_fields = ["use_mixed_precision", "use_wandb", "use_tensorboard"]
            for field in boolean_fields:
                if field in config:
                    value = config[field]
                    if not isinstance(value, bool):
                        raise ConfigValidationError(
                            f"Training field '{field}' must be boolean for '{config_name}'",
                            f"training.{config_name}.{field}", value
                        )
    
    def _validate_data_configs(self, data_config: Dict[str, Any]):
        """Validate data configurations."""
        if not data_config:
            raise ConfigValidationError(
                "At least one data configuration is required",
                "data", None
            )
        
        for config_name, config in data_config.items():
            if not isinstance(config, dict):
                raise ConfigValidationError(
                    f"Data configuration '{config_name}' must be a dictionary",
                    f"data.{config_name}", config
                )
            
            # Validate numeric fields
            numeric_fields = ["max_length", "batch_size", "num_workers"]
            for field in numeric_fields:
                if field in config:
                    value = config[field]
                    if not isinstance(value, int) or value <= 0:
                        raise ConfigValidationError(
                            f"Data field '{field}' must be a positive integer for '{config_name}'",
                            f"data.{config_name}.{field}", value
                        )
    
    def _validate_evaluation_configs(self, evaluation_config: Dict[str, Any]):
        """Validate evaluation configurations."""
        if not evaluation_config:
            raise ConfigValidationError(
                "At least one evaluation configuration is required",
                "evaluation", None
            )
        
        for config_name, config in evaluation_config.items():
            if not isinstance(config, dict):
                raise ConfigValidationError(
                    f"Evaluation configuration '{config_name}' must be a dictionary",
                    f"evaluation.{config_name}", config
                )
            
            # Validate boolean fields
            boolean_fields = ["save_predictions", "save_metrics", "generate_plots", "generate_report"]
            for field in boolean_fields:
                if field in config:
                    value = config[field]
                    if not isinstance(value, bool):
                        raise ConfigValidationError(
                            f"Evaluation field '{field}' must be boolean for '{config_name}'",
                            f"evaluation.{config_name}.{field}", value
                        )
    
    def get_model_config(self, model_name: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        if config is None:
            config = self.load_config()
        
        models_config = config.get("models", {})
        if model_name not in models_config:
            raise ConfigValidationError(
                f"Model configuration '{model_name}' not found",
                f"models.{model_name}", None
            )
        
        return models_config[model_name]
    
    def get_training_config(self, config_name: str = "default", 
                          config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get configuration for a specific training setup."""
        if config is None:
            config = self.load_config()
        
        training_config = config.get("training", {})
        if config_name not in training_config:
            raise ConfigValidationError(
                f"Training configuration '{config_name}' not found",
                f"training.{config_name}", None
            )
        
        return training_config[config_name]
    
    def get_data_config(self, config_name: str = "default", 
                       config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get configuration for a specific data setup."""
        if config is None:
            config = self.load_config()
        
        data_config = config.get("data", {})
        if config_name not in data_config:
            raise ConfigValidationError(
                f"Data configuration '{config_name}' not found",
                f"data.{config_name}", None
            )
        
        return data_config[config_name]
    
    def get_evaluation_config(self, config_name: str = "default", 
                            config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get configuration for a specific evaluation setup."""
        if config is None:
            config = self.load_config()
        
        evaluation_config = config.get("evaluation", {})
        if config_name not in evaluation_config:
            raise ConfigValidationError(
                f"Evaluation configuration '{config_name}' not found",
                f"evaluation.{config_name}", None
            )
        
        return evaluation_config[config_name]
    
    def resolve_device(self, device_config: str) -> str:
        """Resolve device configuration to actual device."""
        match device_config:
    case "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_config
    
    def resolve_torch_dtype(self, dtype_config: str) -> torch.dtype:
        """Resolve torch dtype configuration to actual dtype."""
        match dtype_config:
    case "auto":
            return torch.float16 if torch.cuda.is_available() else torch.float32
        elif dtype_config == "float16":
            return torch.float16
        elif dtype_config == "float32":
            return torch.float32
        else:
            raise ConfigValidationError(
                f"Invalid torch_dtype '{dtype_config}'. Must be 'auto', 'float16', or 'float32'",
                "torch_dtype", dtype_config
            )
    
    def get_ensemble_config(self, config_name: str = "default", 
                          config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get configuration for model ensemble."""
        if config is None:
            config = self.load_config()
        
        ensemble_config = config.get("ensemble", {})
        if config_name not in ensemble_config:
            raise ConfigValidationError(
                f"Ensemble configuration '{config_name}' not found",
                f"ensemble.{config_name}", None
            )
        
        return ensemble_config[config_name]
    
    def get_experiment_tracking_config(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get experiment tracking configuration."""
        if config is None:
            config = self.load_config()
        
        return config.get("experiment_tracking", {})
    
    def get_performance_config(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get performance optimization configuration."""
        if config is None:
            config = self.load_config()
        
        return config.get("performance", {})
    
    def get_logging_config(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get logging configuration."""
        if config is None:
            config = self.load_config()
        
        return config.get("logging", {})
    
    def get_security_config(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get security configuration."""
        if config is None:
            config = self.load_config()
        
        return config.get("security", {})
    
    def get_deployment_config(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get deployment configuration."""
        if config is None:
            config = self.load_config()
        
        return config.get("deployment", {})
    
    def update_config(self, updates: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Update configuration with new values."""
        if config is None:
            config = self.load_config()
        
        updated_config = self._deep_merge(config, updates)
        
        # Re-validate the updated configuration
        self._validate_config(updated_config)
        
        logger.info("Configuration updated", updates=list(updates.keys()))
        return updated_config
    
    def save_config(self, config: Dict[str, Any], filename: str):
        """Save configuration to a file."""
        output_path = self.config_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(config, f, default_flow_style=False, indent=2, allow_unicode=True)
        
        logger.info("Configuration saved", output_file=str(output_path))
    
    def get_config_summary(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        if config is None:
            config = self.load_config()
        
        summary = {
            "app": {
                "name": config.get("app", {}).get("name"),
                "version": config.get("app", {}).get("version"),
                "environment": config.get("app", {}).get("environment")
            },
            "models": list(config.get("models", {}).keys()),
            "training_configs": list(config.get("training", {}).keys()),
            "data_configs": list(config.get("data", {}).keys()),
            "evaluation_configs": list(config.get("evaluation", {}).keys()),
            "ensemble_configs": list(config.get("ensemble", {}).keys()),
            "experiment_tracking": {
                "tensorboard": config.get("experiment_tracking", {}).get("tensorboard", {}).get("enabled"),
                "wandb": config.get("experiment_tracking", {}).get("wandb", {}).get("enabled"),
                "mlflow": config.get("experiment_tracking", {}).get("mlflow", {}).get("enabled")
            },
            "performance": {
                "device": config.get("performance", {}).get("gpu", {}).get("device"),
                "mixed_precision": config.get("performance", {}).get("gpu", {}).get("mixed_precision")
            }
        }
        
        return summary

# Convenience functions
def load_config(environment: str = None) -> Dict[str, Any]:
    """Load configuration with default settings."""
    config_manager = ConfigManager(environment=environment)
    return config_manager.load_config()

def get_model_config(model_name: str, environment: str = None) -> Dict[str, Any]:
    """Get configuration for a specific model."""
    config_manager = ConfigManager(environment=environment)
    config = config_manager.load_config()
    return config_manager.get_model_config(model_name, config)

def get_training_config(config_name: str = "default", environment: str = None) -> Dict[str, Any]:
    """Get configuration for a specific training setup."""
    config_manager = ConfigManager(environment=environment)
    config = config_manager.load_config()
    return config_manager.get_training_config(config_name, config)

def get_data_config(config_name: str = "default", environment: str = None) -> Dict[str, Any]:
    """Get configuration for a specific data setup."""
    config_manager = ConfigManager(environment=environment)
    config = config_manager.load_config()
    return config_manager.get_data_config(config_name, config)

def get_evaluation_config(config_name: str = "default", environment: str = None) -> Dict[str, Any]:
    """Get configuration for a specific evaluation setup."""
    config_manager = ConfigManager(environment=environment)
    config = config_manager.load_config()
    return config_manager.get_evaluation_config(config_name, config) 