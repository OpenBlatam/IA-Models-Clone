"""
Configuration Manager for the ads feature.

This module consolidates advanced configuration management functionality from config_manager.py,
providing YAML-based configuration management with validation and persistence.
"""

import yaml
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime
import copy
from contextlib import contextmanager

from .models import (
    ConfigType, ModelConfig, TrainingConfig, DataConfig, 
    ExperimentConfig, OptimizationConfig, DeploymentConfig, ProjectConfig
)


logger = logging.getLogger(__name__)


class ConfigManager:
    """Advanced configuration manager for YAML-based configuration files."""
    
    def __init__(self, config_dir: str = "./configs"):
        """Initialize the configuration manager.
        
        Args:
            config_dir: Directory to store configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded configurations
        self._config_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, float] = {}
        
        # Default configurations
        self._default_configs = {
            ConfigType.MODEL: self._get_default_model_config,
            ConfigType.TRAINING: self._get_default_training_config,
            ConfigType.DATA: self._get_default_data_config,
            ConfigType.EXPERIMENT: self._get_default_experiment_config,
            ConfigType.OPTIMIZATION: self._get_default_optimization_config,
            ConfigType.DEPLOYMENT: self._get_default_deployment_config,
            ConfigType.PROJECT: self._get_default_project_config,
        }
    
    def create_default_configs(self, project_name: str) -> Dict[str, str]:
        """Create default configuration files for a project.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Dictionary mapping config types to file paths
        """
        project_dir = self.config_dir / project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        
        created_files = {}
        
        for config_type in ConfigType:
            config_func = self._default_configs.get(config_type)
            if config_func:
                config = config_func()
                config.name = project_name
                
                filename = f"{project_name}_{config_type.value}_config.yaml"
                filepath = project_dir / filename
                
                self.save_config(config, filepath, config_type)
                created_files[config_type.value] = str(filepath)
                
                logger.info(f"Created {config_type.value} config: {filepath}")
        
        return created_files
    
    def load_all_configs(self, project_name: str) -> Dict[str, Any]:
        """Load all configuration files for a project.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Dictionary mapping config types to configuration objects
        """
        project_dir = self.config_dir / project_name
        
        if not project_dir.exists():
            raise FileNotFoundError(f"Project directory not found: {project_dir}")
        
        configs = {}
        
        for config_type in ConfigType:
            filename = f"{project_name}_{config_type.value}_config.yaml"
            filepath = project_dir / filename
            
            if filepath.exists():
                try:
                    config = self.load_config(filepath, config_type)
                    configs[config_type.value] = config
                except Exception as e:
                    logger.warning(f"Failed to load {config_type.value} config: {e}")
                    # Load default config as fallback
                    config = self._default_configs[config_type]()
                    config.name = project_name
                    configs[config_type.value] = config
            else:
                # Create default config if file doesn't exist
                config = self._default_configs[config_type]()
                config.name = project_name
                configs[config_type.value] = config
        
        return configs
    
    def load_config(self, filepath: Union[str, Path], config_type: ConfigType) -> Any:
        """Load a configuration from a file.
        
        Args:
            filepath: Path to the configuration file
            filepath: Type of configuration to load
            
        Returns:
            Configuration object
        """
        filepath = Path(filepath)
        
        # Check cache
        cache_key = f"{filepath}_{config_type.value}"
        if cache_key in self._config_cache:
            if filepath.stat().st_mtime <= self._cache_timestamps[cache_key]:
                return self._config_cache[cache_key]
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Convert to appropriate config object
            config = self._dict_to_config(data, config_type)
            
            # Update cache
            self._config_cache[cache_key] = config
            self._cache_timestamps[cache_key] = filepath.stat().st_mtime
            
            logger.info(f"Loaded {config_type.value} config from {filepath}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config from {filepath}: {e}")
            raise
    
    def save_config(self, config: Any, filepath: Union[str, Path], config_type: ConfigType) -> bool:
        """Save a configuration to a file.
        
        Args:
            config: Configuration object to save
            filepath: Path where to save the configuration
            config_type: Type of configuration being saved
            
        Returns:
            True if successful, False otherwise
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Convert config to dictionary
            data = self._config_to_dict(config)
            
            # Add metadata
            data['_metadata'] = {
                'config_type': config_type.value,
                'saved_at': datetime.now().isoformat(),
                'version': '1.0.0'
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            
            # Update cache
            cache_key = f"{filepath}_{config_type.value}"
            self._config_cache[cache_key] = config
            self._cache_timestamps[cache_key] = filepath.stat().st_mtime
            
            logger.info(f"Saved {config_type.value} config to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config to {filepath}: {e}")
            return False
    
    def update_config(self, project_name: str, config_type: ConfigType, updates: Dict[str, Any]) -> bool:
        """Update a configuration with new values.
        
        Args:
            project_name: Name of the project
            config_type: Type of configuration to update
            updates: Dictionary of updates to apply
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load current config
            configs = self.load_all_configs(project_name)
            config = configs.get(config_type.value)
            
            if not config:
                logger.error(f"Config type {config_type.value} not found for project {project_name}")
                return False
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning(f"Unknown config key: {key}")
            
            # Save updated config
            filename = f"{project_name}_{config_type.value}_config.yaml"
            filepath = self.config_dir / project_name / filename
            
            return self.save_config(config, filepath, config_type)
            
        except Exception as e:
            logger.error(f"Failed to update config: {e}")
            return False
    
    def validate_config(self, config: Any, config_type: ConfigType) -> Dict[str, Any]:
        """Validate a configuration object.
        
        Args:
            config: Configuration object to validate
            config_type: Type of configuration
            
        Returns:
            Validation result dictionary
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'config_type': config_type.value
        }
        
        try:
            # Basic validation
            if not hasattr(config, 'name'):
                validation_result['errors'].append("Configuration must have a 'name' attribute")
                validation_result['is_valid'] = False
            
            # Type-specific validation
            if config_type == ConfigType.MODEL:
                self._validate_model_config(config, validation_result)
            elif config_type == ConfigType.TRAINING:
                self._validate_training_config(config, validation_result)
            elif config_type == ConfigType.DATA:
                self._validate_data_config(config, validation_result)
            
        except Exception as e:
            validation_result['errors'].append(f"Validation error: {e}")
            validation_result['is_valid'] = False
        
        return validation_result
    
    def get_config_info(self, project_name: str) -> Dict[str, Any]:
        """Get information about all configurations for a project.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Dictionary with configuration information
        """
        project_dir = self.config_dir / project_name
        
        if not project_dir.exists():
            return {}
        
        config_info = {
            'project_name': project_name,
            'config_dir': str(project_dir),
            'configs': {},
            'last_updated': None
        }
        
        for config_type in ConfigType:
            filename = f"{project_name}_{config_type.value}_config.yaml"
            filepath = project_dir / filename
            
            if filepath.exists():
                stat = filepath.stat()
                config_info['configs'][config_type.value] = {
                    'filepath': str(filepath),
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
                
                if not config_info['last_updated'] or stat.st_mtime > Path(config_info['last_updated']).stat().st_mtime:
                    config_info['last_updated'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        
        return config_info
    
    def cleanup_project(self, project_name: str) -> bool:
        """Clean up all configuration files for a project.
        
        Args:
            project_name: Name of the project
            
        Returns:
            True if successful, False otherwise
        """
        try:
            project_dir = self.config_dir / project_name
            
            if project_dir.exists():
                import shutil
                shutil.rmtree(project_dir)
                
                # Clear cache
                cache_keys = [k for k in self._config_cache.keys() if project_name in k]
                for key in cache_keys:
                    del self._config_cache[key]
                    del self._cache_timestamps[key]
                
                logger.info(f"Cleaned up project: {project_name}")
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup project {project_name}: {e}")
            return False
    
    def _dict_to_config(self, data: Dict[str, Any], config_type: ConfigType) -> Any:
        """Convert dictionary to configuration object."""
        if config_type == ConfigType.MODEL:
            return ModelConfig(**data)
        elif config_type == ConfigType.TRAINING:
            return TrainingConfig(**data)
        elif config_type == ConfigType.DATA:
            return DataConfig(**data)
        elif config_type == ConfigType.EXPERIMENT:
            return ExperimentConfig(**data)
        elif config_type == ConfigType.OPTIMIZATION:
            return OptimizationConfig(**data)
        elif config_type == ConfigType.DEPLOYMENT:
            return DeploymentConfig(**data)
        elif config_type == ConfigType.PROJECT:
            return ProjectConfig(**data)
        else:
            raise ValueError(f"Unknown config type: {config_type}")
    
    def _config_to_dict(self, config: Any) -> Dict[str, Any]:
        """Convert configuration object to dictionary."""
        if hasattr(config, '__dict__'):
            return {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
        elif hasattr(config, '__dataclass_fields__'):
            return {k: getattr(config, k) for k in config.__dataclass_fields__}
        else:
            return dict(config)
    
    def _validate_model_config(self, config: ModelConfig, result: Dict[str, Any]) -> None:
        """Validate model configuration."""
        if config.input_size <= 0:
            result['errors'].append("Input size must be positive")
            result['is_valid'] = False
        
        if config.output_size <= 0:
            result['errors'].append("Output size must be positive")
            result['is_valid'] = False
        
        if config.dropout_rate < 0 or config.dropout_rate > 1:
            result['warnings'].append("Dropout rate should be between 0 and 1")
    
    def _validate_training_config(self, config: TrainingConfig, result: Dict[str, Any]) -> None:
        """Validate training configuration."""
        if config.batch_size <= 0:
            result['errors'].append("Batch size must be positive")
            result['is_valid'] = False
        
        if config.learning_rate <= 0:
            result['errors'].append("Learning rate must be positive")
            result['is_valid'] = False
        
        if config.epochs <= 0:
            result['errors'].append("Number of epochs must be positive")
            result['is_valid'] = False
    
    def _validate_data_config(self, config: DataConfig, result: Dict[str, Any]) -> None:
        """Validate data configuration."""
        if config.batch_size <= 0:
            result['errors'].append("Batch size must be positive")
            result['is_valid'] = False
        
        if config.num_workers < 0:
            result['warnings'].append("Number of workers should be non-negative")
    
    def _get_default_model_config(self) -> ModelConfig:
        """Get default model configuration."""
        return ModelConfig(
            name="default_model",
            type="neural_network",
            architecture="mlp",
            input_size=100,
            output_size=10,
            hidden_sizes=[64, 32]
        )
    
    def _get_default_training_config(self) -> TrainingConfig:
        """Get default training configuration."""
        return TrainingConfig()
    
    def _get_default_data_config(self) -> DataConfig:
        """Get default data configuration."""
        return DataConfig()
    
    def _get_default_experiment_config(self) -> ExperimentConfig:
        """Get default experiment configuration."""
        return ExperimentConfig(
            experiment_name="default_experiment",
            project_name="default_project"
        )
    
    def _get_default_optimization_config(self) -> OptimizationConfig:
        """Get default optimization configuration."""
        return OptimizationConfig()
    
    def _get_default_deployment_config(self) -> DeploymentConfig:
        """Get default deployment configuration."""
        return DeploymentConfig()
    
    def _get_default_project_config(self) -> ProjectConfig:
        """Get default project configuration."""
        return ProjectConfig(
            name="default_project",
            description="Default project configuration"
        )
