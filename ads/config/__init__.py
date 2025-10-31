"""
Unified Configuration package for the ads feature.

This package consolidates all configuration functionality from the scattered implementations:
- config.py (basic settings)
- optimized_config.py (production settings)
- config_manager.py (advanced YAML configuration management)

The new structure follows Clean Architecture principles with clear separation of concerns.
"""

from .settings import Settings, OptimizedSettings, get_settings, get_optimized_settings
from .manager import ConfigManager, ConfigType
from .models import (
    ModelConfig, TrainingConfig, DataConfig, ExperimentConfig, 
    OptimizationConfig, DeploymentConfig, ProjectConfig
)
from .providers import get_llm_config, get_embeddings_config, get_redis_config, get_database_config

__all__ = [
    # Settings
    "Settings",
    "OptimizedSettings", 
    "get_settings",
    "get_optimized_settings",
    
    # Configuration Manager
    "ConfigManager",
    "ConfigType",
    
    # Configuration Models
    "ModelConfig",
    "TrainingConfig", 
    "DataConfig",
    "ExperimentConfig",
    "OptimizationConfig",
    "DeploymentConfig",
    "ProjectConfig",
    
    # Provider Configurations
    "get_llm_config",
    "get_embeddings_config", 
    "get_redis_config",
    "get_database_config"
] 