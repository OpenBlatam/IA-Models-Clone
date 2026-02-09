"""
Configuration Loader
====================

Advanced configuration loading and management utilities.
"""

import json
import os
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

from ..config.enhancer_config import EnhancerConfig

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Advanced configuration loader with multiple sources support.
    
    Supports:
    - Environment variables
    - JSON files
    - YAML files (if pyyaml installed)
    - Default values
    """
    
    @staticmethod
    def load_from_file(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix == ".json":
            return ConfigLoader._load_json(path)
        elif suffix in [".yaml", ".yml"]:
            return ConfigLoader._load_yaml(path)
        else:
            raise ValueError(f"Unsupported configuration file format: {suffix}")
    
    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        """Load JSON configuration file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML is required for YAML configuration files")
    
    @staticmethod
    def load_from_env(prefix: str = "ENHANCER_") -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Args:
            prefix: Environment variable prefix
            
        Returns:
            Configuration dictionary
        """
        config = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to nested dict
                config_key = key[len(prefix):].lower()
                
                # Handle nested keys (e.g., OPENROUTER_API_KEY -> openrouter.api_key)
                keys = config_key.split("_")
                current = config
                
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                
                current[keys[-1]] = value
        
        return config
    
    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries.
        
        Later configs override earlier ones.
        
        Args:
            *configs: Configuration dictionaries to merge
            
        Returns:
            Merged configuration dictionary
        """
        merged = {}
        
        for config in configs:
            merged = ConfigLoader._deep_merge(merged, config)
        
        return merged
    
    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    @staticmethod
    def create_config(
        file_path: Optional[Union[str, Path]] = None,
        env_prefix: str = "ENHANCER_",
        use_env: bool = True,
        defaults: Optional[Dict[str, Any]] = None
    ) -> EnhancerConfig:
        """
        Create EnhancerConfig from multiple sources.
        
        Args:
            file_path: Optional path to configuration file
            env_prefix: Environment variable prefix
            use_env: Whether to load from environment
            defaults: Default configuration values
            
        Returns:
            Configured EnhancerConfig instance
        """
        configs = []
        
        # Add defaults first
        if defaults:
            configs.append(defaults)
        
        # Load from file if provided
        if file_path:
            try:
                file_config = ConfigLoader.load_from_file(file_path)
                configs.append(file_config)
            except FileNotFoundError:
                logger.warning(f"Configuration file not found: {file_path}")
        
        # Load from environment
        if use_env:
            env_config = ConfigLoader.load_from_env(env_prefix)
            if env_config:
                configs.append(env_config)
        
        # Merge all configs
        merged_config = ConfigLoader.merge_configs(*configs) if configs else {}
        
        # Create and return config
        return EnhancerConfig.from_dict(merged_config) if merged_config else EnhancerConfig()




