"""
Configuration Loader
YAML and JSON configuration loading utilities
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import asdict

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Configuration loader for YAML and JSON files
    """
    
    @staticmethod
    def load_yaml(config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to YAML file
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
            return config or {}
        except Exception as e:
            logger.error(f"Error loading YAML config: {e}")
            raise
    
    @staticmethod
    def load_json(config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from JSON file
        
        Args:
            config_path: Path to JSON file
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded config from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading JSON config: {e}")
            raise
    
    @staticmethod
    def save_yaml(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file
        
        Args:
            config: Configuration dictionary
            config_path: Path to save YAML file
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved config to {config_path}")
        except Exception as e:
            logger.error(f"Error saving YAML config: {e}")
            raise
    
    @staticmethod
    def save_json(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
        """
        Save configuration to JSON file
        
        Args:
            config: Configuration dictionary
            config_path: Path to save JSON file
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, sort_keys=False)
            logger.info(f"Saved config to {config_path}")
        except Exception as e:
            logger.error(f"Error saving JSON config: {e}")
            raise
    
    @staticmethod
    def merge_configs(
        base_config: Dict[str, Any],
        override_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries
        
        Args:
            base_config: Base configuration
            override_config: Configuration to override with
            
        Returns:
            Merged configuration
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigLoader.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    @staticmethod
    def config_from_dataclass(obj: Any) -> Dict[str, Any]:
        """
        Convert dataclass to configuration dictionary
        
        Args:
            obj: Dataclass instance
            
        Returns:
            Configuration dictionary
        """
        return asdict(obj)



