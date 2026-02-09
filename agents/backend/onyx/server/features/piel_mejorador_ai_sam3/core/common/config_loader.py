"""
Unified Config Loader for Piel Mejorador AI SAM3
================================================

Consolidated configuration loading utilities.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Type, TypeVar, Union
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConfigLoader:
    """Unified configuration loader."""
    
    @staticmethod
    def load_from_file(
        file_path: Union[str, Path],
        required: bool = False
    ) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            file_path: Path to config file
            required: Whether file is required
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If file is required but not found
        """
        path = Path(file_path)
        
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Required config file not found: {file_path}")
            logger.warning(f"Config file not found: {file_path}, using empty config")
            return {}
        
        try:
            suffix = path.suffix.lower()
            
            if suffix in ['.yaml', '.yml']:
                with open(path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
            elif suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                logger.warning(f"Unsupported config file format: {suffix}")
                return {}
            
            logger.info(f"Loaded configuration from {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading config from {file_path}: {e}")
            if required:
                raise
            return {}
    
    @staticmethod
    def load_from_env(
        prefix: str = "",
        nested: bool = True
    ) -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Args:
            prefix: Environment variable prefix (e.g., "PIEL_MEJORADOR_")
            nested: Whether to create nested structure from underscores
            
        Returns:
            Configuration dictionary
        """
        config = {}
        prefix_upper = prefix.upper()
        
        for key, value in os.environ.items():
            if prefix and not key.startswith(prefix_upper):
                continue
            
            # Remove prefix
            if prefix:
                key = key[len(prefix_upper):]
            
            # Convert to nested structure if requested
            if nested and '_' in key:
                parts = key.lower().split('_')
                current = config
                
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                current[parts[-1]] = ConfigLoader._parse_env_value(value)
            else:
                config[key.lower()] = ConfigLoader._parse_env_value(value)
        
        if config:
            logger.info(f"Loaded {len(config)} configuration values from environment")
        
        return config
    
    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """
        Parse environment variable value.
        
        Args:
            value: Environment variable value
            
        Returns:
            Parsed value (bool, int, float, or str)
        """
        # Boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        
        # String
        return value
    
    @staticmethod
    def merge_configs(
        *configs: Dict[str, Any],
        priority: str = "last"
    ) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries.
        
        Args:
            *configs: Configuration dictionaries to merge
            priority: Merge priority ("first", "last", or "deep")
            
        Returns:
            Merged configuration
        """
        if not configs:
            return {}
        
        if priority == "first":
            configs = reversed(configs)
        elif priority == "last":
            pass  # Use as-is
        elif priority == "deep":
            return ConfigLoader._deep_merge(*configs)
        
        result = {}
        for config in configs:
            result.update(config)
        
        return result
    
    @staticmethod
    def _deep_merge(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge configuration dictionaries."""
        result = {}
        
        for config in configs:
            for key, value in config.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = ConfigLoader._deep_merge(result[key], value)
                else:
                    result[key] = value
        
        return result
    
    @staticmethod
    def validate_required(
        config: Dict[str, Any],
        required_keys: list[str],
        prefix: str = ""
    ) -> list[str]:
        """
        Validate required configuration keys.
        
        Args:
            config: Configuration dictionary
            required_keys: List of required keys (supports dot notation)
            prefix: Optional prefix for error messages
            
        Returns:
            List of missing keys
        """
        missing = []
        
        for key in required_keys:
            if '.' in key:
                # Nested key
                parts = key.split('.')
                current = config
                
                for part in parts:
                    if not isinstance(current, dict) or part not in current:
                        missing.append(key)
                        break
                    current = current[part]
            else:
                if key not in config:
                    missing.append(key)
        
        if missing:
            error_msg = f"Missing required configuration keys: {', '.join(missing)}"
            if prefix:
                error_msg = f"{prefix}: {error_msg}"
            logger.error(error_msg)
        
        return missing




