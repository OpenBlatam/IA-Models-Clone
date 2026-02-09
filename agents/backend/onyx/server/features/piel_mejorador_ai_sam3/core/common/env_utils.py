"""
Environment Utilities for Piel Mejorador AI SAM3
=================================================

Unified environment variable access and utilities.
"""

import os
import logging
from typing import Optional, Any, Dict, List, Union, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class EnvUtils:
    """Unified environment variable utilities."""
    
    @staticmethod
    def get(
        key: str,
        default: Optional[str] = None,
        required: bool = False,
        transform: Optional[Callable[[str], Any]] = None
    ) -> Optional[str]:
        """
        Get environment variable with optional transformation.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            required: Whether variable is required
            transform: Optional transformation function
            
        Returns:
            Environment variable value (transformed if transform provided)
            
        Raises:
            ValueError: If required and not found
        """
        value = os.getenv(key, default)
        
        if required and value is None:
            raise ValueError(f"Required environment variable {key} is not set")
        
        if value is None:
            return None
        
        if transform:
            try:
                return transform(value)
            except Exception as e:
                logger.warning(f"Failed to transform {key}: {e}")
                return default
        
        return value
    
    @staticmethod
    def get_bool(
        key: str,
        default: bool = False
    ) -> bool:
        """
        Get boolean environment variable.
        
        Args:
            key: Environment variable name
            default: Default value
            
        Returns:
            Boolean value
        """
        value = EnvUtils.get(key, str(default))
        if value is None:
            return default
        
        return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
    
    @staticmethod
    def get_int(
        key: str,
        default: Optional[int] = None,
        required: bool = False
    ) -> Optional[int]:
        """
        Get integer environment variable.
        
        Args:
            key: Environment variable name
            default: Default value
            required: Whether variable is required
            
        Returns:
            Integer value
            
        Raises:
            ValueError: If required and not found or invalid
        """
        value = EnvUtils.get(key, str(default) if default is not None else None, required)
        
        if value is None:
            return default
        
        try:
            return int(value)
        except ValueError:
            if required:
                raise ValueError(f"Environment variable {key} must be an integer")
            logger.warning(f"Invalid integer value for {key}: {value}, using default {default}")
            return default
    
    @staticmethod
    def get_float(
        key: str,
        default: Optional[float] = None,
        required: bool = False
    ) -> Optional[float]:
        """
        Get float environment variable.
        
        Args:
            key: Environment variable name
            default: Default value
            required: Whether variable is required
            
        Returns:
            Float value
            
        Raises:
            ValueError: If required and not found or invalid
        """
        value = EnvUtils.get(key, str(default) if default is not None else None, required)
        
        if value is None:
            return default
        
        try:
            return float(value)
        except ValueError:
            if required:
                raise ValueError(f"Environment variable {key} must be a float")
            logger.warning(f"Invalid float value for {key}: {value}, using default {default}")
            return default
    
    @staticmethod
    def get_list(
        key: str,
        separator: str = ",",
        default: Optional[List[str]] = None,
        required: bool = False
    ) -> List[str]:
        """
        Get list from environment variable.
        
        Args:
            key: Environment variable name
            separator: Separator character
            default: Default value
            required: Whether variable is required
            
        Returns:
            List of strings
        """
        value = EnvUtils.get(key, None, required)
        
        if value is None:
            return default or []
        
        return [item.strip() for item in value.split(separator) if item.strip()]
    
    @staticmethod
    def get_path(
        key: str,
        default: Optional[Union[str, Path]] = None,
        required: bool = False,
        must_exist: bool = False
    ) -> Optional[Path]:
        """
        Get path from environment variable.
        
        Args:
            key: Environment variable name
            default: Default value
            required: Whether variable is required
            must_exist: Whether path must exist
            
        Returns:
            Path object
            
        Raises:
            ValueError: If required and not found or must_exist and path doesn't exist
        """
        value = EnvUtils.get(key, str(default) if default else None, required)
        
        if value is None:
            return Path(default) if default else None
        
        path = Path(value)
        
        if must_exist and not path.exists():
            if required:
                raise ValueError(f"Path from {key} does not exist: {path}")
            logger.warning(f"Path from {key} does not exist: {path}")
            return Path(default) if default else None
        
        return path
    
    @staticmethod
    def get_dict(
        prefix: str,
        nested: bool = True
    ) -> Dict[str, Any]:
        """
        Get dictionary from environment variables with prefix.
        
        Args:
            prefix: Environment variable prefix
            nested: Whether to create nested structure from underscores
            
        Returns:
            Dictionary of environment variables
        """
        config = {}
        prefix_upper = prefix.upper()
        
        for key, value in os.environ.items():
            if not key.startswith(prefix_upper):
                continue
            
            # Remove prefix
            key = key[len(prefix_upper):]
            
            # Convert to nested structure if requested
            if nested and '_' in key:
                parts = key.lower().split('_')
                current = config
                
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                current[parts[-1]] = EnvUtils._parse_value(value)
            else:
                config[key.lower()] = EnvUtils._parse_value(value)
        
        return config
    
    @staticmethod
    def _parse_value(value: str) -> Any:
        """
        Parse environment variable value (try to infer type).
        
        Args:
            value: String value
            
        Returns:
            Parsed value (int, float, bool, or str)
        """
        # Try boolean
        if value.lower() in ('true', '1', 'yes', 'on', 'enabled'):
            return True
        if value.lower() in ('false', '0', 'no', 'off', 'disabled'):
            return False
        
        # Try int
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    @staticmethod
    def set(key: str, value: str) -> None:
        """
        Set environment variable.
        
        Args:
            key: Environment variable name
            value: Value to set
        """
        os.environ[key] = value
    
    @staticmethod
    def unset(key: str) -> None:
        """
        Unset environment variable.
        
        Args:
            key: Environment variable name
        """
        os.environ.pop(key, None)
    
    @staticmethod
    def has(key: str) -> bool:
        """
        Check if environment variable exists.
        
        Args:
            key: Environment variable name
            
        Returns:
            True if exists
        """
        return key in os.environ
    
    @staticmethod
    def require(*keys: str) -> Dict[str, str]:
        """
        Require multiple environment variables.
        
        Args:
            *keys: Environment variable names
            
        Returns:
            Dictionary of key-value pairs
            
        Raises:
            ValueError: If any required variable is missing
        """
        missing = []
        result = {}
        
        for key in keys:
            value = os.getenv(key)
            if value is None:
                missing.append(key)
            else:
                result[key] = value
        
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        return result


# Convenience functions
def get_env(key: str, **kwargs) -> Optional[str]:
    """Get environment variable."""
    return EnvUtils.get(key, **kwargs)


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    return EnvUtils.get_bool(key, default)


def get_env_int(key: str, **kwargs) -> Optional[int]:
    """Get integer environment variable."""
    return EnvUtils.get_int(key, **kwargs)


def get_env_float(key: str, **kwargs) -> Optional[float]:
    """Get float environment variable."""
    return EnvUtils.get_float(key, **kwargs)


def get_env_list(key: str, **kwargs) -> List[str]:
    """Get list from environment variable."""
    return EnvUtils.get_list(key, **kwargs)


def get_env_path(key: str, **kwargs) -> Optional[Path]:
    """Get path from environment variable."""
    return EnvUtils.get_path(key, **kwargs)


def require_env(*keys: str) -> Dict[str, str]:
    """Require multiple environment variables."""
    return EnvUtils.require(*keys)




