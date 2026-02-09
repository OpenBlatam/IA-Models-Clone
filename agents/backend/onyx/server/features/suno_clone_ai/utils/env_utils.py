"""
Environment variable utilities.

Consolidates environment variable loading and parsing patterns.
"""

import os
import logging
from typing import Any, Optional, Union, List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class EnvUtils:
    """Utilities for environment variable handling."""
    
    @staticmethod
    def get(
        key: str,
        default: Any = None,
        required: bool = False,
        var_type: type = str
    ) -> Any:
        """
        Get environment variable with type conversion.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            required: Raise error if not found
            var_type: Type to convert to (str, int, float, bool)
        
        Returns:
            Environment variable value (converted to var_type)
        
        Raises:
            ValueError: If required and not found
        """
        value = os.getenv(key, default)
        
        if value is None:
            if required:
                raise ValueError(f"Required environment variable '{key}' not set")
            return default
        
        return EnvUtils._convert_type(value, var_type)
    
    @staticmethod
    def get_bool(
        key: str,
        default: bool = False,
        required: bool = False
    ) -> bool:
        """
        Get boolean environment variable.
        
        Args:
            key: Environment variable name
            default: Default value
            required: Raise error if not found
        
        Returns:
            Boolean value
        """
        return EnvUtils.get(key, default, required, bool)
    
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
            required: Raise error if not found
        
        Returns:
            Integer value
        """
        return EnvUtils.get(key, default, required, int)
    
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
            required: Raise error if not found
        
        Returns:
            Float value
        """
        return EnvUtils.get(key, default, required, float)
    
    @staticmethod
    def get_list(
        key: str,
        separator: str = ",",
        default: Optional[List[str]] = None,
        required: bool = False
    ) -> List[str]:
        """
        Get list from comma-separated environment variable.
        
        Args:
            key: Environment variable name
            separator: Separator character
            default: Default value
            required: Raise error if not found
        
        Returns:
            List of strings
        """
        value = EnvUtils.get(key, default, required, str)
        if value is None:
            return default or []
        if isinstance(value, str):
            return [item.strip() for item in value.split(separator) if item.strip()]
        return value
    
    @staticmethod
    def get_path(
        key: str,
        default: Optional[Union[str, Path]] = None,
        required: bool = False,
        create: bool = False
    ) -> Optional[Path]:
        """
        Get Path from environment variable.
        
        Args:
            key: Environment variable name
            default: Default path
            required: Raise error if not found
            create: Create directory if it doesn't exist
        
        Returns:
            Path object
        """
        value = EnvUtils.get(key, default, required, str)
        if value is None:
            return None
        
        path = Path(value)
        if create and not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")
        
        return path
    
    @staticmethod
    def load_from_file(
        file_path: Union[str, Path],
        override: bool = False
    ) -> Dict[str, str]:
        """
        Load environment variables from .env file.
        
        Args:
            file_path: Path to .env file
            override: Override existing environment variables
        
        Returns:
            Dictionary of loaded variables
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f".env file not found: {file_path}")
            return {}
        
        loaded = {}
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    
                    if override or key not in os.environ:
                        os.environ[key] = value
                        loaded[key] = value
        
        logger.info(f"Loaded {len(loaded)} environment variables from {file_path}")
        return loaded
    
    @staticmethod
    def _convert_type(value: str, var_type: type) -> Any:
        """Convert string value to specified type."""
        if var_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
        elif var_type == int:
            try:
                return int(value)
            except ValueError:
                raise ValueError(f"Cannot convert '{value}' to int")
        elif var_type == float:
            try:
                return float(value)
            except ValueError:
                raise ValueError(f"Cannot convert '{value}' to float")
        else:
            return value


# Convenience functions
def get_env(
    key: str,
    default: Any = None,
    required: bool = False,
    var_type: type = str
) -> Any:
    """Get environment variable."""
    return EnvUtils.get(key, default, required, var_type)


def get_env_bool(key: str, default: bool = False, required: bool = False) -> bool:
    """Get boolean environment variable."""
    return EnvUtils.get_bool(key, default, required)


def get_env_int(
    key: str,
    default: Optional[int] = None,
    required: bool = False
) -> Optional[int]:
    """Get integer environment variable."""
    return EnvUtils.get_int(key, default, required)


def get_env_float(
    key: str,
    default: Optional[float] = None,
    required: bool = False
) -> Optional[float]:
    """Get float environment variable."""
    return EnvUtils.get_float(key, default, required)


def get_env_list(
    key: str,
    separator: str = ",",
    default: Optional[List[str]] = None,
    required: bool = False
) -> List[str]:
    """Get list from environment variable."""
    return EnvUtils.get_list(key, separator, default, required)


def get_env_path(
    key: str,
    default: Optional[Union[str, Path]] = None,
    required: bool = False,
    create: bool = False
) -> Optional[Path]:
    """Get Path from environment variable."""
    return EnvUtils.get_path(key, default, required, create)


def load_env_file(
    file_path: Union[str, Path],
    override: bool = False
) -> Dict[str, str]:
    """Load environment variables from .env file."""
    return EnvUtils.load_from_file(file_path, override)

