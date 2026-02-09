"""
Configuration Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, Type, TypeVar

T = TypeVar('T')


class ConfigSource(str, Enum):
    """Configuration source"""
    ENV = "env"
    FILE = "file"
    DATABASE = "database"
    MEMORY = "memory"
    SECRETS = "secrets"


class Config:
    """Configuration model"""
    
    def __init__(
        self,
        key: str,
        value: Any,
        source: ConfigSource,
        description: Optional[str] = None,
        encrypted: bool = False
    ):
        self.key = key
        self.value = value
        self.source = source
        self.description = description
        self.encrypted = encrypted


class ConfigBase(ABC):
    """Base interface for configuration"""
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        pass
    
    @abstractmethod
    def get_typed(self, key: str, expected_type: Type[T], default: Optional[T] = None) -> T:
        """Get configuration value with type validation"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> bool:
        """Set configuration value"""
        pass
    
    @abstractmethod
    def require(self, *keys: str) -> Dict[str, Any]:
        """Require configuration keys to be present"""
        pass
    
    @abstractmethod
    def reload(self) -> bool:
        """Reload configuration"""
        pass

