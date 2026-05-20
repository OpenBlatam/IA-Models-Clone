"""
Utils Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Callable, Dict, List, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import functools


class ValidationError(Exception):
    """Validation error"""
    pass


class ConfigurationError(Exception):
    """Configuration error"""
    pass


@dataclass
class RetryConfig:
    """Retry configuration"""
    max_retries: int = 3
    backoff_factor: float = 1.0
    retry_on: tuple = (Exception,)
    max_wait_time: Optional[float] = None
    jitter: bool = True


@dataclass
class CacheConfig:
    """Cache configuration"""
    ttl: timedelta
    key_prefix: str = ""
    serialize: bool = True


class UtilityBase(ABC):
    """Base interface for utilities"""
    
    @staticmethod
    @abstractmethod
    def validate_email(email: str) -> bool:
        """Validate email"""
        pass
    
    @staticmethod
    @abstractmethod
    def sanitize_string(value: str) -> str:
        """Sanitize string"""
        pass
    
    @staticmethod
    @abstractmethod
    async def retry(
        func: Callable,
        config: RetryConfig,
        *args,
        **kwargs
    ) -> Any:
        """Retry function with backoff"""
        pass
    
    @staticmethod
    @abstractmethod
    def generate_id(prefix: str = "") -> str:
        """Generate unique ID"""
        pass
    
    @staticmethod
    @abstractmethod
    def format_datetime(dt: datetime, format: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Format datetime"""
        pass
    
    @staticmethod
    @abstractmethod
    def parse_datetime(date_string: str, format: str = "%Y-%m-%d %H:%M:%S") -> datetime:
        """Parse datetime string"""
        pass
    
    @staticmethod
    @abstractmethod
    def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
        """Deep merge dictionaries"""
        pass
    
    @staticmethod
    @abstractmethod
    def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
        """Chunk list into smaller lists"""
        pass

