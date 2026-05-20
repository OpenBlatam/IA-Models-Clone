"""
Adapter Pattern - Adapt different interfaces
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


class IAdapter(ABC):
    """
    Interface for adapters
    """
    
    @abstractmethod
    def adapt(self, data: Any) -> Any:
        """Adapt data from one format to another"""
        pass
    
    @property
    @abstractmethod
    def source_format(self) -> str:
        """Source format name"""
        pass
    
    @property
    @abstractmethod
    def target_format(self) -> str:
        """Target format name"""
        pass


class BaseAdapter(IAdapter):
    """
    Base adapter implementation
    """
    
    def __init__(self, source_format: str, target_format: str):
        self._source_format = source_format
        self._target_format = target_format
    
    @property
    def source_format(self) -> str:
        return self._source_format
    
    @property
    def target_format(self) -> str:
        return self._target_format
    
    def adapt(self, data: Any) -> Any:
        """Base adapt - override in subclasses"""
        return data








