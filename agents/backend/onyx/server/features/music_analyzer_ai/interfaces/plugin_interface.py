"""
Plugin Interfaces - Define contracts for plugins
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class IPlugin(ABC):
    """
    Base interface for all plugins
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version"""
        pass
    
    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize plugin"""
        pass
    
    @abstractmethod
    def execute(self, data: Any) -> Any:
        """Execute plugin logic"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup plugin resources"""
        pass


class IPreprocessingPlugin(IPlugin):
    """
    Interface for preprocessing plugins
    """
    
    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """Preprocess data"""
        pass


class IPostprocessingPlugin(IPlugin):
    """
    Interface for postprocessing plugins
    """
    
    @abstractmethod
    def postprocess(self, data: Any) -> Any:
        """Postprocess data"""
        pass


class IAnalysisPlugin(IPlugin):
    """
    Interface for analysis plugins
    """
    
    @abstractmethod
    def analyze(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Perform analysis"""
        pass









