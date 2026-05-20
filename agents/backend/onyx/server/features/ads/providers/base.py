from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
from ..config.providers import ProviderConfig
from typing import Any, List, Dict, Optional
import asyncio
"""
Base provider for AI services.
"""

logger = logging.getLogger(__name__)

class BaseProvider(ABC):
    """Base class for AI providers."""
    
    def __init__(self, config: ProviderConfig):
        
    """__init__ function."""
self.config = config
        self.logger = logger
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider."""
        pass
    
    @abstractmethod
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    async def generate_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate JSON from prompt."""
        pass
    
    @abstractmethod
    async def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text."""
        pass
    
    @abstractmethod
    async def analyze_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """Analyze text and return insights."""
        pass
    
    @abstractmethod
    async def optimize_text(self, text: str, target: str, **kwargs) -> str:
        """Optimize text for target."""
        pass
    
    @abstractmethod
    async def compare_texts(self, text1: str, text2: str) -> Dict[str, Any]:
        """Compare two texts."""
        pass
    
    @abstractmethod
    async def generate_variations(self, text: str, num_variations: int = 3, **kwargs) -> List[str]:
        """Generate variations of text."""
        pass
    
    @abstractmethod
    async def analyze_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze metrics and return insights."""
        pass
    
    def _validate_initialization(self) -> None:
        """Validate that the provider is initialized."""
        if not self._initialized:
            raise RuntimeError("Provider not initialized. Call initialize() first.")
    
    def _get_config_value(self, key: str, default: Any = None) -> Optional[Dict[str, Any]]:
        """Get configuration value with fallback."""
        return getattr(self.config, key, default)
    
    def _log_operation(self, operation: str, **kwargs) -> None:
        """Log operation details."""
        self.logger.info(f"Executing {operation} with provider {self.__class__.__name__}", extra=kwargs)
    
    def _log_error(self, operation: str, error: Exception) -> None:
        """Log error details."""
        self.logger.error(f"Error in {operation} with provider {self.__class__.__name__}: {str(error)}") 