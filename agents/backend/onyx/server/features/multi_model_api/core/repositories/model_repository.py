"""
Abstract repository interface for model operations
Repository pattern for data access abstraction
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from ...api.schemas import ModelType, ModelResponse, ModelConfig
from ..models import ModelMetadata


class ModelRepository(ABC):
    """Abstract repository for model operations"""
    
    @abstractmethod
    async def execute_model(
        self,
        model_type: ModelType,
        prompt: str,
        **kwargs
    ) -> ModelResponse:
        """
        Execute a model with a prompt
        
        Args:
            model_type: Type of model to execute
            prompt: Input prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            ModelResponse with result
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[ModelMetadata]:
        """
        Get all available models
        
        Returns:
            List of model metadata
        """
        pass
    
    @abstractmethod
    def get_model_metadata(self, model_type: ModelType) -> Optional[ModelMetadata]:
        """
        Get metadata for a specific model
        
        Args:
            model_type: Type of model
            
        Returns:
            ModelMetadata or None if not found
        """
        pass
    
    @abstractmethod
    def get_model_health(self, model_type: ModelType) -> Dict[str, Any]:
        """
        Get health metrics for a model
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary with health metrics
        """
        pass




