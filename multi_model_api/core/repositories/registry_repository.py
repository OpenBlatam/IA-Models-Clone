"""
Repository implementation using ModelRegistry
Concrete implementation of ModelRepository
"""

from typing import List, Optional, Dict, Any
from ...api.schemas import ModelType, ModelResponse
from ..models import ModelRegistry, ModelMetadata
from .model_repository import ModelRepository


class RegistryModelRepository(ModelRepository):
    """Repository implementation using ModelRegistry"""
    
    def __init__(self, registry: ModelRegistry):
        """
        Initialize repository with registry
        
        Args:
            registry: ModelRegistry instance
        """
        self.registry = registry
    
    async def execute_model(
        self,
        model_type: ModelType,
        prompt: str,
        **kwargs
    ) -> ModelResponse:
        """Execute a model using the registry"""
        return await self.registry.execute_model(
            model_type,
            prompt,
            **kwargs
        )
    
    def get_available_models(self) -> List[ModelMetadata]:
        """Get available models from registry"""
        return self.registry.get_available_models()
    
    def get_model_metadata(self, model_type: ModelType) -> Optional[ModelMetadata]:
        """Get model metadata from registry"""
        return self.registry.get_model(model_type)
    
    def get_model_health(self, model_type: ModelType) -> Dict[str, Any]:
        """Get model health from registry"""
        return self.registry.get_model_health(model_type)




