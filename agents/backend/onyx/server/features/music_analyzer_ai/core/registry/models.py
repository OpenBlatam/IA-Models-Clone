"""
Model Registry Module

Model registration functionality.
"""

from typing import Dict, Type, Optional
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Model registration mixin."""
    
    def __init__(self):
        self._models: Dict[str, Type] = {}
    
    def register_model(self, name: str, model_class: Type):
        """
        Register a model class.
        
        Args:
            name: Model name.
            model_class: Model class.
        """
        if name in self._models:
            logger.warning(f"Model {name} already registered, overwriting")
        self._models[name] = model_class
        logger.info(f"Registered model: {name}")
    
    def get_model(self, name: str) -> Optional[Type]:
        """
        Get model class by name.
        
        Args:
            name: Model name.
        
        Returns:
            Model class or None.
        """
        return self._models.get(name)
    
    def list_models(self) -> list:
        """
        List all registered models.
        
        Returns:
            List of model names.
        """
        return list(self._models.keys())



