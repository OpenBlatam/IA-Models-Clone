"""
Factory Module for Deep Learning Generator

Factory pattern implementation for creating generator instances.
"""

import logging
from typing import Optional, Dict, Any, Type
from functools import lru_cache

logger = logging.getLogger(__name__)


class GeneratorFactory:
    """
    Factory class for creating DeepLearningGenerator instances.
    
    Follows the Factory pattern and provides lazy loading of dependencies.
    """
    
    def __init__(self):
        self._generator_class: Optional[Type] = None
        self._generator_available: bool = False
        self._initialize_generator()
    
    def _initialize_generator(self) -> None:
        """Lazy initialization of the generator class."""
        try:
            from ...deep_learning.core import DeepLearningGenerator
            self._generator_class = DeepLearningGenerator
            self._generator_available = True
            logger.info("DeepLearningGenerator successfully loaded")
        except ImportError as e:
            logger.warning(f"DeepLearningGenerator not available: {e}")
            self._generator_available = False
            self._generator_class = None
    
    @property
    def is_available(self) -> bool:
        """Check if generator is available."""
        return self._generator_available
    
    @property
    def generator_class(self) -> Optional[Type]:
        """Get the generator class."""
        return self._generator_class
    
    def create(
        self,
        framework: str = "pytorch",
        model_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enable_advanced_features: bool = True
    ) -> Any:
        """
        Create a DeepLearningGenerator instance.
        
        Args:
            framework: Framework to use
            model_type: Model type (optional)
            config: Additional configuration
            enable_advanced_features: Enable advanced features
            
        Returns:
            DeepLearningGenerator instance
            
        Raises:
            ImportError: If generator is not available
            ValueError: If parameters are invalid
        """
        if not self._generator_available:
            raise ImportError(
                "DeepLearningGenerator is not available. "
                "Check that deep_learning.core module is properly installed."
            )
        
        if config is None:
            config = {}
        
        # Set defaults
        config.setdefault("framework", framework)
        if model_type:
            config.setdefault("model_type", model_type)
        
        # Integrate advanced features if enabled
        if enable_advanced_features:
            self._integrate_advanced_features(config)
        
        try:
            return self._generator_class(**config)
        except Exception as e:
            logger.error(f"Error creating DeepLearningGenerator: {e}", exc_info=True)
            raise
    
    def _integrate_advanced_features(self, config: Dict[str, Any]) -> None:
        """
        Integrate advanced pipeline features into config.
        
        Args:
            config: Configuration dictionary to update
        """
        try:
            # Try relative import first
            from ...robot_movement_ai.core.architecture.pipelines import (
                TrainingPipeline,
                InferencePipeline,
                TransferLearningManager,
                AutoMLPipeline,
                KnowledgeDistillation,
                ReinforcementLearningWrapper
            )
            config["advanced_pipelines"] = {
                "training": TrainingPipeline,
                "inference": InferencePipeline,
                "transfer_learning": TransferLearningManager,
                "automl": AutoMLPipeline,
                "knowledge_distillation": KnowledgeDistillation,
                "reinforcement_learning": ReinforcementLearningWrapper
            }
            logger.info("Advanced pipelines integrated into generator")
        except ImportError:
            logger.debug("Advanced pipelines not available, continuing without them")


# Singleton factory instance
@lru_cache(maxsize=1)
def get_factory() -> GeneratorFactory:
    """Get singleton factory instance."""
    return GeneratorFactory()

