"""
Component Registry Module

Component registration (metrics, callbacks, extractors, augmentations).
"""

from typing import Dict, Type, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class ComponentRegistryMixin:
    """Component registration mixin."""
    
    def __init__(self):
        self._metrics: Dict[str, Type] = {}
        self._callbacks: Dict[str, Type] = {}
        self._extractors: Dict[str, Callable] = {}
        self._augmentations: Dict[str, Callable] = {}
    
    # Metric registration
    def register_metric(self, name: str, metric_class: Type):
        """
        Register a metric class.
        
        Args:
            name: Metric name.
            metric_class: Metric class.
        """
        if name in self._metrics:
            logger.warning(f"Metric {name} already registered, overwriting")
        self._metrics[name] = metric_class
        logger.info(f"Registered metric: {name}")
    
    def get_metric(self, name: str) -> Optional[Type]:
        """
        Get metric class by name.
        
        Args:
            name: Metric name.
        
        Returns:
            Metric class or None.
        """
        return self._metrics.get(name)
    
    # Callback registration
    def register_callback(self, name: str, callback_class: Type):
        """
        Register a callback class.
        
        Args:
            name: Callback name.
            callback_class: Callback class.
        """
        if name in self._callbacks:
            logger.warning(f"Callback {name} already registered, overwriting")
        self._callbacks[name] = callback_class
        logger.info(f"Registered callback: {name}")
    
    def get_callback(self, name: str) -> Optional[Type]:
        """
        Get callback class by name.
        
        Args:
            name: Callback name.
        
        Returns:
            Callback class or None.
        """
        return self._callbacks.get(name)
    
    # Extractor registration
    def register_extractor(self, name: str, extractor_factory: Callable):
        """
        Register a feature extractor factory.
        
        Args:
            name: Extractor name.
            extractor_factory: Extractor factory function.
        """
        if name in self._extractors:
            logger.warning(f"Extractor {name} already registered, overwriting")
        self._extractors[name] = extractor_factory
        logger.info(f"Registered extractor: {name}")
    
    def get_extractor_factory(self, name: str) -> Optional[Callable]:
        """
        Get extractor factory by name.
        
        Args:
            name: Extractor name.
        
        Returns:
            Extractor factory or None.
        """
        return self._extractors.get(name)
    
    # Augmentation registration
    def register_augmentation(self, name: str, augmentation_factory: Callable):
        """
        Register a data augmentation factory.
        
        Args:
            name: Augmentation name.
            augmentation_factory: Augmentation factory function.
        """
        if name in self._augmentations:
            logger.warning(f"Augmentation {name} already registered, overwriting")
        self._augmentations[name] = augmentation_factory
        logger.info(f"Registered augmentation: {name}")
    
    def get_augmentation_factory(self, name: str) -> Optional[Callable]:
        """
        Get augmentation factory by name.
        
        Args:
            name: Augmentation name.
        
        Returns:
            Augmentation factory or None.
        """
        return self._augmentations.get(name)



