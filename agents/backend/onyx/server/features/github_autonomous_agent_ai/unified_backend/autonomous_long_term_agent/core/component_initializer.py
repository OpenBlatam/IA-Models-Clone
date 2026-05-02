"""
Component Initializer
====================

Helper for initializing optional components based on settings.
Centralizes the pattern of conditional component creation.

Single Responsibility: Initialize optional components based on configuration.
"""

import logging
from typing import Optional, TypeVar, Type, Callable

from ..config import settings

logger = logging.getLogger(__name__)

T = TypeVar('T')


def create_optional_component(
    component_class: Type[T],
    enabled_setting: bool,
    component_name: str,
    **kwargs
) -> Optional[T]:
    """
    Create an optional component if enabled in settings.
    
    Args:
        component_class: Class to instantiate
        enabled_setting: Boolean setting that controls if component is enabled
        component_name: Name of component for logging
        **kwargs: Arguments to pass to component constructor
        
    Returns:
        Component instance if enabled, None otherwise
    """
    if not enabled_setting:
        logger.debug(f"{component_name} is disabled in settings")
        return None
    
    try:
        component = component_class(**kwargs)
        logger.debug(f"{component_name} initialized successfully")
        return component
    except Exception as e:
        logger.warning(
            f"Failed to initialize {component_name}: {e}",
            exc_info=True
        )
        return None


def create_optional_component_with_factory(
    factory_func: Callable[[], T],
    enabled_setting: bool,
    component_name: str
) -> Optional[T]:
    """
    Create an optional component using a factory function if enabled.
    
    Args:
        factory_func: Factory function that creates the component
        enabled_setting: Boolean setting that controls if component is enabled
        component_name: Name of component for logging
        
    Returns:
        Component instance if enabled, None otherwise
    """
    if not enabled_setting:
        logger.debug(f"{component_name} is disabled in settings")
        return None
    
    try:
        component = factory_func()
        logger.debug(f"{component_name} initialized successfully")
        return component
    except Exception as e:
        logger.warning(
            f"Failed to initialize {component_name}: {e}",
            exc_info=True
        )
        return None


class ComponentInitializer:
    """
    Helper class for initializing multiple optional components.
    
    Provides a convenient way to initialize all optional components
    used by AutonomousLongTermAgent.
    """
    
    @staticmethod
    def initialize_self_reflection_engine():
        """Initialize SelfReflectionEngine if enabled."""
        from .self_reflection import SelfReflectionEngine
        return create_optional_component(
            SelfReflectionEngine,
            settings.enable_self_reflection,
            "SelfReflectionEngine"
        )
    
    @staticmethod
    def initialize_experience_learning():
        """Initialize ExperienceDrivenLearning if enabled."""
        from .experience_driven_learning import ExperienceDrivenLearning
        return create_optional_component(
            ExperienceDrivenLearning,
            settings.enable_experience_learning,
            "ExperienceDrivenLearning"
        )
    
    @staticmethod
    def initialize_world_model():
        """Initialize ContinualWorldModel if enabled."""
        from .world_model import ContinualWorldModel
        return create_optional_component(
            ContinualWorldModel,
            settings.enable_world_model,
            "ContinualWorldModel"
        )
    
    @staticmethod
    def initialize_all_optional_components():
        """
        Initialize all optional components based on settings.
        
        Returns:
            Dict with component names as keys and component instances (or None) as values
        """
        return {
            "self_reflection_engine": ComponentInitializer.initialize_self_reflection_engine(),
            "experience_learning": ComponentInitializer.initialize_experience_learning(),
            "world_model": ComponentInitializer.initialize_world_model(),
        }

