"""
Component Loader - Dynamic loading of component classes.

Single Responsibility: Import and load component classes from modules.
This eliminates duplication across factories.
"""

from __future__ import annotations

from typing import Dict, Type, Tuple, Optional
from .exceptions import AudioConfigurationError


class ComponentLoader:
    """
    Load components dynamically from modules.
    
    Single Responsibility: Import and load component classes.
    Centralizes all dynamic import logic to eliminate duplication.
    
    Example:
        loader = ComponentLoader()
        separator_class = loader.load_separator("spleeter")
    """
    
    # Centralized mapping of component types to (module_path, class_name)
    # This makes it easy to add new components without modifying factories
    
    SEPARATOR_MAP: Dict[str, Tuple[str, str]] = {
        "spleeter": ("..separators.spleeter_separator", "SpleeterSeparator"),
        "demucs": ("..separators.demucs_separator", "DemucsSeparator"),
        "lalal": ("..separators.lalal_separator", "LALALSeparator"),
    }
    
    MIXER_MAP: Dict[str, Tuple[str, str]] = {
        "simple": ("..mixers.simple_mixer", "SimpleMixer"),
        "advanced": ("..mixers.advanced_mixer", "AdvancedMixer"),
    }
    
    PROCESSOR_MAP: Dict[str, Tuple[str, str]] = {
        "extractor": ("..processors.video_extractor", "VideoAudioExtractor"),
        "converter": ("..processors.format_converter", "AudioFormatConverter"),
        "enhancer": ("..processors.audio_enhancer", "AudioEnhancer"),
    }
    
    @classmethod
    def load_separator(cls, separator_type: str) -> Type:
        """
        Load a separator class dynamically.
        
        Args:
            separator_type: Type of separator ("spleeter", "demucs", "lalal")
            
        Returns:
            Separator class
            
        Raises:
            AudioConfigurationError: If separator type is unknown or import fails
        """
        return cls._load_component(separator_type, cls.SEPARATOR_MAP, "separator")
    
    @classmethod
    def load_mixer(cls, mixer_type: str) -> Type:
        """
        Load a mixer class dynamically.
        
        Args:
            mixer_type: Type of mixer ("simple", "advanced")
            
        Returns:
            Mixer class
            
        Raises:
            AudioConfigurationError: If mixer type is unknown or import fails
        """
        return cls._load_component(mixer_type, cls.MIXER_MAP, "mixer")
    
    @classmethod
    def load_processor(cls, processor_type: str) -> Type:
        """
        Load a processor class dynamically.
        
        Args:
            processor_type: Type of processor ("extractor", "converter", "enhancer")
            
        Returns:
            Processor class
            
        Raises:
            AudioConfigurationError: If processor type is unknown or import fails
        """
        return cls._load_component(processor_type, cls.PROCESSOR_MAP, "processor")
    
    @classmethod
    def _load_component(
        cls,
        component_type: str,
        component_map: Dict[str, Tuple[str, str]],
        component_category: str
    ) -> Type:
        """
        Generic component loading logic.
        
        Args:
            component_type: Type of component
            component_map: Mapping of types to (module_path, class_name)
            component_category: Category name for error messages
            
        Returns:
            Component class
            
        Raises:
            AudioConfigurationError: If component type is unknown or import fails
        """
        component_type = component_type.lower()
        
        if component_type not in component_map:
            available = list(component_map.keys())
            raise AudioConfigurationError(
                f"Unknown {component_category} type: '{component_type}'. "
                f"Available types: {', '.join(available)}"
            )
        
        module_path, class_name = component_map[component_type]
        
        try:
            # Import module dynamically
            module = __import__(module_path, fromlist=[class_name], level=1)
            component_class = getattr(module, class_name)
            
            if not isinstance(component_class, type):
                raise AudioConfigurationError(
                    f"'{class_name}' in module '{module_path}' is not a class"
                )
            
            return component_class
        except ImportError as e:
            raise AudioConfigurationError(
                f"Failed to import {component_category} '{component_type}': {e}. "
                f"Make sure the required dependencies are installed."
            ) from e
        except AttributeError as e:
            raise AudioConfigurationError(
                f"Class '{class_name}' not found in module '{module_path}': {e}"
            ) from e
        except AudioConfigurationError:
            # Re-raise AudioConfigurationError as-is
            raise
        except Exception as e:
            raise AudioConfigurationError(
                f"Unexpected error loading {component_category} '{component_type}': {e}"
            ) from e
    
    @classmethod
    def list_available_separators(cls) -> list[str]:
        """List all available separator types."""
        return list(cls.SEPARATOR_MAP.keys())
    
    @classmethod
    def list_available_mixers(cls) -> list[str]:
        """List all available mixer types."""
        return list(cls.MIXER_MAP.keys())
    
    @classmethod
    def list_available_processors(cls) -> list[str]:
        """List all available processor types."""
        return list(cls.PROCESSOR_MAP.keys())

