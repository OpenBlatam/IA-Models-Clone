"""
Factories Refactorizado - Versión simplificada y optimizada.

Mejoras:
- Eliminada duplicación entre factories
- BaseFactory genérico reutilizable
- Código más limpio y mantenible
"""

from __future__ import annotations

from typing import Dict, Optional, Any, Type, Callable
from pathlib import Path

from .interfaces import IAudioSeparator, IAudioMixer, IAudioProcessor
from .exceptions import AudioSeparationError, AudioConfigurationError
from .config import SeparationConfig, MixingConfig, ProcessorConfig


class BaseFactory:
    """
    Factory base genérico que elimina duplicación.
    
    Responsabilidades:
    - Registro de componentes
    - Creación dinámica de instancias
    - Validación de tipos
    - Manejo de errores consistente
    """
    
    _registry: Dict[str, Type] = {}
    _interface_type: Optional[Type] = None
    _default_config_type: Optional[Type] = None
    _import_mapping: Dict[str, Callable[[], Type]] = {}
    
    @classmethod
    def register(cls, name: str, component_class: Type) -> None:
        """Registra un componente."""
        if cls._interface_type and not issubclass(component_class, cls._interface_type):
            raise TypeError(f"{component_class} must implement {cls._interface_type.__name__}")
        cls._registry[name.lower()] = component_class
    
    @classmethod
    def create(
        cls,
        component_type: str,
        config: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """Crea una instancia del componente."""
        component_type = component_type.lower()
        
        # Intentar importar dinámicamente si no está registrado
        if component_type not in cls._registry:
            cls._try_dynamic_import(component_type)
        
        if component_type not in cls._registry:
            raise AudioConfigurationError(
                f"Unknown {cls._get_component_name()} type: {component_type}"
            )
        
        component_class = cls._registry[component_type]
        
        # Crear configuración por defecto si no se proporciona
        if config is None and cls._default_config_type:
            config = cls._default_config_type(**cls._get_default_config_kwargs(component_type))
        
        try:
            return component_class(config=config, **kwargs)
        except Exception as e:
            raise AudioSeparationError(
                f"Failed to create {cls._get_component_name()} '{component_type}': {e}",
                component=f"{cls.__name__}"
            ) from e
    
    @classmethod
    def _try_dynamic_import(cls, component_type: str) -> None:
        """Intenta importar dinámicamente un componente."""
        if component_type in cls._import_mapping:
            try:
                component_class = cls._import_mapping[component_type]()
                cls.register(component_type, component_class)
            except ImportError as e:
                raise AudioConfigurationError(
                    f"Failed to import {cls._get_component_name()} '{component_type}': {e}"
                ) from e
    
    @classmethod
    def _get_component_name(cls) -> str:
        """Obtiene el nombre del tipo de componente para mensajes de error."""
        return cls.__name__.replace("Factory", "").lower()
    
    @classmethod
    def _get_default_config_kwargs(cls, component_type: str) -> Dict[str, Any]:
        """Obtiene kwargs por defecto para la configuración."""
        return {}
    
    @classmethod
    def list_available(cls) -> list[str]:
        """Lista los componentes disponibles."""
        return list(cls._registry.keys())


class AudioSeparatorFactory(BaseFactory):
    """Factory para crear separadores de audio."""
    
    _registry: Dict[str, Type] = {}
    _interface_type = IAudioSeparator
    _default_config_type = SeparationConfig
    
    _import_mapping = {
        "spleeter": lambda: __import__("..separators.spleeter_separator", fromlist=["SpleeterSeparator"]).SpleeterSeparator,
        "demucs": lambda: __import__("..separators.demucs_separator", fromlist=["DemucsSeparator"]).DemucsSeparator,
        "lalal": lambda: __import__("..separators.lalal_separator", fromlist=["LALALSeparator"]).LALALSeparator,
    }
    
    @classmethod
    def create(
        cls,
        separator_type: str = "auto",
        config: Optional[SeparationConfig] = None,
        **kwargs
    ) -> IAudioSeparator:
        """Crea un separador de audio."""
        separator_type = separator_type.lower()
        
        # Auto-detección
        if separator_type == "auto":
            separator_type = cls._detect_best_separator()
        
        return super().create(separator_type, config, **kwargs)
    
    @classmethod
    def _get_default_config_kwargs(cls, component_type: str) -> Dict[str, Any]:
        """Obtiene kwargs por defecto para SeparationConfig."""
        return {"model_type": component_type}
    
    @classmethod
    def _detect_best_separator(cls) -> str:
        """
        Detecta el mejor separador disponible.
        
        Returns:
            Nombre del separador recomendado
        """
        # Prioridad: demucs > spleeter > lalal
        SEPARATOR_PRIORITY = ["demucs", "spleeter", "lalal"]
        SEPARATOR_IMPORTS = {
            "demucs": "demucs",
            "spleeter": "spleeter",
            "lalal": None,  # LALAL puede requerir API key, no necesita import
        }
        
        for separator_name in SEPARATOR_PRIORITY:
            import_name = SEPARATOR_IMPORTS.get(separator_name)
            if import_name:
                try:
                    __import__(import_name)
                    return separator_name
                except ImportError:
                    continue
            else:
                # LALAL no requiere import, pero puede requerir API key
                return separator_name
        
        return "spleeter"  # Fallback
    
    @classmethod
    def list_available(cls) -> list[str]:
        """
        Lista los separadores disponibles.
        
        Returns:
            Lista de nombres de separadores disponibles
        """
        SEPARATOR_IMPORTS = {
            "demucs": "demucs",
            "spleeter": "spleeter",
            "lalal": None,  # LALAL no requiere import
        }
        
        available = []
        for name, import_name in SEPARATOR_IMPORTS.items():
            if import_name:
                try:
                    __import__(import_name)
                    available.append(name)
                except ImportError:
                    pass
            else:
                # LALAL siempre disponible (puede requerir API key)
                available.append(name)
        
        return available or super().list_available()


class AudioMixerFactory(BaseFactory):
    """Factory para crear mezcladores de audio."""
    
    _registry: Dict[str, Type] = {}
    _interface_type = IAudioMixer
    _default_config_type = MixingConfig
    
    _import_mapping = {
        "simple": lambda: __import__("..mixers.simple_mixer", fromlist=["SimpleMixer"]).SimpleMixer,
        "advanced": lambda: __import__("..mixers.advanced_mixer", fromlist=["AdvancedMixer"]).AdvancedMixer,
    }
    
    @classmethod
    def _get_default_config_kwargs(cls, component_type: str) -> Dict[str, Any]:
        """Obtiene kwargs por defecto para MixingConfig."""
        return {"mixer_type": component_type}


class AudioProcessorFactory(BaseFactory):
    """Factory para crear procesadores de audio."""
    
    _registry: Dict[str, Type] = {}
    _interface_type = IAudioProcessor
    _default_config_type = ProcessorConfig
    
    _import_mapping = {
        "extractor": lambda: __import__("..processors.video_extractor", fromlist=["VideoAudioExtractor"]).VideoAudioExtractor,
        "converter": lambda: __import__("..processors.format_converter", fromlist=["AudioFormatConverter"]).AudioFormatConverter,
        "enhancer": lambda: __import__("..processors.audio_enhancer", fromlist=["AudioEnhancer"]).AudioEnhancer,
    }
    
    @classmethod
    def _get_default_config_kwargs(cls, component_type: str) -> Dict[str, Any]:
        """Obtiene kwargs por defecto para ProcessorConfig."""
        return {"processor_type": component_type}


# ════════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE CONVENIENCIA
# ════════════════════════════════════════════════════════════════════════════════

def create_audio_separator(
    separator_type: str = "auto",
    config: Optional[SeparationConfig] = None,
    **kwargs
) -> IAudioSeparator:
    """Función de conveniencia para crear un separador."""
    return AudioSeparatorFactory.create(separator_type, config, **kwargs)


def create_audio_mixer(
    mixer_type: str = "simple",
    config: Optional[MixingConfig] = None,
    **kwargs
) -> IAudioMixer:
    """Función de conveniencia para crear un mezclador."""
    return AudioMixerFactory.create(mixer_type, config, **kwargs)


def create_audio_processor(
    processor_type: str,
    config: Optional[ProcessorConfig] = None,
    **kwargs
) -> IAudioProcessor:
    """Función de conveniencia para crear un procesador."""
    return AudioProcessorFactory.create(processor_type, config, **kwargs)

