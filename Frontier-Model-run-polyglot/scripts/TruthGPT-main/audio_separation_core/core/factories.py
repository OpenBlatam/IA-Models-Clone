"""
Factories Core - Patrón Factory para Audio Separation Core.

Refactorizado para:
- Eliminar duplicación usando BaseFactory, ComponentRegistry, ComponentLoader, SeparatorDetector
- Separar responsabilidades (SRP)
- Mejorar testabilidad y mantenibilidad
"""

from __future__ import annotations

from typing import Optional, Type, TypeVar, Callable, Dict, Any
from abc import ABC, abstractmethod

from .interfaces import IAudioSeparator, IAudioMixer, IAudioProcessor
from .exceptions import AudioSeparationError, AudioConfigurationError
from .config import SeparationConfig, MixingConfig, ProcessorConfig
from .registry import ComponentRegistry
from .loader import ComponentLoader
from .detector import SeparatorDetector

T = TypeVar('T')


class BaseFactory(ABC):
    """
    Factory base genérico que elimina duplicación entre factories.
    
    Responsabilidades:
    - Registro de componentes
    - Creación dinámica de instancias
    - Validación de tipos
    - Manejo de errores consistente
    
    Las factories específicas solo necesitan implementar:
    - `_get_interface_type()`: Retornar la interfaz que deben implementar
    - `_get_default_config_type()`: Retornar el tipo de configuración por defecto
    - `_get_default_config_kwargs()`: Retornar kwargs para la configuración
    - `_load_component()`: Cargar componente dinámicamente
    """
    
    # Shared instances (class-level for efficiency)
    _registry = ComponentRegistry()
    _loader = ComponentLoader()
    
    @classmethod
    def register(cls, name: str, component_class: Type) -> None:
        """
        Registra un componente.
        
        Args:
            name: Nombre del componente
            component_class: Clase del componente
            
        Raises:
            TypeError: Si la clase no implementa la interfaz requerida
        """
        interface_type = cls._get_interface_type()
        if not issubclass(component_class, interface_type):
            raise TypeError(f"{component_class} must implement {interface_type.__name__}")
        cls._registry.register(name, component_class)
    
    @classmethod
    def create(
        cls,
        component_type: str,
        config: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """
        Crea una instancia del componente.
        
        Args:
            component_type: Tipo de componente
            config: Configuración (opcional)
            **kwargs: Parámetros adicionales
            
        Returns:
            Instancia del componente
            
        Raises:
            AudioConfigurationError: Si el tipo no es soportado
            AudioSeparationError: Si la creación falla
        """
        component_type = component_type.lower()
        
        # Cargar dinámicamente si no está registrado
        if not cls._registry.is_registered(component_type):
            try:
                component_class = cls._load_component(component_type)
                cls.register(component_type, component_class)
            except AudioConfigurationError:
                raise
            except Exception as e:
                raise AudioConfigurationError(
                    f"Failed to load {cls._get_component_name()} '{component_type}': {e}"
                ) from e
        else:
            component_class = cls._registry.get(component_type)
        
        # Crear configuración si no se proporciona
        if config is None:
            config_type = cls._get_default_config_type()
            config_kwargs = cls._get_default_config_kwargs(component_type)
            config = config_type(**config_kwargs)
        
        # Crear y retornar instancia
        try:
            return component_class(config=config, **kwargs)
        except Exception as e:
            raise AudioSeparationError(
                f"Failed to create {cls._get_component_name()} '{component_type}': {e}",
                component=cls.__name__
            ) from e
    
    @classmethod
    def _get_component_name(cls) -> str:
        """Obtiene el nombre del tipo de componente para mensajes de error."""
        return cls.__name__.replace("Factory", "").lower()
    
    # ════════════════════════════════════════════════════════════════════════════
    # MÉTODOS ABSTRACTOS (implementar en subclases)
    # ════════════════════════════════════════════════════════════════════════════
    
    @classmethod
    @abstractmethod
    def _get_interface_type(cls) -> Type:
        """Retorna la interfaz que deben implementar los componentes."""
        pass
    
    @classmethod
    @abstractmethod
    def _get_default_config_type(cls) -> Type:
        """Retorna el tipo de configuración por defecto."""
        pass
    
    @classmethod
    @abstractmethod
    def _load_component(cls, component_type: str) -> Type:
        """Carga un componente dinámicamente."""
        pass
    
    @classmethod
    def _get_default_config_kwargs(cls, component_type: str) -> Dict[str, Any]:
        """Obtiene kwargs por defecto para la configuración."""
        return {}


class AudioSeparatorFactory(BaseFactory):
    """
    Factory para crear separadores de audio.
    
    Refactorizado para usar BaseFactory, ComponentRegistry, ComponentLoader, y SeparatorDetector.
    Single Responsibility: Create separator instances (orchestrates helpers).
    """
    
    # Shared instances (class-level for efficiency)
    _detector = SeparatorDetector()
    
    @classmethod
    def _get_interface_type(cls) -> Type:
        """Retorna la interfaz que deben implementar los separadores."""
        return IAudioSeparator
    
    @classmethod
    def _get_default_config_type(cls) -> Type:
        """Retorna el tipo de configuración por defecto para separadores."""
        return SeparationConfig
    
    @classmethod
    def _load_component(cls, component_type: str) -> Type:
        """Carga un separador dinámicamente."""
        return cls._loader.load_separator(component_type)
    
    @classmethod
    def _get_default_config_kwargs(cls, component_type: str) -> Dict[str, Any]:
        """Obtiene kwargs por defecto para SeparationConfig."""
        return {"model_type": component_type}
    
    @classmethod
    def create(
        cls,
        separator_type: str = "auto",
        config: Optional[SeparationConfig] = None,
        **kwargs
    ) -> IAudioSeparator:
        """
        Crea un separador de audio.
        
        Args:
            separator_type: Tipo de separador ("spleeter", "demucs", "lalal", "auto")
            config: Configuración del separador
            **kwargs: Parámetros adicionales
        
        Returns:
            Instancia del separador
        
        Raises:
            AudioConfigurationError: Si el tipo no es soportado
            AudioSeparationError: Si la creación falla
        """
        # Auto-detección
        if separator_type.lower() == "auto":
            separator_type = cls._detector.detect_best()
        
        return super().create(separator_type, config, **kwargs)
    
    @classmethod
    def list_available(cls) -> list[str]:
        """
        Lista los separadores disponibles.
        
        Returns:
            Lista de nombres de separadores disponibles
        """
        return cls._detector.list_available()


class AudioMixerFactory(BaseFactory):
    """
    Factory para crear mezcladores de audio.
    
    Refactorizado para usar BaseFactory, ComponentRegistry y ComponentLoader.
    Single Responsibility: Create mixer instances (orchestrates helpers).
    """
    
    @classmethod
    def _get_interface_type(cls) -> Type:
        """Retorna la interfaz que deben implementar los mezcladores."""
        return IAudioMixer
    
    @classmethod
    def _get_default_config_type(cls) -> Type:
        """Retorna el tipo de configuración por defecto para mezcladores."""
        return MixingConfig
    
    @classmethod
    def _load_component(cls, component_type: str) -> Type:
        """Carga un mezclador dinámicamente."""
        return cls._loader.load_mixer(component_type)
    
    @classmethod
    def _get_default_config_kwargs(cls, component_type: str) -> Dict[str, Any]:
        """Obtiene kwargs por defecto para MixingConfig."""
        return {"mixer_type": component_type}


class AudioProcessorFactory(BaseFactory):
    """
    Factory para crear procesadores de audio.
    
    Refactorizado para usar BaseFactory, ComponentRegistry y ComponentLoader.
    Single Responsibility: Create processor instances (orchestrates helpers).
    """
    
    @classmethod
    def _get_interface_type(cls) -> Type:
        """Retorna la interfaz que deben implementar los procesadores."""
        return IAudioProcessor
    
    @classmethod
    def _get_default_config_type(cls) -> Type:
        """Retorna el tipo de configuración por defecto para procesadores."""
        return ProcessorConfig
    
    @classmethod
    def _load_component(cls, component_type: str) -> Type:
        """Carga un procesador dinámicamente."""
        return cls._loader.load_processor(component_type)
    
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
    """
    Función de conveniencia para crear un separador.
    
    Args:
        separator_type: Tipo de separador
        config: Configuración
        **kwargs: Parámetros adicionales
    
    Returns:
        Instancia del separador
    """
    return AudioSeparatorFactory.create(separator_type, config, **kwargs)


def create_audio_mixer(
    mixer_type: str = "simple",
    config: Optional[MixingConfig] = None,
    **kwargs
) -> IAudioMixer:
    """
    Función de conveniencia para crear un mezclador.
    
    Args:
        mixer_type: Tipo de mezclador
        config: Configuración
        **kwargs: Parámetros adicionales
    
    Returns:
        Instancia del mezclador
    """
    return AudioMixerFactory.create(mixer_type, config, **kwargs)


def create_audio_processor(
    processor_type: str,
    config: Optional[ProcessorConfig] = None,
    **kwargs
) -> IAudioProcessor:
    """
    Función de conveniencia para crear un procesador.
    
    Args:
        processor_type: Tipo de procesador
        config: Configuración
        **kwargs: Parámetros adicionales
    
    Returns:
        Instancia del procesador
    """
    return AudioProcessorFactory.create(processor_type, config, **kwargs)
