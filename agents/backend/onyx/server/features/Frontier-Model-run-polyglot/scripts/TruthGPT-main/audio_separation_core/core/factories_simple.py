"""
Factories Simplificadas - Funciones simples en lugar de clases complejas.

Refactorizado para:
- Eliminar registro complejo
- Sin auto-detección "mágica"
- Funciones directas y claras
- Más fácil de mantener
"""

from __future__ import annotations

from typing import Optional, Union
from pathlib import Path

from .interfaces import IAudioSeparator, IAudioMixer, IAudioProcessor
from .exceptions import AudioConfigurationError


def create_audio_separator(
    separator_type: str,
    config: Optional[Any] = None,
    **kwargs
) -> IAudioSeparator:
    """
    Crea un separador de audio.
    
    Args:
        separator_type: Tipo de separador ("spleeter", "demucs", "lalal")
        config: Configuración (opcional, se puede pasar como kwargs)
        **kwargs: Parámetros del separador
    
    Returns:
        Instancia del separador
    
    Raises:
        AudioConfigurationError: Si el tipo no es soportado
    
    Examples:
        >>> separator = create_audio_separator("spleeter", use_gpu=True)
        >>> separator = create_audio_separator("demucs", components=["vocals", "drums"])
    """
    separator_type = separator_type.lower()
    
    try:
        if separator_type == "spleeter":
            from ..separators.spleeter_separator import SpleeterSeparator
            return SpleeterSeparator(config=config, **kwargs)
        
        elif separator_type == "demucs":
            from ..separators.demucs_separator import DemucsSeparator
            return DemucsSeparator(config=config, **kwargs)
        
        elif separator_type == "lalal":
            from ..separators.lalal_separator import LALALSeparator
            return LALALSeparator(config=config, **kwargs)
        
        else:
            available = ["spleeter", "demucs", "lalal"]
            raise AudioConfigurationError(
                f"Unknown separator type: {separator_type}. "
                f"Available: {available}"
            )
    except ImportError as e:
        raise AudioConfigurationError(
            f"Failed to import separator '{separator_type}': {e}. "
            f"Make sure the required dependencies are installed."
        ) from e


def create_audio_mixer(
    mixer_type: str = "simple",
    config: Optional[Any] = None,
    **kwargs
) -> IAudioMixer:
    """
    Crea un mezclador de audio.
    
    Args:
        mixer_type: Tipo de mezclador ("simple", "advanced")
        config: Configuración (opcional)
        **kwargs: Parámetros del mezclador
    
    Returns:
        Instancia del mezclador
    
    Examples:
        >>> mixer = create_audio_mixer("simple")
        >>> mixer = create_audio_mixer("advanced", default_volume=0.9)
    """
    mixer_type = mixer_type.lower()
    
    try:
        if mixer_type == "simple":
            from ..mixers.simple_mixer import SimpleMixer
            return SimpleMixer(config=config, **kwargs)
        
        elif mixer_type == "advanced":
            from ..mixers.advanced_mixer import AdvancedMixer
            return AdvancedMixer(config=config, **kwargs)
        
        else:
            available = ["simple", "advanced"]
            raise AudioConfigurationError(
                f"Unknown mixer type: {mixer_type}. "
                f"Available: {available}"
            )
    except ImportError as e:
        raise AudioConfigurationError(
            f"Failed to import mixer '{mixer_type}': {e}"
        ) from e


def create_audio_processor(
    processor_type: str,
    config: Optional[Any] = None,
    **kwargs
) -> IAudioProcessor:
    """
    Crea un procesador de audio.
    
    Args:
        processor_type: Tipo de procesador ("extractor", "converter", "enhancer")
        config: Configuración (opcional)
        **kwargs: Parámetros del procesador
    
    Returns:
        Instancia del procesador
    
    Examples:
        >>> extractor = create_audio_processor("extractor")
        >>> converter = create_audio_processor("converter", output_format="mp3")
    """
    processor_type = processor_type.lower()
    
    try:
        if processor_type == "extractor":
            from ..processors.video_extractor import VideoAudioExtractor
            return VideoAudioExtractor(config=config, **kwargs)
        
        elif processor_type == "converter":
            # TODO: Implementar cuando esté disponible
            raise AudioConfigurationError("AudioFormatConverter not yet implemented")
        
        elif processor_type == "enhancer":
            # TODO: Implementar cuando esté disponible
            raise AudioConfigurationError("AudioEnhancer not yet implemented")
        
        else:
            available = ["extractor"]  # Solo extractor implementado por ahora
            raise AudioConfigurationError(
                f"Unknown processor type: {processor_type}. "
                f"Available: {available}"
            )
    except ImportError as e:
        raise AudioConfigurationError(
            f"Failed to import processor '{processor_type}': {e}"
        ) from e


def list_available_separators() -> list[str]:
    """
    Lista los separadores disponibles en el sistema.
    
    Returns:
        Lista de nombres de separadores disponibles
    """
    available = []
    
    for name in ["spleeter", "demucs", "lalal"]:
        try:
            if name == "spleeter":
                import spleeter
            elif name == "demucs":
                import demucs
            elif name == "lalal":
                # LALAL puede requerir API key, pero el módulo debería estar disponible
                pass
            available.append(name)
        except ImportError:
            pass
    
    return available

