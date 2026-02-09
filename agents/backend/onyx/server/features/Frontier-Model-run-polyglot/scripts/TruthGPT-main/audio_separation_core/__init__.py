"""
Audio Separation Core - Sistema de Separación y Mezcla de Audio con IA

Este módulo proporciona un framework completo para:
- Separar audio de videos en componentes (voces, música, efectos, etc.)
- Mezclar componentes de audio de manera inteligente
- Procesar audio usando modelos de IA optimizados
- Gestionar el ciclo de vida completo del procesamiento de audio

Arquitectura basada en optimization_core para alto rendimiento y modularidad.
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "TruthGPT Team"

# ════════════════════════════════════════════════════════════════════════════════
# CORE COMPONENTS (Eager Import)
# ════════════════════════════════════════════════════════════════════════════════

from .core.interfaces import (
    IAudioComponent,
    IAudioSeparator,
    IAudioMixer,
    IAudioProcessor,
)

from .core.exceptions import (
    AudioSeparationError,
    AudioProcessingError,
    AudioFormatError,
    AudioModelError,
)

from .core.config import (
    AudioConfig,
    SeparationConfig,
    MixingConfig,
    ProcessorConfig,
)

# ════════════════════════════════════════════════════════════════════════════════
# FACTORIES (Eager Import)
# ════════════════════════════════════════════════════════════════════════════════

from .core.factories import (
    create_audio_separator,
    create_audio_mixer,
    create_audio_processor,
    AudioSeparatorFactory,
    AudioMixerFactory,
    AudioProcessorFactory,
)

# ════════════════════════════════════════════════════════════════════════════════
# LAZY IMPORTS
# ════════════════════════════════════════════════════════════════════════════════

_LAZY_IMPORTS = {
    # Separators
    'separators': '.separators',
    'SpleeterSeparator': '.separators.spleeter_separator',
    'DemucsSeparator': '.separators.demucs_separator',
    'LALALSeparator': '.separators.lalal_separator',
    'BaseSeparator': '.separators.base_separator',
    # Mixers
    'mixers': '.mixers',
    'BaseMixer': '.mixers.base_mixer',
    'SimpleMixer': '.mixers.simple_mixer',
    'AdvancedMixer': '.mixers.advanced_mixer',
    # Processors
    'processors': '.processors',
    'VideoAudioExtractor': '.processors.video_extractor',
    'AudioFormatConverter': '.processors.format_converter',
    'AudioEnhancer': '.processors.audio_enhancer',
    # Utils
    'utils': '.utils',
    'audio_utils': '.utils.audio_utils',
    'format_utils': '.utils.format_utils',
    'validation_utils': '.utils.validation_utils',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for audio_separation_core submodules."""
    if name.startswith('_'):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name in _import_cache:
        return _import_cache[name]
    
    module_path = _LAZY_IMPORTS[name]
    try:
        if name in ['separators', 'mixers', 'processors', 'utils']:
            # Import entire module
            module = __import__(module_path, fromlist=[name], level=1)
        else:
            # Import specific class/function
            module = __import__(module_path, fromlist=[name], level=1)
            if hasattr(module, name):
                _import_cache[name] = getattr(module, name)
                return _import_cache[name]
            else:
                # Return module if class not found
                _import_cache[name] = module
                return module
        
        _import_cache[name] = module
        return module
    except (ImportError, AttributeError) as e:
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'. "
            f"Failed to import: {e}"
        ) from e


# ════════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def separate_audio(
    input_path: str,
    output_dir: str = None,
    separator_type: str = "auto",
    components: list[str] = None,
    **kwargs
) -> dict[str, str]:
    """
    Función de conveniencia para separar audio de un video o archivo de audio.
    
    Args:
        input_path: Ruta al archivo de video o audio
        output_dir: Directorio de salida (opcional)
        separator_type: Tipo de separador ("spleeter", "demucs", "lalal", "auto")
        components: Lista de componentes a separar (["vocals", "accompaniment"], etc.)
        **kwargs: Parámetros adicionales para el separador
    
    Returns:
        Diccionario con rutas a los archivos separados:
        {
            "vocals": "path/to/vocals.wav",
            "accompaniment": "path/to/accompaniment.wav",
            ...
        }
    
    Examples:
        >>> # Separar audio de un video
        >>> results = separate_audio("video.mp4", components=["vocals", "music"])
        >>> print(results["vocals"])
        'output/vocals.wav'
    """
    separator = create_audio_separator(separator_type, **kwargs)
    
    if components is None:
        components = ["vocals", "accompaniment"]
    
    return separator.separate(input_path, output_dir, components=components)


def mix_audio(
    audio_files: dict[str, str],
    output_path: str,
    mixer_type: str = "simple",
    volumes: dict[str, float] = None,
    **kwargs
) -> str:
    """
    Función de conveniencia para mezclar múltiples archivos de audio.
    
    Args:
        audio_files: Diccionario de nombre_componente -> ruta_archivo
        output_path: Ruta del archivo de salida
        mixer_type: Tipo de mezclador ("simple", "advanced")
        volumes: Diccionario de volúmenes por componente (0.0-1.0)
        **kwargs: Parámetros adicionales para el mezclador
    
    Returns:
        Ruta al archivo mezclado
    
    Examples:
        >>> # Mezclar voces y música
        >>> mixed = mix_audio(
        ...     {"vocals": "vocals.wav", "music": "music.wav"},
        ...     "output/mixed.wav",
        ...     volumes={"vocals": 0.8, "music": 0.6}
        ... )
    """
    mixer = create_audio_mixer(mixer_type, **kwargs)
    return mixer.mix(audio_files, output_path, volumes=volumes)


def process_video_audio(
    video_path: str,
    output_dir: str = None,
    separate: bool = True,
    components: list[str] = None,
    **kwargs
) -> dict[str, any]:
    """
    Procesa un video completo: extrae audio, separa componentes y prepara para mezcla.
    
    Args:
        video_path: Ruta al archivo de video
        output_dir: Directorio de salida
        separate: Si True, separa el audio en componentes
        components: Componentes a separar
        **kwargs: Parámetros adicionales
    
    Returns:
        Diccionario con información del procesamiento:
        {
            "audio_path": "path/to/extracted_audio.wav",
            "separated": {
                "vocals": "path/to/vocals.wav",
                ...
            },
            "metadata": {...}
        }
    """
    from .processors.video_extractor import VideoAudioExtractor
    
    extractor = VideoAudioExtractor()
    audio_path = extractor.extract(video_path, output_dir)
    
    result = {
        "audio_path": audio_path,
        "metadata": extractor.get_metadata(video_path),
    }
    
    if separate:
        separated = separate_audio(
            audio_path,
            output_dir,
            components=components,
            **kwargs
        )
        result["separated"] = separated
    
    return result


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Core Interfaces
    "IAudioComponent",
    "IAudioSeparator",
    "IAudioMixer",
    "IAudioProcessor",
    # Exceptions
    "AudioSeparationError",
    "AudioProcessingError",
    "AudioFormatError",
    "AudioModelError",
    # Config
    "AudioConfig",
    "SeparationConfig",
    "MixingConfig",
    "ProcessorConfig",
    # Factories
    "create_audio_separator",
    "create_audio_mixer",
    "create_audio_processor",
    "AudioSeparatorFactory",
    "AudioMixerFactory",
    "AudioProcessorFactory",
    # Convenience Functions
    "separate_audio",
    "mix_audio",
    "process_video_audio",
    # Lazy imports (available via __getattr__)
    "separators",
    "mixers",
    "processors",
    "utils",
]




