"""
Configuración Core - Clases de configuración para Audio Separation Core.

Define las clases de configuración para todos los componentes del sistema.
Refactorizado para:
- Usar validadores centralizados
- Consistencia en manejo de errores
- Eliminar duplicación en validación
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from .validators import (
    validate_sample_rate,
    validate_channels,
    validate_bit_depth,
    validate_positive_integer,
    validate_range,
    validate_non_negative,
    validate_choice,
    validate_volume,
)

from .exceptions import AudioConfigurationError


@dataclass
class AudioConfig:
    """
    Configuración base para procesamiento de audio.
    """
    sample_rate: int = 44100
    channels: int = 2
    bit_depth: int = 16
    format: str = "wav"
    normalize: bool = True
    remove_silence: bool = False
    silence_threshold: float = -40.0  # dB
    
    def validate(self) -> None:
        """
        Valida la configuración usando validators centralizados.
        
        Raises:
            ValueError: Si algún parámetro no es válido
        """
        try:
            validate_sample_rate(self.sample_rate)
            validate_channels(self.channels)
            validate_bit_depth(self.bit_depth)
        except ValueError as e:
            raise AudioConfigurationError(str(e)) from e


@dataclass
class SeparationConfig(AudioConfig):
    """
    Configuración para separación de audio.
    """
    model_type: str = "spleeter"  # "spleeter", "demucs", "lalal", "auto"
    model_path: Optional[str] = None
    components: List[str] = field(default_factory=lambda: ["vocals", "accompaniment"])
    use_gpu: bool = True
    batch_size: int = 1
    overlap: float = 0.25  # Overlap entre chunks para mejor calidad
    segment_length: Optional[int] = None  # Longitud de segmentos en segundos
    post_process: bool = True  # Aplicar post-procesamiento para mejorar calidad
    
    # Parámetros específicos por modelo
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """
        Valida la configuración usando validators centralizados.
        
        Raises:
            AudioConfigurationError: Si algún parámetro no es válido
        """
        super().validate()
        try:
            validate_choice(self.model_type, ["spleeter", "demucs", "lalal", "auto"], "model_type")
            validate_range(self.overlap, 0.0, 1.0, "overlap", inclusive=False)
            validate_positive_integer(self.batch_size, "batch_size")
        except ValueError as e:
            raise AudioConfigurationError(str(e)) from e


@dataclass
class MixingConfig(AudioConfig):
    """
    Configuración para mezcla de audio.
    """
    mixer_type: str = "simple"  # "simple", "advanced"
    default_volume: float = 0.8  # Volumen por defecto (0.0-1.0)
    normalize_output: bool = True
    fade_in: float = 0.0  # Segundos de fade in
    fade_out: float = 0.0  # Segundos de fade out
    
    # Efectos opcionales
    apply_reverb: bool = False
    apply_eq: bool = False
    apply_compressor: bool = False
    
    # Parámetros de efectos
    reverb_params: Dict[str, Any] = field(default_factory=dict)
    eq_params: Dict[str, Any] = field(default_factory=dict)
    compressor_params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """
        Valida la configuración usando validators centralizados.
        
        Raises:
            AudioConfigurationError: Si algún parámetro no es válido
        """
        super().validate()
        try:
            validate_volume(self.default_volume, "default_volume")
            validate_non_negative(self.fade_in, "fade_in")
            validate_non_negative(self.fade_out, "fade_out")
        except (ValueError, AudioConfigurationError, Exception) as e:
            # AudioValidationError from validate_volume is converted to AudioConfigurationError
            raise AudioConfigurationError(str(e)) from e


@dataclass
class ProcessorConfig(AudioConfig):
    """
    Configuración para procesadores de audio.
    """
    processor_type: str = "extractor"  # "extractor", "converter", "enhancer"
    output_format: str = "wav"
    quality: str = "high"  # "low", "medium", "high"
    extract_metadata: bool = True
    remove_original: bool = False
    
    # Parámetros específicos
    processor_params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """
        Valida la configuración usando validators centralizados.
        
        Raises:
            AudioConfigurationError: Si algún parámetro no es válido
        """
        super().validate()
        try:
            validate_choice(self.quality, ["low", "medium", "high"], "quality")
        except ValueError as e:
            raise AudioConfigurationError(str(e)) from e




