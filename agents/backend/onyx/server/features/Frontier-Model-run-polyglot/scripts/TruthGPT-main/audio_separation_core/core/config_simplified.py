"""
Configuración Simplificada - Versión más simple y directa.

Refactorizado para:
- Menos parámetros opcionales no utilizados
- Valores por defecto más sensatos
- Validación más simple
- Menos complejidad
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class AudioConfig:
    """
    Configuración base para procesamiento de audio.
    
    Simplificado: Solo parámetros esenciales.
    """
    sample_rate: int = 44100
    channels: int = 2
    format: str = "wav"
    
    def validate(self) -> None:
        """Valida la configuración."""
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels not in [1, 2]:
            raise ValueError("channels must be 1 (mono) or 2 (stereo)")


@dataclass
class SeparationConfig(AudioConfig):
    """
    Configuración para separación de audio.
    
    Simplificado: Solo parámetros que realmente se usan.
    """
    model_type: str = "spleeter"  # "spleeter", "demucs", "lalal"
    components: List[str] = field(default_factory=lambda: ["vocals", "accompaniment"])
    use_gpu: bool = True
    model_path: Optional[str] = None  # Para modelos personalizados
    
    def validate(self) -> None:
        """Valida la configuración."""
        super().validate()
        if self.model_type not in ["spleeter", "demucs", "lalal"]:
            raise ValueError(f"Unsupported model_type: {self.model_type}")


@dataclass
class MixingConfig(AudioConfig):
    """
    Configuración para mezcla de audio.
    
    Simplificado: Solo parámetros esenciales.
    """
    default_volume: float = 0.8  # Volumen por defecto (0.0-1.0)
    normalize_output: bool = True
    fade_in: float = 0.0  # Segundos de fade in
    fade_out: float = 0.0  # Segundos de fade out
    
    def validate(self) -> None:
        """Valida la configuración."""
        super().validate()
        if not 0.0 <= self.default_volume <= 1.0:
            raise ValueError("default_volume must be between 0.0 and 1.0")
        if self.fade_in < 0 or self.fade_out < 0:
            raise ValueError("fade_in and fade_out must be non-negative")


@dataclass
class ProcessorConfig(AudioConfig):
    """
    Configuración para procesadores de audio.
    
    Simplificado: Solo parámetros esenciales.
    """
    output_format: str = "wav"
    extract_metadata: bool = True
    
    def validate(self) -> None:
        """Valida la configuración."""
        super().validate()

