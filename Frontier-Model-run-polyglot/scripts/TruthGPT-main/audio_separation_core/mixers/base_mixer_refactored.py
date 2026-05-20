"""
Base Mixer Refactorizado - Implementación base simplificada.

Mejoras:
1. Eliminada duplicación con BaseSeparator
2. Validación consolidada
3. Código más directo
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

from ..core.base_component import BaseComponent
from ..core.interfaces_refactored import AudioMixer
from ..utils.validation_utils import validate_audio_path, validate_volume


class BaseMixer(AudioMixer):
    """
    Clase base para mezcladores de audio.
    
    Responsabilidades:
    - Validación común de entrada
    - Gestión de volúmenes
    - Delegación a implementación específica
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        default_volume: float = 0.8,
        normalize_output: bool = True,
    ):
        """
        Inicializa el mezclador base.
        
        Args:
            name: Nombre del mezclador
            default_volume: Volumen por defecto (0.0-1.0)
            normalize_output: Si normalizar la salida
        """
        super().__init__(name)
        self.default_volume = default_volume
        self.normalize_output = normalize_output
    
    def mix(
        self,
        audio_files: Dict[str, Union[str, Path]],
        output_path: Union[str, Path],
        volumes: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Mezcla múltiples archivos de audio.
        
        Implementa validación común y delega a _perform_mixing().
        """
        self._ensure_ready()
        
        # Validar archivos de entrada
        validated_files = {}
        for name, path in audio_files.items():
            validated_files[name] = validate_audio_path(path)
        
        # Validar y normalizar volúmenes
        if volumes is None:
            volumes = {}
        
        normalized_volumes = {}
        for name, volume in volumes.items():
            validate_volume(volume, name=f"volume for {name}")
            normalized_volumes[name] = volume
        
        # Aplicar volumen por defecto a componentes sin volumen especificado
        for name in validated_files:
            if name not in normalized_volumes:
                normalized_volumes[name] = self.default_volume
        
        # Validar ruta de salida
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Realizar mezcla
        try:
            result_path = self._perform_mixing(
                validated_files,
                output_path,
                normalized_volumes
            )
            
            if not Path(result_path).exists():
                raise FileNotFoundError(f"Mixed file not created: {result_path}")
            
            return str(result_path)
        except Exception as e:
            self._last_error = str(e)
            raise RuntimeError(f"Mixing failed: {e}") from e
    
    # ════════════════════════════════════════════════════════════════════════════
    # MÉTODOS ABSTRACTOS (implementar en subclases)
    # ════════════════════════════════════════════════════════════════════════════
    
    @abstractmethod
    def _perform_mixing(
        self,
        audio_files: Dict[str, Path],
        output_path: Path,
        volumes: Dict[str, float],
    ) -> str:
        """
        Realiza la mezcla (implementación específica).
        
        Args:
            audio_files: Diccionario de archivos de audio validados
            output_path: Ruta de salida
            volumes: Volúmenes normalizados por componente
        
        Returns:
            Ruta al archivo mezclado
        """
        pass

