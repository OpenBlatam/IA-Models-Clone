"""
Base Mixer - Implementación base para mezcladores de audio.

Refactorizado para:
- Simplificar validación
- Consolidar métodos helper
- Reducir complejidad
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union, Any, List

from ..core.interfaces import IAudioMixer
from ..core.base_component import BaseComponent
from ..core.exceptions import AudioProcessingError, AudioIOError
from ..core.config import MixingConfig
from ..core.validators import validate_path, validate_output_path, validate_volume


class BaseMixer(BaseComponent, IAudioMixer):
    """
    Clase base abstracta para mezcladores de audio.
    
    Responsabilidades:
    - Validación común de entrada
    - Gestión de volúmenes
    - Delegación a implementación específica
    """
    
    # Formatos soportados por defecto
    DEFAULT_SUPPORTED_FORMATS = [".wav", ".mp3", ".flac", ".m4a"]
    
    def __init__(
        self,
        config: Optional[MixingConfig] = None,
        **kwargs
    ):
        """
        Inicializa el mezclador base.
        
        Args:
            config: Configuración del mezclador (opcional)
            **kwargs: Parámetros adicionales
        """
        super().__init__()
        self._config = config or MixingConfig()
        if config:
            self._config.validate()
    
    @property
    def config(self) -> MixingConfig:
        """Configuración del mezclador."""
        return self._config
    
    def mix(
        self,
        audio_files: Dict[str, Union[str, Path]],
        output_path: Union[str, Path],
        volumes: Optional[Dict[str, float]] = None,
        effects: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> str:
        """
        Mezcla múltiples archivos de audio.
        
        Args:
            audio_files: Diccionario de nombre_componente -> ruta_archivo
            output_path: Ruta del archivo de salida
            volumes: Diccionario de volúmenes por componente (0.0-1.0)
            effects: Diccionario de efectos por componente
            **kwargs: Parámetros adicionales
        
        Returns:
            Ruta al archivo mezclado
        """
        self._ensure_ready()
        
        # Validar archivos de entrada
        validated_files = self._validate_audio_files(audio_files)
        
        # Validar y normalizar volúmenes
        normalized_volumes = self._normalize_volumes(volumes, list(validated_files.keys()))
        
        # Preparar ruta de salida
        output_path = self._prepare_output_path(output_path)
        
        try:
            # Realizar mezcla (implementación específica)
            result_path = self._perform_mixing(
                validated_files,
                output_path,
                normalized_volumes,
                effects,
                **kwargs
            )
            
            # Validar que el archivo se creó usando validators centralizados
            validate_path(result_path, must_exist=True, must_be_file=True)
            
            return str(result_path)
        except Exception as e:
            self._handle_error(
                error=e,
                error_class=AudioProcessingError,
                operation="Mixing",
                component=self.name
            )
    
    def get_supported_formats(self) -> List[str]:
        """Obtiene los formatos soportados."""
        return self.DEFAULT_SUPPORTED_FORMATS
    
    def apply_effect(
        self,
        audio_path: Union[str, Path],
        effect_type: str,
        effect_params: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Aplica un efecto a un archivo de audio.
        
        Args:
            audio_path: Ruta al archivo de audio
            effect_type: Tipo de efecto
            effect_params: Parámetros del efecto
            output_path: Ruta de salida (opcional)
        
        Returns:
            Ruta al archivo procesado
        """
        audio_path = self._validate_input_file(audio_path)
        
        if output_path is None:
            output_path = audio_path.parent / f"{audio_path.stem}_{effect_type}{audio_path.suffix}"
        output_path = self._prepare_output_path(output_path)
        
        try:
            return self._apply_effect_impl(audio_path, effect_type, effect_params, output_path)
        except Exception as e:
            self._handle_error(
                error=e,
                error_class=AudioProcessingError,
                operation=f"Applying effect '{effect_type}'",
                component=self.name
            )
    
    # ════════════════════════════════════════════════════════════════════════════
    # MÉTODOS ABSTRACTOS
    # ════════════════════════════════════════════════════════════════════════════
    
    @abstractmethod
    def _perform_mixing(
        self,
        audio_files: Dict[str, Path],
        output_path: Path,
        volumes: Dict[str, float],
        effects: Optional[Dict[str, Dict[str, Any]]],
        **kwargs
    ) -> str:
        """Realiza la mezcla (implementación específica)."""
        pass
    
    def _apply_effect_impl(
        self,
        audio_path: Path,
        effect_type: str,
        effect_params: Dict[str, Any],
        output_path: Path
    ) -> str:
        """Implementación de aplicación de efectos (opcional)."""
        raise NotImplementedError(f"Effect '{effect_type}' not supported by {self.name}")
    
    # ════════════════════════════════════════════════════════════════════════════
    # MÉTODOS HELPER (consolidados)
    # ════════════════════════════════════════════════════════════════════════════
    
    def _validate_audio_files(self, audio_files: Dict[str, Union[str, Path]]) -> Dict[str, Path]:
        """
        Valida que todos los archivos de audio existan.
        
        Args:
            audio_files: Diccionario de archivos a validar
        
        Returns:
            Diccionario con Paths validados
        
        Raises:
            AudioIOError: Si algún archivo no existe
        """
        validated = {}
        for name, path in audio_files.items():
            validated[name] = self._validate_input_file(path)
        return validated
    
    def _validate_input_file(self, path: Union[str, Path]) -> Path:
        """
        Valida que un archivo exista usando validators centralizados.
        
        Args:
            path: Ruta a validar
        
        Returns:
            Path validado
        
        Raises:
            AudioIOError: Si el archivo no existe
        """
        return validate_path(path, must_exist=True, must_be_file=True)
    
    def _normalize_volumes(
        self,
        volumes: Optional[Dict[str, float]],
        component_names: List[str]
    ) -> Dict[str, float]:
        """
        Normaliza volúmenes aplicando valores por defecto donde falten.
        
        Args:
            volumes: Diccionario de volúmenes (puede ser None o incompleto)
            component_names: Lista de nombres de componentes
        
        Returns:
            Diccionario completo de volúmenes normalizados
        
        Raises:
            AudioProcessingError: Si algún volumen está fuera de rango
        """
        volumes = volumes or {}
        normalized = {}
        default_volume = self._config.default_volume
        
        for name in component_names:
            volume = volumes.get(name, default_volume)
            
            # Validar rango usando validators centralizados
            try:
                validate_volume(volume, name=f"Volume for '{name}'")
            except Exception as e:
                raise AudioProcessingError(
                    str(e),
                    component=self.name
                ) from e
            
            normalized[name] = float(volume)
        
        return normalized
    
    def _prepare_output_path(self, output_path: Union[str, Path]) -> Path:
        """
        Prepara la ruta de salida usando validators centralizados.
        
        Args:
            output_path: Ruta de salida
        
        Returns:
            Path validado y preparado
        """
        return validate_output_path(output_path, create_parent=True)
    
    def _ensure_ready(self) -> None:
        """
        Asegura que el mezclador esté listo.
        
        Raises:
            AudioProcessingError: Si el mezclador no está listo
        """
        # Usa el método base con el tipo de excepción específico del dominio
        super()._ensure_ready(exception_type=AudioProcessingError)
    
