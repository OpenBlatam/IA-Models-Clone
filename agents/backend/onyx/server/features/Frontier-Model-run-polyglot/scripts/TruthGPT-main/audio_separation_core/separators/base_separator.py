"""
Base Separator - Implementación base para separadores de audio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from abc import abstractmethod

from ..core.interfaces import IAudioSeparator
from ..core.base_component import BaseComponent
from ..core.exceptions import AudioSeparationError, AudioFormatError, AudioIOError
from ..core.config import SeparationConfig
from ..core.validators import validate_path, validate_format, validate_components, validate_output_dir


class BaseSeparator(BaseComponent, IAudioSeparator):
    """
    Clase base abstracta para separadores de audio.
    
    Proporciona implementaciones comunes y define métodos abstractos
    que deben ser implementados por separadores específicos.
    """
    
    def __init__(
        self,
        config: Optional[SeparationConfig] = None,
        **kwargs
    ):
        """
        Inicializa el separador base.
        
        Args:
            config: Configuración del separador
            **kwargs: Parámetros adicionales
        """
        super().__init__()
        self._config = config or SeparationConfig()
        self._config.validate()
        self._model = None
    
    @property
    def config(self) -> SeparationConfig:
        """Configuración del separador."""
        return self._config
    
    def _do_initialize(self, **kwargs) -> None:
        """
        Implementación específica de inicialización del separador.
        
        Args:
            **kwargs: Parámetros adicionales pasados desde initialize()
        """
        self._model = self._load_model(**kwargs)
    
    def _do_cleanup(self) -> None:
        """Limpia los recursos específicos del separador."""
        if self._model is not None:
            try:
                self._cleanup_model()
            except Exception:
                pass
            finally:
                self._model = None
    
    def separate(
        self,
        input_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        components: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, str]:
        """
        Separa un archivo de audio en componentes.
        
        Args:
            input_path: Ruta al archivo de audio o video
            output_dir: Directorio de salida
            components: Componentes a extraer
            **kwargs: Parámetros adicionales
        
        Returns:
            Diccionario con rutas a los componentes separados
        """
        self._ensure_ready()
        
        # Validar y normalizar entrada usando validators centralizados
        input_path = validate_path(input_path, must_exist=True, must_be_file=True)
        
        # Validar formato usando validators centralizados
        validate_format(input_path, self.get_supported_formats(), self.name)
        
        # Determinar componentes
        if components is None:
            components = self._config.components
        
        # Validar componentes usando validators centralizados
        supported = self.get_supported_components()
        validate_components(components, supported, self.name)
        
        # Determinar y validar directorio de salida
        if output_dir is None:
            output_dir = input_path.parent / f"{input_path.stem}_separated"
        output_dir = validate_output_dir(output_dir, create=True)
        
        try:
            # Realizar separación
            results = self._perform_separation(
                input_path,
                output_dir,
                components,
                **kwargs
            )
            
            # Validar resultados usando validators centralizados
            for component, path in results.items():
                validate_path(path, must_exist=True, must_be_file=True)
            
            return results
        except Exception as e:
            self._handle_error(
                error=e,
                error_class=AudioSeparationError,
                operation="Separation",
                component=self.name
            )
    
    def get_supported_components(self) -> List[str]:
        """
        Obtiene los componentes soportados.
        
        Returns:
            Lista de componentes soportados
        """
        return self._get_default_components()
    
    # Formatos soportados por defecto
    DEFAULT_SUPPORTED_FORMATS = [".wav", ".mp3", ".flac", ".m4a", ".mp4", ".avi", ".mov"]
    
    def get_supported_formats(self) -> List[str]:
        """
        Obtiene los formatos soportados.
        
        Returns:
            Lista de extensiones soportadas
        """
        return self.DEFAULT_SUPPORTED_FORMATS
    
    def estimate_separation_time(
        self,
        input_path: Union[str, Path],
        components: Optional[List[str]] = None
    ) -> float:
        """
        Estima el tiempo de separación.
        
        Args:
            input_path: Ruta al archivo
            components: Componentes a separar
        
        Returns:
            Tiempo estimado en segundos
        """
        # Estimación simple basada en duración del archivo
        # Esto debería ser sobrescrito por implementaciones específicas
        try:
            from ..utils.audio_utils import get_audio_duration
            duration = get_audio_duration(input_path)
            # Estimación: 1x duración para separación básica
            return duration * 1.0
        except Exception:
            # Fallback: estimación conservadora
            return 60.0
    
    # ════════════════════════════════════════════════════════════════════════════
    # MÉTODOS ABSTRACTOS (deben ser implementados por subclases)
    # ════════════════════════════════════════════════════════════════════════════
    
    @abstractmethod
    def _load_model(self, **kwargs):
        """
        Carga el modelo de separación.
        
        Args:
            **kwargs: Parámetros adicionales
        
        Returns:
            Modelo cargado
        """
        pass
    
    @abstractmethod
    def _cleanup_model(self) -> None:
        """Limpia el modelo."""
        pass
    
    @abstractmethod
    def _perform_separation(
        self,
        input_path: Path,
        output_dir: Path,
        components: List[str],
        **kwargs
    ) -> Dict[str, str]:
        """
        Realiza la separación.
        
        Args:
            input_path: Ruta al archivo de entrada
            output_dir: Directorio de salida
            components: Componentes a separar
            **kwargs: Parámetros adicionales
        
        Returns:
            Diccionario con rutas a los componentes separados
        """
        pass
    
    @abstractmethod
    def _get_default_components(self) -> List[str]:
        """
        Obtiene los componentes por defecto.
        
        Returns:
            Lista de componentes por defecto
        """
        pass
    
    # ════════════════════════════════════════════════════════════════════════════
    # MÉTODOS AUXILIARES
    # ════════════════════════════════════════════════════════════════════════════
    
    
    def _get_metrics(self) -> Dict[str, Any]:
        """
        Obtiene métricas del separador.
        
        Returns:
            Diccionario con métricas
        """
        return {
            "model_loaded": self._model is not None,
            "config": {
                "model_type": self._config.model_type,
                "use_gpu": self._config.use_gpu,
            }
        }
    
    def _ensure_ready(self) -> None:
        """
        Asegura que el separador esté listo.
        
        Raises:
            AudioSeparationError: Si el separador no está listo
        """
        # Usa el método base con el tipo de excepción específico del dominio
        super()._ensure_ready(exception_type=AudioSeparationError)




