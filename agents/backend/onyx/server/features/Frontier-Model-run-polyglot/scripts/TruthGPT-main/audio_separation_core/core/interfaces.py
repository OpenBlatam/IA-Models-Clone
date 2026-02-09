"""
Interfaces Core - Definición de contratos para Audio Separation Core.

Este módulo define todas las interfaces abstractas que deben implementar
los componentes del sistema de separación y mezcla de audio.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


class IAudioComponent(ABC):
    """
    Interfaz base para todos los componentes de audio.
    
    Define el contrato básico que deben cumplir todos los componentes:
    - Inicialización y limpieza
    - Estado y salud
    - Metadatos
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Nombre del componente.
        
        Returns:
            Nombre del componente (ej: "SpleeterSeparator", "SimpleMixer")
        """
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """
        Versión del componente.
        
        Returns:
            Versión (ej: "1.0.0")
        """
        pass
    
    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """
        Inicializa el componente.
        
        Args:
            **kwargs: Parámetros de inicialización específicos del componente
        
        Returns:
            True si la inicialización fue exitosa, False en caso contrario
        
        Raises:
            AudioSeparationError: Si la inicialización falla
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Limpia los recursos del componente.
        
        Debe ser idempotente (seguro llamar múltiples veces).
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado del componente.
        
        Returns:
            Diccionario con información de estado:
            {
                "name": str,
                "version": str,
                "initialized": bool,
                "ready": bool,
                "health": str,  # "healthy", "degraded", "unhealthy"
                "metrics": Dict[str, Any],
                "last_error": Optional[str],
                "uptime_seconds": float
            }
        """
        pass


class IAudioSeparator(IAudioComponent):
    """
    Interfaz para separadores de audio.
    
    Un separador de audio toma un archivo de audio/video y lo separa
    en diferentes componentes (voces, música, efectos, etc.).
    """
    
    @abstractmethod
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
            output_dir: Directorio donde guardar los componentes separados
            components: Lista de componentes a extraer (ej: ["vocals", "accompaniment"])
            **kwargs: Parámetros adicionales específicos del separador
        
        Returns:
            Diccionario con nombre_componente -> ruta_archivo:
            {
                "vocals": "path/to/vocals.wav",
                "accompaniment": "path/to/accompaniment.wav",
                ...
            }
        
        Raises:
            AudioSeparationError: Si la separación falla
            AudioFormatError: Si el formato de entrada no es soportado
            AudioIOError: Si hay problemas de lectura/escritura
        """
        pass
    
    @abstractmethod
    def get_supported_components(self) -> List[str]:
        """
        Obtiene la lista de componentes soportados por este separador.
        
        Returns:
            Lista de nombres de componentes (ej: ["vocals", "drums", "bass", "other"])
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Obtiene los formatos de entrada soportados.
        
        Returns:
            Lista de extensiones soportadas (ej: [".wav", ".mp3", ".mp4", ".m4a"])
        """
        pass
    
    @abstractmethod
    def estimate_separation_time(
        self,
        input_path: Union[str, Path],
        components: Optional[List[str]] = None
    ) -> float:
        """
        Estima el tiempo que tomará separar el audio.
        
        Args:
            input_path: Ruta al archivo de entrada
            components: Componentes a separar
        
        Returns:
            Tiempo estimado en segundos
        """
        pass


class IAudioMixer(IAudioComponent):
    """
    Interfaz para mezcladores de audio.
    
    Un mezclador de audio combina múltiples archivos de audio
    en un solo archivo con control de volúmenes, efectos, etc.
    """
    
    @abstractmethod
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
        
        Raises:
            AudioProcessingError: Si la mezcla falla
            AudioFormatError: Si algún formato no es soportado
            AudioIOError: Si hay problemas de lectura/escritura
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Obtiene los formatos de entrada/salida soportados.
        
        Returns:
            Lista de extensiones soportadas
        """
        pass
    
    @abstractmethod
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
            effect_type: Tipo de efecto (ej: "reverb", "eq", "compressor")
            effect_params: Parámetros del efecto
            output_path: Ruta de salida (opcional, si no se proporciona se sobrescribe)
        
        Returns:
            Ruta al archivo procesado
        """
        pass


class IAudioProcessor(IAudioComponent):
    """
    Interfaz para procesadores de audio.
    
    Un procesador de audio realiza operaciones sobre archivos de audio:
    - Extracción de audio de videos
    - Conversión de formatos
    - Mejora de calidad
    - Análisis y metadata
    """
    
    @abstractmethod
    def process(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> str:
        """
        Procesa un archivo de audio o video.
        
        Args:
            input_path: Ruta al archivo de entrada
            output_path: Ruta de salida (opcional)
            **kwargs: Parámetros específicos del procesador
        
        Returns:
            Ruta al archivo procesado
        
        Raises:
            AudioProcessingError: Si el procesamiento falla
            AudioFormatError: Si el formato no es soportado
        """
        pass
    
    @abstractmethod
    def get_metadata(
        self,
        input_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Obtiene metadatos de un archivo de audio o video.
        
        Args:
            input_path: Ruta al archivo
        
        Returns:
            Diccionario con metadatos:
            {
                "duration": float,  # segundos
                "sample_rate": int,
                "channels": int,
                "bit_depth": int,
                "format": str,
                "codec": str,
                "bitrate": int,
                ...
            }
        """
        pass
    
    @abstractmethod
    def validate(
        self,
        input_path: Union[str, Path]
    ) -> bool:
        """
        Valida que un archivo sea procesable.
        
        Args:
            input_path: Ruta al archivo
        
        Returns:
            True si el archivo es válido, False en caso contrario
        """
        pass




