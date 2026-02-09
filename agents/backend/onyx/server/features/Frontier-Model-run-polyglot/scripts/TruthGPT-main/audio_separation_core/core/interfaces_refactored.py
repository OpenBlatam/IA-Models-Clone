"""
Interfaces Refactorizadas - Interfaces simplificadas y más directas.

Cambios principales:
1. Eliminada IAudioComponent redundante (usar BaseComponent)
2. Interfaces más simples y directas
3. Menos métodos abstractos innecesarios
4. Mejor separación de responsabilidades
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from pathlib import Path

from .base_component import BaseComponent


class AudioSeparator(BaseComponent):
    """
    Interfaz para separadores de audio.
    
    Responsabilidad única: Separar audio en componentes.
    
    Simplificado:
    - Métodos esenciales solamente
    - Validación delegada a implementaciones
    - Sin configuración compleja en la interfaz
    """
    
    @abstractmethod
    def separate(
        self,
        input_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        components: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Separa un archivo de audio en componentes.
        
        Args:
            input_path: Ruta al archivo de audio o video
            output_dir: Directorio donde guardar los componentes
            components: Componentes a extraer (None = todos disponibles)
        
        Returns:
            Diccionario con nombre_componente -> ruta_archivo
        
        Raises:
            RuntimeError: Si la separación falla
            FileNotFoundError: Si el archivo no existe
        """
        pass
    
    @abstractmethod
    def get_supported_components(self) -> List[str]:
        """
        Obtiene los componentes soportados.
        
        Returns:
            Lista de nombres de componentes
        """
        pass
    
    def get_supported_formats(self) -> List[str]:
        """
        Obtiene los formatos soportados.
        
        Returns:
            Lista de extensiones soportadas (con punto)
        """
        return [".wav", ".mp3", ".flac", ".m4a", ".mp4", ".avi", ".mov"]


class AudioMixer(BaseComponent):
    """
    Interfaz para mezcladores de audio.
    
    Responsabilidad única: Mezclar múltiples archivos de audio.
    
    Simplificado:
    - Método principal: mix()
    - Efectos opcionales (no requeridos en interfaz)
    - Sin configuración compleja
    """
    
    @abstractmethod
    def mix(
        self,
        audio_files: Dict[str, Union[str, Path]],
        output_path: Union[str, Path],
        volumes: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Mezcla múltiples archivos de audio.
        
        Args:
            audio_files: Diccionario de nombre_componente -> ruta_archivo
            output_path: Ruta del archivo de salida
            volumes: Diccionario de volúmenes por componente (0.0-1.0)
        
        Returns:
            Ruta al archivo mezclado
        
        Raises:
            RuntimeError: Si la mezcla falla
            FileNotFoundError: Si algún archivo no existe
        """
        pass
    
    def get_supported_formats(self) -> List[str]:
        """Obtiene los formatos soportados."""
        return [".wav", ".mp3", ".flac", ".m4a"]


class AudioProcessor(BaseComponent):
    """
    Interfaz para procesadores de audio.
    
    Responsabilidad única: Procesar archivos de audio/video.
    
    Simplificado:
    - Método principal: process()
    - Metadata opcional (no requerido en interfaz base)
    """
    
    @abstractmethod
    def process(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Procesa un archivo de audio o video.
        
        Args:
            input_path: Ruta al archivo de entrada
            output_path: Ruta de salida (opcional)
        
        Returns:
            Ruta al archivo procesado
        
        Raises:
            RuntimeError: Si el procesamiento falla
            FileNotFoundError: Si el archivo no existe
        """
        pass
    
    def get_metadata(
        self,
        input_path: Union[str, Path]
    ) -> Optional[Dict[str, Any]]:
        """
        Obtiene metadatos del archivo (opcional).
        
        Args:
            input_path: Ruta al archivo
        
        Returns:
            Diccionario con metadatos o None si no está disponible
        """
        return None

