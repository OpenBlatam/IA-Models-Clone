"""
Video Audio Extractor - Extrae audio de archivos de video.

Refactorizado para usar BaseComponent y eliminar duplicación.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union, Any

from ..core.interfaces import IAudioProcessor
from ..core.base_component import BaseComponent
from ..core.exceptions import AudioProcessingError, AudioFormatError, AudioIOError
from ..core.config import ProcessorConfig
from ..core.validators import validate_path, validate_output_path


class VideoAudioExtractor(BaseComponent, IAudioProcessor):
    """
    Extractor de audio de videos.
    
    Extrae pistas de audio de archivos de video usando ffmpeg.
    Refactorizado para usar BaseComponent.
    """
    
    def __init__(
        self,
        config: Optional[ProcessorConfig] = None,
        **kwargs
    ):
        """
        Inicializa el extractor de audio.
        
        Args:
            config: Configuración
            **kwargs: Parámetros adicionales
        """
        super().__init__(name="VideoAudioExtractor")
        self._config = config or ProcessorConfig(processor_type="extractor")
        self._config.validate()
    
    @property
    def config(self) -> ProcessorConfig:
        """Configuración del procesador."""
        return self._config
    
    def _do_initialize(self, **kwargs) -> None:
        """
        Verifica que ffmpeg esté disponible.
        
        Raises:
            AudioProcessingError: Si ffmpeg no está disponible
        """
        self._check_ffmpeg_available()
    
    def _check_ffmpeg_available(self) -> None:
        """Verifica que ffmpeg esté disponible."""
        import subprocess
        
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise AudioProcessingError(
                    "ffmpeg is not available. Please install ffmpeg.",
                    component=self.name
                )
        except FileNotFoundError:
            raise AudioProcessingError(
                "ffmpeg is not installed. Please install ffmpeg.",
                component=self.name
            )
    
    def process(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> str:
        """
        Extrae audio de un video.
        
        Args:
            input_path: Ruta al archivo de video
            output_path: Ruta de salida (opcional)
            **kwargs: Parámetros adicionales
        
        Returns:
            Ruta al archivo de audio extraído
        """
        if not self.is_initialized:
            self.initialize()
        
        if not self.is_ready:
            raise AudioProcessingError(
                f"{self.name} is not ready",
                component=self.name
            )
        
        # Validar y normalizar entrada usando validators centralizados
        input_path = validate_path(input_path, must_exist=True, must_be_file=True)
        
        # Determinar y validar ruta de salida
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_audio.{self._config.output_format}"
        output_path = validate_output_path(output_path, create_parent=True)
        
        try:
            # Construir y ejecutar comando ffmpeg
            self._run_ffmpeg_extraction(input_path, output_path)
            
            # Validar que el archivo se creó usando validators centralizados
            validate_path(output_path, must_exist=True, must_be_file=True)
            return str(output_path)
        except subprocess.TimeoutExpired:
            self._set_error("ffmpeg extraction timed out")
            raise AudioProcessingError(
                "Audio extraction timed out (exceeded 5 minutes)",
                component=self.name
            )
        except Exception as e:
            self._set_error(str(e))
            raise AudioProcessingError(
                f"Audio extraction failed: {e}",
                component=self.name
            ) from e
    
    def extract(
        self,
        video_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> str:
        """
        Función de conveniencia para extraer audio.
        
        Args:
            video_path: Ruta al video
            output_dir: Directorio de salida
            **kwargs: Parámetros adicionales
        
        Returns:
            Ruta al archivo de audio extraído
        """
        if output_dir:
            video_path = Path(video_path)
            output_path = Path(output_dir) / f"{video_path.stem}_audio.{self._config.output_format}"
        else:
            output_path = None
        
        return self.process(video_path, output_path, **kwargs)
    
    def get_metadata(
        self,
        input_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Obtiene metadatos del video/audio.
        
        Args:
            input_path: Ruta al archivo
        
        Returns:
            Diccionario con metadatos
        """
        try:
            input_path = Path(input_path)
            data = self._run_ffprobe(input_path)
            return self._extract_metadata_from_probe(data)
        except Exception as e:
            raise AudioProcessingError(
                f"Failed to get metadata: {e}",
                component=self.name
            ) from e
    
    def validate(
        self,
        input_path: Union[str, Path]
    ) -> bool:
        """
        Valida que el archivo sea procesable.
        
        Args:
            input_path: Ruta al archivo
        
        Returns:
            True si el archivo es válido
        """
        try:
            # Usar validators centralizados para validar entrada
            validate_path(input_path, must_exist=True, must_be_file=True)
            
            # Verificar que tenga stream de audio
            metadata = self.get_metadata(input_path)
            return metadata.get("sample_rate", 0) > 0
        except (AudioIOError, AudioProcessingError, AudioFormatError):
            return False
        except Exception:
            return False
    
    # ════════════════════════════════════════════════════════════════════════════
    # MÉTODOS HELPER (consolidados)
    # ════════════════════════════════════════════════════════════════════════════
    
    def _run_ffmpeg_extraction(
        self,
        input_path: Path,
        output_path: Path
    ) -> None:
        """
        Ejecuta ffmpeg para extraer audio.
        
        Args:
            input_path: Ruta al archivo de video
            output_path: Ruta de salida
        
        Raises:
            AudioProcessingError: Si la extracción falla
        """
        import subprocess
        
        cmd = [
            "ffmpeg",
            "-i", str(input_path),
            "-vn",  # Sin video
            "-acodec", "pcm_s16le",  # Codec de audio
            "-ar", str(self._config.sample_rate),
            "-ac", str(self._config.channels),
            "-y",  # Sobrescribir si existe
            str(output_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutos máximo
            )
            
            if result.returncode != 0:
                error_msg = result.stderr or "Unknown ffmpeg error"
                raise AudioProcessingError(
                    f"ffmpeg extraction failed: {error_msg}",
                    component=self.name
                )
        except subprocess.TimeoutExpired:
            raise AudioProcessingError(
                "Audio extraction timed out (exceeded 5 minutes)",
                component=self.name
            )
    
    def _run_ffprobe(self, input_path: Path) -> Dict[str, Any]:
        """
        Ejecuta ffprobe para obtener información del archivo.
        
        Args:
            input_path: Ruta al archivo
        
        Returns:
            Diccionario con datos de ffprobe
        
        Raises:
            AudioProcessingError: Si ffprobe falla
        """
        import subprocess
        import json
        
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(input_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise AudioProcessingError(
                f"ffprobe failed: {result.stderr}",
                component=self.name
            )
        
        return json.loads(result.stdout)
    
    def _extract_metadata_from_probe(self, probe_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrae metadatos de audio de los datos de ffprobe.
        
        Args:
            probe_data: Datos de ffprobe
        
        Returns:
            Diccionario con metadatos de audio
        
        Raises:
            AudioFormatError: Si no se encuentra stream de audio
        """
        # Buscar stream de audio
        audio_stream = None
        for stream in probe_data.get("streams", []):
            if stream.get("codec_type") == "audio":
                audio_stream = stream
                break
        
        if not audio_stream:
            raise AudioFormatError(
                "No audio stream found in file",
                component=self.name
            )
        
        format_info = probe_data.get("format", {})
        
        return {
            "duration": float(format_info.get("duration", 0)),
            "sample_rate": int(audio_stream.get("sample_rate", 0)),
            "channels": int(audio_stream.get("channels", 0)),
            "bit_depth": None,  # No siempre disponible
            "format": format_info.get("format_name", ""),
            "codec": audio_stream.get("codec_name", ""),
            "bitrate": int(format_info.get("bit_rate", 0)) if format_info.get("bit_rate") else None,
        }




