"""
Exportador de audio a múltiples formatos
"""

import logging
from pathlib import Path
from typing import Optional
import subprocess
import numpy as np
import soundfile as sf

from config.settings import settings

logger = logging.getLogger(__name__)


class AudioExporter:
    """Exportador de audio a diferentes formatos"""
    
    def __init__(self):
        self.supported_formats = ["wav", "mp3", "flac", "ogg", "m4a"]
    
    def export_to_format(
        self,
        input_path: str,
        output_path: str,
        format: str = "mp3",
        bitrate: str = "192k",
        sample_rate: Optional[int] = None
    ) -> str:
        """
        Exporta un archivo de audio a otro formato
        
        Args:
            input_path: Ruta del archivo de entrada
            output_path: Ruta del archivo de salida
            format: Formato de salida (mp3, flac, ogg, m4a)
            bitrate: Bitrate para formatos comprimidos
            sample_rate: Sample rate de salida (opcional)
            
        Returns:
            Ruta del archivo exportado
        """
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}. Supported: {self.supported_formats}")
        
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Si el formato es WAV, usar soundfile directamente
        if format == "wav":
            return self._export_to_wav(input_path, output_path, sample_rate)
        
        # Para otros formatos, usar ffmpeg
        return self._export_with_ffmpeg(input_path, output_path, format, bitrate, sample_rate)
    
    def _export_to_wav(
        self,
        input_path: Path,
        output_path: Path,
        sample_rate: Optional[int]
    ) -> str:
        """Exporta a WAV usando soundfile"""
        try:
            audio, sr = sf.read(str(input_path))
            
            # Resample si es necesario
            if sample_rate and sample_rate != sr:
                from scipy import signal
                num_samples = int(len(audio) * sample_rate / sr)
                audio = signal.resample(audio, num_samples)
                sr = sample_rate
            
            # Asegurar que el directorio existe
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Guardar
            sf.write(str(output_path), audio, sr)
            logger.info(f"Exported to WAV: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Error exporting to WAV: {e}")
            raise
    
    def _export_with_ffmpeg(
        self,
        input_path: Path,
        output_path: Path,
        format: str,
        bitrate: str,
        sample_rate: Optional[int]
    ) -> str:
        """Exporta usando ffmpeg"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            cmd = ["ffmpeg", "-i", str(input_path), "-y"]
            
            # Agregar bitrate si es un formato comprimido
            if format in ["mp3", "ogg", "m4a"]:
                cmd.extend(["-b:a", bitrate])
            
            # Agregar sample rate si se especifica
            if sample_rate:
                cmd.extend(["-ar", str(sample_rate)])
            
            # Formato de salida
            cmd.append(str(output_path))
            
            # Ejecutar ffmpeg
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Exported to {format.upper()}: {output_path}")
            return str(output_path)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise RuntimeError(f"FFmpeg conversion failed: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg to export to compressed formats.")
        except Exception as e:
            logger.error(f"Error exporting with ffmpeg: {e}")
            raise
    
    def get_format_info(self, format: str) -> dict:
        """Obtiene información sobre un formato"""
        formats_info = {
            "wav": {
                "name": "WAV",
                "description": "Uncompressed audio format",
                "lossless": True,
                "compression": None
            },
            "mp3": {
                "name": "MP3",
                "description": "Compressed audio format",
                "lossless": False,
                "compression": "lossy"
            },
            "flac": {
                "name": "FLAC",
                "description": "Free Lossless Audio Codec",
                "lossless": True,
                "compression": "lossless"
            },
            "ogg": {
                "name": "OGG Vorbis",
                "description": "Open source compressed audio",
                "lossless": False,
                "compression": "lossy"
            },
            "m4a": {
                "name": "M4A",
                "description": "Apple's audio format",
                "lossless": False,
                "compression": "lossy"
            }
        }
        
        return formats_info.get(format.lower(), {})


# Instancia global
_audio_exporter: Optional[AudioExporter] = None


def get_audio_exporter() -> AudioExporter:
    """Obtiene la instancia global del exportador de audio"""
    global _audio_exporter
    if _audio_exporter is None:
        _audio_exporter = AudioExporter()
    return _audio_exporter

