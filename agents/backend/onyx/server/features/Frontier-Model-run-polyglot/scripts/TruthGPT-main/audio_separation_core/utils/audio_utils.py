"""
Audio Utilities - Utilidades para procesamiento de audio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


def get_audio_duration(
    audio_path: Union[str, Path]
) -> float:
    """
    Obtiene la duración de un archivo de audio en segundos.
    
    Args:
        audio_path: Ruta al archivo de audio
    
    Returns:
        Duración en segundos
    """
    try:
        import librosa
        y, sr = librosa.load(str(audio_path), sr=None)
        return len(y) / sr
    except Exception:
        # Fallback usando ffprobe
        try:
            import subprocess
            import json
            
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(audio_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return float(data.get("format", {}).get("duration", 0))
        except Exception:
            pass
        
        return 0.0


def get_audio_info(
    audio_path: Union[str, Path]
) -> dict:
    """
    Obtiene información de un archivo de audio.
    
    Args:
        audio_path: Ruta al archivo de audio
    
    Returns:
        Diccionario con información del audio
    """
    try:
        import librosa
        import soundfile as sf
        
        info = sf.info(str(audio_path))
        y, sr = librosa.load(str(audio_path), sr=None, duration=1.0)
        
        return {
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "duration": info.duration,
            "format": info.format,
            "subtype": info.subtype,
            "frames": info.frames,
        }
    except Exception as e:
        return {
            "error": str(e)
        }


def normalize_audio_path(
    path: Union[str, Path]
) -> Path:
    """
    Normaliza una ruta de audio.
    
    Args:
        path: Ruta a normalizar
    
    Returns:
        Path normalizado
    """
    return Path(path).resolve()




