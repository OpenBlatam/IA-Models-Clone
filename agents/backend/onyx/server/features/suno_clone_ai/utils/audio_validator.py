"""
Sistema de Validación de Audio

Proporciona:
- Validación de formato
- Validación de calidad
- Detección de corrupción
- Análisis de metadatos
- Verificación de duración
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import librosa
    import soundfile as sf
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    logger.warning("librosa/soundfile not available, audio validation limited")


class AudioValidationResult:
    """Resultado de validación de audio"""
    
    def __init__(self):
        self.valid: bool = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_error(self, error: str):
        """Agrega un error"""
        self.errors.append(error)
        self.valid = False
    
    def add_warning(self, warning: str):
        """Agrega una advertencia"""
        self.warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata
        }


class AudioValidator:
    """Validador de archivos de audio"""
    
    def __init__(
        self,
        min_duration: float = 1.0,
        max_duration: float = 600.0,
        min_sample_rate: int = 16000,
        max_sample_rate: int = 48000,
        allowed_formats: list = None
    ):
        """
        Args:
            min_duration: Duración mínima en segundos
            max_duration: Duración máxima en segundos
            min_sample_rate: Sample rate mínimo
            max_sample_rate: Sample rate máximo
            allowed_formats: Formatos permitidos
        """
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_sample_rate = min_sample_rate
        self.max_sample_rate = max_sample_rate
        self.allowed_formats = allowed_formats or [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
        logger.info("AudioValidator initialized")
    
    def validate_file(
        self,
        file_path: str,
        check_quality: bool = True
    ) -> AudioValidationResult:
        """
        Valida un archivo de audio
        
        Args:
            file_path: Ruta del archivo
            check_quality: Verificar calidad del audio
        
        Returns:
            AudioValidationResult
        """
        result = AudioValidationResult()
        file_path_obj = Path(file_path)
        
        # Verificar que el archivo existe
        if not file_path_obj.exists():
            result.add_error(f"File not found: {file_path}")
            return result
        
        # Verificar extensión
        if file_path_obj.suffix.lower() not in self.allowed_formats:
            result.add_error(
                f"Invalid format: {file_path_obj.suffix}. "
                f"Allowed formats: {', '.join(self.allowed_formats)}"
            )
        
        # Verificar tamaño del archivo
        file_size = file_path_obj.stat().st_size
        if file_size == 0:
            result.add_error("File is empty")
            return result
        
        if file_size > 100 * 1024 * 1024:  # 100MB
            result.add_warning(f"Large file size: {file_size / (1024*1024):.2f}MB")
        
        if not AUDIO_LIBS_AVAILABLE:
            result.add_warning("Audio libraries not available, limited validation")
            return result
        
        # Validar contenido del audio
        try:
            audio_data, sample_rate = self._load_audio(file_path)
            
            # Validar sample rate
            if sample_rate < self.min_sample_rate:
                result.add_error(
                    f"Sample rate too low: {sample_rate}Hz (minimum: {self.min_sample_rate}Hz)"
                )
            elif sample_rate > self.max_sample_rate:
                result.add_warning(
                    f"Sample rate high: {sample_rate}Hz (maximum recommended: {self.max_sample_rate}Hz)"
                )
            
            # Calcular duración
            duration = len(audio_data) / sample_rate
            
            # Validar duración
            if duration < self.min_duration:
                result.add_error(
                    f"Duration too short: {duration:.2f}s (minimum: {self.min_duration}s)"
                )
            elif duration > self.max_duration:
                result.add_error(
                    f"Duration too long: {duration:.2f}s (maximum: {self.max_duration}s)"
                )
            
            # Guardar metadatos
            result.metadata = {
                "duration": duration,
                "sample_rate": sample_rate,
                "channels": 1 if len(audio_data.shape) == 1 else audio_data.shape[1],
                "samples": len(audio_data),
                "file_size": file_size
            }
            
            # Verificar calidad si se solicita
            if check_quality:
                quality_issues = self._check_quality(audio_data, sample_rate)
                for issue in quality_issues:
                    result.add_warning(issue)
            
            # Verificar corrupción
            if np.isnan(audio_data).any() or np.isinf(audio_data).any():
                result.add_error("Audio contains NaN or Inf values (corrupted)")
            
            # Verificar silencio
            if np.max(np.abs(audio_data)) < 0.001:
                result.add_warning("Audio appears to be silent or very quiet")
        
        except Exception as e:
            result.add_error(f"Error reading audio file: {str(e)}")
        
        return result
    
    def validate_data(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        check_quality: bool = True
    ) -> AudioValidationResult:
        """
        Valida datos de audio en memoria
        
        Args:
            audio_data: Datos de audio (numpy array)
            sample_rate: Sample rate
            check_quality: Verificar calidad
        
        Returns:
            AudioValidationResult
        """
        result = AudioValidationResult()
        
        # Validar sample rate
        if sample_rate < self.min_sample_rate:
            result.add_error(
                f"Sample rate too low: {sample_rate}Hz (minimum: {self.min_sample_rate}Hz)"
            )
        elif sample_rate > self.max_sample_rate:
            result.add_warning(
                f"Sample rate high: {sample_rate}Hz (maximum recommended: {self.max_sample_rate}Hz)"
            )
        
        # Calcular duración
        duration = len(audio_data) / sample_rate
        
        # Validar duración
        if duration < self.min_duration:
            result.add_error(
                f"Duration too short: {duration:.2f}s (minimum: {self.min_duration}s)"
            )
        elif duration > self.max_duration:
            result.add_error(
                f"Duration too long: {duration:.2f}s (maximum: {self.max_duration}s)"
            )
        
        # Guardar metadatos
        result.metadata = {
            "duration": duration,
            "sample_rate": sample_rate,
            "channels": 1 if len(audio_data.shape) == 1 else audio_data.shape[1],
            "samples": len(audio_data)
        }
        
        # Verificar corrupción
        if np.isnan(audio_data).any() or np.isinf(audio_data).any():
            result.add_error("Audio contains NaN or Inf values (corrupted)")
        
        # Verificar silencio
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude < 0.001:
            result.add_warning("Audio appears to be silent or very quiet")
        elif max_amplitude > 0.95:
            result.add_warning("Audio may be clipping (amplitude too high)")
        
        # Verificar calidad
        if check_quality:
            quality_issues = self._check_quality(audio_data, sample_rate)
            for issue in quality_issues:
                result.add_warning(issue)
        
        return result
    
    def _load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Carga un archivo de audio"""
        audio_data, sample_rate = librosa.load(file_path, sr=None, mono=False)
        return audio_data, sample_rate
    
    def _check_quality(self, audio_data: np.ndarray, sample_rate: int) -> list:
        """Verifica la calidad del audio"""
        issues = []
        
        # Verificar rango dinámico
        if len(audio_data.shape) == 1:
            audio_mono = audio_data
        else:
            audio_mono = np.mean(audio_data, axis=0)
        
        dynamic_range = np.max(audio_mono) - np.min(audio_mono)
        if dynamic_range < 0.1:
            issues.append("Low dynamic range - audio may be compressed or normalized too much")
        
        # Verificar clipping
        if np.any(np.abs(audio_mono) > 0.99):
            issues.append("Audio may contain clipping")
        
        # Verificar ruido de fondo
        # Calcular RMS en segmentos silenciosos
        threshold = np.percentile(np.abs(audio_mono), 10)
        quiet_segments = audio_mono[np.abs(audio_mono) < threshold]
        if len(quiet_segments) > 0:
            noise_level = np.std(quiet_segments)
            if noise_level > 0.01:
                issues.append(f"High background noise level: {noise_level:.4f}")
        
        return issues


# Instancia global
_audio_validator: Optional[AudioValidator] = None


def get_audio_validator(
    min_duration: float = 1.0,
    max_duration: float = 600.0
) -> AudioValidator:
    """Obtiene la instancia global del validador de audio"""
    global _audio_validator
    if _audio_validator is None:
        _audio_validator = AudioValidator(
            min_duration=min_duration,
            max_duration=max_duration
        )
    return _audio_validator

