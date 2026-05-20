"""
Servicio de análisis de archivos de audio locales
"""

import logging
from typing import Dict, List, Any, Optional
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioFileAnalyzer:
    """Analiza archivos de audio locales sin necesidad de Spotify"""
    
    def __init__(self):
        self.logger = logger
        self.supported_formats = ['.mp3', '.wav', '.flac', '.m4a', '.ogg']
    
    def analyze_audio_file(self, file_path: str) -> Dict[str, Any]:
        """Analiza un archivo de audio local"""
        try:
            if not os.path.exists(file_path):
                return {"error": f"Archivo no encontrado: {file_path}"}
            
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_formats:
                return {"error": f"Formato no soportado: {file_ext}. Formatos soportados: {', '.join(self.supported_formats)}"}
            
            # Obtener información básica del archivo
            file_info = self._get_file_info(file_path)
            
            # Análisis básico (simplificado sin librerías de audio)
            analysis = {
                "file_info": file_info,
                "format": file_ext,
                "supported": True,
                "analysis_available": False,
                "note": "Análisis completo requiere librerías de audio como librosa o pydub"
            }
            
            return {
                "success": True,
                "analysis": analysis
            }
        except Exception as e:
            self.logger.error(f"Error analyzing audio file: {e}")
            return {"error": str(e)}
    
    def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Obtiene información básica del archivo"""
        try:
            stat = os.stat(file_path)
            file_size = stat.st_size
            file_size_mb = file_size / (1024 * 1024)
            
            return {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size_mb, 2),
                "exists": True
            }
        except Exception as e:
            self.logger.error(f"Error getting file info: {e}")
            return {"error": str(e)}
    
    def validate_audio_file(self, file_path: str) -> Dict[str, Any]:
        """Valida un archivo de audio"""
        try:
            if not os.path.exists(file_path):
                return {
                    "valid": False,
                    "error": "File not found"
                }
            
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext not in self.supported_formats:
                return {
                    "valid": False,
                    "error": f"Unsupported format: {file_ext}",
                    "supported_formats": self.supported_formats
                }
            
            return {
                "valid": True,
                "format": file_ext,
                "file_info": self._get_file_info(file_path)
            }
        except Exception as e:
            self.logger.error(f"Error validating audio file: {e}")
            return {
                "valid": False,
                "error": str(e)
            }
    
    def get_supported_formats(self) -> List[str]:
        """Obtiene lista de formatos soportados"""
        return self.supported_formats

