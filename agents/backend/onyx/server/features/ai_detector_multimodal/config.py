"""
Configuración para el detector multimodal de IA
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuración de la aplicación"""
    
    app_name: str = "AI Detector Multimodal"
    app_version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Configuración de modelos
    enable_text_detection: bool = True
    enable_image_detection: bool = True
    enable_audio_detection: bool = True
    enable_video_detection: bool = True
    
    # Thresholds de detección
    ai_threshold: float = 0.5  # Porcentaje mínimo para considerar como IA
    confidence_threshold: float = 0.6  # Confianza mínima para reportar
    
    # Configuración de procesamiento
    max_content_length: int = 100000  # Longitud máxima de contenido
    batch_size: int = 10  # Tamaño de batch por defecto
    parallel_processing: bool = True
    
    # Configuración de logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Instancia global de configuración
settings = Settings()






