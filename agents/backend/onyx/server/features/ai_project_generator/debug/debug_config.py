"""
Debug Configuration - Configuración de debugging
================================================

Configuración para habilitar/deshabilitar características de debugging.
"""

import os
from pydantic_settings import BaseSettings
from typing import Optional


class DebugConfig(BaseSettings):
    """Configuración de debugging"""
    
    # Habilitar/deshabilitar
    debug_enabled: bool = False
    debug_logging: bool = False
    error_tracking: bool = True
    profiling: bool = False
    request_debugging: bool = False
    
    # Niveles
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_to_file: bool = True
    log_dir: str = "logs"
    
    # Error tracking
    max_error_history: int = 1000
    error_grouping: bool = True
    
    # Profiling
    profile_enabled: bool = False
    profile_slow_threshold: float = 1.0  # segundos
    
    # Request debugging
    log_requests: bool = False
    log_responses: bool = False
    log_request_body: bool = False
    log_response_body: bool = False
    
    # Performance debugging
    track_slow_requests: bool = True
    slow_request_threshold: float = 1.0  # segundos
    
    class Config:
        env_prefix = "DEBUG_"
        case_sensitive = False


def get_debug_config() -> DebugConfig:
    """Obtiene configuración de debugging"""
    return DebugConfig()


def is_debug_enabled() -> bool:
    """Verifica si debugging está habilitado"""
    config = get_debug_config()
    return config.debug_enabled or os.getenv("DEBUG", "false").lower() == "true"















