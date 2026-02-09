"""
Environment Configuration
========================

Configuración del entorno con validación.
"""

import os
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class EnvironmentConfig:
    """Configuración del entorno."""
    
    # Directorios
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / "models"
    CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
    OUTPUTS_DIR = BASE_DIR / "outputs"
    LOGS_DIR = BASE_DIR / "logs"
    EXPERIMENTS_DIR = BASE_DIR / "experiments"
    DATA_DIR = BASE_DIR / "data"
    
    @staticmethod
    def ensure_directories():
        """Asegurar que los directorios existan."""
        directories = [
            EnvironmentConfig.MODELS_DIR,
            EnvironmentConfig.CHECKPOINTS_DIR,
            EnvironmentConfig.OUTPUTS_DIR,
            EnvironmentConfig.LOGS_DIR,
            EnvironmentConfig.EXPERIMENTS_DIR,
            EnvironmentConfig.DATA_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directorio verificado: {directory}")
    
    @staticmethod
    def validate_environment() -> tuple[bool, list[str]]:
        """
        Validar entorno.
        
        Returns:
            (is_valid, errors)
        """
        errors = []
        
        # Verificar variables requeridas
        if not os.getenv("OPENROUTER_API_KEY"):
            errors.append("OPENROUTER_API_KEY no configurada")
        
        # Verificar directorios
        EnvironmentConfig.ensure_directories()
        
        # Verificar permisos de escritura
        try:
            test_file = EnvironmentConfig.LOGS_DIR / ".test"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            errors.append(f"Sin permisos de escritura en logs: {str(e)}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def get_environment_info() -> dict:
        """Obtener información del entorno."""
        import platform
        import sys
        
        info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "base_dir": str(EnvironmentConfig.BASE_DIR),
            "models_dir": str(EnvironmentConfig.MODELS_DIR),
            "checkpoints_dir": str(EnvironmentConfig.CHECKPOINTS_DIR),
            "outputs_dir": str(EnvironmentConfig.OUTPUTS_DIR),
            "logs_dir": str(EnvironmentConfig.LOGS_DIR),
            "experiments_dir": str(EnvironmentConfig.EXPERIMENTS_DIR),
            "data_dir": str(EnvironmentConfig.DATA_DIR)
        }
        
        # GPU info
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu"] = {
                    "available": True,
                    "device_count": torch.cuda.device_count(),
                    "device_name": torch.cuda.get_device_name(0)
                }
            else:
                info["gpu"] = {"available": False}
        except:
            info["gpu"] = {"available": False, "error": "torch no disponible"}
        
        return info




