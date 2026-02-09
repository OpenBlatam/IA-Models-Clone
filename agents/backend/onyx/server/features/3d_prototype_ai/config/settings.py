"""
Settings - Configuración del sistema de prototipos 3D
======================================================
"""

from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    """Configuración de la aplicación"""
    
    # General
    app_name: str = "3D Prototype AI"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8030
    
    # Output
    output_dir: str = "output/prototypes"
    
    # Material Database
    material_db_path: Optional[str] = None
    
    # CAD Generation
    cad_output_format: str = "STL"  # STL, STEP, OBJ
    cad_output_dir: str = "output/cad_files"
    
    # AI/LLM Integration (para futuras mejoras)
    llm_provider: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Instancia global de configuración
settings = Settings()




