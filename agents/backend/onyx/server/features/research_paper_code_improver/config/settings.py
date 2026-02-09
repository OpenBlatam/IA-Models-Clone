"""
Settings - Configuración del sistema
=====================================
"""

import os
from typing import Optional

try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback para versiones antiguas de pydantic
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Configuración del sistema"""
    
    # API Settings
    API_HOST: str = os.getenv("HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("PORT", 8030))
    
    # GitHub Settings
    GITHUB_TOKEN: Optional[str] = os.getenv("GITHUB_TOKEN")
    
    # Model Settings
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-4")
    BASE_MODEL: Optional[str] = os.getenv("BASE_MODEL")
    
    # Paths
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    PAPERS_DIR: str = os.getenv("PAPERS_DIR", "data/papers")
    MODELS_DIR: str = os.getenv("MODELS_DIR", "data/models")
    EMBEDDINGS_DIR: str = os.getenv("EMBEDDINGS_DIR", "data/embeddings")
    
    # Training Settings
    DEFAULT_EPOCHS: int = int(os.getenv("DEFAULT_EPOCHS", "3"))
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", str(50 * 1024 * 1024)))  # 50MB
    
    # Vector DB Settings
    VECTOR_DB_TYPE: str = os.getenv("VECTOR_DB_TYPE", "chroma")  # chroma, pinecone, faiss
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "data/vector_db")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Instancia global de configuración
settings = Settings()

