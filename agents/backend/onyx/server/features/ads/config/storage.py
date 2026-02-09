from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Storage configuration for the ads module.
"""

class DatabaseSettings(BaseModel):
    """Database settings."""
    url: str = Field(default="postgresql+asyncpg://user:pass@localhost/db")
    pool_size: int = Field(default=20)
    max_overflow: int = Field(default=10)
    echo: bool = Field(default=False)
    pool_timeout: int = Field(default=30)
    pool_recycle: int = Field(default=1800)

class RedisSettings(BaseModel):
    """Redis settings."""
    url: str = Field(default="redis://localhost:6379/0")
    pool_size: int = Field(default=10)
    socket_timeout: int = Field(default=5)
    socket_connect_timeout: int = Field(default=5)
    retry_on_timeout: bool = Field(default=True)

class VectorStoreSettings(BaseModel):
    """Vector store settings."""
    type: str = Field(default="faiss")
    path: str = Field(default="./vector_store")
    dimension: int = Field(default=1536)
    metric: str = Field(default="cosine")

class StorageSettings(BaseSettings):
    """Storage settings."""
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    
    # Cache settings
    cache_ttl: int = Field(default=3600)
    cache_prefix: str = Field(default="ads_")
    enable_cache: bool = Field(default=True)
    
    # File storage settings
    upload_dir: str = Field(default="./uploads")
    max_file_size: int = Field(default=10 * 1024 * 1024)  # 10MB
    allowed_extensions: list[str] = Field(default=["txt", "pdf", "doc", "docx"])
    
    @dataclass
class Config:
        env_prefix = "STORAGE_"
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global storage settings instance
storage_settings = StorageSettings() 