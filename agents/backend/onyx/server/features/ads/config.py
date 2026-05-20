from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
TIMEOUT_SECONDS = 60

from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any
import os
from functools import lru_cache
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Configuration for Onyx ads functionality.
"""

class Settings(BaseSettings):
    """Settings for Onyx ads functionality."""
    
    # API Settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Database Settings
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./onyx.db")
    
    # Storage Settings
    storage_path: str = os.getenv("STORAGE_PATH", "./storage")
    
    # LangChain Settings
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = "gpt-4-turbo-preview"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 2000
    
    # Vector Store Settings
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "./vector_store")
    
    # Cache Settings
    cache_ttl: int = 3600  # 1 hour
    
    @dataclass
class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

def get_llm():
    """Get LLM instance."""
    settings = get_settings()
    
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        max_tokens=settings.openai_max_tokens,
        api_key=settings.openai_api_key
    )

def get_embeddings():
    """Get embeddings instance."""
    settings = get_settings()
    
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=settings.openai_api_key
    ) 