from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import Optional, List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
API configuration for the ads module.
"""

class SecuritySettings(BaseModel):
    """Security settings."""
    secret_key: str = Field(default="your-secret-key")
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30)
    refresh_token_expire_days: int = Field(default=7)
    enable_rate_limiting: bool = Field(default=True)
    rate_limit_per_minute: int = Field(default=100)

class CORSettings(BaseModel):
    """CORS settings."""
    allow_origins: List[str] = Field(default=["*"])
    allow_credentials: bool = Field(default=True)
    allow_methods: List[str] = Field(default=["*"])
    allow_headers: List[str] = Field(default=["*"])
    expose_headers: List[str] = Field(default=["*"])
    max_age: int = Field(default=600)

class APISettings(BaseSettings):
    """API settings."""
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    cors: CORSettings = Field(default_factory=CORSettings)
    
    # API versioning
    version: str = Field(default="1.0.0")
    prefix: str = Field(default="/api/v1/ads")
    
    # Documentation
    title: str = Field(default="Onyx Ads API")
    description: str = Field(default="API for Onyx - AI-powered content generation and analysis")
    docs_url: str = Field(default="/docs")
    redoc_url: str = Field(default="/redoc")
    openapi_url: str = Field(default="/openapi.json")
    
    # Response settings
    default_response_model: str = Field(default="json")
    enable_response_compression: bool = Field(default=True)
    response_timeout: int = Field(default=30)
    
    # Validation settings
    enable_request_validation: bool = Field(default=True)
    enable_response_validation: bool = Field(default=True)
    
    @dataclass
class Config:
        env_prefix = "API_"
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global API settings instance
api_settings = APISettings() 