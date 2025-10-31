from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Type, TypeVar, Union
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from .model_types import (
from datetime import datetime
from typing import List, Optional
import logging
import os
from typing import Any, List, Dict, Optional
import asyncio
"""
Model Configuration - Onyx Integration
Configuration settings for model operations.
"""
    JsonDict, JsonList, JsonValue, FieldType, FieldValue,
    ModelId, ModelKey, ModelValue, ModelData, ModelList, ModelDict,
    IndexField, IndexValue, IndexKey, IndexData, IndexList, IndexDict,
    CacheKey, CacheValue, CacheData, CacheList, CacheDict,
    ValidationRule, ValidationRules, ValidationError, ValidationErrors,
    EventName, EventData, EventHandler, EventHandlers,
    ModelStatus, ModelCategory, ModelPermission,
    OnyxBaseModel, ModelField, ModelSchema, ModelRegistry,
    ModelCache, ModelIndex, ModelEvent, ModelValidation, ModelFactory
)

T = TypeVar('T', bound="OnyxBaseModel")

class ModelConfig:
    """Model configuration settings."""
    
    # Base settings
    BASE_DIR: Path = Path(__file__).parent.parent.parent.parent
    CONFIG_DIR: Path = BASE_DIR / "config"
    LOG_DIR: Path = BASE_DIR / "logs"
    DATA_DIR: Path = BASE_DIR / "data"
    
    # Redis settings
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    REDIS_SSL: bool = os.getenv("REDIS_SSL", "false").lower() == "true"
    
    # Cache settings
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
    CACHE_PREFIX: str = os.getenv("CACHE_PREFIX", "onyx:")
    CACHE_MAX_SIZE: int = int(os.getenv("CACHE_MAX_SIZE", "1000"))
    CACHE_CLEANUP_INTERVAL: int = int(os.getenv("CACHE_CLEANUP_INTERVAL", "300"))  # 5 minutes
    
    # Index settings
    INDEX_PREFIX: str = os.getenv("INDEX_PREFIX", "onyx:index:")
    INDEX_TTL: int = int(os.getenv("INDEX_TTL", "86400"))  # 24 hours
    INDEX_BATCH_SIZE: int = int(os.getenv("INDEX_BATCH_SIZE", "100"))
    INDEX_CLEANUP_INTERVAL: int = int(os.getenv("INDEX_CLEANUP_INTERVAL", "3600"))  # 1 hour
    
    # Validation settings
    VALIDATION_STRICT: bool = os.getenv("VALIDATION_STRICT", "true").lower() == "true"
    VALIDATION_CACHE_SIZE: int = int(os.getenv("VALIDATION_CACHE_SIZE", "1000"))
    VALIDATION_CACHE_TTL: int = int(os.getenv("VALIDATION_CACHE_TTL", "3600"))  # 1 hour
    
    # Event settings
    EVENT_QUEUE_SIZE: int = int(os.getenv("EVENT_QUEUE_SIZE", "1000"))
    EVENT_BATCH_SIZE: int = int(os.getenv("EVENT_BATCH_SIZE", "100"))
    EVENT_PROCESSING_INTERVAL: int = int(os.getenv("EVENT_PROCESSING_INTERVAL", "60"))  # 1 minute
    
    # Logging settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    LOG_FILE: str = os.getenv("LOG_FILE", "onyx.log")
    LOG_MAX_SIZE: int = int(os.getenv("LOG_MAX_SIZE", "10485760"))  # 10 MB
    LOG_BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    
    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key")
    TOKEN_EXPIRY: int = int(os.getenv("TOKEN_EXPIRY", "3600"))  # 1 hour
    PASSWORD_SALT: str = os.getenv("PASSWORD_SALT", "your-password-salt")
    PASSWORD_ITERATIONS: int = int(os.getenv("PASSWORD_ITERATIONS", "100000"))
    
    # API settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_WORKERS: int = int(os.getenv("API_WORKERS", "4"))
    API_TIMEOUT: int = int(os.getenv("API_TIMEOUT", "30"))
    API_MAX_REQUESTS: int = int(os.getenv("API_MAX_REQUESTS", "1000"))
    
    # Database settings
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "onyx")
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "postgres")
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "20"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    
    # Model settings
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "1.0.0")
    MODEL_PREFIX: str = os.getenv("MODEL_PREFIX", "onyx:model:")
    MODEL_TTL: int = int(os.getenv("MODEL_TTL", "86400"))  # 24 hours
    MODEL_BATCH_SIZE: int = int(os.getenv("MODEL_BATCH_SIZE", "100"))
    MODEL_CLEANUP_INTERVAL: int = int(os.getenv("MODEL_CLEANUP_INTERVAL", "3600"))  # 1 hour
    
    @classmethod
    def load_config(cls, config_file: Optional[str] = None) -> None:
        """Load configuration from file."""
        if config_file is None:
            config_file = cls.CONFIG_DIR / "config.json"
        
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                config = json.load(f)
                
                for key, value in config.items():
                    if hasattr(cls, key):
                        setattr(cls, key, value)
    
    @classmethod
    def save_config(cls, config_file: Optional[str] = None) -> None:
        """Save configuration to file."""
        if config_file is None:
            config_file = cls.CONFIG_DIR / "config.json"
        
        config = {
            key: value
            for key, value in cls.__dict__.items()
            if not key.startswith("_") and isinstance(value, (str, int, float, bool, list, dict))
        }
        
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        with open(config_file, "w") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(config, f, indent=4)
    
    @classmethod
    def setup_logging(cls) -> None:
        """Setup logging configuration."""
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL),
            format=cls.LOG_FORMAT,
            handlers=[
                logging.FileHandler(cls.LOG_DIR / cls.LOG_FILE),
                logging.StreamHandler()
            ]
        )
    
    @classmethod
    def get_redis_config(cls) -> Dict[str, Any]:
        """Get Redis configuration."""
        return {
            "host": cls.REDIS_HOST,
            "port": cls.REDIS_PORT,
            "db": cls.REDIS_DB,
            "password": cls.REDIS_PASSWORD,
            "ssl": cls.REDIS_SSL
        }
    
    @classmethod
    def get_cache_config(cls) -> Dict[str, Any]:
        """Get cache configuration."""
        return {
            "ttl": cls.CACHE_TTL,
            "prefix": cls.CACHE_PREFIX,
            "max_size": cls.CACHE_MAX_SIZE,
            "cleanup_interval": cls.CACHE_CLEANUP_INTERVAL
        }
    
    @classmethod
    def get_index_config(cls) -> Dict[str, Any]:
        """Get index configuration."""
        return {
            "prefix": cls.INDEX_PREFIX,
            "ttl": cls.INDEX_TTL,
            "batch_size": cls.INDEX_BATCH_SIZE,
            "cleanup_interval": cls.INDEX_CLEANUP_INTERVAL
        }
    
    @classmethod
    def get_validation_config(cls) -> Dict[str, Any]:
        """Get validation configuration."""
        return {
            "strict": cls.VALIDATION_STRICT,
            "cache_size": cls.VALIDATION_CACHE_SIZE,
            "cache_ttl": cls.VALIDATION_CACHE_TTL
        }
    
    @classmethod
    def get_event_config(cls) -> Dict[str, Any]:
        """Get event configuration."""
        return {
            "queue_size": cls.EVENT_QUEUE_SIZE,
            "batch_size": cls.EVENT_BATCH_SIZE,
            "processing_interval": cls.EVENT_PROCESSING_INTERVAL
        }
    
    @classmethod
    def get_logging_config(cls) -> Dict[str, Any]:
        """Get logging configuration."""
        return {
            "level": cls.LOG_LEVEL,
            "format": cls.LOG_FORMAT,
            "file": cls.LOG_FILE,
            "max_size": cls.LOG_MAX_SIZE,
            "backup_count": cls.LOG_BACKUP_COUNT
        }
    
    @classmethod
    def get_security_config(cls) -> Dict[str, Any]:
        """Get security configuration."""
        return {
            "secret_key": cls.SECRET_KEY,
            "token_expiry": cls.TOKEN_EXPIRY,
            "password_salt": cls.PASSWORD_SALT,
            "password_iterations": cls.PASSWORD_ITERATIONS
        }
    
    @classmethod
    async def get_api_config(cls) -> Dict[str, Any]:
        """Get API configuration."""
        return {
            "host": cls.API_HOST,
            "port": cls.API_PORT,
            "workers": cls.API_WORKERS,
            "timeout": cls.API_TIMEOUT,
            "max_requests": cls.API_MAX_REQUESTS
        }
    
    @classmethod
    def get_db_config(cls) -> Dict[str, Any]:
        """Get database configuration."""
        return {
            "host": cls.DB_HOST,
            "port": cls.DB_PORT,
            "name": cls.DB_NAME,
            "user": cls.DB_USER,
            "password": cls.DB_PASSWORD,
            "pool_size": cls.DB_POOL_SIZE,
            "max_overflow": cls.DB_MAX_OVERFLOW
        }
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "version": cls.MODEL_VERSION,
            "prefix": cls.MODEL_PREFIX,
            "ttl": cls.MODEL_TTL,
            "batch_size": cls.MODEL_BATCH_SIZE,
            "cleanup_interval": cls.MODEL_CLEANUP_INTERVAL
        }

# Example usage:
"""

# Set environment variables
os.environ["REDIS_HOST"] = "localhost"
os.environ["REDIS_PORT"] = "6379"
os.environ["LOG_LEVEL"] = "INFO"
os.environ["API_PORT"] = "8000"

# Load configuration
ModelConfig.load_config()

# Setup logging
ModelConfig.setup_logging()
logger = logging.getLogger(__name__)

# Get configurations
redis_config = ModelConfig.get_redis_config()
logger.info(f"Redis config: {redis_config}")

cache_config = ModelConfig.get_cache_config()
logger.info(f"Cache config: {cache_config}")

index_config = ModelConfig.get_index_config()
logger.info(f"Index config: {index_config}")

validation_config = ModelConfig.get_validation_config()
logger.info(f"Validation config: {validation_config}")

event_config = ModelConfig.get_event_config()
logger.info(f"Event config: {event_config}")

logging_config = ModelConfig.get_logging_config()
logger.info(f"Logging config: {logging_config}")

security_config = ModelConfig.get_security_config()
logger.info(f"Security config: {security_config}")

api_config = ModelConfig.get_api_config()
logger.info(f"API config: {api_config}")

db_config = ModelConfig.get_db_config()
logger.info(f"Database config: {db_config}")

model_config = ModelConfig.get_model_config()
logger.info(f"Model config: {model_config}")

# Save configuration
ModelConfig.save_config()
""" 