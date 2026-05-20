"""
BUL Configuration
=================

Configuration management for the BUL system.
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from pydantic import BaseSettings, Field, validator
from pydantic_settings import BaseSettings as PydanticBaseSettings

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class APIConfig:
    """API configuration"""
    openrouter_api_key: str = ""
    openai_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openai_base_url: str = "https://api.openai.com/v1"
    default_model: str = "openai/gpt-4"
    fallback_model: str = "openai/gpt-3.5-turbo"
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout: int = 30

@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str = "sqlite:///bul.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600

@dataclass
class ServerConfig:
    """Server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    cors_origins: list = field(default_factory=lambda: ["*"])

@dataclass
class CacheConfig:
    """Cache configuration"""
    enabled: bool = True
    backend: str = "memory"  # memory, redis
    redis_url: str = "redis://localhost:6379/0"
    default_ttl: int = 3600  # 1 hour
    max_size: int = 1000

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

@dataclass
class BULConfig:
    """Main BUL configuration"""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    api: APIConfig = field(default_factory=APIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Business logic configuration
    max_documents_per_batch: int = 10
    max_document_length: int = 50000
    default_language: str = "es"
    supported_languages: list = field(default_factory=lambda: ["es", "en", "pt", "fr"])
    
    # Agent configuration
    agent_auto_selection: bool = True
    agent_fallback_enabled: bool = True
    max_agent_retries: int = 3
    
    # Document generation configuration
    default_format: str = "markdown"
    supported_formats: list = field(default_factory=lambda: ["markdown", "html", "pdf", "docx"])
    quality_threshold: float = 0.7
    max_processing_time: int = 300  # 5 minutes

class ConfigManager:
    """Configuration manager for BUL system"""
    
    def __init__(self):
        self.config: Optional[BULConfig] = None
        self._load_from_environment()
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        try:
            # API Configuration
            api_config = APIConfig(
                openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                default_model=os.getenv("DEFAULT_MODEL", "openai/gpt-4"),
                fallback_model=os.getenv("FALLBACK_MODEL", "openai/gpt-3.5-turbo"),
                max_tokens=int(os.getenv("MAX_TOKENS", "4000")),
                temperature=float(os.getenv("TEMPERATURE", "0.7")),
                timeout=int(os.getenv("API_TIMEOUT", "30"))
            )
            
            # Database Configuration
            database_config = DatabaseConfig(
                url=os.getenv("DATABASE_URL", "sqlite:///bul.db"),
                echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
                pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
                max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
                pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
                pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "3600"))
            )
            
            # Server Configuration
            server_config = ServerConfig(
                host=os.getenv("HOST", "0.0.0.0"),
                port=int(os.getenv("PORT", "8000")),
                workers=int(os.getenv("WORKERS", "1")),
                reload=os.getenv("RELOAD", "false").lower() == "true",
                log_level=os.getenv("LOG_LEVEL", "info"),
                cors_origins=os.getenv("CORS_ORIGINS", "*").split(",")
            )
            
            # Cache Configuration
            cache_config = CacheConfig(
                enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
                backend=os.getenv("CACHE_BACKEND", "memory"),
                redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                default_ttl=int(os.getenv("CACHE_TTL", "3600")),
                max_size=int(os.getenv("CACHE_MAX_SIZE", "1000"))
            )
            
            # Logging Configuration
            logging_config = LoggingConfig(
                level=os.getenv("LOG_LEVEL", "INFO"),
                format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
                file_path=os.getenv("LOG_FILE_PATH"),
                max_file_size=int(os.getenv("LOG_MAX_FILE_SIZE", str(10 * 1024 * 1024))),
                backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5"))
            )
            
            # Environment
            env_str = os.getenv("ENVIRONMENT", "development").lower()
            environment = Environment.DEVELOPMENT
            if env_str == "staging":
                environment = Environment.STAGING
            elif env_str == "production":
                environment = Environment.PRODUCTION
            
            # Main Configuration
            self.config = BULConfig(
                environment=environment,
                debug=os.getenv("DEBUG", "false").lower() == "true",
                api=api_config,
                database=database_config,
                server=server_config,
                cache=cache_config,
                logging=logging_config,
                max_documents_per_batch=int(os.getenv("MAX_DOCUMENTS_PER_BATCH", "10")),
                max_document_length=int(os.getenv("MAX_DOCUMENT_LENGTH", "50000")),
                default_language=os.getenv("DEFAULT_LANGUAGE", "es"),
                supported_languages=os.getenv("SUPPORTED_LANGUAGES", "es,en,pt,fr").split(","),
                agent_auto_selection=os.getenv("AGENT_AUTO_SELECTION", "true").lower() == "true",
                agent_fallback_enabled=os.getenv("AGENT_FALLBACK_ENABLED", "true").lower() == "true",
                max_agent_retries=int(os.getenv("MAX_AGENT_RETRIES", "3")),
                default_format=os.getenv("DEFAULT_FORMAT", "markdown"),
                supported_formats=os.getenv("SUPPORTED_FORMATS", "markdown,html,pdf,docx").split(","),
                quality_threshold=float(os.getenv("QUALITY_THRESHOLD", "0.7")),
                max_processing_time=int(os.getenv("MAX_PROCESSING_TIME", "300"))
            )
            
            logger.info(f"Configuration loaded for environment: {environment.value}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Use default configuration
            self.config = BULConfig()
    
    def get_config(self) -> BULConfig:
        """Get the current configuration"""
        if self.config is None:
            self._load_from_environment()
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        if self.config is None:
            self.config = BULConfig()
        
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")
    
    def save_to_file(self, file_path: str):
        """Save configuration to file"""
        if self.config is None:
            return
        
        try:
            config_dict = self._config_to_dict(self.config)
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            logger.info(f"Configuration saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def load_from_file(self, file_path: str):
        """Load configuration from file"""
        try:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
            
            # Convert dict back to BULConfig
            self.config = self._dict_to_config(config_dict)
            logger.info(f"Configuration loaded from {file_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from file: {e}")
    
    def _config_to_dict(self, config: BULConfig) -> Dict[str, Any]:
        """Convert BULConfig to dictionary"""
        result = {}
        for field_name, field_value in config.__dict__.items():
            if hasattr(field_value, '__dict__'):
                result[field_name] = self._config_to_dict(field_value)
            elif isinstance(field_value, Enum):
                result[field_name] = field_value.value
            else:
                result[field_name] = field_value
        return result
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> BULConfig:
        """Convert dictionary to BULConfig"""
        # This is a simplified version - in practice, you'd want more robust conversion
        return BULConfig()

# Global configuration manager
_config_manager: Optional[ConfigManager] = None

def get_config() -> BULConfig:
    """Get the global configuration"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.get_config()

def get_config_manager() -> ConfigManager:
    """Get the global configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager



