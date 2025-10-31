"""
BUL Optimized Configuration
==========================

Clean, modular configuration for the BUL system.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseSettings, Field

class BULConfig(BaseSettings):
    """Optimized BUL system configuration."""
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="BUL_API_HOST")
    api_port: int = Field(default=8000, env="BUL_API_PORT")
    debug_mode: bool = Field(default=False, env="BUL_DEBUG")
    
    # AI Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, env="OPENROUTER_API_KEY")
    
    # Processing Configuration
    max_concurrent_tasks: int = Field(default=5, env="BUL_MAX_CONCURRENT_TASKS")
    task_timeout: int = Field(default=300, env="BUL_TASK_TIMEOUT")
    max_retries: int = Field(default=3, env="BUL_MAX_RETRIES")
    
    # Document Configuration
    supported_formats: List[str] = Field(default=["markdown", "html", "pdf"])
    output_directory: str = Field(default="generated_documents", env="BUL_OUTPUT_DIR")
    max_document_size: int = Field(default=10485760, env="BUL_MAX_DOCUMENT_SIZE")  # 10MB
    
    # Business Areas Configuration
    enabled_business_areas: List[str] = Field(default=[
        "marketing", "sales", "operations", "hr", "finance"
    ])
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="BUL_LOG_LEVEL")
    log_file: str = Field(default="bul.log", env="BUL_LOG_FILE")
    
    # Security Configuration
    secret_key: str = Field(default="your-secret-key-here", env="BUL_SECRET_KEY")
    cors_origins: List[str] = Field(default=["*"], env="BUL_CORS_ORIGINS")
    
    # Performance Configuration
    cache_ttl: int = Field(default=3600, env="BUL_CACHE_TTL")  # 1 hour
    rate_limit: int = Field(default=100, env="BUL_RATE_LIMIT")  # requests per minute
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any errors."""
        errors = []
        
        # Check required API keys
        if not self.openai_api_key and not self.openrouter_api_key:
            errors.append("Either OPENAI_API_KEY or OPENROUTER_API_KEY must be set")
        
        # Check port range
        if not (1 <= self.api_port <= 65535):
            errors.append("API port must be between 1 and 65535")
        
        # Check concurrent tasks
        if self.max_concurrent_tasks < 1:
            errors.append("Max concurrent tasks must be at least 1")
        
        # Check timeout
        if self.task_timeout < 30:
            errors.append("Task timeout must be at least 30 seconds")
        
        # Check output directory
        output_path = Path(self.output_directory)
        if not output_path.exists():
            try:
                output_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create output directory: {e}")
        
        return errors
    
    def get_business_area_config(self, area: str) -> Dict[str, Any]:
        """Get configuration for a specific business area."""
        return {
            "enabled": area in self.enabled_business_areas,
            "priority": 1 if area in ["marketing", "sales", "finance"] else 2
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "api_host": self.api_host,
            "api_port": self.api_port,
            "debug_mode": self.debug_mode,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "task_timeout": self.task_timeout,
            "supported_formats": self.supported_formats,
            "output_directory": self.output_directory,
            "enabled_business_areas": self.enabled_business_areas,
            "log_level": self.log_level,
            "cache_ttl": self.cache_ttl,
            "rate_limit": self.rate_limit
        }

# Global configuration instance
config = BULConfig()

def get_config() -> BULConfig:
    """Get the global configuration instance."""
    return config

def load_config_from_file(config_path: str) -> BULConfig:
    """Load configuration from a file."""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Implementation would depend on config file format (JSON, YAML, etc.)
    # For now, return default config
    return BULConfig()

