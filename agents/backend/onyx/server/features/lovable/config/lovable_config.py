"""
Configuration management for Lovable Community SAM3.
"""

import os
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class LovableConfig:
    """Configuration class for Lovable Community SAM3."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.
        
        Args:
            config_dict: Optional dictionary with configuration values
        """
        # Database configuration
        self.database_url = os.getenv(
            "DATABASE_URL",
            config_dict.get("database_url") if config_dict else None or "sqlite:///./lovable_community.db"
        )
        
        # API configuration
        self.api_host = os.getenv(
            "API_HOST",
            config_dict.get("api_host") if config_dict else None or "0.0.0.0"
        )
        self.api_port = int(os.getenv(
            "API_PORT",
            config_dict.get("api_port") if config_dict else None or "8000"
        ))
        
        # OpenRouter configuration
        self.openrouter_api_key = os.getenv(
            "OPENROUTER_API_KEY",
            config_dict.get("openrouter_api_key") if config_dict else None or ""
        )
        self.openrouter_base_url = os.getenv(
            "OPENROUTER_BASE_URL",
            config_dict.get("openrouter_base_url") if config_dict else None or "https://openrouter.ai/api/v1"
        )
        
        # TruthGPT configuration
        self.truthgpt_api_key = os.getenv(
            "TRUTHGPT_API_KEY",
            config_dict.get("truthgpt_api_key") if config_dict else None or ""
        )
        self.truthgpt_base_url = os.getenv(
            "TRUTHGPT_BASE_URL",
            config_dict.get("truthgpt_base_url") if config_dict else None or "https://api.truthgpt.com/v1"
        )
        
        # Task manager configuration
        self.max_workers = int(os.getenv(
            "MAX_WORKERS",
            config_dict.get("max_workers") if config_dict else None or "4"
        ))
        self.task_queue_size = int(os.getenv(
            "TASK_QUEUE_SIZE",
            config_dict.get("task_queue_size") if config_dict else None or "100"
        ))
        
        # Cache configuration
        self.cache_ttl = int(os.getenv(
            "CACHE_TTL",
            config_dict.get("cache_ttl") if config_dict else None or "3600"
        ))
        self.cache_max_size = int(os.getenv(
            "CACHE_MAX_SIZE",
            config_dict.get("cache_max_size") if config_dict else None or "1000"
        ))
        
        # Logging configuration
        self.log_level = os.getenv(
            "LOG_LEVEL",
            config_dict.get("log_level") if config_dict else None or "INFO"
        )
    
    def validate(self) -> bool:
        """
        Validate configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If required configuration is missing
        """
        if not self.database_url:
            raise ValueError("DATABASE_URL is required")
        
        logger.info("Configuration validated successfully")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return {
            "database_url": self.database_url,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "openrouter_api_key": "***" if self.openrouter_api_key else "",
            "truthgpt_api_key": "***" if self.truthgpt_api_key else "",
            "max_workers": self.max_workers,
            "task_queue_size": self.task_queue_size,
            "cache_ttl": self.cache_ttl,
            "cache_max_size": self.cache_max_size,
            "log_level": self.log_level
        }




