"""
BUL Configuration
================

Centralized configuration for the BUL system.
"""

import os
from typing import List, Dict, Any
from pydantic import BaseSettings, Field
from pathlib import Path

class BULConfig(BaseSettings):
    """BUL system configuration."""
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="BUL_API_HOST")
    api_port: int = Field(default=8000, env="BUL_API_PORT")
    debug_mode: bool = Field(default=False, env="BUL_DEBUG")
    
    # AI Configuration
    openai_api_key: str = Field(env="OPENAI_API_KEY")
    openrouter_api_key: str = Field(env="OPENROUTER_API_KEY")
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///bul.db", env="BUL_DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379", env="BUL_REDIS_URL")
    
    # Processing Configuration
    max_concurrent_tasks: int = Field(default=5, env="BUL_MAX_CONCURRENT_TASKS")
    task_timeout: int = Field(default=300, env="BUL_TASK_TIMEOUT")  # 5 minutes
    max_retries: int = Field(default=3, env="BUL_MAX_RETRIES")
    
    # Document Configuration
    supported_formats: List[str] = Field(default=["markdown", "html", "pdf", "docx"])
    max_document_size: int = Field(default=10485760, env="BUL_MAX_DOCUMENT_SIZE")  # 10MB
    
    # Business Areas
    enabled_business_areas: List[str] = Field(default=[
        "marketing", "sales", "operations", "hr", "finance",
        "legal", "technical", "content", "strategy", "customer_service"
    ])
    
    # Document Types
    document_types: Dict[str, List[str]] = Field(default={
        "marketing": ["strategy", "campaign", "content", "analysis"],
        "sales": ["proposal", "presentation", "playbook", "forecast"],
        "operations": ["manual", "procedure", "workflow", "report"],
        "hr": ["policy", "training", "job_description", "evaluation"],
        "finance": ["budget", "forecast", "analysis", "report"],
        "legal": ["contract", "policy", "compliance", "agreement"],
        "technical": ["documentation", "specification", "guide", "troubleshooting"],
        "content": ["article", "blog", "whitepaper", "case_study"],
        "strategy": ["plan", "roadmap", "initiative", "assessment"],
        "customer_service": ["faq", "guide", "policy", "training"]
    })
    
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
        
        return errors
    
    def get_business_area_config(self, area: str) -> Dict[str, Any]:
        """Get configuration for a specific business area."""
        return {
            "enabled": area in self.enabled_business_areas,
            "document_types": self.document_types.get(area, []),
            "priority": 1  # Default priority
        }
    
    def get_document_type_config(self, doc_type: str) -> Dict[str, Any]:
        """Get configuration for a specific document type."""
        return {
            "supported_formats": self.supported_formats,
            "max_size": self.max_document_size,
            "timeout": self.task_timeout
        }

# Global configuration instance
config = BULConfig()

def get_config() -> BULConfig:
    """Get the global configuration instance."""
    return config


