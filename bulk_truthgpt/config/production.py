"""
Production Configuration
========================

Production configuration for Bulk TruthGPT system.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path

class ProductionConfig:
    """Production configuration settings."""
    
    # System Configuration
    SYSTEM_NAME = "Bulk TruthGPT"
    VERSION = "1.0.0"
    ENVIRONMENT = "production"
    DEBUG = False
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_WORKERS = int(os.getenv("API_WORKERS", "4"))
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/bulk_truthgpt")
    DATABASE_POOL_SIZE = int(os.getenv("DATABASE_POOL_SIZE", "20"))
    DATABASE_MAX_OVERFLOW = int(os.getenv("DATABASE_MAX_OVERFLOW", "30"))
    
    # Redis Configuration
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_POOL_SIZE = int(os.getenv("REDIS_POOL_SIZE", "10"))
    
    # TruthGPT Configuration
    TRUTHGPT_MODEL_PATH = os.getenv("TRUTHGPT_MODEL_PATH", "./models/truthgpt")
    TRUTHGPT_DEVICE = os.getenv("TRUTHGPT_DEVICE", "cuda")
    TRUTHGPT_BATCH_SIZE = int(os.getenv("TRUTHGPT_BATCH_SIZE", "4"))
    TRUTHGPT_MAX_LENGTH = int(os.getenv("TRUTHGPT_MAX_LENGTH", "2048"))
    
    # Generation Configuration
    MAX_DOCUMENTS_PER_TASK = int(os.getenv("MAX_DOCUMENTS_PER_TASK", "1000"))
    MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "10"))
    DOCUMENT_GENERATION_TIMEOUT = int(os.getenv("DOCUMENT_GENERATION_TIMEOUT", "300"))
    
    # Quality Configuration
    MIN_QUALITY_SCORE = float(os.getenv("MIN_QUALITY_SCORE", "0.7"))
    QUALITY_CHECK_INTERVAL = int(os.getenv("QUALITY_CHECK_INTERVAL", "10"))
    AUTO_OPTIMIZATION_ENABLED = os.getenv("AUTO_OPTIMIZATION_ENABLED", "true").lower() == "true"
    
    # Monitoring Configuration
    METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"
    METRICS_PORT = int(os.getenv("METRICS_PORT", "9090"))
    HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv("LOG_FORMAT", "json")
    LOG_FILE = os.getenv("LOG_FILE", "./logs/bulk_truthgpt.log")
    LOG_MAX_SIZE = int(os.getenv("LOG_MAX_SIZE", "100MB"))
    LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    
    # Security Configuration
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Notification Configuration
    NOTIFICATION_ENABLED = os.getenv("NOTIFICATION_ENABLED", "true").lower() == "true"
    EMAIL_SMTP_HOST = os.getenv("EMAIL_SMTP_HOST", "smtp.gmail.com")
    EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
    EMAIL_USERNAME = os.getenv("EMAIL_USERNAME", "")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
    
    # Analytics Configuration
    ANALYTICS_ENABLED = os.getenv("ANALYTICS_ENABLED", "true").lower() == "true"
    ANALYTICS_RETENTION_DAYS = int(os.getenv("ANALYTICS_RETENTION_DAYS", "30"))
    
    # Learning Configuration
    LEARNING_ENABLED = os.getenv("LEARNING_ENABLED", "true").lower() == "true"
    LEARNING_BATCH_SIZE = int(os.getenv("LEARNING_BATCH_SIZE", "100"))
    LEARNING_UPDATE_INTERVAL = int(os.getenv("LEARNING_UPDATE_INTERVAL", "3600"))
    
    # Optimization Configuration
    OPTIMIZATION_ENABLED = os.getenv("OPTIMIZATION_ENABLED", "true").lower() == "true"
    OPTIMIZATION_INTERVAL = int(os.getenv("OPTIMIZATION_INTERVAL", "7200"))
    OPTIMIZATION_MAX_ITERATIONS = int(os.getenv("OPTIMIZATION_MAX_ITERATIONS", "100"))
    
    # Storage Configuration
    STORAGE_PATH = os.getenv("STORAGE_PATH", "./storage")
    TEMPLATES_PATH = os.getenv("TEMPLATES_PATH", "./templates")
    MODELS_PATH = os.getenv("MODELS_PATH", "./models")
    KNOWLEDGE_PATH = os.getenv("KNOWLEDGE_PATH", "./knowledge_base")
    
    # Performance Configuration
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    
    # Rate Limiting
    RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))
    
    @classmethod
    def get_database_config(cls) -> Dict[str, Any]:
        """Get database configuration."""
        return {
            "url": cls.DATABASE_URL,
            "pool_size": cls.DATABASE_POOL_SIZE,
            "max_overflow": cls.DATABASE_MAX_OVERFLOW,
            "echo": cls.DEBUG
        }
    
    @classmethod
    def get_redis_config(cls) -> Dict[str, Any]:
        """Get Redis configuration."""
        return {
            "url": cls.REDIS_URL,
            "pool_size": cls.REDIS_POOL_SIZE,
            "decode_responses": True
        }
    
    @classmethod
    def get_truthgpt_config(cls) -> Dict[str, Any]:
        """Get TruthGPT configuration."""
        return {
            "model_path": cls.TRUTHGPT_MODEL_PATH,
            "device": cls.TRUTHGPT_DEVICE,
            "batch_size": cls.TRUTHGPT_BATCH_SIZE,
            "max_length": cls.TRUTHGPT_MAX_LENGTH
        }
    
    @classmethod
    def get_generation_config(cls) -> Dict[str, Any]:
        """Get generation configuration."""
        return {
            "max_documents_per_task": cls.MAX_DOCUMENTS_PER_TASK,
            "max_concurrent_tasks": cls.MAX_CONCURRENT_TASKS,
            "timeout": cls.DOCUMENT_GENERATION_TIMEOUT,
            "min_quality_score": cls.MIN_QUALITY_SCORE
        }
    
    @classmethod
    def get_monitoring_config(cls) -> Dict[str, Any]:
        """Get monitoring configuration."""
        return {
            "enabled": cls.METRICS_ENABLED,
            "port": cls.METRICS_PORT,
            "health_check_interval": cls.HEALTH_CHECK_INTERVAL
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
            "access_token_expire_minutes": cls.ACCESS_TOKEN_EXPIRE_MINUTES,
            "cors_origins": cls.CORS_ORIGINS
        }
    
    @classmethod
    def get_notification_config(cls) -> Dict[str, Any]:
        """Get notification configuration."""
        return {
            "enabled": cls.NOTIFICATION_ENABLED,
            "email": {
                "smtp_host": cls.EMAIL_SMTP_HOST,
                "smtp_port": cls.EMAIL_SMTP_PORT,
                "username": cls.EMAIL_USERNAME,
                "password": cls.EMAIL_PASSWORD
            }
        }
    
    @classmethod
    def get_analytics_config(cls) -> Dict[str, Any]:
        """Get analytics configuration."""
        return {
            "enabled": cls.ANALYTICS_ENABLED,
            "retention_days": cls.ANALYTICS_RETENTION_DAYS
        }
    
    @classmethod
    def get_learning_config(cls) -> Dict[str, Any]:
        """Get learning configuration."""
        return {
            "enabled": cls.LEARNING_ENABLED,
            "batch_size": cls.LEARNING_BATCH_SIZE,
            "update_interval": cls.LEARNING_UPDATE_INTERVAL
        }
    
    @classmethod
    def get_optimization_config(cls) -> Dict[str, Any]:
        """Get optimization configuration."""
        return {
            "enabled": cls.OPTIMIZATION_ENABLED,
            "interval": cls.OPTIMIZATION_INTERVAL,
            "max_iterations": cls.OPTIMIZATION_MAX_ITERATIONS
        }
    
    @classmethod
    def get_storage_config(cls) -> Dict[str, Any]:
        """Get storage configuration."""
        return {
            "storage_path": cls.STORAGE_PATH,
            "templates_path": cls.TEMPLATES_PATH,
            "models_path": cls.MODELS_PATH,
            "knowledge_path": cls.KNOWLEDGE_PATH
        }
    
    @classmethod
    def get_performance_config(cls) -> Dict[str, Any]:
        """Get performance configuration."""
        return {
            "cache_ttl": cls.CACHE_TTL,
            "cache_max_size": cls.CACHE_MAX_SIZE,
            "request_timeout": cls.REQUEST_TIMEOUT,
            "max_retries": cls.MAX_RETRIES
        }
    
    @classmethod
    def get_rate_limit_config(cls) -> Dict[str, Any]:
        """Get rate limiting configuration."""
        return {
            "enabled": cls.RATE_LIMIT_ENABLED,
            "requests": cls.RATE_LIMIT_REQUESTS,
            "window": cls.RATE_LIMIT_WINDOW
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration."""
        try:
            # Check required paths
            required_paths = [
                cls.STORAGE_PATH,
                cls.TEMPLATES_PATH,
                cls.MODELS_PATH,
                cls.KNOWLEDGE_PATH
            ]
            
            for path in required_paths:
                Path(path).mkdir(parents=True, exist_ok=True)
            
            # Check required environment variables
            required_vars = [
                "SECRET_KEY",
                "DATABASE_URL",
                "REDIS_URL"
            ]
            
            for var in required_vars:
                if not os.getenv(var):
                    raise ValueError(f"Required environment variable {var} not set")
            
            return True
            
        except Exception as e:
            print(f"Configuration validation failed: {str(e)}")
            return False
    
    @classmethod
    def get_all_config(cls) -> Dict[str, Any]:
        """Get all configuration."""
        return {
            "system": {
                "name": cls.SYSTEM_NAME,
                "version": cls.VERSION,
                "environment": cls.ENVIRONMENT,
                "debug": cls.DEBUG
            },
            "api": {
                "host": cls.API_HOST,
                "port": cls.API_PORT,
                "workers": cls.API_WORKERS
            },
            "database": cls.get_database_config(),
            "redis": cls.get_redis_config(),
            "truthgpt": cls.get_truthgpt_config(),
            "generation": cls.get_generation_config(),
            "monitoring": cls.get_monitoring_config(),
            "logging": cls.get_logging_config(),
            "security": cls.get_security_config(),
            "notification": cls.get_notification_config(),
            "analytics": cls.get_analytics_config(),
            "learning": cls.get_learning_config(),
            "optimization": cls.get_optimization_config(),
            "storage": cls.get_storage_config(),
            "performance": cls.get_performance_config(),
            "rate_limit": cls.get_rate_limit_config()
        }











