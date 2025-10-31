"""
Configuration Management
========================

Ultra-advanced configuration management with environment support.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

class Environment(str, Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

@dataclass
class BaseConfig:
    """Base configuration."""
    # Flask
    SECRET_KEY: str = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG: bool = False
    TESTING: bool = False
    
    # Database
    SQLALCHEMY_DATABASE_URI: str = os.getenv('DATABASE_URL', 'sqlite:///bulk_truthgpt.db')
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False
    SQLALCHEMY_ENGINE_OPTIONS: Dict[str, Any] = field(default_factory=lambda: {
        'pool_pre_ping': True,
        'pool_recycle': 300,
        'pool_timeout': 20,
        'max_overflow': 0
    })
    
    # JWT
    JWT_SECRET_KEY: str = os.getenv('JWT_SECRET_KEY', 'jwt-secret-key')
    JWT_ACCESS_TOKEN_EXPIRES: int = 3600  # 1 hour
    JWT_REFRESH_TOKEN_EXPIRES: int = 2592000  # 30 days
    JWT_BLACKLIST_ENABLED: bool = True
    JWT_BLACKLIST_TOKEN_CHECKS: list = field(default_factory=lambda: ['access', 'refresh'])
    
    # Redis
    REDIS_URL: Optional[str] = os.getenv('REDIS_URL')
    REDIS_HOST: str = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT: int = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB: int = int(os.getenv('REDIS_DB', 0))
    
    # Caching
    CACHE_TYPE: str = 'redis'
    CACHE_REDIS_URL: Optional[str] = os.getenv('REDIS_URL')
    CACHE_DEFAULT_TIMEOUT: int = 300
    
    # CORS
    CORS_ORIGINS: list = field(default_factory=lambda: ['*'])
    
    # Rate Limiting
    RATELIMIT_STORAGE_URL: Optional[str] = os.getenv('REDIS_URL')
    
    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT: str = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    
    # Application
    VERSION: str = os.getenv('VERSION', '1.0.0')
    API_TITLE: str = 'Bulk TruthGPT API'
    API_VERSION: str = 'v1'
    
    # Optimization
    ENABLE_OPTIMIZATION: bool = True
    ENABLE_CACHING: bool = True
    ENABLE_COMPRESSION: bool = True
    ENABLE_MONITORING: bool = True
    
    # Security
    ENABLE_SECURITY: bool = True
    ENABLE_ENCRYPTION: bool = True
    ENABLE_AUTHENTICATION: bool = True
    
    # Performance
    ENABLE_GPU: bool = os.getenv('ENABLE_GPU', 'false').lower() == 'true'
    ENABLE_DISTRIBUTED: bool = os.getenv('ENABLE_DISTRIBUTED', 'false').lower() == 'true'
    ENABLE_QUANTUM: bool = os.getenv('ENABLE_QUANTUM', 'false').lower() == 'true'
    ENABLE_EDGE: bool = os.getenv('ENABLE_EDGE', 'false').lower() == 'true'

@dataclass
class DevelopmentConfig(BaseConfig):
    """Development configuration."""
    DEBUG: bool = True
    TESTING: bool = False
    
    # Database
    SQLALCHEMY_DATABASE_URI: str = os.getenv('DATABASE_URL', 'sqlite:///bulk_truthgpt_dev.db')
    
    # Logging
    LOG_LEVEL: str = 'DEBUG'
    
    # CORS
    CORS_ORIGINS: list = field(default_factory=lambda: ['http://localhost:3000', 'http://localhost:8080'])
    
    # Development features
    ENABLE_DEBUG_TOOLBAR: bool = True
    ENABLE_PROFILER: bool = True

@dataclass
class TestingConfig(BaseConfig):
    """Testing configuration."""
    DEBUG: bool = False
    TESTING: bool = True
    
    # Database
    SQLALCHEMY_DATABASE_URI: str = 'sqlite:///:memory:'
    
    # JWT
    JWT_ACCESS_TOKEN_EXPIRES: int = 300  # 5 minutes for testing
    
    # Caching
    CACHE_TYPE: str = 'simple'
    
    # Testing features
    ENABLE_OPTIMIZATION: bool = False
    ENABLE_CACHING: bool = False
    ENABLE_MONITORING: bool = False

@dataclass
class ProductionConfig(BaseConfig):
    """Production configuration."""
    DEBUG: bool = False
    TESTING: bool = False
    
    # Security
    SECRET_KEY: str = os.getenv('SECRET_KEY')
    JWT_SECRET_KEY: str = os.getenv('JWT_SECRET_KEY')
    
    # Database
    SQLALCHEMY_DATABASE_URI: str = os.getenv('DATABASE_URL')
    
    # Redis
    REDIS_URL: str = os.getenv('REDIS_URL')
    CACHE_REDIS_URL: str = os.getenv('REDIS_URL')
    RATELIMIT_STORAGE_URL: str = os.getenv('REDIS_URL')
    
    # CORS
    CORS_ORIGINS: list = field(default_factory=lambda: os.getenv('CORS_ORIGINS', '').split(','))
    
    # Logging
    LOG_LEVEL: str = 'WARNING'
    
    # Production features
    ENABLE_DEBUG_TOOLBAR: bool = False
    ENABLE_PROFILER: bool = False

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config(config_name: str = 'default'):
    """Get configuration class."""
    return config.get(config_name, config['default'])









