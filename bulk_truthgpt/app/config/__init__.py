"""
Configuration module for Ultimate Enhanced Supreme Production system
"""

import os
from pathlib import Path

class Config:
    """Base configuration class."""
    
    # Application
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'ultimate-enhanced-supreme-secret-key'
    FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///ultimate_enhanced_supreme.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # JWT
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'jwt-secret-string'
    JWT_ACCESS_TOKEN_EXPIRES = int(os.environ.get('JWT_ACCESS_TOKEN_EXPIRES', 3600))
    
    # Cache
    CACHE_TYPE = os.environ.get('CACHE_TYPE', 'simple')
    CACHE_DEFAULT_TIMEOUT = int(os.environ.get('CACHE_DEFAULT_TIMEOUT', 300))
    
    # Ultimate Enhanced Supreme Configuration
    ULTIMATE_ENHANCED_SUPREME_CONFIG_PATH = os.environ.get(
        'ULTIMATE_ENHANCED_SUPREME_CONFIG_PATH',
        str(Path(__file__).parent.parent.parent / 'ultimate_enhanced_supreme_production_config.yaml')
    )
    
    # Performance Configuration
    MAX_CONCURRENT_GENERATIONS = int(os.environ.get('MAX_CONCURRENT_GENERATIONS', 10000))
    MAX_DOCUMENTS_PER_QUERY = int(os.environ.get('MAX_DOCUMENTS_PER_QUERY', 1000000))
    MAX_CONTINUOUS_DOCUMENTS = int(os.environ.get('MAX_CONTINUOUS_DOCUMENTS', 10000000))
    GENERATION_TIMEOUT = float(os.environ.get('GENERATION_TIMEOUT', 300.0))
    OPTIMIZATION_TIMEOUT = float(os.environ.get('OPTIMIZATION_TIMEOUT', 60.0))
    
    # Monitoring Configuration
    MONITORING_INTERVAL = float(os.environ.get('MONITORING_INTERVAL', 1.0))
    HEALTH_CHECK_INTERVAL = float(os.environ.get('HEALTH_CHECK_INTERVAL', 5.0))
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'logs/ultimate_enhanced_supreme.log')

class DevelopmentConfig(Config):
    """Development configuration."""
    
    DEBUG = True
    TESTING = False
    
    # Development database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or 'sqlite:///ultimate_enhanced_supreme_dev.db'
    
    # Development cache
    CACHE_TYPE = 'simple'
    
    # Development logging
    LOG_LEVEL = 'DEBUG'

class TestingConfig(Config):
    """Testing configuration."""
    
    TESTING = True
    DEBUG = True
    
    # Testing database
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    
    # Testing cache
    CACHE_TYPE = 'simple'
    
    # Testing JWT
    JWT_ACCESS_TOKEN_EXPIRES = 300  # 5 minutes for testing
    
    # Testing logging
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration."""
    
    DEBUG = False
    TESTING = False
    
    # Production database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    
    # Production cache
    CACHE_TYPE = os.environ.get('CACHE_TYPE', 'redis')
    CACHE_REDIS_URL = os.environ.get('CACHE_REDIS_URL')
    
    # Production JWT
    JWT_ACCESS_TOKEN_EXPIRES = int(os.environ.get('JWT_ACCESS_TOKEN_EXPIRES', 3600))
    
    # Production logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    # Production security
    SECRET_KEY = os.environ.get('SECRET_KEY')
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY')

class StagingConfig(Config):
    """Staging configuration."""
    
    DEBUG = False
    TESTING = False
    
    # Staging database
    SQLALCHEMY_DATABASE_URI = os.environ.get('STAGING_DATABASE_URL')
    
    # Staging cache
    CACHE_TYPE = os.environ.get('CACHE_TYPE', 'redis')
    CACHE_REDIS_URL = os.environ.get('CACHE_REDIS_URL')
    
    # Staging logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')









