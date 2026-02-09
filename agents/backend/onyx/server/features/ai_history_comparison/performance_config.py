"""
Performance Configuration for AI History Comparison System
Configuración de Rendimiento para el Sistema de Comparación de Historial de IA
"""

from pydantic import BaseSettings
from typing import Dict, Any
import os

class PerformanceConfig(BaseSettings):
    """Performance optimization settings"""
    
    # FastAPI Performance Settings
    fastapi_debug: bool = False
    fastapi_reload: bool = False
    fastapi_workers: int = 1
    
    # Uvicorn Performance Settings
    uvicorn_host: str = "0.0.0.0"
    uvicorn_port: int = 8000
    uvicorn_workers: int = 1
    uvicorn_loop: str = "uvloop"  # Faster event loop
    uvicorn_http: str = "httptools"  # Faster HTTP parser
    
    # Database Performance Settings
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30
    db_pool_recycle: int = 3600
    
    # Redis Performance Settings
    redis_max_connections: int = 20
    redis_socket_timeout: int = 5
    redis_socket_connect_timeout: int = 5
    
    # Caching Settings
    cache_ttl: int = 300  # 5 minutes
    cache_max_size: int = 1000
    
    # Logging Performance
    log_level: str = "INFO"
    log_format: str = "json"  # Faster than text
    log_rotation: str = "10 MB"
    log_retention: str = "7 days"
    
    # AI/ML Performance Settings
    ml_batch_size: int = 32
    ml_max_workers: int = 4
    ml_cache_models: bool = True
    
    # API Performance Settings
    api_timeout: int = 30
    api_max_retries: int = 3
    api_rate_limit: int = 100  # requests per minute
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Performance optimization functions
def get_optimized_uvicorn_config() -> Dict[str, Any]:
    """Get optimized Uvicorn configuration"""
    config = PerformanceConfig()
    
    return {
        "host": config.uvicorn_host,
        "port": config.uvicorn_port,
        "workers": config.uvicorn_workers,
        "loop": config.uvicorn_loop,
        "http": config.uvicorn_http,
        "access_log": False,  # Disable access logs for performance
        "log_level": config.log_level.lower(),
        "reload": config.fastapi_reload,
        "reload_dirs": ["."] if config.fastapi_reload else None,
    }

def get_optimized_database_config() -> Dict[str, Any]:
    """Get optimized database configuration"""
    config = PerformanceConfig()
    
    return {
        "pool_size": config.db_pool_size,
        "max_overflow": config.db_max_overflow,
        "pool_timeout": config.db_pool_timeout,
        "pool_recycle": config.db_pool_recycle,
        "pool_pre_ping": True,  # Verify connections
        "echo": False,  # Disable SQL logging for performance
    }

def get_optimized_redis_config() -> Dict[str, Any]:
    """Get optimized Redis configuration"""
    config = PerformanceConfig()
    
    return {
        "max_connections": config.redis_max_connections,
        "socket_timeout": config.redis_socket_timeout,
        "socket_connect_timeout": config.redis_socket_connect_timeout,
        "retry_on_timeout": True,
        "health_check_interval": 30,
    }

# Environment-specific optimizations
def get_environment_optimizations() -> Dict[str, Any]:
    """Get environment-specific optimizations"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return {
            "debug": False,
            "reload": False,
            "workers": 4,
            "log_level": "WARNING",
            "access_log": False,
        }
    elif env == "staging":
        return {
            "debug": False,
            "reload": False,
            "workers": 2,
            "log_level": "INFO",
            "access_log": True,
        }
    else:  # development
        return {
            "debug": True,
            "reload": True,
            "workers": 1,
            "log_level": "DEBUG",
            "access_log": True,
        }

# Quick performance check
def check_performance_requirements() -> Dict[str, bool]:
    """Check if performance requirements are met"""
    import psutil
    import sys
    
    checks = {
        "python_version": sys.version_info >= (3, 8),
        "memory_gb": psutil.virtual_memory().total >= 2 * 1024**3,  # 2GB
        "cpu_cores": psutil.cpu_count() >= 2,
        "disk_space_gb": psutil.disk_usage('/').free >= 1 * 1024**3,  # 1GB
    }
    
    return checks

if __name__ == "__main__":
    # Print current configuration
    config = PerformanceConfig()
    print("🚀 Performance Configuration:")
    print(f"   FastAPI Debug: {config.fastapi_debug}")
    print(f"   Uvicorn Workers: {config.uvicorn_workers}")
    print(f"   Database Pool Size: {config.db_pool_size}")
    print(f"   Cache TTL: {config.cache_ttl}s")
    print(f"   Log Level: {config.log_level}")
    
    # Check requirements
    print("\n📊 Performance Requirements Check:")
    checks = check_performance_requirements()
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check}: {passed}")
    
    # Print optimized configs
    print("\n⚡ Optimized Configurations:")
    print("   Uvicorn:", get_optimized_uvicorn_config())
    print("   Database:", get_optimized_database_config())
    print("   Redis:", get_optimized_redis_config())







