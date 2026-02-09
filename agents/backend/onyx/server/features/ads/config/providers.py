"""
Provider Configurations for the ads feature.

This module consolidates provider configuration functions from optimized_config.py,
providing clean interfaces for LLM, embeddings, Redis, and database configurations.
"""

from typing import Dict, Any, Optional
from .settings import get_optimized_settings

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    # Fallback for environments without langchain
    ChatOpenAI = None
    OpenAIEmbeddings = None


def get_llm_config():
    """Get LLM configuration with optimized settings.
    
    Returns:
        ChatOpenAI instance configured with optimized settings
        
    Raises:
        ImportError: If langchain_openai is not available
    """
    if ChatOpenAI is None:
        raise ImportError("langchain_openai is required for LLM configuration")
    
    settings = get_optimized_settings()
    
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        max_tokens=settings.openai_max_tokens,
        api_key=settings.openai_api_key,
        timeout=settings.openai_timeout,
        max_retries=settings.openai_max_retries
    )


def get_embeddings_config():
    """Get embeddings configuration with optimized settings.
    
    Returns:
        OpenAIEmbeddings instance configured with optimized settings
        
    Raises:
        ImportError: If langchain_openai is not available
    """
    if OpenAIEmbeddings is None:
        raise ImportError("langchain_openai is required for embeddings configuration")
    
    settings = get_optimized_settings()
    
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        chunk_size=settings.chunk_size
    )


def get_redis_config() -> Dict[str, Any]:
    """Get Redis configuration with optimized settings.
    
    Returns:
        Dictionary with Redis configuration parameters
    """
    settings = get_optimized_settings()
    
    return {
        "url": settings.redis_url,
        "max_connections": settings.redis_max_connections,
        "socket_timeout": settings.redis_socket_timeout,
        "socket_connect_timeout": settings.redis_socket_connect_timeout,
        "encoding": "utf-8",
        "decode_responses": True
    }


def get_database_config() -> Dict[str, Any]:
    """Get database configuration with optimized settings.
    
    Returns:
        Dictionary with database configuration parameters
    """
    settings = get_optimized_settings()
    
    return {
        "url": settings.database_url,
        "pool_size": settings.database_pool_size,
        "max_overflow": settings.database_max_overflow,
        "pool_timeout": settings.database_pool_timeout,
        "pool_recycle": settings.database_pool_recycle,
        "echo": settings.debug
    }


def get_storage_config() -> Dict[str, Any]:
    """Get storage configuration with optimized settings.
    
    Returns:
        Dictionary with storage configuration parameters
    """
    settings = get_optimized_settings()
    
    return {
        "storage_path": settings.storage_path,
        "storage_url": settings.storage_url,
        "max_file_size": settings.max_file_size,
        "allowed_file_types": settings.allowed_file_types,
        "max_image_size": settings.max_image_size,
        "max_image_size_bytes": settings.max_image_size_bytes,
        "jpeg_quality": settings.jpeg_quality,
        "png_optimize": settings.png_optimize,
        "image_cache_ttl": settings.image_cache_ttl
    }


def get_api_config() -> Dict[str, Any]:
    """Get API configuration with optimized settings.
    
    Returns:
        Dictionary with API configuration parameters
    """
    settings = get_optimized_settings()
    
    return {
        "host": settings.host,
        "port": settings.port,
        "workers": settings.workers,
        "max_requests": settings.max_requests,
        "max_requests_jitter": settings.max_requests_jitter,
        "max_concurrent_requests": settings.max_concurrent_requests,
        "request_timeout": settings.request_timeout,
        "connection_timeout": settings.connection_timeout,
        "keepalive_timeout": settings.keepalive_timeout
    }


def get_monitoring_config() -> Dict[str, Any]:
    """Get monitoring configuration with optimized settings.
    
    Returns:
        Dictionary with monitoring configuration parameters
    """
    settings = get_optimized_settings()
    
    return {
        "log_level": settings.log_level,
        "log_format": settings.log_format,
        "sentry_dsn": settings.sentry_dsn,
        "prometheus_enabled": settings.prometheus_enabled,
        "health_check_interval": settings.health_check_interval
    }


def get_security_config() -> Dict[str, Any]:
    """Get security configuration with optimized settings.
    
    Returns:
        Dictionary with security configuration parameters
    """
    settings = get_optimized_settings()
    
    return {
        "cors_origins": settings.cors_origins,
        "cors_allow_credentials": settings.cors_allow_credentials,
        "cors_allow_methods": settings.cors_allow_methods,
        "cors_allow_headers": settings.cors_allow_headers
    }


def get_rate_limiting_config() -> Dict[str, Any]:
    """Get rate limiting configuration with optimized settings.
    
    Returns:
        Dictionary with rate limiting configuration parameters
    """
    settings = get_optimized_settings()
    
    return {
        "rate_limits": settings.rate_limits,
        "max_concurrent_requests": settings.max_concurrent_requests
    }


def get_background_tasks_config() -> Dict[str, Any]:
    """Get background tasks configuration with optimized settings.
    
    Returns:
        Dictionary with background tasks configuration parameters
    """
    settings = get_optimized_settings()
    
    return {
        "background_task_workers": settings.background_task_workers,
        "task_queue_size": settings.task_queue_size,
        "task_timeout": settings.task_timeout
    }


def get_analytics_config() -> Dict[str, Any]:
    """Get analytics configuration with optimized settings.
    
    Returns:
        Dictionary with analytics configuration parameters
    """
    settings = get_optimized_settings()
    
    return {
        "analytics_enabled": settings.analytics_enabled,
        "analytics_retention_days": settings.analytics_retention_days,
        "analytics_batch_size": settings.analytics_batch_size,
        "analytics_flush_interval": settings.analytics_flush_interval
    }


def get_file_processing_config() -> Dict[str, Any]:
    """Get file processing configuration with optimized settings.
    
    Returns:
        Dictionary with file processing configuration parameters
    """
    settings = get_optimized_settings()
    
    return {
        "chunk_size": settings.chunk_size,
        "max_memory_usage": settings.max_memory_usage,
        "temp_file_cleanup_interval": settings.temp_file_cleanup_interval
    }


def get_cache_config() -> Dict[str, Any]:
    """Get cache configuration with optimized settings.
    
    Returns:
        Dictionary with cache configuration parameters
    """
    settings = get_optimized_settings()
    
    return {
        "cache_ttl": settings.cache_ttl,
        "cache_max_size": settings.cache_max_size,
        "cache_cleanup_interval": settings.cache_cleanup_interval
    }


def get_environment_config() -> Dict[str, Any]:
    """Get environment configuration with optimized settings.
    
    Returns:
        Dictionary with environment configuration parameters
    """
    settings = get_optimized_settings()
    
    return {
        "environment": settings.environment.value,
        "debug": settings.debug
    } 