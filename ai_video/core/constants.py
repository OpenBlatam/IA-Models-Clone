from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import os
from pathlib import Path
from typing import List, Dict, Any
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
AI Video System - Constants

Production-ready constants and configuration defaults.
"""


# System constants
SYSTEM_NAME = "Onyx AI Video System"
DEFAULT_CONFIG_PATH = "./config.yaml"
DEFAULT_LOG_LEVEL = "INFO"

# Version information
VERSION = "1.0.0"
AUTHOR = "AI Video Team"
DESCRIPTION = "Production-ready AI video generation system"

# Default configuration
DEFAULT_CONFIG = {
    "plugins": {
        "auto_discover": True,
        "auto_load": True,
        "validation_level": "standard",
        "plugin_dirs": ["./plugins"],
        "enable_events": True,
        "enable_metrics": True
    },
    "workflow": {
        "max_concurrent_workflows": 5,
        "workflow_timeout": 300,
        "enable_retry": True,
        "max_retries": 3,
        "extraction_timeout": 60,
        "max_content_length": 50000,
        "enable_language_detection": True,
        "default_duration": 30.0,
        "default_resolution": "1920x1080",
        "default_quality": "high",
        "enable_avatar_selection": True,
        "enable_caching": True,
        "cache_ttl": 3600,
        "enable_metrics": True,
        "enable_monitoring": True
    },
    "ai": {
        "default_model": "gpt-4",
        "fallback_model": "gpt-3.5-turbo",
        "max_tokens": 4000,
        "temperature": 0.7,
        "api_timeout": 30,
        "api_retries": 3,
        "enable_streaming": False,
        "enable_content_optimization": True,
        "enable_short_video_optimization": True,
        "enable_langchain_analysis": True,
        "suggestion_count": 3,
        "enable_music_suggestions": True,
        "enable_visual_suggestions": True,
        "enable_transition_suggestions": True
    },
    "storage": {
        "local_storage_path": "./storage",
        "temp_directory": "./temp",
        "output_directory": "./output",
        "max_file_size": 104857600,
        "allowed_formats": ["mp4", "avi", "mov", "mkv"],
        "enable_compression": True,
        "auto_cleanup": True,
        "cleanup_interval": 86400,
        "max_age_days": 7
    },
    "security": {
        "enable_auth": False,
        "auth_token_expiry": 3600,
        "enable_url_validation": True,
        "allowed_domains": [],
        "blocked_domains": [],
        "enable_content_filtering": True,
        "filter_inappropriate_content": True,
        "enable_nsfw_detection": False,
        "enable_rate_limiting": True,
        "max_requests_per_minute": 60,
        "max_requests_per_hour": 1000
    },
    "monitoring": {
        "log_level": "INFO",
        "log_file": None,
        "enable_structured_logging": True,
        "enable_metrics": True,
        "metrics_port": 9090,
        "enable_prometheus": True,
        "enable_health_checks": True,
        "health_check_interval": 300,
        "enable_alerts": False,
        "alert_webhook_url": None
    },
    "environment": "production",
    "debug": False,
    "version": VERSION
}

# File and format constants
SUPPORTED_FORMATS = {
    "video": ["mp4", "avi", "mov", "mkv", "webm", "flv"],
    "audio": ["mp3", "wav", "aac", "ogg", "flac"],
    "image": ["jpg", "jpeg", "png", "gif", "webp", "svg"],
    "document": ["pdf", "doc", "docx", "txt", "md"]
}

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
DEFAULT_TIMEOUT = 30  # seconds
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 1  # seconds

# Workflow constants
WORKFLOW_STAGES = [
    "initializing",
    "plugins_loading", 
    "plugins_ready",
    "extracting",
    "suggesting",
    "generating",
    "completed",
    "failed",
    "cancelled",
    "plugin_error"
]

WORKFLOW_STATUSES = {
    "PENDING": "pending",
    "EXTRACTING": "extracting",
    "SUGGESTING": "suggesting", 
    "GENERATING": "generating",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "CANCELLED": "cancelled"
}

WORKFLOW_PRIORITIES = [
    "low",
    "normal",
    "high",
    "urgent"
]

# Plugin constants
PLUGIN_CATEGORIES = [
    "extractor",
    "suggestion_engine", 
    "video_generator",
    "processor",
    "analyzer",
    "utility"
]

PLUGIN_TYPES = [
    "extractor",
    "suggestion_engine", 
    "video_generator",
    "processor",
    "analyzer",
    "utility"
]

PLUGIN_STATUSES = [
    "active",
    "inactive",
    "error",
    "loading",
    "ready"
]

VALIDATION_LEVELS = [
    "basic",
    "standard",
    "strict", 
    "security"
]

# Security constants
SECURITY_CHECKS = [
    "url_validation",
    "content_filtering",
    "nsfw_detection",
    "rate_limiting",
    "authentication",
    "authorization"
]

SECURITY_LEVELS = [
    "low",
    "medium",
    "high",
    "critical"
]

ENCRYPTION_ALGORITHMS = [
    "AES-256-GCM",
    "ChaCha20-Poly1305",
    "AES-256-CBC"
]

HASH_ALGORITHMS = [
    "SHA-256",
    "SHA-512",
    "bcrypt",
    "argon2"
]

# Performance constants
PERFORMANCE_THRESHOLDS = {
    "max_response_time": 30.0,  # seconds
    "max_memory_usage": 0.8,    # 80% of available memory
    "max_cpu_usage": 0.9,       # 90% of available CPU
    "max_disk_usage": 0.9,      # 90% of available disk space
    "min_success_rate": 0.95    # 95% success rate
}

CACHE_TTL_DEFAULTS = {
    "workflow_state": 3600,      # 1 hour
    "plugin_config": 86400,      # 24 hours
    "extracted_content": 1800,   # 30 minutes
    "generated_video": 604800,   # 1 week
    "system_stats": 300,         # 5 minutes
    "health_status": 60          # 1 minute
}

RATE_LIMIT_DEFAULTS = {
    "requests_per_minute": 60,
    "requests_per_hour": 1000,
    "requests_per_day": 10000
}

# Monitoring constants
METRICS_NAMES = {
    "workflow_duration": "ai_video_workflow_duration_seconds",
    "workflow_success_rate": "ai_video_workflow_success_rate",
    "plugin_usage": "ai_video_plugin_usage_total",
    "plugin_errors": "ai_video_plugin_errors_total",
    "extraction_duration": "ai_video_extraction_duration_seconds",
    "generation_duration": "ai_video_generation_duration_seconds",
    "memory_usage": "ai_video_memory_usage_bytes",
    "cpu_usage": "ai_video_cpu_usage_percent"
}

METRIC_TYPES = [
    "counter",
    "gauge",
    "histogram",
    "summary"
]

ALERT_SEVERITIES = [
    "info",
    "warning",
    "error",
    "critical"
]

HEALTH_STATUSES = [
    "healthy",
    "degraded",
    "unhealthy",
    "unknown"
]

# Error codes
ERROR_CODES = {
    "CONFIG_ERROR": "CONFIG_ERROR",
    "PLUGIN_ERROR": "PLUGIN_ERROR", 
    "WORKFLOW_ERROR": "WORKFLOW_ERROR",
    "VALIDATION_ERROR": "VALIDATION_ERROR",
    "EXTRACTION_ERROR": "EXTRACTION_ERROR",
    "GENERATION_ERROR": "GENERATION_ERROR",
    "STORAGE_ERROR": "STORAGE_ERROR",
    "SECURITY_ERROR": "SECURITY_ERROR",
    "PERFORMANCE_ERROR": "PERFORMANCE_ERROR",
    "RESOURCE_ERROR": "RESOURCE_ERROR",
    "UNKNOWN_ERROR": "UNKNOWN_ERROR"
}

# Environment variables
ENV_VARS = {
    "CONFIG_FILE": "AI_VIDEO_CONFIG_FILE",
    "LOG_LEVEL": "AI_VIDEO_LOG_LEVEL",
    "ENVIRONMENT": "AI_VIDEO_ENVIRONMENT",
    "DEBUG": "AI_VIDEO_DEBUG",
    "STORAGE_PATH": "AI_VIDEO_STORAGE_PATH",
    "TEMP_DIR": "AI_VIDEO_TEMP_DIR",
    "OUTPUT_DIR": "AI_VIDEO_OUTPUT_DIR",
    "PLUGIN_DIRS": "AI_VIDEO_PLUGIN_DIRS",
    "API_KEY": "AI_VIDEO_API_KEY",
    "SECRET_KEY": "AI_VIDEO_SECRET_KEY"
}

# Default paths
DEFAULT_PATHS = {
    "config": "./config",
    "plugins": "./plugins", 
    "storage": "./storage",
    "temp": "./temp",
    "output": "./output",
    "logs": "./logs",
    "examples": "./examples"
}

# Cache constants
CACHE_KEYS = {
    "workflow_state": "workflow_state:{workflow_id}",
    "plugin_config": "plugin_config:{plugin_name}",
    "extracted_content": "extracted_content:{url_hash}",
    "generated_video": "generated_video:{video_id}",
    "system_stats": "system_stats",
    "health_status": "health_status"
}

CACHE_TTL = {
    "workflow_state": 3600,      # 1 hour
    "plugin_config": 86400,      # 24 hours
    "extracted_content": 1800,   # 30 minutes
    "generated_video": 604800,   # 1 week
    "system_stats": 300,         # 5 minutes
    "health_status": 60          # 1 minute
}

# Rate limiting constants
RATE_LIMITS = {
    "default": {
        "requests_per_minute": 60,
        "requests_per_hour": 1000,
        "requests_per_day": 10000
    },
    "extraction": {
        "requests_per_minute": 30,
        "requests_per_hour": 500,
        "requests_per_day": 5000
    },
    "generation": {
        "requests_per_minute": 10,
        "requests_per_hour": 100,
        "requests_per_day": 1000
    }
}

# Health check constants
HEALTH_CHECK_INTERVALS = {
    "system": 300,    # 5 minutes
    "plugins": 60,    # 1 minute
    "storage": 300,   # 5 minutes
    "network": 60,    # 1 minute
    "database": 300   # 5 minutes
}

# Logging constants
LOG_FORMATS = {
    "simple": "%(asctime)s - %(levelname)s - %(message)s",
    "detailed": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "json": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
}

LOG_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50
}

# Validation constants
VALIDATION_RULES = {
    "url": {
        "max_length": 2048,
        "allowed_schemes": ["http", "https"],
        "required_fields": ["scheme", "netloc"]
    },
    "filename": {
        "max_length": 255,
        "allowed_chars": "a-zA-Z0-9._-",
        "forbidden_chars": ["<", ">", ":", "\"", "|", "?", "*"]
    },
    "workflow_id": {
        "max_length": 64,
        "pattern": r"^[a-zA-Z0-9_-]+$"
    },
    "plugin_name": {
        "max_length": 50,
        "pattern": r"^[a-zA-Z][a-zA-Z0-9_-]*$"
    }
}

# API constants
API_ENDPOINTS = {
    "health": "/health",
    "status": "/status",
    "metrics": "/metrics",
    "workflows": "/workflows",
    "plugins": "/plugins",
    "config": "/config"
}

API_RESPONSE_CODES = {
    "SUCCESS": 200,
    "CREATED": 201,
    "BAD_REQUEST": 400,
    "UNAUTHORIZED": 401,
    "FORBIDDEN": 403,
    "NOT_FOUND": 404,
    "CONFLICT": 409,
    "INTERNAL_ERROR": 500,
    "SERVICE_UNAVAILABLE": 503
}

# Utility functions
def get_default_path(path_type: str) -> Path:
    """Get default path for a given type."""
    return Path(DEFAULT_PATHS.get(path_type, "./")) 