"""
Constants and configuration values for Blaze AI.

This module provides common constants, default values, and configuration
parameters used throughout the Blaze AI system.
"""

# =============================================================================
# System Constants
# =============================================================================

# Version information
VERSION = "2.0.0"
AUTHOR = "Blaze AI Team"
DESCRIPTION = "Advanced AI Module for Content Generation and Analysis"

# Default configuration values
DEFAULT_SYSTEM_MODE = "development"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_API_PORT = 8000
DEFAULT_GRADIO_PORT = 7860
DEFAULT_METRICS_PORT = 9090

# =============================================================================
# Performance Constants
# =============================================================================

# Memory thresholds (percentages)
MEMORY_WARNING_THRESHOLD = 80.0
MEMORY_CRITICAL_THRESHOLD = 95.0
GPU_MEMORY_WARNING_THRESHOLD = 85.0
GPU_MEMORY_CRITICAL_THRESHOLD = 95.0

# Cache settings
DEFAULT_CACHE_TTL = 3600  # 1 hour
DEFAULT_CACHE_MAX_SIZE = 10000
DEFAULT_MODEL_CACHE_SIZE = 100

# Batch size limits
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 512
DEFAULT_BATCH_SIZE = 32

# =============================================================================
# Training Constants
# =============================================================================

# Learning rate bounds
MIN_LEARNING_RATE = 1e-7
MAX_LEARNING_RATE = 1.0
DEFAULT_LEARNING_RATE = 1e-4

# Training limits
MAX_EPOCHS = 10000
MAX_STEPS = 1000000
DEFAULT_EPOCHS = 100
DEFAULT_STEPS = 10000

# Gradient settings
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
MAX_GRADIENT_ACCUMULATION_STEPS = 128
DEFAULT_GRADIENT_CLIP_NORM = 1.0

# Early stopping
DEFAULT_PATIENCE = 10
MIN_PATIENCE = 1
MAX_PATIENCE = 100

# =============================================================================
# Model Constants
# =============================================================================

# Supported model types
SUPPORTED_MODEL_TYPES = [
    "transformer",
    "diffusion",
    "classification",
    "generation",
    "embedding"
]

# Supported precision types
SUPPORTED_PRECISIONS = [
    "float16",
    "float32",
    "bfloat16",
    "int8"
]

# Device types
SUPPORTED_DEVICES = [
    "cpu",
    "cuda",
    "mps",
    "auto"
]

# =============================================================================
# API Constants
# =============================================================================

# HTTP status codes
HTTP_OK = 200
HTTP_CREATED = 201
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_INTERNAL_ERROR = 500

# Rate limiting
DEFAULT_RATE_LIMIT_PER_MINUTE = 100
MAX_RATE_LIMIT_PER_MINUTE = 10000
MIN_RATE_LIMIT_PER_MINUTE = 1

# Request limits
DEFAULT_MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
MAX_MAX_REQUEST_SIZE = 100 * 1024 * 1024     # 100MB
MIN_MAX_REQUEST_SIZE = 1024 * 1024            # 1MB

# =============================================================================
# File and Path Constants
# =============================================================================

# Default directories
DEFAULT_DATA_DIR = "./data"
DEFAULT_MODELS_DIR = "./models"
DEFAULT_CHECKPOINTS_DIR = "./checkpoints"
DEFAULT_LOGS_DIR = "./logs"
DEFAULT_CACHE_DIR = "./cache"

# File extensions
SUPPORTED_CONFIG_EXTENSIONS = [".yaml", ".yml", ".json"]
SUPPORTED_MODEL_EXTENSIONS = [".pt", ".pth", ".bin", ".safetensors"]
SUPPORTED_DATA_EXTENSIONS = [".json", ".csv", ".txt", ".parquet"]

# =============================================================================
# Monitoring Constants
# =============================================================================

# Monitoring intervals
DEFAULT_MONITORING_INTERVAL = 5  # seconds
MIN_MONITORING_INTERVAL = 1      # seconds
MAX_MONITORING_INTERVAL = 300    # seconds

# Health check intervals
DEFAULT_HEALTH_CHECK_INTERVAL = 30  # seconds
MIN_HEALTH_CHECK_INTERVAL = 5       # seconds
MAX_HEALTH_CHECK_INTERVAL = 300     # seconds

# Metric retention
DEFAULT_METRIC_RETENTION_HOURS = 24
MIN_METRIC_RETENTION_HOURS = 1
MAX_METRIC_RETENTION_HOURS = 168  # 1 week

# =============================================================================
# Security Constants
# =============================================================================

# Encryption
DEFAULT_ENCRYPTION_ALGORITHM = "AES-256"
SUPPORTED_ENCRYPTION_ALGORITHMS = ["AES-128", "AES-256", "ChaCha20"]

# Authentication
DEFAULT_TOKEN_EXPIRY_HOURS = 24
MIN_TOKEN_EXPIRY_HOURS = 1
MAX_TOKEN_EXPIRY_HOURS = 168  # 1 week

# =============================================================================
# Error Messages
# =============================================================================

# Common error messages
ERROR_MODEL_NOT_FOUND = "Model not found"
ERROR_INVALID_CONFIG = "Invalid configuration"
ERROR_DEVICE_NOT_AVAILABLE = "Device not available"
ERROR_MEMORY_INSUFFICIENT = "Insufficient memory"
ERROR_INVALID_INPUT = "Invalid input data"
ERROR_SERVICE_UNAVAILABLE = "Service temporarily unavailable"

# =============================================================================
# Success Messages
# =============================================================================

# Common success messages
SUCCESS_MODEL_LOADED = "Model loaded successfully"
SUCCESS_TRAINING_STARTED = "Training started successfully"
SUCCESS_INFERENCE_COMPLETED = "Inference completed successfully"
SUCCESS_CONFIG_SAVED = "Configuration saved successfully"

# =============================================================================
# Export Constants
# =============================================================================

__all__ = [
    # System
    "VERSION", "AUTHOR", "DESCRIPTION",
    "DEFAULT_SYSTEM_MODE", "DEFAULT_LOG_LEVEL",
    "DEFAULT_API_PORT", "DEFAULT_GRADIO_PORT", "DEFAULT_METRICS_PORT",
    
    # Performance
    "MEMORY_WARNING_THRESHOLD", "MEMORY_CRITICAL_THRESHOLD",
    "GPU_MEMORY_WARNING_THRESHOLD", "GPU_MEMORY_CRITICAL_THRESHOLD",
    "DEFAULT_CACHE_TTL", "DEFAULT_CACHE_MAX_SIZE", "DEFAULT_MODEL_CACHE_SIZE",
    "MIN_BATCH_SIZE", "MAX_BATCH_SIZE", "DEFAULT_BATCH_SIZE",
    
    # Training
    "MIN_LEARNING_RATE", "MAX_LEARNING_RATE", "DEFAULT_LEARNING_RATE",
    "MAX_EPOCHS", "MAX_STEPS", "DEFAULT_EPOCHS", "DEFAULT_STEPS",
    "DEFAULT_GRADIENT_ACCUMULATION_STEPS", "MAX_GRADIENT_ACCUMULATION_STEPS",
    "DEFAULT_GRADIENT_CLIP_NORM", "DEFAULT_PATIENCE", "MIN_PATIENCE", "MAX_PATIENCE",
    
    # Model
    "SUPPORTED_MODEL_TYPES", "SUPPORTED_PRECISIONS", "SUPPORTED_DEVICES",
    
    # API
    "HTTP_OK", "HTTP_CREATED", "HTTP_BAD_REQUEST", "HTTP_UNAUTHORIZED",
    "HTTP_FORBIDDEN", "HTTP_NOT_FOUND", "HTTP_INTERNAL_ERROR",
    "DEFAULT_RATE_LIMIT_PER_MINUTE", "MAX_RATE_LIMIT_PER_MINUTE", "MIN_RATE_LIMIT_PER_MINUTE",
    "DEFAULT_MAX_REQUEST_SIZE", "MAX_MAX_REQUEST_SIZE", "MIN_MAX_REQUEST_SIZE",
    
    # File and Path
    "DEFAULT_DATA_DIR", "DEFAULT_MODELS_DIR", "DEFAULT_CHECKPOINTS_DIR",
    "DEFAULT_LOGS_DIR", "DEFAULT_CACHE_DIR",
    "SUPPORTED_CONFIG_EXTENSIONS", "SUPPORTED_MODEL_EXTENSIONS", "SUPPORTED_DATA_EXTENSIONS",
    
    # Monitoring
    "DEFAULT_MONITORING_INTERVAL", "MIN_MONITORING_INTERVAL", "MAX_MONITORING_INTERVAL",
    "DEFAULT_HEALTH_CHECK_INTERVAL", "MIN_HEALTH_CHECK_INTERVAL", "MAX_HEALTH_CHECK_INTERVAL",
    "DEFAULT_METRIC_RETENTION_HOURS", "MIN_METRIC_RETENTION_HOURS", "MAX_METRIC_RETENTION_HOURS",
    
    # Security
    "DEFAULT_ENCRYPTION_ALGORITHM", "SUPPORTED_ENCRYPTION_ALGORITHMS",
    "DEFAULT_TOKEN_EXPIRY_HOURS", "MIN_TOKEN_EXPIRY_HOURS", "MAX_TOKEN_EXPIRY_HOURS",
    
    # Messages
    "ERROR_MODEL_NOT_FOUND", "ERROR_INVALID_CONFIG", "ERROR_DEVICE_NOT_AVAILABLE",
    "ERROR_MEMORY_INSUFFICIENT", "ERROR_INVALID_INPUT", "ERROR_SERVICE_UNAVAILABLE",
    "SUCCESS_MODEL_LOADED", "SUCCESS_TRAINING_STARTED", "SUCCESS_INFERENCE_COMPLETED",
    "SUCCESS_CONFIG_SAVED"
]


