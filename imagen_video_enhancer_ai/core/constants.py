"""
Constants for Imagen Video Enhancer AI
=====================================

Application-wide constants.
"""

# Default values
DEFAULT_MAX_PARALLEL_TASKS = 5
DEFAULT_OUTPUT_DIR = "enhancer_output"
DEFAULT_CACHE_TTL_HOURS = 24
DEFAULT_MAX_RETRIES = 3
DEFAULT_RATE_LIMIT_RPS = 10.0
DEFAULT_RATE_LIMIT_BURST = 20

# Directory names
DIR_RESULTS = "results"
DIR_TASKS = "tasks"
DIR_STORAGE = "storage"
DIR_UPLOADS = "uploads"
DIR_CACHE = "cache"
DIR_LOGS = "logs"

# Output directories list
OUTPUT_DIRECTORIES = [
    DIR_RESULTS,
    DIR_TASKS,
    DIR_STORAGE,
    DIR_UPLOADS,
    DIR_CACHE,
    DIR_LOGS
]

# File size limits (MB)
DEFAULT_MAX_IMAGE_SIZE_MB = 50
DEFAULT_MAX_VIDEO_SIZE_MB = 500
DEFAULT_MAX_FILE_SIZE_MB = 500

# Supported formats
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"]
SUPPORTED_VIDEO_FORMATS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]

# Enhancement types
ENHANCEMENT_TYPES = [
    "general",
    "sharpness",
    "colors",
    "denoise",
    "upscale",
    "restore"
]

# Service types
SERVICE_ENHANCE_IMAGE = "enhance_image"
SERVICE_ENHANCE_VIDEO = "enhance_video"
SERVICE_UPSCALE = "upscale"
SERVICE_DENOISE = "denoise"
SERVICE_RESTORE = "restore"
SERVICE_COLOR_CORRECTION = "color_correction"

# Task priorities
PRIORITY_LOW = 0
PRIORITY_NORMAL = 5
PRIORITY_HIGH = 10

# Retry strategies
RETRY_STRATEGY_IMMEDIATE = "immediate"
RETRY_STRATEGY_EXPONENTIAL_BACKOFF = "exponential_backoff"
RETRY_STRATEGY_FIXED_DELAY = "fixed_delay"
RETRY_STRATEGY_LINEAR_BACKOFF = "linear_backoff"

# Logging
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
DEFAULT_LOG_BACKUP_COUNT = 5

# Metrics
METRICS_MAX_POINTS = 10000
METRICS_HISTORY_LIMIT = 1000

# Events
EVENT_HISTORY_LIMIT = 1000

# Backup
DEFAULT_BACKUP_DIR = "backups"




