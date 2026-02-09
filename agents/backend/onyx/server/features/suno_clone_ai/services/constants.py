"""
Constantes del Sistema

Define constantes reutilizables en todo el sistema.
"""

from enum import Enum


class ServiceStatus(Enum):
    """Estados de servicios"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class CacheLevel(Enum):
    """Niveles de caché"""
    L1 = "l1"  # Memoria
    L2 = "l2"  # Redis
    L3 = "l3"  # Disco


class ProcessingPriority(Enum):
    """Prioridades de procesamiento"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


# Constantes de configuración
DEFAULT_CACHE_TTL = 3600  # 1 hora
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30  # segundos

# Límites
MAX_PROMPT_LENGTH = 1000
MIN_PROMPT_LENGTH = 1
MAX_AUDIO_DURATION = 3600  # 1 hora en segundos
MIN_AUDIO_DURATION = 1  # 1 segundo

# Formatos soportados
SUPPORTED_AUDIO_FORMATS = ['wav', 'mp3', 'ogg', 'flac', 'm4a']
SUPPORTED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'gif', 'webp']

# Tamaños
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_BATCH_SIZE = 100

# Timeouts
GENERATION_TIMEOUT = 300  # 5 minutos
UPLOAD_TIMEOUT = 60  # 1 minuto

# Rate limits
DEFAULT_RATE_LIMIT = 100  # requests per minute
PREMIUM_RATE_LIMIT = 1000

# Paginación
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

