from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from enum import Enum, auto
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🎯 DOMAIN ENUMS - Tipos del Dominio NLP
======================================

Enumeraciones que definen los tipos y constantes del dominio.
"""



class AnalysisType(Enum):
    """Tipos de análisis NLP disponibles."""
    SENTIMENT = auto()
    READABILITY = auto()
    KEYWORDS = auto()
    LANGUAGE_DETECTION = auto()
    QUALITY_ASSESSMENT = auto()
    SEMANTIC_SIMILARITY = auto()
    ENTITY_EXTRACTION = auto()
    SUMMARIZATION = auto()
    TOXICITY = auto()
    EMOTION = auto()


class ProcessingTier(Enum):
    """Tiers de procesamiento según velocidad/calidad."""
    ULTRA_FAST = "ultra_fast"      # < 0.1ms, calidad básica
    BALANCED = "balanced"          # < 1ms, calidad buena  
    HIGH_QUALITY = "high_quality"  # < 10ms, calidad alta
    RESEARCH_GRADE = "research"     # < 100ms, calidad máxima


class CacheStrategy(Enum):
    """Estrategias de cache disponibles."""
    MEMORY_ONLY = "memory"
    REDIS_FALLBACK = "redis_fallback" 
    DISTRIBUTED = "distributed"
    PERSISTENT = "persistent"
    NO_CACHE = "no_cache"


class AnalysisStatus(Enum):
    """Estados de un análisis."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


class ErrorType(Enum):
    """Tipos de errores del sistema."""
    VALIDATION_ERROR = "validation"
    PROCESSING_ERROR = "processing"
    TIMEOUT_ERROR = "timeout"
    CACHE_ERROR = "cache"
    RATE_LIMIT_ERROR = "rate_limit"
    CIRCUIT_BREAKER_ERROR = "circuit_breaker"
    SYSTEM_ERROR = "system"


class MetricType(Enum):
    """Tipos de métricas del sistema."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CACHE_HIT_RATE = "cache_hit_rate"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"


class LogLevel(Enum):
    """Niveles de logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Environment(Enum):
    """Entornos de deployment."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production" 