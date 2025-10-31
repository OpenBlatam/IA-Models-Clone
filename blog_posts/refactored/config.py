from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Configuración centralizada para el sistema NLP ultra-optimizado.
"""


class CacheBackend(Enum):
    """Tipos de backend de cache disponibles."""
    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"

class ModelType(Enum):
    """Tipos de modelos NLP disponibles."""
    LIGHTWEIGHT = "lightweight"
    STANDARD = "standard"
    ADVANCED = "advanced"

@dataclass
class CacheConfig:
    """Configuración del sistema de cache."""
    backend: CacheBackend = CacheBackend.HYBRID
    memory_size: int = 10000
    ttl_seconds: int = 3600
    redis_url: str = "redis://localhost:6379"
    redis_prefix: str = "nlp"
    redis_timeout: int = 5

@dataclass
class ModelConfig:
    """Configuración de modelos NLP."""
    type: ModelType = ModelType.LIGHTWEIGHT
    quantization_enabled: bool = True
    async_loading: bool = True
    preload_critical: bool = True
    cache_size: int = 50
    
    # Modelos específicos
    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    embedding_model: str = "all-MiniLM-L6-v2"
    spacy_model: str = "en_core_web_sm"

@dataclass
class PerformanceConfig:
    """Configuración de rendimiento."""
    max_workers: int = 4
    batch_size: int = 32
    timeout_seconds: int = 30
    enable_monitoring: bool = True
    log_level: str = "INFO"

@dataclass
class AnalysisConfig:
    """Configuración de análisis NLP."""
    enable_sentiment: bool = True
    enable_readability: bool = True
    enable_keywords: bool = True
    enable_language_detection: bool = True
    enable_advanced_features: bool = False
    
    # Parámetros específicos
    keyword_max_count: int = 5
    keyword_max_ngram: int = 2
    readability_target_grade: int = 8

@dataclass
class NLPConfig:
    """Configuración principal del sistema NLP."""
    cache: CacheConfig = field(default_factory=CacheConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    
    @classmethod
    def from_env(cls) -> 'NLPConfig':
        """Crear configuración desde variables de entorno."""
        config = cls()
        
        # Cache config
        if os.getenv('NLP_CACHE_BACKEND'):
            config.cache.backend = CacheBackend(os.getenv('NLP_CACHE_BACKEND'))
        if os.getenv('NLP_CACHE_SIZE'):
            config.cache.memory_size = int(os.getenv('NLP_CACHE_SIZE'))
        if os.getenv('NLP_CACHE_TTL'):
            config.cache.ttl_seconds = int(os.getenv('NLP_CACHE_TTL'))
        if os.getenv('REDIS_URL'):
            config.cache.redis_url = os.getenv('REDIS_URL')
        
        # Performance config
        if os.getenv('NLP_MAX_WORKERS'):
            config.performance.max_workers = int(os.getenv('NLP_MAX_WORKERS'))
        if os.getenv('NLP_BATCH_SIZE'):
            config.performance.batch_size = int(os.getenv('NLP_BATCH_SIZE'))
        if os.getenv('NLP_LOG_LEVEL'):
            config.performance.log_level = os.getenv('NLP_LOG_LEVEL')
        
        # Model config
        if os.getenv('NLP_MODEL_TYPE'):
            config.models.type = ModelType(os.getenv('NLP_MODEL_TYPE'))
        if os.getenv('NLP_QUANTIZATION'):
            config.models.quantization_enabled = os.getenv('NLP_QUANTIZATION').lower() == 'true'
        
        return config
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NLPConfig':
        """Crear configuración desde diccionario."""
        config = cls()
        
        if 'cache' in data:
            cache_data = data['cache']
            if 'backend' in cache_data:
                config.cache.backend = CacheBackend(cache_data['backend'])
            if 'memory_size' in cache_data:
                config.cache.memory_size = cache_data['memory_size']
            if 'ttl_seconds' in cache_data:
                config.cache.ttl_seconds = cache_data['ttl_seconds']
        
        if 'performance' in data:
            perf_data = data['performance']
            if 'max_workers' in perf_data:
                config.performance.max_workers = perf_data['max_workers']
            if 'batch_size' in perf_data:
                config.performance.batch_size = perf_data['batch_size']
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Exportar configuración como diccionario."""
        return {
            'cache': {
                'backend': self.cache.backend.value,
                'memory_size': self.cache.memory_size,
                'ttl_seconds': self.cache.ttl_seconds,
                'redis_url': self.cache.redis_url,
                'redis_prefix': self.cache.redis_prefix
            },
            'models': {
                'type': self.models.type.value,
                'quantization_enabled': self.models.quantization_enabled,
                'async_loading': self.models.async_loading,
                'sentiment_model': self.models.sentiment_model,
                'embedding_model': self.models.embedding_model
            },
            'performance': {
                'max_workers': self.performance.max_workers,
                'batch_size': self.performance.batch_size,
                'timeout_seconds': self.performance.timeout_seconds,
                'enable_monitoring': self.performance.enable_monitoring
            },
            'analysis': {
                'enable_sentiment': self.analysis.enable_sentiment,
                'enable_readability': self.analysis.enable_readability,
                'enable_keywords': self.analysis.enable_keywords,
                'keyword_max_count': self.analysis.keyword_max_count
            }
        }

# Configuración global por defecto
DEFAULT_CONFIG = NLPConfig()

# Configuraciones predefinidas para diferentes entornos
DEVELOPMENT_CONFIG = NLPConfig(
    cache=CacheConfig(
        backend=CacheBackend.MEMORY,
        memory_size=5000,
        ttl_seconds=1800
    ),
    performance=PerformanceConfig(
        max_workers=2,
        batch_size=16,
        enable_monitoring=True,
        log_level="DEBUG"
    ),
    models=ModelConfig(
        type=ModelType.LIGHTWEIGHT,
        quantization_enabled=False,
        preload_critical=False
    )
)

PRODUCTION_CONFIG = NLPConfig(
    cache=CacheConfig(
        backend=CacheBackend.HYBRID,
        memory_size=20000,
        ttl_seconds=7200
    ),
    performance=PerformanceConfig(
        max_workers=8,
        batch_size=64,
        enable_monitoring=True,
        log_level="INFO"
    ),
    models=ModelConfig(
        type=ModelType.STANDARD,
        quantization_enabled=True,
        preload_critical=True
    ),
    analysis=AnalysisConfig(
        enable_advanced_features=True
    )
)

def get_config(environment: Optional[str] = None) -> NLPConfig:
    """
    Obtener configuración según el entorno.
    
    Args:
        environment: 'development', 'production', o None para usar variables de entorno
    
    Returns:
        Configuración NLP apropiada
    """
    if environment == 'development':
        return DEVELOPMENT_CONFIG
    elif environment == 'production':
        return PRODUCTION_CONFIG
    else:
        # Intentar cargar desde variables de entorno
        try:
            return NLPConfig.from_env()
        except Exception:
            return DEFAULT_CONFIG 