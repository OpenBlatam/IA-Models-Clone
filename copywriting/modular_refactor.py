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

import asyncio
import json
import time
import logging
import hashlib
import os
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import uuid
            import orjson
            import msgspec
            import blake3
            import xxhash
            import mmh3
            import lz4.frame
            import zstandard as zstd
            import gzip
            import redis
                import uvloop
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULAR REFACTOR - Arquitectura Completamente Modular
====================================================

Sistema de copywriting con arquitectura modular completa.
Cada componente es independiente y reutilizable.

Architecture:
├── OptimizationEngine (Motor de optimización)
├── CacheManager (Gestión de cache multi-nivel)
├── DataModels (Modelos de datos)
├── ContentGenerator (Generador de contenido)
├── MetricsCollector (Colector de métricas)
├── ConfigManager (Gestor de configuración)
└── CopywritingService (Servicio principal)
"""


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS - Enumeraciones del sistema
# ============================================================================

class PerformanceTier(Enum):
    """Tiers de performance"""
    ULTRA_MAXIMUM = "ULTRA MAXIMUM"
    MAXIMUM = "MAXIMUM"
    ULTRA = "ULTRA"
    OPTIMIZED = "OPTIMIZED"
    ENHANCED = "ENHANCED"
    STANDARD = "STANDARD"

class ToneType(Enum):
    """Tipos de tono"""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    URGENT = "urgent"
    FRIENDLY = "friendly"
    TECHNICAL = "technical"
    CREATIVE = "creative"

class LanguageType(Enum):
    """Idiomas soportados"""
    SPANISH = "es"
    ENGLISH = "en"
    FRENCH = "fr"
    PORTUGUESE = "pt"

# ============================================================================
# DATA MODELS - Modelos de datos modulares
# ============================================================================

@dataclass
class CopywritingRequest:
    """Modelo de request modular con validación"""
    prompt: str
    tone: str = ToneType.PROFESSIONAL.value
    language: str = LanguageType.SPANISH.value
    use_case: str = "general"
    target_length: Optional[int] = None
    keywords: List[str] = field(default_factory=list)
    use_cache: bool = True
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self) -> Any:
        """Validación automática"""
        if not self.prompt or len(self.prompt.strip()) == 0:
            raise ValueError("Prompt cannot be empty")
        if self.target_length and self.target_length <= 0:
            raise ValueError("Target length must be positive")
    
    def to_cache_key(self) -> str:
        """Generar clave de cache única"""
        components = [
            self.prompt, self.tone, self.language, self.use_case,
            str(self.target_length) if self.target_length else "none",
            "|".join(sorted(self.keywords)) if self.keywords else "none"
        ]
        return "|".join(components)

@dataclass
class CopywritingResponse:
    """Modelo de response con metadata completa"""
    content: str
    request_id: str
    generation_time_ms: float
    cache_hit: bool
    optimization_score: float
    performance_tier: str
    word_count: int = 0
    character_count: int = 0
    compression_ratio: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self) -> Any:
        """Calcular métricas automáticamente"""
        if self.word_count == 0:
            self.word_count = len(self.content.split())
        if self.character_count == 0:
            self.character_count = len(self.content)

# ============================================================================
# CONFIG MANAGER - Gestor de configuración modular
# ============================================================================

class ConfigManager:
    """Gestor de configuración centralizado"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializar gestor de configuración
        
        Args:
            config: Configuración personalizada (opcional)
        """
        self.config = self._merge_configs(config or {})
    
    def _merge_configs(self, custom_config: Dict[str, Any]) -> Dict[str, Any]:
        """Fusionar configuración por defecto con personalizada"""
        default_config = {
            "optimization": {
                "preferred_json": "auto",
                "preferred_hash": "auto", 
                "preferred_compression": "auto"
            },
            "cache": {
                "memory_cache_size": 1000,
                "cache_ttl": 3600,
                "compression_threshold": 1024
            },
            "redis": {
                "url": os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
                "timeout": 5
            },
            "content": {
                "max_prompt_length": 1000,
                "max_target_length": 1000,
                "default_tone": ToneType.PROFESSIONAL.value
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
        
        # Deep merge
        merged = default_config.copy()
        for key, value in custom_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key].update(value)
            else:
                merged[key] = value
        
        return merged
    
    def get(self, path: str, default: Any = None) -> Optional[Dict[str, Any]]:
        """
        Obtener valor de configuración por path
        
        Args:
            path: Path separado por puntos (ej: "cache.memory_cache_size")
            default: Valor por defecto si no existe
            
        Returns:
            Valor de configuración
        """
        keys = path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value

# ============================================================================
# OPTIMIZATION ENGINE - Motor de optimización modular
# ============================================================================

class OptimizationEngine:
    """Motor de optimización completamente modular"""
    
    def __init__(self, config_manager: ConfigManager):
        """
        Inicializar motor de optimización
        
        Args:
            config_manager: Gestor de configuración
        """
        self.config = config_manager
        self.libraries = self._scan_libraries()
        
        # Setup handlers
        self.json_handler = self._setup_json_handler()
        self.hash_handler = self._setup_hash_handler()
        self.compression_handler = self._setup_compression_handler()
        self.cache_handler = self._setup_cache_handler()
        
        # Calculate metrics
        self.optimization_score = self._calculate_score()
        self.performance_tier = self._determine_tier()
        
        logger.info(f"OptimizationEngine: {self.optimization_score:.1f}/100 - {self.performance_tier.value}")
    
    def _scan_libraries(self) -> Dict[str, bool]:
        """Escanear librerías disponibles"""
        target_libs = [
            "orjson", "msgspec", "blake3", "xxhash", "mmh3",
            "lz4", "zstandard", "numba", "polars", "duckdb",
            "redis", "uvloop", "rapidfuzz", "psutil"
        ]
        
        libraries = {}
        for lib in target_libs:
            try:
                __import__(lib)
                libraries[lib] = True
            except ImportError:
                libraries[lib] = False
        
        available = sum(libraries.values())
        logger.info(f"Libraries available: {available}/{len(target_libs)}")
        return libraries
    
    def _setup_json_handler(self) -> Dict[str, Any]:
        """Configurar handler JSON"""
        preferred = self.config.get("optimization.preferred_json", "auto")
        
        if (preferred == "orjson" or preferred == "auto") and self.libraries.get("orjson"):
            return {
                "dumps": lambda x: orjson.dumps(x).decode(),
                "loads": orjson.loads,
                "name": "orjson",
                "speed": 5.0
            }
        elif (preferred == "msgspec" or preferred == "auto") and self.libraries.get("msgspec"):
            enc, dec = msgspec.json.Encoder(), msgspec.json.Decoder()
            return {
                "dumps": lambda x: enc.encode(x).decode(),
                "loads": dec.decode,
                "name": "msgspec",
                "speed": 6.0
            }
        else:
            return {
                "dumps": json.dumps,
                "loads": json.loads,
                "name": "json",
                "speed": 1.0
            }
    
    def _setup_hash_handler(self) -> Dict[str, Any]:
        """Configurar handler de hash"""
        preferred = self.config.get("optimization.preferred_hash", "auto")
        
        if (preferred == "blake3" or preferred == "auto") and self.libraries.get("blake3"):
            return {
                "hash": lambda x: blake3.blake3(x.encode()).hexdigest(),
                "name": "blake3",
                "speed": 8.0
            }
        elif (preferred == "xxhash" or preferred == "auto") and self.libraries.get("xxhash"):
            return {
                "hash": lambda x: xxhash.xxh64(x.encode()).hexdigest(),
                "name": "xxhash",
                "speed": 6.0
            }
        elif self.libraries.get("mmh3"):
            return {
                "hash": lambda x: str(mmh3.hash128(x.encode())),
                "name": "mmh3",
                "speed": 3.0
            }
        else:
            return {
                "hash": lambda x: hashlib.sha256(x.encode()).hexdigest(),
                "name": "sha256",
                "speed": 1.0
            }
    
    def _setup_compression_handler(self) -> Dict[str, Any]:
        """Configurar handler de compresión"""
        preferred = self.config.get("optimization.preferred_compression", "auto")
        
        if (preferred == "lz4" or preferred == "auto") and self.libraries.get("lz4"):
            return {
                "compress": lz4.frame.compress,
                "decompress": lz4.frame.decompress,
                "name": "lz4",
                "speed": 10.0
            }
        elif (preferred == "zstandard" or preferred == "auto") and self.libraries.get("zstandard"):
            comp = zstd.ZstdCompressor(level=1)
            decomp = zstd.ZstdDecompressor()
            return {
                "compress": comp.compress,
                "decompress": decomp.decompress,
                "name": "zstandard",
                "speed": 5.0
            }
        else:
            return {
                "compress": gzip.compress,
                "decompress": gzip.decompress,
                "name": "gzip",
                "speed": 1.0
            }
    
    def _setup_cache_handler(self) -> Optional[Any]:
        """Configurar handler de cache Redis"""
        if not self.libraries.get("redis"):
            return None
        
        try:
            redis_url = self.config.get("redis.url")
            timeout = self.config.get("redis.timeout", 5)
            
            client = redis.from_url(redis_url, decode_responses=True, socket_timeout=timeout)
            client.ping()
            logger.info("Redis connected successfully")
            return client
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            return None
    
    def _calculate_score(self) -> float:
        """Calcular score de optimización"""
        score = 0.0
        
        # Base optimizations
        score += self.json_handler["speed"] * 5
        score += self.hash_handler["speed"] * 3
        score += self.compression_handler["speed"] * 2
        
        # Library bonuses
        bonuses = {
            "polars": 15, "duckdb": 10, "numba": 12, "uvloop": 8,
            "rapidfuzz": 5, "psutil": 2
        }
        
        for lib, bonus in bonuses.items():
            if self.libraries.get(lib):
                score += bonus
        
        # Redis bonus
        if self.cache_handler:
            score += 8
        
        return min(score, 100.0)
    
    def _determine_tier(self) -> PerformanceTier:
        """Determinar tier de performance"""
        if self.optimization_score >= 95:
            return PerformanceTier.ULTRA_MAXIMUM
        elif self.optimization_score >= 85:
            return PerformanceTier.MAXIMUM
        elif self.optimization_score >= 70:
            return PerformanceTier.ULTRA
        elif self.optimization_score >= 50:
            return PerformanceTier.OPTIMIZED
        elif self.optimization_score >= 30:
            return PerformanceTier.ENHANCED
        else:
            return PerformanceTier.STANDARD

# ============================================================================
# CACHE MANAGER - Gestor de cache modular
# ============================================================================

class CacheManager:
    """Gestor de cache multi-nivel modular"""
    
    def __init__(self, optimization_engine: OptimizationEngine, config_manager: ConfigManager):
        """
        Inicializar gestor de cache
        
        Args:
            optimization_engine: Motor de optimización
            config_manager: Gestor de configuración
        """
        self.engine = optimization_engine
        self.config = config_manager
        
        # Cache configuration
        self.memory_size = self.config.get("cache.memory_cache_size", 1000)
        self.ttl = self.config.get("cache.cache_ttl", 3600)
        self.compression_threshold = self.config.get("cache.compression_threshold", 1024)
        
        # Storage layers
        self.memory_cache: Dict[str, Any] = {}
        self.timestamps: Dict[str, float] = {}
        self.compressed_cache: Dict[str, bytes] = {}
        
        # External cache
        self.redis = optimization_engine.cache_handler
        
        # Metrics
        self.metrics = {
            "memory_hits": 0, "compressed_hits": 0, "redis_hits": 0,
            "misses": 0, "sets": 0, "errors": 0
        }
        
        logger.info(f"CacheManager: Memory + Compression + {'Redis' if self.redis else 'No Redis'}")
    
    def _generate_key(self, key: str) -> str:
        """Generar clave de cache"""
        return self.engine.hash_handler["hash"](key)[:16]
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtener del cache multi-nivel"""
        cache_key = self._generate_key(key)
        
        try:
            # L1: Memory cache
            if cache_key in self.memory_cache:
                if time.time() - self.timestamps.get(cache_key, 0) < self.ttl:
                    self.metrics["memory_hits"] += 1
                    return self.memory_cache[cache_key]
                else:
                    self._evict_memory(cache_key)
            
            # L2: Compressed cache
            if cache_key in self.compressed_cache:
                try:
                    compressed = self.compressed_cache[cache_key]
                    decompressed = self.engine.compression_handler["decompress"](compressed)
                    value = self.engine.json_handler["loads"](decompressed.decode())
                    
                    self._store_memory(cache_key, value)
                    self.metrics["compressed_hits"] += 1
                    return value
                except Exception:
                    del self.compressed_cache[cache_key]
            
            # L3: Redis cache
            if self.redis:
                try:
                    data = self.redis.get(f"modular:{cache_key}")
                    if data:
                        value = self.engine.json_handler["loads"](data)
                        await self.set(key, value)
                        self.metrics["redis_hits"] += 1
                        return value
                except Exception:
                    pass
            
            self.metrics["misses"] += 1
            return None
            
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any) -> bool:
        """Almacenar en cache multi-nivel"""
        cache_key = self._generate_key(key)
        
        try:
            # Store in memory
            self._store_memory(cache_key, value)
            
            # Store compressed
            self._store_compressed(cache_key, value)
            
            # Store in Redis
            if self.redis:
                try:
                    data = self.engine.json_handler["dumps"](value)
                    self.redis.setex(f"modular:{cache_key}", self.ttl, data)
                except Exception:
                    pass
            
            self.metrics["sets"] += 1
            return True
            
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Cache set error: {e}")
            return False
    
    def _store_memory(self, cache_key: str, value: Any):
        """Almacenar en memoria con LRU"""
        if len(self.memory_cache) >= self.memory_size:
            oldest = min(self.timestamps.keys(), key=self.timestamps.get)
            self._evict_memory(oldest)
        
        self.memory_cache[cache_key] = value
        self.timestamps[cache_key] = time.time()
    
    def _evict_memory(self, cache_key: str):
        """Evictar de memoria"""
        self.memory_cache.pop(cache_key, None)
        self.timestamps.pop(cache_key, None)
    
    def _store_compressed(self, cache_key: str, value: Any):
        """Almacenar comprimido"""
        try:
            json_data = self.engine.json_handler["dumps"](value).encode()
            if len(json_data) >= self.compression_threshold:
                compressed = self.engine.compression_handler["compress"](json_data)
                self.compressed_cache[cache_key] = compressed
        except Exception:
            pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas"""
        total = sum([self.metrics["memory_hits"], self.metrics["compressed_hits"], 
                    self.metrics["redis_hits"], self.metrics["misses"]])
        
        hit_rate = 0.0
        if total > 0:
            hits = self.metrics["memory_hits"] + self.metrics["compressed_hits"] + self.metrics["redis_hits"]
            hit_rate = (hits / total) * 100
        
        return {
            "hit_rate_percent": hit_rate,
            "total_requests": total,
            "memory_size": len(self.memory_cache),
            "compressed_size": len(self.compressed_cache),
            "redis_available": self.redis is not None,
            **self.metrics
        }

# ============================================================================
# CONTENT GENERATOR - Generador de contenido modular
# ============================================================================

class ContentGenerator:
    """Generador de contenido modular y configurable"""
    
    def __init__(self, config_manager: ConfigManager):
        """
        Inicializar generador de contenido
        
        Args:
            config_manager: Gestor de configuración
        """
        self.config = config_manager
        self.templates = self._load_templates()
        
        logger.info("ContentGenerator initialized")
    
    def _load_templates(self) -> Dict[str, str]:
        """Cargar templates de contenido"""
        return {
            ToneType.PROFESSIONAL.value: (
                "Como experto en {use_case}, presento {prompt}. "
                "Esta solución profesional está diseñada para maximizar resultados "
                "y generar un impacto significativo en su industria."
            ),
            ToneType.CASUAL.value: (
                "¡Hola! Te cuento sobre {prompt}. "
                "Es algo realmente genial para {use_case} que te va a fascinar. "
                "Definitivamente vale la pena conocer más detalles."
            ),
            ToneType.URGENT.value: (
                "⚡ ¡OPORTUNIDAD ÚNICA! {prompt} - "
                "Solución revolucionaria para {use_case} disponible por tiempo limitado. "
                "No dejes pasar esta oportunidad excepcional."
            ),
            ToneType.FRIENDLY.value: (
                "¡Hola amigo! Te quiero compartir {prompt}. "
                "Como alguien que se preocupa por tu éxito en {use_case}, "
                "esto realmente puede cambiar tu perspectiva."
            ),
            ToneType.TECHNICAL.value: (
                "Análisis técnico: {prompt} representa una solución avanzada "
                "para {use_case} con métricas optimizadas y resultados medibles "
                "basados en las mejores prácticas de la industria."
            ),
            ToneType.CREATIVE.value: (
                "¡Imagina las posibilidades! {prompt} abre un mundo de oportunidades "
                "en {use_case}. Una propuesta innovadora que desafía los límites "
                "y redefine lo que es posible."
            )
        }
    
    async def generate(self, request: CopywritingRequest) -> str:
        """
        Generar contenido basado en request
        
        Args:
            request: Request de copywriting
            
        Returns:
            Contenido generado
        """f"
        # Get template
        template = self.templates.get(request.tone, self.templates[ToneType.PROFESSIONAL.value])
        
        # Generate base content
        content = template"
        
        # Add keywords if specified
        if request.keywords:
            keywords_text = ", ".join(request.keywords)
            content += f" Palabras clave relevantes: {keywords_text}."
        
        # Adjust length if needed
        if request.target_length:
            content = self._adjust_length(content, request.target_length)
        
        # Simulate processing time
        await asyncio.sleep(0.005)
        
        return content
    
    def _adjust_length(self, content: str, target_length: int) -> str:
        """Ajustar longitud del contenido"""
        current_length = len(content.split())
        
        if current_length < target_length:
            extension = (
                " Esta propuesta integral abarca múltiples aspectos fundamentales, "
                "proporcionando una visión completa y detallada que garantiza "
                "resultados excepcionales y sostenibles a largo plazo."
            )
            content += extension
        
        return content

# ============================================================================
# METRICS COLLECTOR - Colector de métricas modular
# ============================================================================

class MetricsCollector:
    """Colector de métricas modular"""
    
    def __init__(self) -> Any:
        """Inicializar colector de métricas"""
        self.service_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "average_response_time_ms": 0.0,
            "start_time": time.time()
        }
        
        logger.info("MetricsCollector initialized")
    
    def record_request(self, success: bool = True, cache_hit: bool = False, response_time_ms: float = 0.0):
        """Registrar una request"""
        self.service_metrics["total_requests"] += 1
        
        if success:
            self.service_metrics["successful_requests"] += 1
            self._update_average_response_time(response_time_ms)
        else:
            self.service_metrics["failed_requests"] += 1
        
        if cache_hit:
            self.service_metrics["cache_hits"] += 1
    
    def _update_average_response_time(self, response_time_ms: float):
        """Actualizar tiempo promedio de respuesta"""
        current_avg = self.service_metrics["average_response_time_ms"]
        total_requests = self.service_metrics["successful_requests"]
        
        if total_requests == 1:
            self.service_metrics["average_response_time_ms"] = response_time_ms
        else:
            self.service_metrics["average_response_time_ms"] = (
                (current_avg * (total_requests - 1) + response_time_ms) / total_requests
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas actuales"""
        uptime = time.time() - self.service_metrics["start_time"]
        
        return {
            **self.service_metrics,
            "uptime_seconds": uptime,
            "success_rate_percent": (
                (self.service_metrics["successful_requests"] / max(self.service_metrics["total_requests"], 1)) * 100
            ),
            "cache_hit_rate_percent": (
                (self.service_metrics["cache_hits"] / max(self.service_metrics["total_requests"], 1)) * 100
            )
        }

# ============================================================================
# MODULAR COPYWRITING SERVICE - Servicio principal modular
# ============================================================================

class ModularCopywritingService:
    """Servicio de copywriting completamente modular"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializar servicio modular
        
        Args:
            config: Configuración personalizada (opcional)
        """
        # Initialize modules
        self.config_manager = ConfigManager(config)
        self.optimization_engine = OptimizationEngine(self.config_manager)
        self.cache_manager = CacheManager(self.optimization_engine, self.config_manager)
        self.content_generator = ContentGenerator(self.config_manager)
        self.metrics_collector = MetricsCollector()
        
        # Setup uvloop if available
        if self.optimization_engine.libraries.get("uvloop"):
            try:
                uvloop.install()
                logger.info("uvloop event loop activated")
            except Exception:
                pass
        
        logger.info("ModularCopywritingService initialized successfully")
        self._show_status()
    
    async def generate_copy(self, request: CopywritingRequest) -> CopywritingResponse:
        """
        Generar copy con arquitectura modular
        
        Args:
            request: Request de copywriting
            
        Returns:
            Response con contenido generado
        """
        start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = request.to_cache_key()
            
            # Check cache first
            cached_content = None
            if request.use_cache:
                cached_content = await self.cache_manager.get(cache_key)
                if cached_content:
                    generation_time_ms = (time.time() - start_time) * 1000
                    
                    response = CopywritingResponse(
                        content=cached_content["content"],
                        request_id=request.request_id,
                        generation_time_ms=generation_time_ms,
                        cache_hit=True,
                        optimization_score=self.optimization_engine.optimization_score,
                        performance_tier=self.optimization_engine.performance_tier.value,
                        word_count=cached_content["word_count"],
                        character_count=cached_content["character_count"],
                        compression_ratio=cached_content.get("compression_ratio")
                    )
                    
                    self.metrics_collector.record_request(success=True, cache_hit=True, response_time_ms=generation_time_ms)
                    return response
            
            # Generate new content
            content = await self.content_generator.generate(request)
            generation_time_ms = (time.time() - start_time) * 1000
            
            # Calculate compression ratio
            compression_ratio = None
            if len(content) >= 100:
                try:
                    original = content.encode()
                    compressed = self.optimization_engine.compression_handler["compress"](original)
                    compression_ratio = len(compressed) / len(original)
                except Exception:
                    pass
            
            # Create response
            response = CopywritingResponse(
                content=content,
                request_id=request.request_id,
                generation_time_ms=generation_time_ms,
                cache_hit=False,
                optimization_score=self.optimization_engine.optimization_score,
                performance_tier=self.optimization_engine.performance_tier.value,
                compression_ratio=compression_ratio
            )
            
            # Cache the result
            if request.use_cache:
                cache_data = {
                    "content": content,
                    "word_count": response.word_count,
                    "character_count": response.character_count,
                    "compression_ratio": compression_ratio
                }
                await self.cache_manager.set(cache_key, cache_data)
            
            # Record metrics
            self.metrics_collector.record_request(success=True, cache_hit=False, response_time_ms=generation_time_ms)
            
            return response
            
        except Exception as e:
            self.metrics_collector.record_request(success=False)
            logger.error(f"Content generation failed: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check completo del sistema modular"""
        try:
            # Test basic functionality
            test_request = CopywritingRequest(
                prompt="Health check test",
                tone=ToneType.PROFESSIONAL.value,
                use_cache=False
            )
            
            start_time = time.time()
            response = await self.generate_copy(test_request)
            test_time = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "test_response_time_ms": test_time,
                "modules": {
                    "optimization_engine": {
                        "score": self.optimization_engine.optimization_score,
                        "tier": self.optimization_engine.performance_tier.value,
                        "libraries_available": sum(self.optimization_engine.libraries.values())
                    },
                    "cache_manager": self.cache_manager.get_metrics(),
                    "content_generator": {"status": "operational"},
                    "metrics_collector": self.metrics_collector.get_metrics()
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def _show_status(self) -> Any:
        """Mostrar estado del sistema modular"""
        print("\n" + "="*70)
        print("🏗️ MODULAR COPYWRITING SERVICE - REFACTORED ARCHITECTURE")
        print("="*70)
        print(f"📊 Optimization Score: {self.optimization_engine.optimization_score:.1f}/100")
        print(f"🏆 Performance Tier: {self.optimization_engine.performance_tier.value}")
        print(f"\n🔧 Modular Components:")
        print(f"   ✅ ConfigManager: Centralized configuration")
        print(f"   ✅ OptimizationEngine: {self.optimization_engine.json_handler['name']} + {self.optimization_engine.hash_handler['name']}")
        print(f"   ✅ CacheManager: Multi-level caching")
        print(f"   ✅ ContentGenerator: Template-based generation")
        print(f"   ✅ MetricsCollector: Performance tracking")
        print(f"   ✅ Redis: {'Available' if self.cache_manager.redis else 'Not Available'}")
        print("="*70)

# ============================================================================
# DEMO - Demostración del sistema modular
# ============================================================================

async def modular_demo():
    """Demo del sistema completamente modular"""
    
    print("🏗️ MODULAR REFACTOR DEMO")
    print("="*50)
    print("Sistema completamente modular y refactorizado")
    print("✅ Arquitectura de componentes independientes")
    print("✅ Configuración centralizada")
    print("✅ Máxima reutilización y mantenibilidad")
    print("="*50)
    
    # Initialize with custom config
    config = {
        "cache": {
            "memory_cache_size": 2000,
            "cache_ttl": 7200
        },
        "optimization": {
            "preferred_json": "orjson",
            "preferred_hash": "blake3"
        }
    }
    
    service = ModularCopywritingService(config)
    
    # Health check
    health = await service.health_check()
    print(f"\n🏥 System Status: {health['status'].upper()}")
    print(f"📊 Test Response Time: {health['test_response_time_ms']:.1f}ms")
    
    # Test requests with different configurations
    test_requests = [
        CopywritingRequest(
            prompt="Lanzamiento revolucionario de IA modular",
            tone=ToneType.PROFESSIONAL.value,
            use_case="tech_launch",
            keywords=["innovación", "modular", "arquitectura"]
        ),
        CopywritingRequest(
            prompt="Oferta limitada sistema modular",
            tone=ToneType.URGENT.value,
            use_case="promotion",
            target_length=60
        ),
        CopywritingRequest(
            prompt="Descubre arquitectura modular",
            tone=ToneType.CREATIVE.value,
            use_case="educational"
        ),
        CopywritingRequest(
            prompt="Análisis técnico modularidad",
            tone=ToneType.TECHNICAL.value,
            use_case="documentation"
        )
    ]
    
    print(f"\n📝 MODULAR TESTING:")
    print("-" * 35)
    
    for i, request in enumerate(test_requests, 1):
        response = await service.generate_copy(request)
        print(f"\n{i}. {request.tone.upper()} - {request.use_case}")
        print(f"   Content: {response.content[:90]}...")
        print(f"   Time: {response.generation_time_ms:.1f}ms")
        print(f"   Cache: {'✅' if response.cache_hit else '❌'}")
        print(f"   Words: {response.word_count}")
        if response.compression_ratio:
            print(f"   Compression: {response.compression_ratio:.2f}")
    
    # Test cache effectiveness
    print(f"\n🔄 MODULAR CACHE TEST:")
    print("-" * 25)
    cache_test = await service.generate_copy(test_requests[0])
    print(f"   Cache Hit: {'✅' if cache_test.cache_hit else '❌'}")
    print(f"   Response Time: {cache_test.generation_time_ms:.1f}ms")
    
    # Final health check
    final_health = await service.health_check()
    modules = final_health["modules"]
    
    print(f"\n📊 MODULAR METRICS:")
    print("-" * 25)
    print(f"   Optimization Score: {modules['optimization_engine']['score']:.1f}/100")
    print(f"   Performance Tier: {modules['optimization_engine']['tier']}")
    print(f"   Libraries Available: {modules['optimization_engine']['libraries_available']}")
    print(f"   Cache Hit Rate: {modules['cache_manager']['hit_rate_percent']:.1f}%")
    print(f"   Total Requests: {modules['metrics_collector']['total_requests']}")
    print(f"   Success Rate: {modules['metrics_collector']['success_rate_percent']:.1f}%")
    print(f"   Average Response: {modules['metrics_collector']['average_response_time_ms']:.1f}ms")
    
    print(f"\n🎉 MODULAR REFACTOR COMPLETED!")
    print("🏗️ Arquitectura completamente modular")
    print("🔧 Componentes independientes y reutilizables")
    print("⚡ Máximo rendimiento y mantenibilidad")
    print("📊 Configuración centralizada y flexible")

async def main():
    """Función principal"""
    await modular_demo()

match __name__:
    case "__main__":
    asyncio.run(main()) 