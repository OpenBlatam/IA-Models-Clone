from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Protocol
from dataclasses import dataclass
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Abstract interfaces for the ultra-optimized SEO service.
Defines contracts for all core components.
"""



@dataclass
class SEOMetrics:
    """Métricas de rendimiento ultra-optimizadas."""
    load_time: float
    memory_usage: float
    cache_hit: bool
    processing_time: float
    elements_extracted: int
    compression_ratio: float
    network_latency: float


class HTMLParser(ABC):
    """Interfaz abstracta para parsers HTML ultra-rápidos."""
    
    @abstractmethod
    def parse(self, html_content: str, url: str) -> Dict[str, Any]:
        """Parsea el contenido HTML con máxima velocidad."""
        pass
    
    @abstractmethod
    def get_parser_name(self) -> str:
        """Retorna el nombre del parser."""
        pass


class HTTPClient(ABC):
    """Interfaz abstracta para clientes HTTP ultra-optimizados."""
    
    @abstractmethod
    async async def fetch(self, url: str) -> Optional[str]:
        """Obtiene contenido HTML con throttling y retry."""
        pass
    
    @abstractmethod
    async def measure_load_time(self, url: str) -> Optional[float]:
        """Mide tiempo de carga ultra-optimizado."""
        pass
    
    @abstractmethod
    async def close(self) -> Any:
        """Cierra la sesión HTTP."""
        pass


class CacheManager(ABC):
    """Interfaz abstracta para gestores de caché ultra-optimizados."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Obtiene datos del caché con descompresión."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any):
        """Almacena datos en caché con compresión."""
        pass
    
    @abstractmethod
    def clear(self) -> int:
        """Limpia el caché y retorna elementos eliminados."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del caché."""
        pass


class SEOAnalyzer(ABC):
    """Interfaz abstracta para analizadores SEO ultra-optimizados."""
    
    @abstractmethod
    async def analyze(self, seo_data: Dict[str, Any], url: str) -> Dict[str, Any]:
        """Analiza datos SEO con máxima eficiencia."""
        pass
    
    @abstractmethod
    def get_analyzer_name(self) -> str:
        """Retorna el nombre del analizador."""
        pass


class PerformanceTracker(Protocol):
    """Protocolo para tracking de rendimiento."""
    
    def start_timer(self, name: str):
        """Inicia un timer."""
        pass
    
    def end_timer(self, name: str) -> float:
        """Termina un timer y retorna la duración."""
        pass
    
    def record_metric(self, name: str, value: float):
        """Registra una métrica."""
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtiene todas las métricas."""
        pass


class ErrorHandler(Protocol):
    """Protocolo para manejo de errores."""
    
    def handle_error(self, error: Exception, context: str) -> None:
        """Maneja un error con contexto."""
        pass
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determina si se debe reintentar."""
        pass
    
    def get_retry_delay(self, attempt: int) -> float:
        """Obtiene el delay para reintento."""
        pass


class ConfigurationProvider(Protocol):
    """Protocolo para proveedores de configuración."""
    
    def get(self, key: str, default: Any = None) -> Optional[Dict[str, Any]]:
        """Obtiene un valor de configuración."""
        pass
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Obtiene un valor entero de configuración."""
        pass
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Obtiene un valor float de configuración."""
        pass
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Obtiene un valor booleano de configuración."""
        pass 