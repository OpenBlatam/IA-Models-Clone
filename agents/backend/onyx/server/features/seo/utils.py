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

import time
import hashlib
import re
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse, urljoin
import logging
from dataclasses import dataclass
from enum import Enum
        from collections import Counter
from typing import Any, List, Dict, Optional
import asyncio
"""
Utilidades refactorizadas para el Servicio SEO Ultra-Optimizado.
"""


logger = logging.getLogger(__name__)


class URLStatus(Enum):
    """Estados de validación de URL."""
    VALID = "valid"
    INVALID = "invalid"
    REDIRECT = "redirect"
    ERROR = "error"


@dataclass
class URLInfo:
    """Información de una URL."""
    original: str
    normalized: str
    domain: str
    protocol: str
    path: str
    status: URLStatus
    redirect_url: Optional[str] = None


class URLUtils:
    """Utilidades para manejo de URLs."""
    
    @staticmethod
    def normalize_url(url: str) -> str:
        """Normaliza una URL."""
        if not url:
            return ""
        
        # Remover espacios
        url = url.strip()
        
        # Agregar protocolo si no tiene
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        return url
    
    @staticmethod
    def parse_url(url: str) -> URLInfo:
        """Parsea y valida una URL."""
        try:
            normalized = URLUtils.normalize_url(url)
            parsed = urlparse(normalized)
            
            return URLInfo(
                original=url,
                normalized=normalized,
                domain=parsed.netloc,
                protocol=parsed.scheme,
                path=parsed.path,
                status=URLStatus.VALID
            )
        except Exception as e:
            logger.error(f"Error parseando URL {url}: {e}")
            return URLInfo(
                original=url,
                normalized="",
                domain="",
                protocol="",
                path="",
                status=URLStatus.ERROR
            )
    
    @staticmethod
    def is_internal_link(base_url: str, link_url: str) -> bool:
        """Verifica si un enlace es interno."""
        try:
            base_domain = urlparse(base_url).netloc
            link_domain = urlparse(link_url).netloc
            return base_domain == link_domain
        except:
            return False
    
    @staticmethod
    def get_domain_from_url(url: str) -> str:
        """Extrae el dominio de una URL."""
        try:
            return urlparse(url).netloc
        except:
            return ""


class TextUtils:
    """Utilidades para procesamiento de texto."""
    
    @staticmethod
    def clean_text(text: str, max_length: int = 1000) -> str:
        """Limpia y normaliza texto."""
        if not text:
            return ""
        
        # Remover espacios extra
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Limitar longitud
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """Extrae palabras clave del texto."""
        if not text:
            return []
        
        # Limpiar texto
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Dividir en palabras
        words = clean_text.split()
        
        # Filtrar palabras cortas y comunes
        stop_words = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 'las', 'una', 'como', 'pero', 'sus', 'me', 'hasta', 'hay', 'donde', 'han', 'quien', 'están', 'estado', 'desde', 'todo', 'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante', 'ellos', 'e', 'esto', 'mí', 'antes', 'algunos', 'qué', 'unos', 'yo', 'otro', 'otras', 'otra', 'él', 'tanto', 'esa', 'estos', 'mucho', 'quienes', 'nada', 'muchos', 'cual', 'poco', 'ella', 'estar', 'estas', 'algunas', 'algo', 'nosotros'}
        
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        # Contar frecuencia
        word_counts = Counter(keywords)
        
        # Retornar las más frecuentes
        return [word for word, count in word_counts.most_common(max_keywords)]
    
    @staticmethod
    def calculate_readability_score(text: str) -> float:
        """Calcula un score de legibilidad básico."""
        if not text:
            return 0.0
        
        sentences = re.split(r'[.!?]+', text)
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Score simple: menor longitud de oración = mejor legibilidad
        if avg_sentence_length <= 10:
            return 1.0
        elif avg_sentence_length <= 15:
            return 0.8
        elif avg_sentence_length <= 20:
            return 0.6
        else:
            return 0.4


class PerformanceUtils:
    """Utilidades para medición de rendimiento."""
    
    @staticmethod
    def measure_time(func) -> Any:
        """Decorator para medir tiempo de ejecución."""
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            logger.info(f"{func.__name__} ejecutado en {end_time - start_time:.3f}s")
            return result
        return wrapper
    
    @staticmethod
    def calculate_performance_score(load_time: float, content_length: int, image_count: int) -> float:
        """Calcula un score de rendimiento."""
        score = 100.0
        
        # Penalizar tiempo de carga
        if load_time > 5:
            score -= 30
        elif load_time > 3:
            score -= 15
        elif load_time > 2:
            score -= 5
        
        # Penalizar contenido muy corto
        if content_length < 300:
            score -= 20
        elif content_length < 500:
            score -= 10
        
        # Penalizar muchas imágenes sin optimizar
        if image_count > 20:
            score -= 10
        
        return max(0, score)


class CacheUtils:
    """Utilidades para manejo de cache."""
    
    @staticmethod
    def generate_cache_key(url: str, options: Dict[str, Any]) -> str:
        """Genera una clave única para el cache."""
        # Crear string de opciones
        options_str = str(sorted(options.items()))
        
        # Combinar URL y opciones
        combined = f"{url}_{options_str}"
        
        # Generar hash
        return hashlib.md5(combined.encode()).hexdigest()
    
    @staticmethod
    def should_cache_response(response: Any, max_age: int = 3600) -> bool:
        """Determina si una respuesta debe ser cachead."""
        # Verificar si la respuesta es exitosa
        if hasattr(response, 'success') and not response.success:
            return False
        
        # Verificar si tiene datos válidos
        if hasattr(response, 'data') and not response.data:
            return False
        
        return True


class ValidationUtils:
    """Utilidades para validación."""
    
    @staticmethod
    def validate_seo_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Valida datos SEO."""
        errors = []
        
        # Validar título
        if not data.get('title'):
            errors.append("Falta título de página")
        elif len(data['title']) > 60:
            errors.append("Título demasiado largo")
        
        # Validar meta descripción
        if not data.get('meta_description'):
            errors.append("Falta meta descripción")
        elif len(data['meta_description']) > 160:
            errors.append("Meta descripción demasiado larga")
        
        # Validar headers
        h1_count = len(data.get('h1_tags', []))
        if h1_count == 0:
            errors.append("No hay tags H1")
        elif h1_count > 1:
            errors.append("Múltiples tags H1")
        
        # Validar contenido
        content_length = data.get('content_length', 0)
        if content_length < 300:
            errors.append("Contenido muy corto")
        
        # Validar imágenes
        images = data.get('images', [])
        images_without_alt = len([img for img in images if not img.get('alt')])
        if images_without_alt > 0:
            errors.append(f"{images_without_alt} imágenes sin alt text")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_url_batch(urls: List[str], max_urls: int = 20) -> Tuple[bool, List[str]]:
        """Valida un lote de URLs."""
        errors = []
        
        if not urls:
            errors.append("Lista de URLs vacía")
            return False, errors
        
        if len(urls) > max_urls:
            errors.append(f"Máximo {max_urls} URLs permitidas")
        
        for i, url in enumerate(urls):
            if not url:
                errors.append(f"URL {i+1} está vacía")
            elif not URLUtils.normalize_url(url):
                errors.append(f"URL {i+1} inválida: {url}")
        
        return len(errors) == 0, errors


class MetricsUtils:
    """Utilidades para métricas."""
    
    @staticmethod
    def calculate_seo_score(data: Dict[str, Any]) -> float:
        """Calcula un score SEO básico."""
        score = 50.0  # Score base
        
        # Título
        if data.get('title'):
            score += 10
            if 30 <= len(data['title']) <= 60:
                score += 5
        
        # Meta descripción
        if data.get('meta_description'):
            score += 10
            if 120 <= len(data['meta_description']) <= 160:
                score += 5
        
        # Headers
        h1_count = len(data.get('h1_tags', []))
        if h1_count == 1:
            score += 10
        elif h1_count > 1:
            score -= 5
        
        h2_count = len(data.get('h2_tags', []))
        if h2_count > 0:
            score += 5
        
        # Contenido
        content_length = data.get('content_length', 0)
        if content_length >= 300:
            score += 10
        if content_length >= 1000:
            score += 5
        
        # Imágenes
        images = data.get('images', [])
        if images:
            images_with_alt = len([img for img in images if img.get('alt')])
            alt_percentage = images_with_alt / len(images) if images else 0
            score += alt_percentage * 10
        
        # Enlaces
        links = data.get('links', [])
        if links:
            internal_links = len([link for link in links if link.get('is_internal')])
            if internal_links > 0:
                score += 5
        
        # Palabras clave
        if data.get('keywords'):
            score += 5
        
        return min(100, max(0, score))
    
    @staticmethod
    def format_performance_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Formatea métricas de rendimiento."""
        return {
            "timestamp": time.time(),
            "memory_usage_mb": round(metrics.get('memory_usage', 0) / 1024 / 1024, 2),
            "response_time_ms": round(metrics.get('response_time', 0) * 1000, 2),
            "cache_hit_rate": round(metrics.get('cache_hit_rate', 0), 2),
            "error_rate": round(metrics.get('error_rate', 0), 2),
            "requests_per_second": round(metrics.get('requests_per_second', 0), 2)
        }


class LoggingUtils:
    """Utilidades para logging."""
    
    @staticmethod
    def setup_logging(level: str = "INFO", format_type: str = "default"):
        """Configura el sistema de logging."""
        log_format = {
            "default": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "json": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}',
            "simple": "%(levelname)s - %(message)s"
        }
        
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=log_format.get(format_type, log_format["default"]),
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    @staticmethod
    def log_performance(operation: str, duration: float, **kwargs):
        """Log de métricas de rendimiento."""
        logger.info(f"Performance: {operation} took {duration:.3f}s", extra={
            "operation": operation,
            "duration": duration,
            **kwargs
        })
    
    @staticmethod
    def log_error(operation: str, error: Exception, **kwargs):
        """Log de errores."""
        logger.error(f"Error in {operation}: {str(error)}", extra={
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            **kwargs
        })


# Funciones de conveniencia
def normalize_and_validate_url(url: str) -> URLInfo:
    """Normaliza y valida una URL."""
    return URLUtils.parse_url(url)


def calculate_seo_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calcula métricas SEO completas."""
    return {
        "seo_score": MetricsUtils.calculate_seo_score(data),
        "readability_score": TextUtils.calculate_readability_score(data.get('content', '')),
        "performance_score": PerformanceUtils.calculate_performance_score(
            data.get('load_time', 0),
            data.get('content_length', 0),
            len(data.get('images', []))
        ),
        "validation": ValidationUtils.validate_seo_data(data)
    }


def format_response(data: Any, cache_hit: bool = False, processing_time: float = 0) -> Dict[str, Any]:
    """Formatea una respuesta estándar."""
    response = {
        "success": True,
        "data": data,
        "timestamp": time.time(),
        "processing_time": round(processing_time, 3)
    }
    
    if cache_hit:
        response["cache_hit"] = True
    
    return response 