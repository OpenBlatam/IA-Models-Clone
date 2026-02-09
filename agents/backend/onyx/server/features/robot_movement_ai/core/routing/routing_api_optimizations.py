"""
Routing API Optimizations
==========================

Optimizaciones para APIs y endpoints.
Incluye: Request validation, Response caching, API versioning, etc.
"""

import logging
import time
import hashlib
import json
from typing import Dict, Any, List, Optional, Callable, Tuple
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)


class RequestValidator:
    """Validador de requests de API."""
    
    def __init__(self):
        """Inicializar validador."""
        self.validation_rules: Dict[str, Dict[str, Any]] = {}
    
    def add_validation_rule(self, endpoint: str, field: str, rule: Dict[str, Any]):
        """
        Agregar regla de validación.
        
        Args:
            endpoint: Endpoint a validar
            field: Campo a validar
            rule: Regla de validación (type, required, min, max, etc.)
        """
        if endpoint not in self.validation_rules:
            self.validation_rules[endpoint] = {}
        self.validation_rules[endpoint][field] = rule
    
    def validate_request(self, endpoint: str, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validar request.
        
        Args:
            endpoint: Endpoint
            data: Datos del request
        
        Returns:
            (is_valid, error_message)
        """
        if endpoint not in self.validation_rules:
            return True, None
        
        rules = self.validation_rules[endpoint]
        
        for field, rule in rules.items():
            if rule.get('required', False) and field not in data:
                return False, f"Field '{field}' is required"
            
            if field in data:
                value = data[field]
                
                # Validar tipo
                expected_type = rule.get('type')
                if expected_type and not isinstance(value, expected_type):
                    return False, f"Field '{field}' must be of type {expected_type.__name__}"
                
                # Validar min/max
                if isinstance(value, (int, float)):
                    if 'min' in rule and value < rule['min']:
                        return False, f"Field '{field}' must be >= {rule['min']}"
                    if 'max' in rule and value > rule['max']:
                        return False, f"Field '{field}' must be <= {rule['max']}"
                
                # Validar longitud
                if isinstance(value, str):
                    if 'min_length' in rule and len(value) < rule['min_length']:
                        return False, f"Field '{field}' must have length >= {rule['min_length']}"
                    if 'max_length' in rule and len(value) > rule['max_length']:
                        return False, f"Field '{field}' must have length <= {rule['max_length']}"
        
        return True, None


class ResponseCache:
    """Cache de respuestas de API."""
    
    def __init__(self, max_size: int = 1000, ttl: float = 300.0):
        """
        Inicializar cache de respuestas.
        
        Args:
            max_size: Tamaño máximo del cache
            ttl: Time to live en segundos
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self.lock = threading.Lock()
    
    def _generate_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generar clave de cache."""
        key_data = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, endpoint: str, params: Dict[str, Any]) -> Optional[Any]:
        """Obtener respuesta del cache."""
        key = self._generate_key(endpoint, params)
        
        with self.lock:
            if key in self.cache:
                # Verificar TTL
                if time.time() - self.timestamps[key] < self.ttl:
                    # Mover al final (LRU)
                    self.cache.move_to_end(key)
                    return self.cache[key]
                else:
                    # Expirar
                    del self.cache[key]
                    del self.timestamps[key]
            
            return None
    
    def put(self, endpoint: str, params: Dict[str, Any], response: Any):
        """Guardar respuesta en cache."""
        key = self._generate_key(endpoint, params)
        
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Evictar LRU
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
            
            self.cache[key] = response
            self.timestamps[key] = time.time()
    
    def clear(self):
        """Limpiar cache."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()


class APIVersionManager:
    """Gestor de versiones de API."""
    
    def __init__(self, default_version: str = "v1"):
        """
        Inicializar gestor de versiones.
        
        Args:
            default_version: Versión por defecto
        """
        self.default_version = default_version
        self.versions: Dict[str, Dict[str, Callable]] = {}
    
    def register_endpoint(self, version: str, endpoint: str, handler: Callable):
        """
        Registrar endpoint para una versión.
        
        Args:
            version: Versión de API
            endpoint: Endpoint
            handler: Handler del endpoint
        """
        if version not in self.versions:
            self.versions[version] = {}
        self.versions[version][endpoint] = handler
    
    def get_handler(self, version: str, endpoint: str) -> Optional[Callable]:
        """
        Obtener handler para versión y endpoint.
        
        Args:
            version: Versión de API
            endpoint: Endpoint
        
        Returns:
            Handler o None
        """
        version = version or self.default_version
        
        if version in self.versions and endpoint in self.versions[version]:
            return self.versions[version][endpoint]
        
        return None


class APIOptimizer:
    """Optimizador completo de API."""
    
    def __init__(self, enable_response_cache: bool = True):
        """
        Inicializar optimizador de API.
        
        Args:
            enable_response_cache: Habilitar cache de respuestas
        """
        self.request_validator = RequestValidator()
        self.response_cache = ResponseCache() if enable_response_cache else None
        self.version_manager = APIVersionManager()
        self.stats = {
            'total_requests': 0,
            'cached_responses': 0,
            'validation_errors': 0
        }
    
    def validate_and_cache(
        self,
        endpoint: str,
        params: Dict[str, Any],
        response_func: Callable
    ) -> Any:
        """
        Validar request y obtener respuesta (con cache).
        
        Args:
            endpoint: Endpoint
            params: Parámetros del request
            response_func: Función para generar respuesta
        
        Returns:
            Respuesta
        """
        self.stats['total_requests'] += 1
        
        # Validar request
        is_valid, error = self.request_validator.validate_request(endpoint, params)
        if not is_valid:
            self.stats['validation_errors'] += 1
            raise ValueError(error)
        
        # Intentar obtener del cache
        if self.response_cache:
            cached_response = self.response_cache.get(endpoint, params)
            if cached_response is not None:
                self.stats['cached_responses'] += 1
                return cached_response
        
        # Generar respuesta
        response = response_func()
        
        # Guardar en cache
        if self.response_cache:
            self.response_cache.put(endpoint, params, response)
        
        return response
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        stats = self.stats.copy()
        if self.response_cache:
            stats['cache_size'] = len(self.response_cache.cache)
            stats['cache_hit_rate'] = (
                self.stats['cached_responses'] / max(self.stats['total_requests'], 1)
            )
        return stats

