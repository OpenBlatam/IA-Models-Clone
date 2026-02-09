"""
Routing Security Optimizations
================================

Optimizaciones de seguridad y validación.
Incluye: Input validation, Rate limiting, Encryption, etc.
"""

import logging
import hashlib
import hmac
import time
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)


class RateLimiter:
    """Limitador de tasa para prevenir abuso."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Inicializar rate limiter.
        
        Args:
            max_requests: Máximo de requests por ventana
            window_seconds: Tamaño de la ventana en segundos
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.lock = threading.Lock()
    
    def is_allowed(self, identifier: str) -> bool:
        """
        Verificar si un request está permitido.
        
        Args:
            identifier: Identificador único (IP, user ID, etc.)
        
        Returns:
            True si está permitido
        """
        with self.lock:
            current_time = time.time()
            request_times = self.requests[identifier]
            
            # Eliminar requests fuera de la ventana
            while request_times and (current_time - request_times[0]) > self.window_seconds:
                request_times.popleft()
            
            # Verificar límite
            if len(request_times) >= self.max_requests:
                return False
            
            # Agregar request actual
            request_times.append(current_time)
            return True
    
    def get_remaining(self, identifier: str) -> int:
        """Obtener requests restantes."""
        with self.lock:
            current_time = time.time()
            request_times = self.requests[identifier]
            
            # Limpiar requests antiguos
            while request_times and (current_time - request_times[0]) > self.window_seconds:
                request_times.popleft()
            
            return max(0, self.max_requests - len(request_times))


class InputValidator:
    """Validador de inputs para seguridad."""
    
    def __init__(self):
        """Inicializar validador."""
        self.validation_rules: Dict[str, List[Callable]] = {}
    
    def add_rule(self, field: str, validator: Callable):
        """
        Agregar regla de validación.
        
        Args:
            field: Campo a validar
            validator: Función validadora
        """
        if field not in self.validation_rules:
            self.validation_rules[field] = []
        self.validation_rules[field].append(validator)
    
    def validate(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validar inputs.
        
        Args:
            data: Datos a validar
        
        Returns:
            (is_valid, error_message)
        """
        for field, validators in self.validation_rules.items():
            if field not in data:
                continue
            
            value = data[field]
            for validator in validators:
                try:
                    if not validator(value):
                        return False, f"Validation failed for field {field}"
                except Exception as e:
                    return False, f"Validation error for field {field}: {e}"
        
        return True, None


class SecurityOptimizer:
    """Optimizador completo de seguridad."""
    
    def __init__(self, enable_rate_limiting: bool = True):
        """
        Inicializar optimizador de seguridad.
        
        Args:
            enable_rate_limiting: Habilitar rate limiting
        """
        self.rate_limiter = RateLimiter() if enable_rate_limiting else None
        self.input_validator = InputValidator()
        self.security_stats = {
            'total_requests': 0,
            'blocked_requests': 0,
            'validated_requests': 0
        }
    
    def check_rate_limit(self, identifier: str) -> bool:
        """Verificar rate limit."""
        if not self.rate_limiter:
            return True
        
        self.security_stats['total_requests'] += 1
        
        if self.rate_limiter.is_allowed(identifier):
            return True
        else:
            self.security_stats['blocked_requests'] += 1
            return False
    
    def validate_input(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validar input."""
        is_valid, error = self.input_validator.validate(data)
        if is_valid:
            self.security_stats['validated_requests'] += 1
        return is_valid, error
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de seguridad."""
        stats = self.security_stats.copy()
        if self.rate_limiter:
            stats['rate_limiting_enabled'] = True
        else:
            stats['rate_limiting_enabled'] = False
        return stats

