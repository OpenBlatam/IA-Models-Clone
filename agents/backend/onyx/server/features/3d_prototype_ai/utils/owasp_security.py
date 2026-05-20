"""
OWASP Security - Validación de seguridad OWASP
==============================================

Implementa:
- Validación de entrada
- Protección contra inyección
- Sanitización de datos
- Content Security Policy
"""

import logging
import re
import html
from typing import Dict, Optional, List, Any
from fastapi import Request, HTTPException, status

logger = logging.getLogger(__name__)


class OWASPSecurityValidator:
    """Validador de seguridad OWASP"""
    
    # Patrones peligrosos
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
        r"(--|#|/\*|\*/)",
        r"(\b(UNION|OR|AND)\b.*\b(SELECT|INSERT|UPDATE|DELETE)\b)",
    ]
    
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$(){}]",
        r"\b(cat|ls|pwd|whoami|id|uname|ps|kill|rm|mv|cp)\b",
    ]
    
    def __init__(self):
        self.sql_patterns = [re.compile(p, re.IGNORECASE) for p in self.SQL_INJECTION_PATTERNS]
        self.xss_patterns = [re.compile(p, re.IGNORECASE) for p in self.XSS_PATTERNS]
        self.cmd_patterns = [re.compile(p, re.IGNORECASE) for p in self.COMMAND_INJECTION_PATTERNS]
    
    def validate_input(self, value: str, input_type: str = "text") -> bool:
        """Valida entrada contra patrones de ataque"""
        if not isinstance(value, str):
            value = str(value)
        
        # Validar SQL injection
        for pattern in self.sql_patterns:
            if pattern.search(value):
                logger.warning(f"SQL injection attempt detected: {value[:50]}")
                return False
        
        # Validar XSS
        for pattern in self.xss_patterns:
            if pattern.search(value):
                logger.warning(f"XSS attempt detected: {value[:50]}")
                return False
        
        # Validar command injection (solo para ciertos tipos)
        if input_type in ["command", "filename"]:
            for pattern in self.cmd_patterns:
                if pattern.search(value):
                    logger.warning(f"Command injection attempt detected: {value[:50]}")
                    return False
        
        return True
    
    def sanitize_input(self, value: str) -> str:
        """Sanitiza entrada"""
        if not isinstance(value, str):
            value = str(value)
        
        # Escapar HTML
        value = html.escape(value)
        
        # Remover caracteres peligrosos
        value = re.sub(r'[<>"\']', '', value)
        
        return value
    
    def validate_json(self, data: Dict) -> Dict[str, List[str]]:
        """Valida un objeto JSON completo"""
        errors = {
            "sql_injection": [],
            "xss": [],
            "command_injection": []
        }
        
        def validate_value(key: str, value: Any):
            if isinstance(value, str):
                if not self.validate_input(value):
                    # Detectar tipo de ataque
                    for pattern in self.sql_patterns:
                        if pattern.search(value):
                            errors["sql_injection"].append(f"{key}: {value[:50]}")
                            break
                    for pattern in self.xss_patterns:
                        if pattern.search(value):
                            errors["xss"].append(f"{key}: {value[:50]}")
                            break
            elif isinstance(value, dict):
                for k, v in value.items():
                    validate_value(f"{key}.{k}", v)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    validate_value(f"{key}[{i}]", item)
        
        for key, value in data.items():
            validate_value(key, value)
        
        return errors


class DDoSProtectionMiddleware:
    """Middleware de protección DDoS"""
    
    def __init__(self, max_requests_per_minute: int = 60, 
                 max_requests_per_hour: int = 1000):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_hour = max_requests_per_hour
        self.request_counts: Dict[str, List[float]] = {}
    
    async def check_rate_limit(self, request: Request) -> bool:
        """Verifica rate limiting"""
        client_ip = request.client.host
        
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []
        
        import time
        current_time = time.time()
        
        # Limpiar requests antiguos
        self.request_counts[client_ip] = [
            t for t in self.request_counts[client_ip]
            if current_time - t < 3600  # Última hora
        ]
        
        # Verificar límites
        minute_requests = sum(
            1 for t in self.request_counts[client_ip]
            if current_time - t < 60
        )
        
        hour_requests = len(self.request_counts[client_ip])
        
        if minute_requests >= self.max_requests_per_minute:
            logger.warning(f"Rate limit exceeded for {client_ip}: {minute_requests} requests/min")
            return False
        
        if hour_requests >= self.max_requests_per_hour:
            logger.warning(f"Rate limit exceeded for {client_ip}: {hour_requests} requests/hour")
            return False
        
        # Registrar request
        self.request_counts[client_ip].append(current_time)
        return True


class SecurityHeadersMiddleware:
    """Middleware para headers de seguridad OWASP"""
    
    @staticmethod
    def get_security_headers() -> Dict[str, str]:
        """Obtiene headers de seguridad recomendados por OWASP"""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self' data:; "
                "connect-src 'self'; "
                "frame-ancestors 'none';"
            ),
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": (
                "geolocation=(), "
                "microphone=(), "
                "camera=(), "
                "payment=(), "
                "usb=()"
            ),
            "X-Permitted-Cross-Domain-Policies": "none"
        }


# Instancias globales
owasp_validator = OWASPSecurityValidator()
ddos_protection = DDoSProtectionMiddleware()




