"""
Security Manager - Sistema de seguridad avanzado
================================================
"""

import logging
import hashlib
import secrets
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)


class SecurityManager:
    """
    Gestiona seguridad avanzada del sistema.
    """
    
    def __init__(self):
        """Inicializar gestor de seguridad"""
        self.blocked_ips: List[str] = []
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=15)
    
    def validate_input(
        self,
        input_data: str,
        input_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Valida entrada de usuario.
        
        Args:
            input_data: Datos de entrada
            input_type: Tipo de entrada
            
        Returns:
            Resultado de validación
        """
        validation = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validar longitud
        if len(input_data) > 1000000:  # 1MB
            validation["valid"] = False
            validation["errors"].append("Input demasiado grande")
        
        # Detectar patrones peligrosos
        dangerous_patterns = [
            (r"<script", "Posible XSS"),
            (r"javascript:", "Posible XSS"),
            (r"eval\(", "Código peligroso"),
            (r"exec\(", "Código peligroso"),
            (r"__import__", "Importación peligrosa")
        ]
        
        for pattern, description in dangerous_patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                validation["warnings"].append(description)
        
        # Validación específica por tipo
        if input_type == "code":
            code_validation = self._validate_code_security(input_data)
            validation["errors"].extend(code_validation["errors"])
            validation["warnings"].extend(code_validation["warnings"])
        
        validation["valid"] = len(validation["errors"]) == 0
        
        return validation
    
    def _validate_code_security(self, code: str) -> Dict[str, List[str]]:
        """Valida seguridad de código"""
        errors = []
        warnings = []
        
        # Detectar imports peligrosos
        dangerous_imports = [
            "os.system", "subprocess.call", "subprocess.run",
            "eval", "exec", "__import__"
        ]
        
        for dangerous in dangerous_imports:
            if dangerous in code:
                warnings.append(f"Uso de {dangerous} detectado")
        
        # Detectar acceso a archivos sensibles
        sensitive_paths = [
            "/etc/passwd", "/etc/shadow", "/proc",
            "C:\\Windows\\System32"
        ]
        
        for path in sensitive_paths:
            if path in code:
                errors.append(f"Acceso a ruta sensible: {path}")
        
        return {"errors": errors, "warnings": warnings}
    
    def check_rate_limit(
        self,
        identifier: str,
        max_requests: int = 100,
        window_seconds: int = 60
    ) -> tuple[bool, Optional[str]]:
        """
        Verifica rate limit avanzado.
        
        Args:
            identifier: Identificador (IP, user_id, etc.)
            max_requests: Máximo de peticiones
            window_seconds: Ventana de tiempo en segundos
            
        Returns:
            (allowed, reason)
        """
        now = datetime.now()
        window_start = now - timedelta(seconds=window_seconds)
        
        # Obtener intentos recientes
        if identifier not in self.failed_attempts:
            self.failed_attempts[identifier] = []
        
        recent_attempts = [
            attempt for attempt in self.failed_attempts[identifier]
            if attempt > window_start
        ]
        
        if len(recent_attempts) >= max_requests:
            return False, f"Rate limit excedido: {len(recent_attempts)}/{max_requests} en {window_seconds}s"
        
        return True, None
    
    def record_failed_attempt(self, identifier: str):
        """
        Registra intento fallido.
        
        Args:
            identifier: Identificador
        """
        if identifier not in self.failed_attempts:
            self.failed_attempts[identifier] = []
        
        self.failed_attempts[identifier].append(datetime.now())
        
        # Limpiar intentos antiguos
        cutoff = datetime.now() - self.lockout_duration
        self.failed_attempts[identifier] = [
            attempt for attempt in self.failed_attempts[identifier]
            if attempt > cutoff
        ]
        
        # Bloquear si excede límite
        if len(self.failed_attempts[identifier]) >= self.max_failed_attempts:
            if identifier not in self.blocked_ips:
                self.blocked_ips.append(identifier)
                logger.warning(f"IP bloqueada por múltiples intentos fallidos: {identifier}")
    
    def is_blocked(self, identifier: str) -> bool:
        """
        Verifica si un identificador está bloqueado.
        
        Args:
            identifier: Identificador
            
        Returns:
            True si está bloqueado
        """
        return identifier in self.blocked_ips
    
    def unblock(self, identifier: str) -> bool:
        """
        Desbloquea un identificador.
        
        Args:
            identifier: Identificador
            
        Returns:
            True si se desbloqueó
        """
        if identifier in self.blocked_ips:
            self.blocked_ips.remove(identifier)
            if identifier in self.failed_attempts:
                del self.failed_attempts[identifier]
            logger.info(f"Identificador desbloqueado: {identifier}")
            return True
        return False
    
    def sanitize_output(self, data: Any) -> Any:
        """
        Sanitiza datos de salida.
        
        Args:
            data: Datos a sanitizar
            
        Returns:
            Datos sanitizados
        """
        if isinstance(data, str):
            # Escapar caracteres peligrosos
            data = data.replace("<", "&lt;").replace(">", "&gt;")
            data = data.replace("'", "&#39;").replace('"', "&quot;")
        elif isinstance(data, dict):
            return {k: self.sanitize_output(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_output(item) for item in data]
        
        return data
    
    def generate_secure_token(self, length: int = 32) -> str:
        """
        Genera token seguro.
        
        Args:
            length: Longitud del token
            
        Returns:
            Token seguro
        """
        return secrets.token_urlsafe(length)
    
    def hash_sensitive_data(self, data: str) -> str:
        """
        Hashea datos sensibles.
        
        Args:
            data: Datos a hashear
            
        Returns:
            Hash SHA-256
        """
        return hashlib.sha256(data.encode()).hexdigest()

