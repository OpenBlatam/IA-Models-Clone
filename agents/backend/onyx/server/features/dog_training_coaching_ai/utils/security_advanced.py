"""
Advanced Security Utilities
============================
Utilidades avanzadas de seguridad.
"""

import hashlib
import hmac
import secrets
import base64
import json
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
import re

from .logger import get_logger

logger = get_logger(__name__)


class TokenGenerator:
    """Generador de tokens seguros."""
    
    def __init__(self, secret_key: str):
        """
        Inicializar generador de tokens.
        
        Args:
            secret_key: Clave secreta
        """
        self.secret_key = secret_key.encode() if isinstance(secret_key, str) else secret_key
    
    def generate_token(self, data: Dict[str, Any], expires_in: Optional[int] = None) -> str:
        """
        Generar token firmado.
        
        Args:
            data: Datos a incluir en el token
            expires_in: Tiempo de expiración en segundos
            
        Returns:
            Token firmado
        """
        if expires_in:
            data["exp"] = (datetime.now() + timedelta(seconds=expires_in)).timestamp()
        
        data["iat"] = datetime.now().timestamp()
        
        # Codificar datos
        payload = base64.urlsafe_b64encode(
            json.dumps(data).encode()
        ).decode()
        
        # Generar firma
        signature = hmac.new(
            self.secret_key,
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"{payload}.{signature}"
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verificar token.
        
        Args:
            token: Token a verificar
            
        Returns:
            Datos del token o None si inválido
        """
        try:
            parts = token.split(".")
            if len(parts) != 2:
                return None
            
            payload, signature = parts
            
            # Verificar firma
            expected_signature = hmac.new(
                self.secret_key,
                payload.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                return None
            
            # Decodificar payload
            data_str = base64.urlsafe_b64decode(payload).decode()
            data = json.loads(data_str)
            
            # Verificar expiración
            if "exp" in data:
                if datetime.now().timestamp() > data["exp"]:
                    return None
            
            return data
        
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None


class InputSanitizer:
    """Sanitizador avanzado de inputs."""
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """
        Sanitizar string.
        
        Args:
            value: Valor a sanitizar
            max_length: Longitud máxima
            
        Returns:
            String sanitizado
        """
        # Remover caracteres de control
        value = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
        
        # Limitar longitud
        if len(value) > max_length:
            value = value[:max_length]
        
        # Remover espacios múltiples
        value = re.sub(r'\s+', ' ', value)
        
        return value.strip()
    
    @staticmethod
    def sanitize_html(value: str) -> str:
        """
        Sanitizar HTML.
        
        Args:
            value: HTML a sanitizar
            
        Returns:
            HTML sanitizado
        """
        # Remover tags peligrosos
        dangerous_tags = ['script', 'iframe', 'object', 'embed', 'link', 'style']
        
        for tag in dangerous_tags:
            value = re.sub(
                f'<{tag}[^>]*>.*?</{tag}>',
                '',
                value,
                flags=re.IGNORECASE | re.DOTALL
            )
        
        return value
    
    @staticmethod
    def sanitize_sql_input(value: str) -> str:
        """
        Sanitizar input SQL.
        
        Args:
            value: Input a sanitizar
            
        Returns:
            Input sanitizado
        """
        # Escapar comillas simples
        value = value.replace("'", "''")
        
        # Remover caracteres peligrosos
        value = re.sub(r'[;--]', '', value)
        
        return value
    
    @staticmethod
    def sanitize_path(value: str) -> str:
        """
        Sanitizar path de archivo.
        
        Args:
            value: Path a sanitizar
            
        Returns:
            Path sanitizado
        """
        # Remover componentes peligrosos
        value = value.replace('..', '')
        value = value.replace('//', '/')
        
        # Remover caracteres especiales
        value = re.sub(r'[^a-zA-Z0-9/._-]', '', value)
        
        return value


class SecurityAuditor:
    """Auditor de seguridad."""
    
    def __init__(self):
        self.audit_log: List[Dict[str, Any]] = []
    
    def log_security_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        severity: str = "info"
    ):
        """
        Registrar evento de seguridad.
        
        Args:
            event_type: Tipo de evento
            details: Detalles del evento
            severity: Severidad (info, warning, error)
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "severity": severity,
            "details": details
        }
        
        self.audit_log.append(event)
        logger.warning(f"Security event: {event_type}", **details)
    
    def detect_suspicious_activity(
        self,
        activity: Dict[str, Any]
    ) -> bool:
        """
        Detectar actividad sospechosa.
        
        Args:
            activity: Actividad a analizar
            
        Returns:
            True si es sospechosa
        """
        # Detectar múltiples intentos fallidos
        failed_attempts = sum(
            1 for event in self.audit_log
            if event.get("event_type") == "auth_failure"
        )
        
        if failed_attempts > 5:
            self.log_security_event(
                "suspicious_activity",
                {"reason": "multiple_failed_attempts", "count": failed_attempts},
                severity="error"
            )
            return True
        
        # Detectar patrones sospechosos
        suspicious_patterns = [
            "sql injection",
            "xss",
            "csrf",
            "path traversal"
        ]
        
        activity_str = str(activity).lower()
        for pattern in suspicious_patterns:
            if pattern in activity_str:
                self.log_security_event(
                    "suspicious_activity",
                    {"reason": f"pattern_detected: {pattern}"},
                    severity="error"
                )
                return True
        
        return False
    
    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Obtener log de auditoría.
        
        Args:
            limit: Límite de eventos
            
        Returns:
            Log de auditoría
        """
        return self.audit_log[-limit:]


def generate_secure_password(length: int = 16) -> str:
    """
    Generar contraseña segura.
    
    Args:
        length: Longitud de la contraseña
        
    Returns:
        Contraseña segura
    """
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def hash_password(password: str, salt: Optional[str] = None) -> Dict[str, str]:
    """
    Hashear contraseña.
    
    Args:
        password: Contraseña a hashear
        salt: Salt opcional
        
    Returns:
        Hash y salt
    """
    if salt is None:
        salt = secrets.token_hex(16)
    
    password_bytes = password.encode()
    salt_bytes = salt.encode()
    
    hash_obj = hashlib.pbkdf2_hmac('sha256', password_bytes, salt_bytes, 100000)
    hash_hex = hash_obj.hex()
    
    return {
        "hash": hash_hex,
        "salt": salt
    }


def verify_password(password: str, hash_value: str, salt: str) -> bool:
    """
    Verificar contraseña.
    
    Args:
        password: Contraseña a verificar
        hash_value: Hash almacenado
        salt: Salt usado
        
    Returns:
        True si la contraseña es correcta
    """
    result = hash_password(password, salt)
    return hmac.compare_digest(result["hash"], hash_value)


