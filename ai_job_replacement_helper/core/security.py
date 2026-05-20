"""
Security Service - Mejoras de seguridad
==========================================

Servicio para funciones de seguridad avanzadas.
"""

import logging
import hashlib
import secrets
import re
from typing import Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SecurityService:
    """Servicio de seguridad"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.failed_login_attempts: dict = {}  # user_id -> count
        self.locked_accounts: dict = {}  # user_id -> unlock_time
        logger.info("SecurityService initialized")
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash de contraseña con salt"""
        if not salt:
            salt = secrets.token_hex(16)
        
        # Usar PBKDF2 en producción (aquí simplificado)
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        )
        
        return password_hash.hex(), salt
    
    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verificar contraseña"""
        new_hash, _ = self.hash_password(password, salt)
        return new_hash == password_hash
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generar token seguro"""
        return secrets.token_urlsafe(length)
    
    def sanitize_input(self, text: str) -> str:
        """Sanitizar input del usuario"""
        # Remover caracteres peligrosos
        text = re.sub(r'[<>"\']', '', text)
        # Remover scripts
        text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        return text.strip()
    
    def check_rate_limit(self, identifier: str, max_attempts: int = 5, window_minutes: int = 15) -> Tuple[bool, Optional[str]]:
        """Verificar rate limit para login"""
        now = datetime.now()
        
        # Verificar si la cuenta está bloqueada
        if identifier in self.locked_accounts:
            unlock_time = self.locked_accounts[identifier]
            if now < unlock_time:
                remaining = (unlock_time - now).seconds // 60
                return False, f"Account locked. Try again in {remaining} minutes."
            else:
                # Desbloquear
                del self.locked_accounts[identifier]
                self.failed_login_attempts[identifier] = 0
        
        # Verificar intentos fallidos
        attempts = self.failed_login_attempts.get(identifier, 0)
        
        if attempts >= max_attempts:
            # Bloquear cuenta
            self.locked_accounts[identifier] = now + timedelta(minutes=window_minutes)
            return False, f"Too many failed attempts. Account locked for {window_minutes} minutes."
        
        return True, None
    
    def record_failed_login(self, identifier: str):
        """Registrar intento de login fallido"""
        self.failed_login_attempts[identifier] = self.failed_login_attempts.get(identifier, 0) + 1
    
    def reset_login_attempts(self, identifier: str):
        """Resetear intentos de login"""
        if identifier in self.failed_login_attempts:
            del self.failed_login_attempts[identifier]
    
    def validate_csrf_token(self, token: str, session_token: str) -> bool:
        """Validar token CSRF"""
        return token == session_token and len(token) >= 32
    
    def check_sql_injection(self, text: str) -> bool:
        """Detectar posibles SQL injection"""
        dangerous_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
            r"(--|#|/\*|\*/)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def check_xss(self, text: str) -> bool:
        """Detectar posibles XSS"""
        xss_patterns = [
            r"<script.*?>",
            r"javascript:",
            r"onerror\s*=",
            r"onload\s*=",
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False


# Instancia global
security_service = SecurityService()




