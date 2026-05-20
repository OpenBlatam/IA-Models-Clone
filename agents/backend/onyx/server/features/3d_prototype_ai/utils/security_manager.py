"""
Security Manager - Sistema de seguridad avanzada
=================================================
"""

import logging
import hashlib
import secrets
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import base64

logger = logging.getLogger(__name__)


class SecurityManager:
    """Sistema de seguridad avanzada"""
    
    def __init__(self):
        self.secrets: Dict[str, str] = {}
        self.encryption_key: Optional[bytes] = None
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Dict[str, str]:
        """Hashea una contraseña"""
        if not salt:
            salt = secrets.token_hex(16)
        
        # Usar SHA256 (en producción usar bcrypt o argon2)
        hash_obj = hashlib.sha256()
        hash_obj.update((password + salt).encode())
        password_hash = hash_obj.hexdigest()
        
        return {
            "hash": password_hash,
            "salt": salt
        }
    
    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verifica una contraseña"""
        result = self.hash_password(password, salt)
        return result["hash"] == password_hash
    
    def generate_api_key(self, length: int = 32) -> str:
        """Genera una API key"""
        return secrets.token_urlsafe(length)
    
    def generate_secret(self, length: int = 64) -> str:
        """Genera un secreto"""
        return secrets.token_hex(length)
    
    def encrypt_data(self, data: str, key: Optional[bytes] = None) -> str:
        """Encripta datos (simplificado)"""
        if not key:
            key = self.encryption_key or secrets.token_bytes(32)
            self.encryption_key = key
        
        # En producción usar AES o similar
        # Por ahora, solo base64 encoding (NO es encriptación real)
        encoded = base64.b64encode(data.encode()).decode()
        return encoded
    
    def decrypt_data(self, encrypted_data: str, key: Optional[bytes] = None) -> str:
        """Desencripta datos"""
        try:
            decoded = base64.b64decode(encrypted_data).decode()
            return decoded
        except Exception as e:
            logger.error(f"Error desencriptando: {e}")
            raise
    
    def store_secret(self, key: str, value: str, encrypt: bool = True):
        """Almacena un secreto"""
        if encrypt:
            value = self.encrypt_data(value)
        
        self.secrets[key] = value
        logger.info(f"Secreto almacenado: {key}")
    
    def get_secret(self, key: str, decrypt: bool = True) -> Optional[str]:
        """Obtiene un secreto"""
        value = self.secrets.get(key)
        if not value:
            return None
        
        if decrypt:
            try:
                return self.decrypt_data(value)
            except:
                return value
        
        return value
    
    def generate_csrf_token(self) -> str:
        """Genera token CSRF"""
        return secrets.token_hex(32)
    
    def validate_csrf_token(self, token: str, stored_token: str) -> bool:
        """Valida token CSRF"""
        return secrets.compare_digest(token, stored_token)
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitiza nombre de archivo"""
        import re
        # Remover caracteres peligrosos
        sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Limitar longitud
        if len(sanitized) > 255:
            sanitized = sanitized[:255]
        return sanitized
    
    def check_password_strength(self, password: str) -> Dict[str, Any]:
        """Verifica fortaleza de contraseña"""
        score = 0
        feedback = []
        
        if len(password) >= 8:
            score += 1
        else:
            feedback.append("La contraseña debe tener al menos 8 caracteres")
        
        if any(c.isupper() for c in password):
            score += 1
        else:
            feedback.append("Agrega letras mayúsculas")
        
        if any(c.islower() for c in password):
            score += 1
        else:
            feedback.append("Agrega letras minúsculas")
        
        if any(c.isdigit() for c in password):
            score += 1
        else:
            feedback.append("Agrega números")
        
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            score += 1
        else:
            feedback.append("Agrega caracteres especiales")
        
        if score <= 2:
            strength = "weak"
        elif score <= 3:
            strength = "medium"
        else:
            strength = "strong"
        
        return {
            "strength": strength,
            "score": score,
            "max_score": 5,
            "feedback": feedback
        }




