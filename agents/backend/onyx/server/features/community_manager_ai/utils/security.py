"""
Security Utils - Utilidades de Seguridad
==========================================

Utilidades para seguridad y encriptación.
"""

import hashlib
import hmac
import base64
import re
from typing import Optional, Tuple, List
from cryptography.fernet import Fernet
import os
import logging

logger = logging.getLogger(__name__)


class SecurityUtils:
    """Utilidades de seguridad"""
    
    @staticmethod
    def encrypt_credentials(credentials: dict, key: Optional[str] = None) -> str:
        """
        Encriptar credenciales
        
        Args:
            credentials: Dict con credenciales
            key: Clave de encriptación (opcional, usa env var)
            
        Returns:
            String encriptado (base64)
        """
        try:
            if key is None:
                key = os.getenv("ENCRYPTION_KEY")
                if not key:
                    # Generar clave si no existe
                    key = Fernet.generate_key().decode()
                    logger.warning("Usando clave generada, guarda ENCRYPTION_KEY en .env")
            
            if isinstance(key, str):
                key = key.encode()
            
            fernet = Fernet(key)
            
            import json
            credentials_json = json.dumps(credentials)
            encrypted = fernet.encrypt(credentials_json.encode())
            
            return base64.b64encode(encrypted).decode()
            
        except Exception as e:
            logger.error(f"Error encriptando credenciales: {e}")
            raise
    
    @staticmethod
    def decrypt_credentials(encrypted_data: str, key: Optional[str] = None) -> dict:
        """
        Desencriptar credenciales
        
        Args:
            encrypted_data: Datos encriptados (base64)
            key: Clave de encriptación
            
        Returns:
            Dict con credenciales
        """
        try:
            if key is None:
                key = os.getenv("ENCRYPTION_KEY")
                if not key:
                    raise ValueError("ENCRYPTION_KEY no configurada")
            
            if isinstance(key, str):
                key = key.encode()
            
            fernet = Fernet(key)
            
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted = fernet.decrypt(encrypted_bytes)
            
            import json
            return json.loads(decrypted.decode())
            
        except Exception as e:
            logger.error(f"Error desencriptando credenciales: {e}")
            raise
    
    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None):
        """
        Hashear contraseña
        
        Args:
            password: Contraseña
            salt: Salt opcional
            
        Returns:
            Tuple (hash, salt)
        """
        if salt is None:
            salt = os.urandom(32).hex()
        
        # Usar PBKDF2
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt.encode(),
            100000  # Iteraciones
        )
        
        hash_hex = key.hex()
        return hash_hex, salt
    
    @staticmethod
    def verify_password(password: str, hash_value: str, salt: str) -> bool:
        """
        Verificar contraseña
        
        Args:
            password: Contraseña a verificar
            hash_value: Hash almacenado
            salt: Salt usado
            
        Returns:
            True si la contraseña es correcta
        """
        computed_hash, _ = SecurityUtils.hash_password(password, salt)
        return hmac.compare_digest(computed_hash, hash_value)
    
    @staticmethod
    def generate_api_key() -> str:
        """
        Generar API key
        
        Returns:
            API key generada
        """
        return base64.urlsafe_b64encode(os.urandom(32)).decode().rstrip('=')
    
    @staticmethod
    def sanitize_input(input_str: str) -> str:
        """
        Sanitizar input del usuario
        
        Args:
            input_str: String a sanitizar
            
        Returns:
            String sanitizado
        """
        # Remover caracteres peligrosos
        dangerous_chars = ['<', '>', '"', "'", '&']
        for char in dangerous_chars:
            input_str = input_str.replace(char, '')
        
        # Limitar longitud
        max_length = 10000
        if len(input_str) > max_length:
            input_str = input_str[:max_length]
        
        return input_str.strip()
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """
        Generar token seguro aleatorio
        
        Args:
            length: Longitud del token en bytes
            
        Returns:
            Token generado (hex)
        """
        return os.urandom(length).hex()
    
    @staticmethod
    def verify_hmac_signature(
        data: str,
        signature: str,
        secret: str
    ) -> bool:
        """
        Verificar firma HMAC
        
        Args:
            data: Datos a verificar
            signature: Firma proporcionada
            secret: Secreto compartido
            
        Returns:
            True si la firma es válida
        """
        expected_signature = hmac.new(
            secret.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(expected_signature, signature)
    
    @staticmethod
    def generate_hmac_signature(data: str, secret: str) -> str:
        """
        Generar firma HMAC
        
        Args:
            data: Datos a firmar
            secret: Secreto compartido
            
        Returns:
            Firma HMAC
        """
        return hmac.new(
            secret.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
    
    @staticmethod
    def mask_sensitive_data(data: str, visible_chars: int = 4) -> str:
        """
        Enmascarar datos sensibles
        
        Args:
            data: Datos a enmascarar
            visible_chars: Caracteres visibles al inicio
            
        Returns:
            Datos enmascarados
        """
        if len(data) <= visible_chars:
            return "*" * len(data)
        
        return data[:visible_chars] + "*" * (len(data) - visible_chars)
    
    @staticmethod
    def validate_password_strength(password: str) -> Tuple[bool, List[str]]:
        """
        Validar fortaleza de contraseña
        
        Args:
            password: Contraseña a validar
            
        Returns:
            Tuple (is_strong, list_of_issues)
        """
        issues = []
        
        if len(password) < 8:
            issues.append("La contraseña debe tener al menos 8 caracteres")
        
        if not re.search(r'[A-Z]', password):
            issues.append("La contraseña debe contener al menos una mayúscula")
        
        if not re.search(r'[a-z]', password):
            issues.append("La contraseña debe contener al menos una minúscula")
        
        if not re.search(r'\d', password):
            issues.append("La contraseña debe contener al menos un número")
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            issues.append("La contraseña debe contener al menos un carácter especial")
        
        return len(issues) == 0, issues

