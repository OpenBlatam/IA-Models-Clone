"""
Encryption Utilities - Utilidades de Encriptación
=================================================

Utilidades para encriptación, hashing y manejo seguro de datos.
"""

import hashlib
import secrets
import base64
import logging
from typing import Optional, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

# Intentar importar cryptography
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    _has_cryptography = True
except ImportError:
    _has_cryptography = False
    logger.warning("cryptography not available. Encryption features will be limited.")


def hash_string(data: str, algorithm: str = "sha256") -> str:
    """
    Hash de string usando algoritmo especificado.
    
    Args:
        data: String a hashear
        algorithm: Algoritmo (sha256, sha512, md5)
        
    Returns:
        Hash hexadecimal
    """
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(data.encode('utf-8'))
    return hash_obj.hexdigest()


def hash_file(filepath: str, algorithm: str = "sha256") -> str:
    """
    Hash de archivo usando algoritmo especificado.
    
    Args:
        filepath: Ruta del archivo
        algorithm: Algoritmo (sha256, sha512, md5)
        
    Returns:
        Hash hexadecimal
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def generate_salt(length: int = 32) -> str:
    """
    Generar salt aleatorio.
    
    Args:
        length: Longitud del salt en bytes
        
    Returns:
        Salt hexadecimal
    """
    return secrets.token_hex(length)


def hash_with_salt(data: str, salt: Optional[str] = None) -> tuple[str, str]:
    """
    Hash de datos con salt.
    
    Args:
        data: Datos a hashear
        salt: Salt opcional (se genera si no se proporciona)
        
    Returns:
        Tupla (hash, salt)
    """
    if salt is None:
        salt = generate_salt()
    
    hash_obj = hashlib.sha256()
    hash_obj.update(f"{data}{salt}".encode('utf-8'))
    return hash_obj.hexdigest(), salt


def verify_hash(data: str, hash_value: str, salt: str) -> bool:
    """
    Verificar hash contra datos.
    
    Args:
        data: Datos originales
        hash_value: Hash a verificar
        salt: Salt usado
        
    Returns:
        True si coincide
    """
    computed_hash, _ = hash_with_salt(data, salt)
    return computed_hash == hash_value


class Encryptor:
    """
    Encriptador usando Fernet (symmetric encryption).
    
    Requiere cryptography library.
    """
    
    def __init__(self, key: Optional[bytes] = None, password: Optional[str] = None):
        """
        Inicializar encriptador.
        
        Args:
            key: Clave Fernet (32 bytes base64)
            password: Contraseña para derivar clave
        """
        if not _has_cryptography:
            raise ImportError("cryptography is required for encryption")
        
        if key:
            self.key = key
        elif password:
            # Derivar clave desde contraseña
            salt = b'cursor_agent_salt'  # En producción, usar salt aleatorio
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key_material = kdf.derive(password.encode())
            self.key = base64.urlsafe_b64encode(key_material)
        else:
            # Generar nueva clave
            self.key = Fernet.generate_key()
        
        self.cipher = Fernet(self.key)
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """
        Encriptar datos.
        
        Args:
            data: Datos a encriptar
            
        Returns:
            String encriptado (base64)
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        encrypted = self.cipher.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode('utf-8')
    
    def decrypt(self, encrypted_data: str) -> str:
        """
        Desencriptar datos.
        
        Args:
            encrypted_data: Datos encriptados (base64)
            
        Returns:
            String desencriptado
        """
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
        decrypted = self.cipher.decrypt(encrypted_bytes)
        return decrypted.decode('utf-8')
    
    def get_key(self) -> str:
        """Obtener clave como string (para almacenamiento)"""
        return self.key.decode('utf-8')


def generate_api_key(length: int = 32) -> str:
    """
    Generar API key aleatoria.
    
    Args:
        length: Longitud en bytes
        
    Returns:
        API key (base64 URL-safe)
    """
    return secrets.token_urlsafe(length)


def generate_secure_token(length: int = 32) -> str:
    """
    Generar token seguro aleatorio.
    
    Args:
        length: Longitud en bytes
        
    Returns:
        Token (hexadecimal)
    """
    return secrets.token_hex(length)




