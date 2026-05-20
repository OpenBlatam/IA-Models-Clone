"""
Encryption Utilities
====================
Utilidades para encriptación (básicas, para datos no sensibles).
"""

import hashlib
import hmac
import base64
from typing import Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


def hash_sha256(data: str) -> str:
    """
    Hashear datos usando SHA-256.
    
    Args:
        data: Datos a hashear
        
    Returns:
        Hash hexadecimal
    """
    return hashlib.sha256(data.encode()).hexdigest()


def hash_sha512(data: str) -> str:
    """
    Hashear datos usando SHA-512.
    
    Args:
        data: Datos a hashear
        
    Returns:
        Hash hexadecimal
    """
    return hashlib.sha512(data.encode()).hexdigest()


def hmac_sign(data: str, secret: str, algorithm: str = "sha256") -> str:
    """
    Firmar datos usando HMAC.
    
    Args:
        data: Datos a firmar
        secret: Secreto para firma
        algorithm: Algoritmo (sha256, sha512)
        
    Returns:
        Firma hexadecimal
    """
    hash_func = hashlib.sha256 if algorithm == "sha256" else hashlib.sha512
    signature = hmac.new(
        secret.encode(),
        data.encode(),
        hash_func
    ).hexdigest()
    return signature


def verify_hmac(data: str, signature: str, secret: str, algorithm: str = "sha256") -> bool:
    """
    Verificar firma HMAC.
    
    Args:
        data: Datos originales
        signature: Firma a verificar
        secret: Secreto
        algorithm: Algoritmo
        
    Returns:
        True si la firma es válida
    """
    expected_signature = hmac_sign(data, secret, algorithm)
    return hmac.compare_digest(expected_signature, signature)


def generate_key_from_password(password: str, salt: Optional[bytes] = None) -> bytes:
    """
    Generar clave desde contraseña usando PBKDF2.
    
    Args:
        password: Contraseña
        salt: Salt (opcional, se genera si no se proporciona)
        
    Returns:
        Clave generada
    """
    if salt is None:
        salt = b'salt_12345678'  # En producción, usar salt aleatorio
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key


def encrypt_fernet(data: str, key: Optional[bytes] = None) -> str:
    """
    Encriptar datos usando Fernet.
    
    Args:
        data: Datos a encriptar
        key: Clave (opcional, se genera si no se proporciona)
        
    Returns:
        Datos encriptados en base64
    """
    if key is None:
        key = Fernet.generate_key()
    
    fernet = Fernet(key)
    encrypted = fernet.encrypt(data.encode())
    return base64.urlsafe_b64encode(encrypted).decode()


def decrypt_fernet(encrypted_data: str, key: bytes) -> str:
    """
    Desencriptar datos usando Fernet.
    
    Args:
        encrypted_data: Datos encriptados en base64
        key: Clave
        
    Returns:
        Datos desencriptados
    """
    fernet = Fernet(key)
    encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
    decrypted = fernet.decrypt(encrypted_bytes)
    return decrypted.decode()

