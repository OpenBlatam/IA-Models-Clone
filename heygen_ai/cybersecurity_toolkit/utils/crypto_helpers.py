from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import hashlib
import hmac
import base64
import secrets
import os
from typing import Dict, List, Optional, Any, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import logging
from typing import Any, List, Dict, Optional
import asyncio
"""
Crypto Helpers Module
====================

Cryptographic utility functions with guard clauses, proper async/def usage,
and comprehensive error handling.
"""


logger = logging.getLogger(__name__)

def validate_encryption_key(key: Union[str, bytes]) -> Dict[str, Any]:
    """
    Validate encryption key with guard clauses.
    
    Args:
        key: Encryption key to validate
        
    Returns:
        Validation result dictionary
    """
    # Guard clause: Check if key is provided
    if not key:
        return {
            "is_valid": False,
            "error": "Encryption key is required",
            "error_type": "MissingEncryptionKey"
        }
    
    # Guard clause: Check key type
    if not isinstance(key, (str, bytes)):
        return {
            "is_valid": False,
            "error": "Encryption key must be string or bytes",
            "error_type": "InvalidKeyType"
        }
    
    # Guard clause: Check key length
    key_length = len(key) if isinstance(key, bytes) else len(key.encode())
    if key_length < 16:
        return {
            "is_valid": False,
            "error": "Encryption key too short (minimum 16 bytes)",
            "error_type": "KeyTooShort"
        }
    
    if key_length > 64:
        return {
            "is_valid": False,
            "error": "Encryption key too long (maximum 64 bytes)",
            "error_type": "KeyTooLong"
        }
    
    return {
        "is_valid": True,
        "key_length": key_length,
        "message": "Encryption key is valid"
    }

def hash_password(password: str, algorithm: str = "sha256", salt: Optional[bytes] = None) -> Dict[str, Any]:
    """
    Hash password with salt (CPU-bound operation).
    
    Args:
        password: Password to hash
        algorithm: Hashing algorithm
        salt: Optional salt bytes
        
    Returns:
        Hashing result dictionary
    """
    # Guard clause: Check if password is provided
    if not password:
        return {
            "success": False,
            "error": "Password is required",
            "error_type": "MissingPassword"
        }
    
    # Guard clause: Check password type
    if not isinstance(password, str):
        return {
            "success": False,
            "error": "Password must be a string",
            "error_type": "InvalidPasswordType"
        }
    
    # Guard clause: Check password length
    if len(password) < 1:
        return {
            "success": False,
            "error": "Password cannot be empty",
            "error_type": "EmptyPassword"
        }
    
    if len(password) > 1000:
        return {
            "success": False,
            "error": "Password too long (maximum 1000 characters)",
            "error_type": "PasswordTooLong"
        }
    
    # Guard clause: Validate algorithm
    valid_algorithms = ["md5", "sha1", "sha256", "sha512", "blake2b"]
    if algorithm not in valid_algorithms:
        return {
            "success": False,
            "error": f"Invalid algorithm. Must be one of: {valid_algorithms}",
            "error_type": "InvalidAlgorithm"
        }
    
    try:
        # Generate salt if not provided
        if salt is None:
            salt = secrets.token_bytes(16)
        
        # Guard clause: Validate salt
        if not isinstance(salt, bytes) or len(salt) < 8:
            return {
                "success": False,
                "error": "Salt must be at least 8 bytes",
                "error_type": "InvalidSalt"
            }
        
        # Hash password with salt
        password_bytes = password.encode('utf-8')
        
        if algorithm == "md5":
            hash_obj = hashlib.md5()
        elif algorithm == "sha1":
            hash_obj = hashlib.sha1()
        elif algorithm == "sha256":
            hash_obj = hashlib.sha256()
        elif algorithm == "sha512":
            hash_obj = hashlib.sha512()
        elif algorithm == "blake2b":
            hash_obj = hashlib.blake2b()
        else:
            return {
                "success": False,
                "error": f"Unsupported algorithm: {algorithm}",
                "error_type": "UnsupportedAlgorithm"
            }
        
        # Update hash with salt and password
        hash_obj.update(salt + password_bytes)
        hashed_password = hash_obj.hexdigest()
        
        return {
            "success": True,
            "hashed_password": hashed_password,
            "salt": base64.b64encode(salt).decode('utf-8'),
            "algorithm": algorithm,
            "hash_length": len(hashed_password)
        }
        
    except Exception as e:
        logger.error(f"Password hashing failed: {str(e)}")
        return {
            "success": False,
            "error": f"Password hashing failed: {str(e)}",
            "error_type": "HashingError"
        }

def encrypt_data(data: Union[str, bytes], key: Union[str, bytes], algorithm: str = "fernet") -> Dict[str, Any]:
    """
    Encrypt data with specified algorithm (CPU-bound operation).
    
    Args:
        data: Data to encrypt
        key: Encryption key
        algorithm: Encryption algorithm
        
    Returns:
        Encryption result dictionary
    """
    # Guard clause: Check if data is provided
    if not data:
        return {
            "success": False,
            "error": "Data is required",
            "error_type": "MissingData"
        }
    
    # Guard clause: Check data type
    if not isinstance(data, (str, bytes)):
        return {
            "success": False,
            "error": "Data must be string or bytes",
            "error_type": "InvalidDataType"
        }
    
    # Guard clause: Validate encryption key
    key_validation = validate_encryption_key(key)
    if not key_validation["is_valid"]:
        return {
            "success": False,
            "error": key_validation["error"],
            "error_type": key_validation["error_type"]
        }
    
    # Guard clause: Validate algorithm
    valid_algorithms = ["fernet", "aes"]
    if algorithm not in valid_algorithms:
        return {
            "success": False,
            "error": f"Invalid algorithm. Must be one of: {valid_algorithms}",
            "error_type": "InvalidAlgorithm"
        }
    
    try:
        # Convert data to bytes if needed
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
        
        # Convert key to bytes if needed
        if isinstance(key, str):
            key_bytes = key.encode('utf-8')
        else:
            key_bytes = key
        
        if algorithm == "fernet":
            # Generate Fernet key from password
            salt = os.urandom(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            fernet_key = base64.urlsafe_b64encode(kdf.derive(key_bytes))
            
            # Encrypt with Fernet
            fernet = Fernet(fernet_key)
            encrypted_data = fernet.encrypt(data_bytes)
            
            return {
                "success": True,
                "encrypted_data": base64.b64encode(encrypted_data).decode('utf-8'),
                "salt": base64.b64encode(salt).decode('utf-8'),
                "algorithm": algorithm
            }
        
        elif algorithm == "aes":
            # Generate AES key and IV
            salt = os.urandom(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            aes_key = kdf.derive(key_bytes)
            iv = os.urandom(16)
            
            # Encrypt with AES
            cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            
            # Pad data to block size
            block_size = 16
            padding_length = block_size - (len(data_bytes) % block_size)
            padded_data = data_bytes + bytes([padding_length] * padding_length)
            
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            
            return {
                "success": True,
                "encrypted_data": base64.b64encode(encrypted_data).decode('utf-8'),
                "iv": base64.b64encode(iv).decode('utf-8'),
                "salt": base64.b64encode(salt).decode('utf-8'),
                "algorithm": algorithm
            }
        
    except Exception as e:
        logger.error(f"Data encryption failed: {str(e)}")
        return {
            "success": False,
            "error": f"Data encryption failed: {str(e)}",
            "error_type": "EncryptionError"
        }

def decrypt_data(encrypted_data: str, key: Union[str, bytes], salt: str, 
                algorithm: str = "fernet", iv: Optional[str] = None) -> Dict[str, Any]:
    """
    Decrypt data with specified algorithm (CPU-bound operation).
    
    Args:
        encrypted_data: Base64 encoded encrypted data
        key: Encryption key
        salt: Base64 encoded salt
        algorithm: Encryption algorithm
        iv: Base64 encoded IV (for AES)
        
    Returns:
        Decryption result dictionary
    """
    # Guard clause: Check if encrypted data is provided
    if not encrypted_data:
        return {
            "success": False,
            "error": "Encrypted data is required",
            "error_type": "MissingEncryptedData"
        }
    
    # Guard clause: Check if salt is provided
    if not salt:
        return {
            "success": False,
            "error": "Salt is required",
            "error_type": "MissingSalt"
        }
    
    # Guard clause: Validate encryption key
    key_validation = validate_encryption_key(key)
    if not key_validation["is_valid"]:
        return {
            "success": False,
            "error": key_validation["error"],
            "error_type": key_validation["error_type"]
        }
    
    # Guard clause: Validate algorithm
    valid_algorithms = ["fernet", "aes"]
    if algorithm not in valid_algorithms:
        return {
            "success": False,
            "error": f"Invalid algorithm. Must be one of: {valid_algorithms}",
            "error_type": "InvalidAlgorithm"
        }
    
    try:
        # Convert key to bytes if needed
        if isinstance(key, str):
            key_bytes = key.encode('utf-8')
        else:
            key_bytes = key
        
        # Decode base64 data
        try:
            encrypted_bytes = base64.b64decode(encrypted_data)
            salt_bytes = base64.b64decode(salt)
        except Exception:
            return {
                "success": False,
                "error": "Invalid base64 encoding",
                "error_type": "InvalidBase64"
            }
        
        if algorithm == "fernet":
            # Generate Fernet key from password
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt_bytes,
                iterations=100000,
                backend=default_backend()
            )
            fernet_key = base64.urlsafe_b64encode(kdf.derive(key_bytes))
            
            # Decrypt with Fernet
            fernet = Fernet(fernet_key)
            decrypted_data = fernet.decrypt(encrypted_bytes)
            
            return {
                "success": True,
                "decrypted_data": decrypted_data.decode('utf-8'),
                "algorithm": algorithm
            }
        
        elif algorithm == "aes":
            # Guard clause: Check if IV is provided for AES
            if not iv:
                return {
                    "success": False,
                    "error": "IV is required for AES decryption",
                    "error_type": "MissingIV"
                }
            
            try:
                iv_bytes = base64.b64decode(iv)
            except Exception:
                return {
                    "success": False,
                    "error": "Invalid IV base64 encoding",
                    "error_type": "InvalidIV"
                }
            
            # Generate AES key
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt_bytes,
                iterations=100000,
                backend=default_backend()
            )
            aes_key = kdf.derive(key_bytes)
            
            # Decrypt with AES
            cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv_bytes), backend=default_backend())
            decryptor = cipher.decryptor()
            
            decrypted_padded = decryptor.update(encrypted_bytes) + decryptor.finalize()
            
            # Remove padding
            padding_length = decrypted_padded[-1]
            decrypted_data = decrypted_padded[:-padding_length]
            
            return {
                "success": True,
                "decrypted_data": decrypted_data.decode('utf-8'),
                "algorithm": algorithm
            }
        
    except Exception as e:
        logger.error(f"Data decryption failed: {str(e)}")
        return {
            "success": False,
            "error": f"Data decryption failed: {str(e)}",
            "error_type": "DecryptionError"
        }

def generate_secure_random_string(length: int = 32) -> Dict[str, Any]:
    """
    Generate secure random string (CPU-bound operation).
    
    Args:
        length: Length of random string
        
    Returns:
        Generation result dictionary
    """
    # Guard clause: Validate length
    if not isinstance(length, int):
        return {
            "success": False,
            "error": "Length must be an integer",
            "error_type": "InvalidLengthType"
        }
    
    if length <= 0:
        return {
            "success": False,
            "error": "Length must be positive",
            "error_type": "InvalidLength"
        }
    
    if length > 1000:
        return {
            "success": False,
            "error": "Length too large (maximum 1000)",
            "error_type": "LengthTooLarge"
        }
    
    try:
        # Generate secure random string
        random_string = secrets.token_urlsafe(length)
        
        return {
            "success": True,
            "random_string": random_string,
            "length": len(random_string)
        }
        
    except Exception as e:
        logger.error(f"Random string generation failed: {str(e)}")
        return {
            "success": False,
            "error": f"Random string generation failed: {str(e)}",
            "error_type": "GenerationError"
        }

def verify_password_hash(password: str, hashed_password: str, salt: str, 
                        algorithm: str = "sha256") -> Dict[str, Any]:
    """
    Verify password against hash (CPU-bound operation).
    
    Args:
        password: Password to verify
        hashed_password: Stored hash
        salt: Stored salt
        algorithm: Hashing algorithm
        
    Returns:
        Verification result dictionary
    """
    # Guard clause: Check if password is provided
    if not password:
        return {
            "success": False,
            "error": "Password is required",
            "error_type": "MissingPassword"
        }
    
    # Guard clause: Check if hash is provided
    if not hashed_password:
        return {
            "success": False,
            "error": "Hashed password is required",
            "error_type": "MissingHash"
        }
    
    # Guard clause: Check if salt is provided
    if not salt:
        return {
            "success": False,
            "error": "Salt is required",
            "error_type": "MissingSalt"
        }
    
    try:
        # Hash the provided password with the stored salt
        hash_result = hash_password(password, algorithm, base64.b64decode(salt))
        
        if not hash_result["success"]:
            return hash_result
        
        # Compare hashes
        is_valid = hmac.compare_digest(hash_result["hashed_password"], hashed_password)
        
        return {
            "success": True,
            "is_valid": is_valid,
            "algorithm": algorithm
        }
        
    except Exception as e:
        logger.error(f"Password verification failed: {str(e)}")
        return {
            "success": False,
            "error": f"Password verification failed: {str(e)}",
            "error_type": "VerificationError"
        }

# --- Named Exports ---

__all__ = [
    'validate_encryption_key',
    'hash_password',
    'encrypt_data',
    'decrypt_data',
    'generate_secure_random_string',
    'verify_password_hash'
] 