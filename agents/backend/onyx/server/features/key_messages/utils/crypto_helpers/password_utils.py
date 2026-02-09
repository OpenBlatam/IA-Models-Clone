from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from typing import Optional
from pydantic import BaseModel, field_validator
import structlog
import bcrypt
import hashlib
import secrets
import base64
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.backends import default_backend
        import string
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Password utilities for cybersecurity tools.
"""

logger = structlog.get_logger(__name__)

class PasswordHashInput(BaseModel):
    """Input model for password hashing."""
    password: str
    algorithm: str = "bcrypt"
    salt_rounds: int = 12
    encoding: str = "utf-8"
    
    @field_validator('password')
    def validate_password(cls, v) -> bool:
        if not v:
            raise ValueError("Password cannot be empty")
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v
    
    @field_validator('algorithm')
    def validate_algorithm(cls, v) -> bool:
        valid_algorithms = ["bcrypt", "sha256", "sha512", "pbkdf2"]
        if v not in valid_algorithms:
            raise ValueError(f"Algorithm must be one of: {valid_algorithms}")
        return v
    
    @field_validator('salt_rounds')
    def validate_salt_rounds(cls, v) -> bool:
        if v < 4 or v > 31:
            raise ValueError("Salt rounds must be between 4 and 31")
        return v

class PasswordVerifyInput(BaseModel):
    """Input model for password verification."""
    password: str
    hashed_password: str
    algorithm: str = "bcrypt"
    encoding: str = "utf-8"
    
    @field_validator('password')
    def validate_password(cls, v) -> bool:
        if not v:
            raise ValueError("Password cannot be empty")
        return v
    
    @field_validator('hashed_password')
    def validate_hashed_password(cls, v) -> bool:
        if not v:
            raise ValueError("Hashed password cannot be empty")
        return v

class PasswordHashResult(BaseModel):
    """Result model for password hashing."""
    hashed_password: str
    salt: Optional[str] = None
    algorithm: str
    salt_rounds: Optional[int] = None
    is_successful: bool
    error_message: Optional[str] = None

class PasswordVerifyResult(BaseModel):
    """Result model for password verification."""
    is_valid: bool
    algorithm: str
    is_successful: bool
    error_message: Optional[str] = None

def hash_password(input_data: PasswordHashInput) -> PasswordHashResult:
    """
    RORO: Receive PasswordHashInput, return PasswordHashResult
    
    Hash a password using the specified algorithm.
    """
    try:
        if input_data.algorithm == "bcrypt":
            return hash_with_bcrypt(input_data)
        elif input_data.algorithm == "sha256":
            return hash_with_sha256(input_data)
        elif input_data.algorithm == "sha512":
            return hash_with_sha512(input_data)
        elif input_data.algorithm == "pbkdf2":
            return hash_with_pbkdf2(input_data)
        else:
            raise ValueError(f"Unsupported algorithm: {input_data.algorithm}")
            
    except Exception as e:
        logger.error("Password hashing failed", error=str(e))
        return PasswordHashResult(
            hashed_password="",
            algorithm=input_data.algorithm,
            is_successful=False,
            error_message=str(e)
        )

def verify_password(input_data: PasswordVerifyInput) -> PasswordVerifyResult:
    """
    RORO: Receive PasswordVerifyInput, return PasswordVerifyResult
    
    Verify a password against its hash.
    """
    try:
        if input_data.algorithm == "bcrypt":
            return verify_with_bcrypt(input_data)
        elif input_data.algorithm == "sha256":
            return verify_with_sha256(input_data)
        elif input_data.algorithm == "sha512":
            return verify_with_sha512(input_data)
        elif input_data.algorithm == "pbkdf2":
            return verify_with_pbkdf2(input_data)
        else:
            raise ValueError(f"Unsupported algorithm: {input_data.algorithm}")
            
    except Exception as e:
        logger.error("Password verification failed", error=str(e))
        return PasswordVerifyResult(
            is_valid=False,
            algorithm=input_data.algorithm,
            is_successful=False,
            error_message=str(e)
        )

def hash_with_bcrypt(input_data: PasswordHashInput) -> PasswordHashResult:
    """Hash password using bcrypt."""
    try:
        password_bytes = input_data.password.encode(input_data.encoding)
        salt = bcrypt.gensalt(rounds=input_data.salt_rounds)
        hashed = bcrypt.hashpw(password_bytes, salt)
        
        return PasswordHashResult(
            hashed_password=hashed.decode(input_data.encoding),
            salt=salt.decode(input_data.encoding),
            algorithm="bcrypt",
            salt_rounds=input_data.salt_rounds,
            is_successful=True
        )
        
    except Exception as e:
        logger.error("Bcrypt hashing failed", error=str(e))
        raise

def verify_with_bcrypt(input_data: PasswordVerifyInput) -> PasswordVerifyResult:
    """Verify password using bcrypt."""
    try:
        password_bytes = input_data.password.encode(input_data.encoding)
        hashed_bytes = input_data.hashed_password.encode(input_data.encoding)
        
        is_valid = bcrypt.checkpw(password_bytes, hashed_bytes)
        
        return PasswordVerifyResult(
            is_valid=is_valid,
            algorithm="bcrypt",
            is_successful=True
        )
        
    except Exception as e:
        logger.error("Bcrypt verification failed", error=str(e))
        raise

def hash_with_sha256(input_data: PasswordHashInput) -> PasswordHashResult:
    """Hash password using SHA-256 with salt."""
    try:
        # Generate salt
        salt = secrets.token_hex(16)
        
        # Combine password and salt
        password_salt = input_data.password + salt
        password_bytes = password_salt.encode(input_data.encoding)
        
        # Hash
        hash_obj = hashlib.sha256()
        hash_obj.update(password_bytes)
        hashed = hash_obj.hexdigest()
        
        return PasswordHashResult(
            hashed_password=hashed,
            salt=salt,
            algorithm="sha256",
            is_successful=True
        )
        
    except Exception as e:
        logger.error("SHA-256 hashing failed", error=str(e))
        raise

def verify_with_sha256(input_data: PasswordVerifyInput) -> PasswordVerifyResult:
    """Verify password using SHA-256."""
    try:
        # Extract salt from hashed password (assuming format: hash:salt)
        if ':' in input_data.hashed_password:
            stored_hash, salt = input_data.hashed_password.split(':', 1)
        else:
            # If no salt in stored hash, assume it's just the hash
            stored_hash = input_data.hashed_password
            salt = ""
        
        # Hash the provided password with the same salt
        password_salt = input_data.password + salt
        password_bytes = password_salt.encode(input_data.encoding)
        
        hash_obj = hashlib.sha256()
        hash_obj.update(password_bytes)
        computed_hash = hash_obj.hexdigest()
        
        is_valid = secrets.compare_digest(stored_hash, computed_hash)
        
        return PasswordVerifyResult(
            is_valid=is_valid,
            algorithm="sha256",
            is_successful=True
        )
        
    except Exception as e:
        logger.error("SHA-256 verification failed", error=str(e))
        raise

def hash_with_sha512(input_data: PasswordHashInput) -> PasswordHashResult:
    """Hash password using SHA-512 with salt."""
    try:
        # Generate salt
        salt = secrets.token_hex(16)
        
        # Combine password and salt
        password_salt = input_data.password + salt
        password_bytes = password_salt.encode(input_data.encoding)
        
        # Hash
        hash_obj = hashlib.sha512()
        hash_obj.update(password_bytes)
        hashed = hash_obj.hexdigest()
        
        return PasswordHashResult(
            hashed_password=hashed,
            salt=salt,
            algorithm="sha512",
            is_successful=True
        )
        
    except Exception as e:
        logger.error("SHA-512 hashing failed", error=str(e))
        raise

def verify_with_sha512(input_data: PasswordVerifyInput) -> PasswordVerifyResult:
    """Verify password using SHA-512."""
    try:
        # Extract salt from hashed password (assuming format: hash:salt)
        if ':' in input_data.hashed_password:
            stored_hash, salt = input_data.hashed_password.split(':', 1)
        else:
            # If no salt in stored hash, assume it's just the hash
            stored_hash = input_data.hashed_password
            salt = ""
        
        # Hash the provided password with the same salt
        password_salt = input_data.password + salt
        password_bytes = password_salt.encode(input_data.encoding)
        
        hash_obj = hashlib.sha512()
        hash_obj.update(password_bytes)
        computed_hash = hash_obj.hexdigest()
        
        is_valid = secrets.compare_digest(stored_hash, computed_hash)
        
        return PasswordVerifyResult(
            is_valid=is_valid,
            algorithm="sha512",
            is_successful=True
        )
        
    except Exception as e:
        logger.error("SHA-512 verification failed", error=str(e))
        raise

def hash_with_pbkdf2(input_data: PasswordHashInput) -> PasswordHashResult:
    """Hash password using PBKDF2."""
    try:
        
        # Generate salt
        salt = secrets.token_bytes(16)
        
        # Create PBKDF2 instance
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        # Hash password
        password_bytes = input_data.password.encode(input_data.encoding)
        hashed = kdf.derive(password_bytes)
        
        # Encode as base64
        hashed_b64 = base64.b64encode(hashed).decode(input_data.encoding)
        salt_b64 = base64.b64encode(salt).decode(input_data.encoding)
        
        return PasswordHashResult(
            hashed_password=hashed_b64,
            salt=salt_b64,
            algorithm="pbkdf2",
            salt_rounds=100000,
            is_successful=True
        )
        
    except Exception as e:
        logger.error("PBKDF2 hashing failed", error=str(e))
        raise

def verify_with_pbkdf2(input_data: PasswordVerifyInput) -> PasswordVerifyResult:
    """Verify password using PBKDF2."""
    try:
        
        # Extract salt from hashed password (assuming format: hash:salt)
        if ':' in input_data.hashed_password:
            stored_hash_b64, salt_b64 = input_data.hashed_password.split(':', 1)
        else:
            # If no salt in stored hash, assume it's just the hash
            stored_hash_b64 = input_data.hashed_password
            salt_b64 = ""
        
        # Decode from base64
        stored_hash = base64.b64decode(stored_hash_b64.encode(input_data.encoding))
        salt = base64.b64decode(salt_b64.encode(input_data.encoding)) if salt_b64 else b""
        
        # Create PBKDF2 instance
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        # Hash the provided password
        password_bytes = input_data.password.encode(input_data.encoding)
        computed_hash = kdf.derive(password_bytes)
        
        # Compare hashes
        is_valid = secrets.compare_digest(stored_hash, computed_hash)
        
        return PasswordVerifyResult(
            is_valid=is_valid,
            algorithm="pbkdf2",
            is_successful=True
        )
        
    except Exception as e:
        logger.error("PBKDF2 verification failed", error=str(e))
        raise

def generate_secure_password(length: int = 16, include_symbols: bool = True) -> str:
    """
    Generate a secure random password.
    
    Args:
        length: Length of the password
        include_symbols: Whether to include special symbols
        
    Returns:
        Generated password
    """
    try:
        
        # Define character sets
        lowercase = string.ascii_lowercase
        uppercase = string.ascii_uppercase
        digits = string.digits
        symbols = "!@#$%^&*()_+-=[]{}|;:,.<>?" if include_symbols else ""
        
        # Ensure at least one character from each set
        password_chars = [
            secrets.choice(lowercase),
            secrets.choice(uppercase),
            secrets.choice(digits)
        ]
        
        if include_symbols:
            password_chars.append(secrets.choice(symbols))
        
        # Fill the rest randomly
        all_chars = lowercase + uppercase + digits + symbols
        remaining_length = length - len(password_chars)
        
        for _ in range(remaining_length):
            password_chars.append(secrets.choice(all_chars))
        
        # Shuffle the password
        password_list = list(password_chars)
        secrets.SystemRandom().shuffle(password_list)
        
        return ''.join(password_list)
        
    except Exception as e:
        logger.error("Password generation failed", error=str(e))
        raise 