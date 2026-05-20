"""
Hashing utilities
Data hashing functions
"""

from typing import Optional
import hashlib
import hmac
import secrets


def hash_md5(data: str) -> str:
    """
    Hash data using MD5
    
    Args:
        data: Data to hash
    
    Returns:
        MD5 hash
    """
    return hashlib.md5(data.encode()).hexdigest()


def hash_sha1(data: str) -> str:
    """
    Hash data using SHA1
    
    Args:
        data: Data to hash
    
    Returns:
        SHA1 hash
    """
    return hashlib.sha1(data.encode()).hexdigest()


def hash_sha256(data: str) -> str:
    """
    Hash data using SHA256
    
    Args:
        data: Data to hash
    
    Returns:
        SHA256 hash
    """
    return hashlib.sha256(data.encode()).hexdigest()


def hash_sha512(data: str) -> str:
    """
    Hash data using SHA512
    
    Args:
        data: Data to hash
    
    Returns:
        SHA512 hash
    """
    return hashlib.sha512(data.encode()).hexdigest()


def hash_blake2b(data: str, digest_size: int = 64) -> str:
    """
    Hash data using BLAKE2b
    
    Args:
        data: Data to hash
        digest_size: Digest size in bytes
    
    Returns:
        BLAKE2b hash
    """
    return hashlib.blake2b(data.encode(), digest_size=digest_size).hexdigest()


def hash_file(file_path: str, algorithm: str = "sha256") -> str:
    """
    Hash file
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm
    
    Returns:
        File hash
    """
    try:
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    except Exception:
        return ""


def hmac_hash(data: str, key: str, algorithm: str = "sha256") -> str:
    """
    Generate HMAC hash
    
    Args:
        data: Data to hash
        key: Secret key
        algorithm: Hash algorithm
    
    Returns:
        HMAC hash
    """
    return hmac.new(
        key.encode(),
        data.encode(),
        algorithm
    ).hexdigest()


def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """
    Hash password with salt
    
    Args:
        password: Password to hash
        salt: Optional salt (generated if not provided)
    
    Returns:
        Tuple of (hash, salt)
    """
    if salt is None:
        salt = secrets.token_hex(16)
    
    hash_obj = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode(),
        salt.encode(),
        100000  # iterations
    )
    
    return hash_obj.hex(), salt


def verify_password(password: str, hash_value: str, salt: str) -> bool:
    """
    Verify password against hash
    
    Args:
        password: Password to verify
        hash_value: Stored hash
        salt: Stored salt
    
    Returns:
        True if password matches
    """
    new_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(new_hash, hash_value)


def generate_salt(length: int = 32) -> str:
    """
    Generate random salt
    
    Args:
        length: Salt length
    
    Returns:
        Generated salt
    """
    return secrets.token_hex(length)


def checksum(data: str) -> int:
    """
    Calculate simple checksum
    
    Args:
        data: Data to checksum
    
    Returns:
        Checksum value
    """
    return sum(ord(c) for c in data) % 256


def hash_multiple(data: str, algorithms: list[str]) -> dict[str, str]:
    """
    Hash data with multiple algorithms
    
    Args:
        data: Data to hash
        algorithms: List of algorithms
    
    Returns:
        Dictionary of algorithm -> hash
    """
    result = {}
    
    for algo in algorithms:
        if algo == "md5":
            result[algo] = hash_md5(data)
        elif algo == "sha1":
            result[algo] = hash_sha1(data)
        elif algo == "sha256":
            result[algo] = hash_sha256(data)
        elif algo == "sha512":
            result[algo] = hash_sha512(data)
        elif algo == "blake2b":
            result[algo] = hash_blake2b(data)
    
    return result

