"""
Hashing Utilities
================
Optimized hashing functions for cache keys and data integrity.

Uses MD5 for fast, consistent hashing. Suitable for cache keys and
non-cryptographic use cases. For security-sensitive operations, use
proper cryptographic hashing (SHA-256, etc.).
"""

import hashlib


def hash_string(value: str, length: int = 32) -> str:
    """
    Hash a string using MD5 (optimized).
    
    Args:
        value: String to hash
        length: Length of hash to return (default 32 for full MD5)
        
    Returns:
        Hexadecimal hash string
    """
    if not value:
        return ""
    
    # Use MD5 for speed (cryptographic security not needed for cache keys)
    return hashlib.md5(value.encode()).hexdigest()[:length]


def hash_data(*args, **kwargs) -> str:
    """
    Hash multiple arguments into a single hash (optimized).
    
    Args:
        *args: Positional arguments to hash
        **kwargs: Keyword arguments to hash
        
    Returns:
        MD5 hash string (32 characters)
    """
    # Fast path: single string argument
    if len(args) == 1 and not kwargs and isinstance(args[0], str):
        return hash_string(args[0])
    
    # Fast path: single non-string argument
    if len(args) == 1 and not kwargs:
        return hash_string(str(args[0]))
    
    # Build key parts efficiently
    key_parts = []
    for arg in args:
        if isinstance(arg, str):
            key_parts.append(arg)
        elif isinstance(arg, (int, float, bool)):
            key_parts.append(str(arg))
        elif isinstance(arg, (list, tuple)):
            # Optimize: use join directly for lists/tuples
            key_parts.append(','.join(str(x) for x in arg))
        elif arg is None:
            key_parts.append('None')
        else:
            # Fallback for other types
            key_parts.append(str(arg))
    
    if kwargs:
        # Sort for consistent keys (optimized: use list comprehension)
        key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
    
    key_data = '|'.join(key_parts)
    return hash_string(key_data)

