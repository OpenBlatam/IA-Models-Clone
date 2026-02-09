"""Random utilities."""

import random
import secrets
import string
from typing import List, Optional


def random_string(length: int = 8, charset: str = string.ascii_letters + string.digits) -> str:
    """
    Generate random string.
    
    Args:
        length: String length
        charset: Character set to use
        
    Returns:
        Random string
    """
    return ''.join(secrets.choice(charset) for _ in range(length))


def random_int(min_value: int = 0, max_value: int = 100) -> int:
    """
    Generate random integer.
    
    Args:
        min_value: Minimum value
        max_value: Maximum value
        
    Returns:
        Random integer
    """
    return random.randint(min_value, max_value)


def random_float(min_value: float = 0.0, max_value: float = 1.0) -> float:
    """
    Generate random float.
    
    Args:
        min_value: Minimum value
        max_value: Maximum value
        
    Returns:
        Random float
    """
    return random.uniform(min_value, max_value)


def random_choice(items: List) -> any:
    """
    Get random choice from list.
    
    Args:
        items: List of items
        
    Returns:
        Random item
    """
    return random.choice(items)


def random_choices(items: List, k: int = 1) -> List:
    """
    Get random choices from list (with replacement).
    
    Args:
        items: List of items
        k: Number of choices
        
    Returns:
        List of random items
    """
    return random.choices(items, k=k)


def random_sample(items: List, k: int) -> List:
    """
    Get random sample from list (without replacement).
    
    Args:
        items: List of items
        k: Sample size
        
    Returns:
        List of random items
    """
    return random.sample(items, k)


def shuffle_list(items: List) -> List:
    """
    Shuffle list (returns new list).
    
    Args:
        items: List to shuffle
        
    Returns:
        Shuffled list
    """
    shuffled = items.copy()
    random.shuffle(shuffled)
    return shuffled


def random_uuid() -> str:
    """
    Generate random UUID string.
    
    Returns:
        UUID string
    """
    import uuid
    return str(uuid.uuid4())


def random_hex(length: int = 32) -> str:
    """
    Generate random hexadecimal string.
    
    Args:
        length: String length
        
    Returns:
        Random hex string
    """
    return secrets.token_hex(length // 2)


def random_bytes(length: int = 16) -> bytes:
    """
    Generate random bytes.
    
    Args:
        length: Number of bytes
        
    Returns:
        Random bytes
    """
    return secrets.token_bytes(length)

