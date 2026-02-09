"""
Generation utilities
Data generation functions
"""

from typing import Optional, List
import random
import string
import uuid
import secrets
from datetime import datetime, timedelta


def generate_id(length: int = 8, prefix: Optional[str] = None) -> str:
    """
    Generate random ID
    
    Args:
        length: ID length
        prefix: Optional prefix
    
    Returns:
        Generated ID
    """
    chars = string.ascii_letters + string.digits
    id_str = ''.join(secrets.choice(chars) for _ in range(length))
    
    if prefix:
        return f"{prefix}_{id_str}"
    
    return id_str


def generate_uuid(version: int = 4) -> str:
    """
    Generate UUID
    
    Args:
        version: UUID version (1, 3, 4, 5)
    
    Returns:
        UUID string
    """
    if version == 1:
        return str(uuid.uuid1())
    if version == 4:
        return str(uuid.uuid4())
    
    return str(uuid.uuid4())


def generate_token(length: int = 32, url_safe: bool = True) -> str:
    """
    Generate secure random token
    
    Args:
        length: Token length
        url_safe: Use URL-safe characters
    
    Returns:
        Generated token
    """
    if url_safe:
        return secrets.token_urlsafe(length)
    
    return secrets.token_hex(length)


def generate_password(
    length: int = 12,
    include_uppercase: bool = True,
    include_lowercase: bool = True,
    include_digits: bool = True,
    include_special: bool = True
) -> str:
    """
    Generate random password
    
    Args:
        length: Password length
        include_uppercase: Include uppercase letters
        include_lowercase: Include lowercase letters
        include_digits: Include digits
        include_special: Include special characters
    
    Returns:
        Generated password
    """
    chars = ""
    
    if include_uppercase:
        chars += string.ascii_uppercase
    if include_lowercase:
        chars += string.ascii_lowercase
    if include_digits:
        chars += string.digits
    if include_special:
        chars += "!@#$%^&*()_+-=[]{}|;:,.<>?"
    
    if not chars:
        chars = string.ascii_letters + string.digits
    
    return ''.join(secrets.choice(chars) for _ in range(length))


def generate_email(domain: str = "example.com") -> str:
    """
    Generate random email
    
    Args:
        domain: Email domain
    
    Returns:
        Generated email
    """
    username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"{username}@{domain}"


def generate_phone(country_code: str = "+1") -> str:
    """
    Generate random phone number
    
    Args:
        country_code: Country code
    
    Returns:
        Generated phone number
    """
    area_code = ''.join(random.choices(string.digits, k=3))
    exchange = ''.join(random.choices(string.digits, k=3))
    number = ''.join(random.choices(string.digits, k=4))
    
    return f"{country_code}{area_code}{exchange}{number}"


def generate_string(
    length: int = 10,
    chars: Optional[str] = None
) -> str:
    """
    Generate random string
    
    Args:
        length: String length
        chars: Character set (default: letters + digits)
    
    Returns:
        Generated string
    """
    if chars is None:
        chars = string.ascii_letters + string.digits
    
    return ''.join(secrets.choice(chars) for _ in range(length))


def generate_number(min_val: int = 0, max_val: int = 100) -> int:
    """
    Generate random number
    
    Args:
        min_val: Minimum value
        max_val: Maximum value
    
    Returns:
        Generated number
    """
    return secrets.randbelow(max_val - min_val + 1) + min_val


def generate_float(min_val: float = 0.0, max_val: float = 1.0, decimals: int = 2) -> float:
    """
    Generate random float
    
    Args:
        min_val: Minimum value
        max_val: Maximum value
        decimals: Number of decimal places
    
    Returns:
        Generated float
    """
    value = random.uniform(min_val, max_val)
    return round(value, decimals)


def generate_date(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> datetime:
    """
    Generate random date
    
    Args:
        start_date: Start date (default: 1 year ago)
        end_date: End date (default: now)
    
    Returns:
        Generated date
    """
    if end_date is None:
        end_date = datetime.now()
    
    if start_date is None:
        start_date = end_date - timedelta(days=365)
    
    time_between = end_date - start_date
    days_between = time_between.days
    random_days = random.randrange(days_between)
    
    return start_date + timedelta(days=random_days)


def generate_list(
    generator: callable,
    count: int = 5,
    *args,
    **kwargs
) -> List:
    """
    Generate list using generator function
    
    Args:
        generator: Generator function
        count: Number of items
        *args: Positional arguments for generator
        **kwargs: Keyword arguments for generator
    
    Returns:
        Generated list
    """
    return [generator(*args, **kwargs) for _ in range(count)]


def generate_sequence(start: int = 0, step: int = 1, count: int = 10) -> List[int]:
    """
    Generate sequence of numbers
    
    Args:
        start: Starting number
        step: Step size
        count: Number of items
    
    Returns:
        Generated sequence
    """
    return [start + (step * i) for i in range(count)]


def generate_choices(items: List, count: int = 1, allow_duplicates: bool = True) -> List:
    """
    Generate random choices from list
    
    Args:
        items: List of items
        count: Number of choices
        allow_duplicates: Allow duplicate choices
    
    Returns:
        Generated choices
    """
    if allow_duplicates:
        return random.choices(items, k=count)
    
    return random.sample(items, min(count, len(items)))

