"""
ID Generator - Generador de IDs Únicos
======================================

Sistema para generar IDs únicos de diferentes tipos.
"""

import logging
import uuid
import time
import random
import string
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_uuid() -> str:
    """
    Generar UUID v4.
    
    Returns:
        UUID como string
    """
    return str(uuid.uuid4())


def generate_short_uuid(length: int = 8) -> str:
    """
    Generar UUID corto.
    
    Args:
        length: Longitud del UUID
        
    Returns:
        UUID corto
    """
    return uuid.uuid4().hex[:length]


def generate_timestamp_id(prefix: str = "") -> str:
    """
    Generar ID basado en timestamp.
    
    Args:
        prefix: Prefijo opcional
        
    Returns:
        ID con timestamp
    """
    timestamp = int(time.time() * 1000)  # milisegundos
    random_part = random.randint(1000, 9999)
    return f"{prefix}{timestamp}{random_part}" if prefix else f"{timestamp}{random_part}"


def generate_nanoid(size: int = 21) -> str:
    """
    Generar NanoID (similar a UUID pero más corto).
    
    Args:
        size: Tamaño del ID
        
    Returns:
        NanoID
    """
    alphabet = string.ascii_letters + string.digits + '-_'
    return ''.join(random.choice(alphabet) for _ in range(size))


def generate_sequential_id(prefix: str = "", start: int = 1) -> str:
    """
    Generar ID secuencial.
    
    Args:
        prefix: Prefijo opcional
        start: Número inicial
        
    Returns:
        ID secuencial
    """
    # Nota: En producción, esto debería usar un contador persistente
    timestamp = int(time.time())
    return f"{prefix}{timestamp}{start}" if prefix else f"{timestamp}{start}"


def generate_random_string(length: int = 16, include_symbols: bool = False) -> str:
    """
    Generar string aleatorio.
    
    Args:
        length: Longitud del string
        include_symbols: Si incluir símbolos
        
    Returns:
        String aleatorio
    """
    chars = string.ascii_letters + string.digits
    if include_symbols:
        chars += string.punctuation
    
    return ''.join(random.choice(chars) for _ in range(length))


def generate_hex_id(length: int = 16) -> str:
    """
    Generar ID hexadecimal.
    
    Args:
        length: Longitud del ID
        
    Returns:
        ID hexadecimal
    """
    return ''.join(random.choice(string.hexdigits) for _ in range(length))


def generate_base62_id(length: int = 16) -> str:
    """
    Generar ID base62 (0-9, a-z, A-Z).
    
    Args:
        length: Longitud del ID
        
    Returns:
        ID base62
    """
    chars = string.digits + string.ascii_letters
    return ''.join(random.choice(chars) for _ in range(length))


def generate_snowflake_id() -> int:
    """
    Generar ID tipo Snowflake (Twitter).
    
    Returns:
        ID Snowflake
    """
    # Implementación simplificada
    # Snowflake real usa: timestamp (41 bits) + machine ID (10 bits) + sequence (12 bits)
    timestamp = int(time.time() * 1000)  # milisegundos
    machine_id = random.randint(0, 1023)  # 10 bits
    sequence = random.randint(0, 4095)  # 12 bits
    
    # Combinar en un ID de 64 bits
    snowflake = (timestamp << 22) | (machine_id << 12) | sequence
    return snowflake


def generate_ulid() -> str:
    """
    Generar ULID (Universally Unique Lexicographically Sortable Identifier).
    
    Returns:
        ULID como string
    """
    # ULID: 26 caracteres base32
    # 10 caracteres timestamp (milliseconds) + 16 caracteres random
    timestamp = int(time.time() * 1000)
    
    # Convertir timestamp a base32
    base32_chars = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    timestamp_part = ""
    temp = timestamp
    for _ in range(10):
        timestamp_part = base32_chars[temp % 32] + timestamp_part
        temp //= 32
    
    # Parte aleatoria
    random_part = ''.join(random.choice(base32_chars) for _ in range(16))
    
    return timestamp_part + random_part


class IDGenerator:
    """
    Generador de IDs con diferentes estrategias.
    """
    
    def __init__(self, strategy: str = "uuid"):
        """
        Inicializar generador.
        
        Args:
            strategy: Estrategia (uuid, nanoid, timestamp, snowflake, ulid)
        """
        self.strategy = strategy
        self._counter = 0
    
    def generate(self, **kwargs) -> str:
        """
        Generar ID según estrategia.
        
        Args:
            **kwargs: Argumentos adicionales según estrategia
            
        Returns:
            ID generado
        """
        if self.strategy == "uuid":
            return generate_uuid()
        elif self.strategy == "nanoid":
            size = kwargs.get("size", 21)
            return generate_nanoid(size)
        elif self.strategy == "timestamp":
            prefix = kwargs.get("prefix", "")
            return generate_timestamp_id(prefix)
        elif self.strategy == "snowflake":
            return str(generate_snowflake_id())
        elif self.strategy == "ulid":
            return generate_ulid()
        elif self.strategy == "short_uuid":
            length = kwargs.get("length", 8)
            return generate_short_uuid(length)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")




