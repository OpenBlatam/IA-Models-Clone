"""
ID Generator Utilities for Piel Mejorador AI SAM3
==================================================

Unified ID generation utilities.
"""

import uuid
import secrets
import hashlib
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class IDGenerator:
    """Unified ID generator with multiple strategies."""
    
    @staticmethod
    def uuid4() -> str:
        """Generate UUID4."""
        return str(uuid.uuid4())
    
    @staticmethod
    def uuid4_hex() -> str:
        """Generate UUID4 as hex string (no dashes)."""
        return uuid.uuid4().hex
    
    @staticmethod
    def token_urlsafe(length: int = 32) -> str:
        """Generate URL-safe token."""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def token_hex(length: int = 32) -> str:
        """Generate hex token."""
        return secrets.token_hex(length)
    
    @staticmethod
    def short_id(length: int = 16) -> str:
        """Generate short ID."""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def hash_id(data: str, length: int = 16) -> str:
        """
        Generate ID from hash of data.
        
        Args:
            data: Data to hash
            length: Length of resulting ID
            
        Returns:
            Hashed ID
        """
        hash_obj = hashlib.sha256(data.encode())
        return hash_obj.hexdigest()[:length]
    
    @staticmethod
    def timestamp_id(prefix: str = "") -> str:
        """
        Generate ID with timestamp.
        
        Args:
            prefix: Optional prefix
            
        Returns:
            Timestamp-based ID
        """
        timestamp = int(datetime.now().timestamp() * 1000000)  # Microseconds
        random_part = secrets.token_hex(8)
        return f"{prefix}{timestamp}_{random_part}" if prefix else f"{timestamp}_{random_part}"
    
    @staticmethod
    def composite_id(*parts: str, separator: str = "_") -> str:
        """
        Generate composite ID from parts.
        
        Args:
            *parts: ID parts
            separator: Separator between parts
            
        Returns:
            Composite ID
        """
        return separator.join(str(part) for part in parts if part)
    
    @staticmethod
    def task_id() -> str:
        """Generate task ID."""
        return IDGenerator.uuid4()
    
    @staticmethod
    def session_id() -> str:
        """Generate session ID."""
        return IDGenerator.token_urlsafe(24)
    
    @staticmethod
    def correlation_id() -> str:
        """Generate correlation ID."""
        return IDGenerator.uuid4()
    
    @staticmethod
    def api_key() -> str:
        """Generate API key."""
        return IDGenerator.token_urlsafe(32)
    
    @staticmethod
    def api_key_id() -> str:
        """Generate API key ID."""
        return IDGenerator.token_urlsafe(16)


# Convenience functions
def generate_id() -> str:
    """Generate UUID4 ID."""
    return IDGenerator.uuid4()


def generate_task_id() -> str:
    """Generate task ID."""
    return IDGenerator.task_id()


def generate_session_id() -> str:
    """Generate session ID."""
    return IDGenerator.session_id()


def generate_correlation_id() -> str:
    """Generate correlation ID."""
    return IDGenerator.correlation_id()




