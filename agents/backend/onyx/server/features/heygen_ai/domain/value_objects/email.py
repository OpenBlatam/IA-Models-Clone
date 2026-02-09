from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import re
from typing import Any
from dataclasses import dataclass
from ..exceptions.domain_errors import ValueObjectValidationError
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Email Value Object

Represents a validated email address.
"""




@dataclass(frozen=True)
class Email:
    """
    Email value object that ensures valid email format.
    
    Business Rules:
    - Must be a valid email format
    - Must not be empty
    - Maximum length of 254 characters (RFC 5321)
    - Case insensitive comparison
    """
    
    value: str
    
    def __post_init__(self) -> None:
        """Validate email format after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate email address."""
        if not self.value:
            raise ValueObjectValidationError("Email cannot be empty")
        
        if not isinstance(self.value, str):
            raise ValueObjectValidationError("Email must be a string")
        
        # Remove leading/trailing whitespace
        email = self.value.strip()
        if len(email) != len(self.value):
            raise ValueObjectValidationError("Email cannot have leading or trailing whitespace")
        
        # Check length
        if len(email) > 254:
            raise ValueObjectValidationError("Email cannot exceed 254 characters")
        
        # Check basic format with regex
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            raise ValueObjectValidationError("Invalid email format")
        
        # Additional validations
        local_part, domain = email.rsplit('@', 1)
        
        # Validate local part (before @)
        if len(local_part) > 64:
            raise ValueObjectValidationError("Email local part cannot exceed 64 characters")
        
        if local_part.startswith('.') or local_part.endswith('.'):
            raise ValueObjectValidationError("Email local part cannot start or end with a period")
        
        if '..' in local_part:
            raise ValueObjectValidationError("Email local part cannot contain consecutive periods")
        
        # Validate domain part (after @)
        if len(domain) > 253:
            raise ValueObjectValidationError("Email domain cannot exceed 253 characters")
        
        # Check for valid domain format
        domain_parts = domain.split('.')
        for part in domain_parts:
            if len(part) == 0:
                raise ValueObjectValidationError("Email domain parts cannot be empty")
            if len(part) > 63:
                raise ValueObjectValidationError("Email domain part cannot exceed 63 characters")
            if not re.match(r'^[a-zA-Z0-9-]+$', part):
                raise ValueObjectValidationError("Email domain part contains invalid characters")
            if part.startswith('-') or part.endswith('-'):
                raise ValueObjectValidationError("Email domain part cannot start or end with hyphen")
    
    @property
    def normalized(self) -> str:
        """Get normalized email (lowercase)."""
        return self.value.lower()
    
    @property
    def local_part(self) -> str:
        """Get local part of email (before @)."""
        return self.value.split('@')[0]
    
    @property
    def domain(self) -> str:
        """Get domain part of email (after @)."""
        return self.value.split('@')[1]
    
    def __str__(self) -> str:
        """String representation."""
        return self.value
    
    def __eq__(self, other: Any) -> bool:
        """Compare emails (case insensitive)."""
        if not isinstance(other, Email):
            return False
        return self.normalized == other.normalized
    
    def __hash__(self) -> int:
        """Hash based on normalized email."""
        return hash(self.normalized)
    
    @classmethod
    def create(cls, email: str) -> 'Email':
        """Create Email from string with validation."""
        return cls(email)
    
    @classmethod
    def from_string(cls, email: str) -> 'Email':
        """Alias for create method."""
        return cls.create(email) 