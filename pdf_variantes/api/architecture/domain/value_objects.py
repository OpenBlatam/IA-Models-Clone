"""
Domain Layer - Value Objects
Immutable value objects with validation
"""

from dataclasses import dataclass
from typing import Optional
import re


@dataclass(frozen=True)
class DocumentId:
    """Document ID value object"""
    value: str
    
    def __post_init__(self):
        if not self.value or not self.value.strip():
            raise ValueError("Document ID cannot be empty")
        if len(self.value) < 10:
            raise ValueError("Document ID must be at least 10 characters")
    
    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class UserId:
    """User ID value object"""
    value: str
    
    def __post_init__(self):
        if not self.value or not self.value.strip():
            raise ValueError("User ID cannot be empty")
    
    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Filename:
    """Filename value object with validation"""
    value: str
    
    def __post_init__(self):
        if not self.value or not self.value.strip():
            raise ValueError("Filename cannot be empty")
        
        # Sanitize filename
        cleaned = self.value.strip()
        cleaned = cleaned.replace('..', '').replace('/', '').replace('\\', '')
        
        if len(cleaned) > 500:
            raise ValueError("Filename too long (max 500 characters)")
        
        # Validate extension
        if not cleaned.lower().endswith('.pdf'):
            raise ValueError("File must be a PDF")
        
        # Use the cleaned value
        object.__setattr__(self, 'value', cleaned)
    
    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class FileSize:
    """File size value object"""
    bytes: int
    max_bytes: int = 100 * 1024 * 1024  # 100MB default
    
    def __post_init__(self):
        if self.bytes < 0:
            raise ValueError("File size cannot be negative")
        if self.bytes > self.max_bytes:
            raise ValueError(f"File size exceeds maximum: {self.max_bytes} bytes")
    
    @property
    def megabytes(self) -> float:
        """Get size in megabytes"""
        return self.bytes / (1024 * 1024)
    
    def __str__(self) -> str:
        return f"{self.megabytes:.2f} MB"


@dataclass(frozen=True)
class RelevanceScore:
    """Relevance score value object (0.0 - 1.0)"""
    value: float
    min_value: float = 0.0
    max_value: float = 1.0
    
    def __post_init__(self):
        if not (self.min_value <= self.value <= self.max_value):
            raise ValueError(
                f"Relevance score must be between {self.min_value} and {self.max_value}"
            )
    
    def is_relevant(self, threshold: float = 0.5) -> bool:
        """Check if score is above threshold"""
        return self.value >= threshold
    
    def __float__(self) -> float:
        return self.value


@dataclass(frozen=True)
class SimilarityScore:
    """Similarity score value object (0.0 - 1.0)"""
    value: float
    
    def __post_init__(self):
        if not (0.0 <= self.value <= 1.0):
            raise ValueError("Similarity score must be between 0.0 and 1.0")
    
    def is_similar(self, threshold: float = 0.7) -> bool:
        """Check if similarity is above threshold"""
        return self.value >= threshold
    
    def __float__(self) -> float:
        return self.value






