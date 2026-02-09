from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, validator, Field
from domain.value_objects.url import URL
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Analyze URL Request DTO
Data Transfer Object with validation
"""



class AnalyzeURLRequest(BaseModel):
    """
    Analyze URL request DTO
    
    This DTO represents the request for analyzing a URL with SEO data.
    """
    
    url: str = Field(..., description="URL to analyze", min_length=1, max_length=2048)
    include_content: bool = Field(True, description="Include content analysis")
    include_links: bool = Field(True, description="Include links analysis")
    include_meta: bool = Field(True, description="Include meta tags analysis")
    max_links: int = Field(100, description="Maximum number of links to analyze", ge=0, le=1000)
    timeout: float = Field(10.0, description="Request timeout in seconds", ge=1.0, le=60.0)
    
    @validator('url')
    def validate_url(cls, v) -> bool:
        """Validate URL format"""
        try:
            url_obj = URL.create(v)
            if not url_obj.is_valid():
                raise ValueError("Invalid URL format")
            return v
        except ValueError as e:
            raise ValueError(f"Invalid URL: {str(e)}")
    
    @validator('max_links')
    def validate_max_links(cls, v) -> bool:
        """Validate max links"""
        if v < 0:
            raise ValueError("Max links cannot be negative")
        if v > 1000:
            raise ValueError("Max links cannot exceed 1000")
        return v
    
    @validator('timeout')
    def validate_timeout(cls, v) -> bool:
        """Validate timeout"""
        if v < 1.0:
            raise ValueError("Timeout must be at least 1 second")
        if v > 60.0:
            raise ValueError("Timeout cannot exceed 60 seconds")
        return v
    
    def get_url_object(self) -> URL:
        """
        Get URL as domain object
        
        Returns:
            URL: URL domain object
        """
        return URL.create(self.url)
    
    class Config:
        """Pydantic configuration"""
        json_schema_extra = {
            "example": {
                "url": "https://example.com",
                "include_content": True,
                "include_links": True,
                "include_meta": True,
                "max_links": 100,
                "timeout": 10.0
            }
        }


@dataclass
class AnalyzeURLRequestDomain:
    """
    Domain version of analyze URL request
    
    This is used internally by the domain layer.
    """
    
    url: URL
    include_content: bool
    include_links: bool
    include_meta: bool
    max_links: int
    timeout: float
    
    @classmethod
    def from_dto(cls, dto: AnalyzeURLRequest) -> "AnalyzeURLRequestDomain":
        """
        Create domain request from DTO
        
        Args:
            dto: Request DTO
            
        Returns:
            AnalyzeURLRequestDomain: Domain request
        """
        return cls(
            url=dto.get_url_object(),
            include_content=dto.include_content,
            include_links=dto.include_links,
            include_meta=dto.include_meta,
            max_links=dto.max_links,
            timeout=dto.timeout
        )
    
    def to_dto(self) -> AnalyzeURLRequest:
        """
        Convert to DTO
        
        Returns:
            AnalyzeURLRequest: Request DTO
        """
        return AnalyzeURLRequest(
            url=str(self.url),
            include_content=self.include_content,
            include_links=self.include_links,
            include_meta=self.include_meta,
            max_links=self.max_links,
            timeout=self.timeout
        ) 