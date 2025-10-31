from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from dataclasses import dataclass
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Analyze URL Response DTO
Data Transfer Object for analysis results
"""



class AnalyzeURLResponse(BaseModel):
    """
    Analyze URL response DTO
    
    This DTO represents the response from analyzing a URL with SEO data.
    """
    
    # Basic information
    url: str = Field(..., description="Analyzed URL")
    title: Optional[str] = Field(None, description="Page title")
    description: Optional[str] = Field(None, description="Meta description")
    keywords: Optional[str] = Field(None, description="Meta keywords")
    
    # Meta tags
    meta_tags: Dict[str, str] = Field(default_factory=dict, description="Meta tags")
    
    # Links
    links: List[str] = Field(default_factory=list, description="Page links")
    
    # Content analysis
    content_length: int = Field(0, description="Content length in characters")
    
    # Performance metrics
    processing_time: float = Field(0.0, description="Processing time in seconds")
    cache_hit: bool = Field(False, description="Whether result was from cache")
    
    # SEO score
    score: int = Field(0, description="SEO score (0-100)")
    grade: str = Field("F", description="SEO grade (A+, A, B, C, D, F)")
    level: str = Field("Failing", description="Performance level")
    
    # Issues and recommendations
    issues: List[str] = Field(default_factory=list, description="SEO issues found")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    
    # Timestamps
    timestamp: str = Field(..., description="Analysis timestamp")
    created_at: str = Field(..., description="Creation timestamp")
    
    class Config:
        """Pydantic configuration"""
        json_schema_extra = {
            "example": {
                "url": "https://example.com",
                "title": "Example Page",
                "description": "This is an example page",
                "keywords": "example, test, demo",
                "meta_tags": {
                    "og:title": "Example Page",
                    "og:description": "This is an example page"
                },
                "links": ["https://example.com/page1", "https://example.com/page2"],
                "content_length": 1500,
                "processing_time": 0.5,
                "cache_hit": False,
                "score": 85,
                "grade": "A",
                "level": "Excellent",
                "issues": ["Missing viewport meta tag"],
                "recommendations": ["Add viewport meta tag for mobile optimization"],
                "timestamp": "2024-01-01T12:00:00Z",
                "created_at": "2024-01-01T12:00:00Z"
            }
        }


@dataclass
class AnalyzeURLResponseDomain:
    """
    Domain version of analyze URL response
    
    This is used internally by the domain layer.
    """
    
    url: str
    title: Optional[str]
    description: Optional[str]
    keywords: Optional[str]
    meta_tags: Dict[str, str]
    links: List[str]
    content_length: int
    processing_time: float
    cache_hit: bool
    score: int
    grade: str
    level: str
    issues: List[str]
    recommendations: List[str]
    timestamp: datetime
    created_at: datetime
    
    @classmethod
    def from_dto(cls, dto: AnalyzeURLResponse) -> "AnalyzeURLResponseDomain":
        """
        Create domain response from DTO
        
        Args:
            dto: Response DTO
            
        Returns:
            AnalyzeURLResponseDomain: Domain response
        """
        return cls(
            url=dto.url,
            title=dto.title,
            description=dto.description,
            keywords=dto.keywords,
            meta_tags=dto.meta_tags,
            links=dto.links,
            content_length=dto.content_length,
            processing_time=dto.processing_time,
            cache_hit=dto.cache_hit,
            score=dto.score,
            grade=dto.grade,
            level=dto.level,
            issues=dto.issues,
            recommendations=dto.recommendations,
            timestamp=datetime.fromisoformat(dto.timestamp.replace('Z', '+00:00')),
            created_at=datetime.fromisoformat(dto.created_at.replace('Z', '+00:00'))
        )
    
    def to_dto(self) -> AnalyzeURLResponse:
        """
        Convert to DTO
        
        Returns:
            AnalyzeURLResponse: Response DTO
        """
        return AnalyzeURLResponse(
            url=self.url,
            title=self.title,
            description=self.description,
            keywords=self.keywords,
            meta_tags=self.meta_tags,
            links=self.links,
            content_length=self.content_length,
            processing_time=self.processing_time,
            cache_hit=self.cache_hit,
            score=self.score,
            grade=self.grade,
            level=self.level,
            issues=self.issues,
            recommendations=self.recommendations,
            timestamp=self.timestamp.isoformat(),
            created_at=self.created_at.isoformat()
        ) 