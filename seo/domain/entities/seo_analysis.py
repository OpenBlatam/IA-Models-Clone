from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from domain.value_objects.url import URL
from domain.value_objects.meta_tags import MetaTags
from domain.value_objects.seo_score import SEOScore
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
SEO Analysis Domain Entity
Clean Architecture with Domain-Driven Design
"""



@dataclass(frozen=True)
class SEOAnalysis:
    """
    SEO Analysis domain entity
    
    This is the core business entity that represents an SEO analysis
    of a web page. It contains all the essential SEO data and business logic.
    """
    
    # Core attributes
    url: URL
    title: Optional[str]
    description: Optional[str]
    keywords: Optional[str]
    meta_tags: MetaTags
    links: List[str]
    content_length: int
    processing_time: float
    created_at: datetime
    
    def __post_init__(self) -> Any:
        """Validate entity after initialization"""
        if not self.url.is_valid():
            raise ValueError("Invalid URL")
        
        if self.content_length < 0:
            raise ValueError("Content length cannot be negative")
        
        if self.processing_time < 0:
            raise ValueError("Processing time cannot be negative")
    
    def is_valid(self) -> bool:
        """
        Validate SEO analysis
        
        Returns:
            bool: True if analysis is valid
        """
        return (
            self.url.is_valid() and
            self.content_length > 0 and
            self.processing_time >= 0
        )
    
    def get_score(self) -> SEOScore:
        """
        Calculate SEO score based on analysis data
        
        Returns:
            SEOScore: Calculated SEO score
        """
        score = 0
        max_score = 100
        
        # Title analysis (20 points)
        if self.title:
            score += 10
            if len(self.title) >= 30 and len(self.title) <= 60:
                score += 10  # Optimal title length
        
        # Description analysis (20 points)
        if self.description:
            score += 10
            if len(self.description) >= 120 and len(self.description) <= 160:
                score += 10  # Optimal description length
        
        # Keywords analysis (10 points)
        if self.keywords:
            score += 5
            if len(self.keywords.split(',')) <= 10:
                score += 5  # Optimal keyword count
        
        # Meta tags analysis (15 points)
        meta_score = self.meta_tags.get_score()
        score += meta_score
        
        # Links analysis (20 points)
        if self.links:
            score += min(len(self.links) * 2, 20)  # Max 20 points for links
        
        # Content analysis (15 points)
        if self.content_length > 300:
            score += 15  # Good content length
        elif self.content_length > 100:
            score += 10  # Minimum content length
        elif self.content_length > 0:
            score += 5   # Some content
        
        return SEOScore(min(score, max_score))
    
    def get_issues(self) -> List[str]:
        """
        Get list of SEO issues found
        
        Returns:
            List[str]: List of SEO issues
        """
        issues = []
        
        # Title issues
        if not self.title:
            issues.append("Missing title tag")
        elif len(self.title) < 30:
            issues.append("Title too short (should be 30-60 characters)")
        elif len(self.title) > 60:
            issues.append("Title too long (should be 30-60 characters)")
        
        # Description issues
        if not self.description:
            issues.append("Missing meta description")
        elif len(self.description) < 120:
            issues.append("Description too short (should be 120-160 characters)")
        elif len(self.description) > 160:
            issues.append("Description too long (should be 120-160 characters)")
        
        # Keywords issues
        if self.keywords:
            keyword_count = len(self.keywords.split(','))
            if keyword_count > 10:
                issues.append(f"Too many keywords ({keyword_count}, should be ≤10)")
        
        # Content issues
        if self.content_length < 300:
            issues.append("Content too short (should be at least 300 characters)")
        
        # Link issues
        if not self.links:
            issues.append("No internal or external links found")
        
        return issues
    
    def get_recommendations(self) -> List[str]:
        """
        Get SEO improvement recommendations
        
        Returns:
            List[str]: List of recommendations
        """
        recommendations = []
        score = self.get_score()
        
        if score.value < 50:
            recommendations.append("Critical SEO issues detected - immediate action required")
        elif score.value < 70:
            recommendations.append("Moderate SEO issues - improvements recommended")
        elif score.value < 90:
            recommendations.append("Minor SEO issues - fine-tuning recommended")
        else:
            recommendations.append("Excellent SEO score - maintain current practices")
        
        # Specific recommendations based on issues
        issues = self.get_issues()
        for issue in issues:
            if "Missing title" in issue:
                recommendations.append("Add a compelling title tag (30-60 characters)")
            elif "Title too short" in issue:
                recommendations.append("Expand title to 30-60 characters")
            elif "Title too long" in issue:
                recommendations.append("Shorten title to 30-60 characters")
            elif "Missing meta description" in issue:
                recommendations.append("Add a descriptive meta description (120-160 characters)")
            elif "Description too short" in issue:
                recommendations.append("Expand meta description to 120-160 characters")
            elif "Description too long" in issue:
                recommendations.append("Shorten meta description to 120-160 characters")
            elif "Too many keywords" in issue:
                recommendations.append("Reduce keywords to 10 or fewer")
            elif "Content too short" in issue:
                recommendations.append("Add more relevant content (at least 300 characters)")
            elif "No internal or external links" in issue:
                recommendations.append("Add relevant internal and external links")
        
        return recommendations
    
    def get_summary(self) -> dict:
        """
        Get analysis summary
        
        Returns:
            dict: Analysis summary
        """
        return {
            "url": str(self.url),
            "score": self.get_score().value,
            "grade": self.get_score().get_grade(),
            "content_length": self.content_length,
            "links_count": len(self.links),
            "meta_tags_count": len(self.meta_tags.tags),
            "processing_time": self.processing_time,
            "issues_count": len(self.get_issues()),
            "recommendations_count": len(self.get_recommendations()),
            "created_at": self.created_at.isoformat()
        }
    
    def to_dict(self) -> dict:
        """
        Convert entity to dictionary
        
        Returns:
            dict: Entity as dictionary
        """
        return {
            "url": str(self.url),
            "title": self.title,
            "description": self.description,
            "keywords": self.keywords,
            "meta_tags": self.meta_tags.to_dict(),
            "links": self.links,
            "content_length": self.content_length,
            "processing_time": self.processing_time,
            "score": self.get_score().value,
            "grade": self.get_score().get_grade(),
            "issues": self.get_issues(),
            "recommendations": self.get_recommendations(),
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def create(
        cls,
        url: URL,
        title: Optional[str] = None,
        description: Optional[str] = None,
        keywords: Optional[str] = None,
        meta_tags: Optional[MetaTags] = None,
        links: Optional[List[str]] = None,
        content_length: int = 0,
        processing_time: float = 0.0
    ) -> "SEOAnalysis":
        """
        Factory method to create SEO analysis
        
        Args:
            url: URL to analyze
            title: Page title
            description: Meta description
            keywords: Meta keywords
            meta_tags: Meta tags
            links: List of links
            content_length: Content length
            processing_time: Processing time
            
        Returns:
            SEOAnalysis: New SEO analysis instance
        """
        return cls(
            url=url,
            title=title,
            description=description,
            keywords=keywords,
            meta_tags=meta_tags or MetaTags({}),
            links=links or [],
            content_length=content_length,
            processing_time=processing_time,
            created_at=datetime.utcnow()
        )
    
    def __eq__(self, other) -> bool:
        """Compare two SEO analyses"""
        if not isinstance(other, SEOAnalysis):
            return False
        
        return (
            self.url == other.url and
            self.title == other.title and
            self.description == other.description and
            self.keywords == other.keywords and
            self.meta_tags == other.meta_tags and
            self.links == other.links and
            self.content_length == other.content_length and
            self.created_at == other.created_at
        )
    
    def __hash__(self) -> int:
        """Hash SEO analysis"""
        return hash((
            self.url,
            self.title,
            self.description,
            self.keywords,
            self.meta_tags,
            tuple(self.links),
            self.content_length,
            self.created_at
        ))
    
    def __str__(self) -> str:
        """String representation"""
        return f"SEOAnalysis(url={self.url}, score={self.get_score().value})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (
            f"SEOAnalysis("
            f"url={self.url}, "
            f"title={self.title}, "
            f"description={self.description}, "
            f"keywords={self.keywords}, "
            f"meta_tags={self.meta_tags}, "
            f"links_count={len(self.links)}, "
            f"content_length={self.content_length}, "
            f"processing_time={self.processing_time}, "
            f"score={self.get_score().value}, "
            f"created_at={self.created_at}"
            f")"
        ) 