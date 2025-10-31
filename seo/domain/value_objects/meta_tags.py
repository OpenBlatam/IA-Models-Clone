from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from collections import defaultdict
        import re
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Meta Tags Value Object
Domain-Driven Design with SEO-specific business logic
"""



@dataclass(frozen=True)
class MetaTags:
    """
    Meta tags value object with SEO-specific business logic
    
    This value object encapsulates meta tag validation, analysis,
    and SEO-specific business rules.
    """
    
    tags: Dict[str, str]
    
    def __post_init__(self) -> Any:
        """Validate meta tags after initialization"""
        if not isinstance(self.tags, dict):
            raise ValueError("Tags must be a dictionary")
    
    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get meta tag value
        
        Args:
            name: Meta tag name
            default: Default value if not found
            
        Returns:
            Optional[str]: Meta tag value
        """
        return self.tags.get(name, default)
    
    def has(self, name: str) -> bool:
        """
        Check if meta tag exists
        
        Args:
            name: Meta tag name
            
        Returns:
            bool: True if meta tag exists
        """
        return name in self.tags
    
    def get_title(self) -> Optional[str]:
        """
        Get page title from meta tags
        
        Returns:
            Optional[str]: Page title
        """
        return self.get('title') or self.get('og:title')
    
    def get_description(self) -> Optional[str]:
        """
        Get page description from meta tags
        
        Returns:
            Optional[str]: Page description
        """
        return self.get('description') or self.get('og:description')
    
    def get_keywords(self) -> Optional[str]:
        """
        Get page keywords from meta tags
        
        Returns:
            Optional[str]: Page keywords
        """
        return self.get('keywords')
    
    def get_author(self) -> Optional[str]:
        """
        Get page author from meta tags
        
        Returns:
            Optional[str]: Page author
        """
        return self.get('author') or self.get('og:author')
    
    def get_image(self) -> Optional[str]:
        """
        Get page image from meta tags
        
        Returns:
            Optional[str]: Page image URL
        """
        return self.get('og:image') or self.get('twitter:image')
    
    def get_type(self) -> Optional[str]:
        """
        Get page type from meta tags
        
        Returns:
            Optional[str]: Page type
        """
        return self.get('og:type') or self.get('twitter:card')
    
    def get_url(self) -> Optional[str]:
        """
        Get page URL from meta tags
        
        Returns:
            Optional[str]: Page URL
        """
        return self.get('og:url') or self.get('canonical')
    
    def get_site_name(self) -> Optional[str]:
        """
        Get site name from meta tags
        
        Returns:
            Optional[str]: Site name
        """
        return self.get('og:site_name')
    
    def get_locale(self) -> Optional[str]:
        """
        Get page locale from meta tags
        
        Returns:
            Optional[str]: Page locale
        """
        return self.get('og:locale') or self.get('lang')
    
    def get_robots(self) -> Optional[str]:
        """
        Get robots directive from meta tags
        
        Returns:
            Optional[str]: Robots directive
        """
        return self.get('robots')
    
    def get_viewport(self) -> Optional[str]:
        """
        Get viewport meta tag
        
        Returns:
            Optional[str]: Viewport directive
        """
        return self.get('viewport')
    
    def get_charset(self) -> Optional[str]:
        """
        Get character encoding from meta tags
        
        Returns:
            Optional[str]: Character encoding
        """
        return self.get('charset')
    
    def get_og_tags(self) -> Dict[str, str]:
        """
        Get all Open Graph tags
        
        Returns:
            Dict[str, str]: Open Graph tags
        """
        return {k: v for k, v in self.tags.items() if k.startswith('og:')}
    
    def get_twitter_tags(self) -> Dict[str, str]:
        """
        Get all Twitter Card tags
        
        Returns:
            Dict[str, str]: Twitter Card tags
        """
        return {k: v for k, v in self.tags.items() if k.startswith('twitter:')}
    
    def get_seo_tags(self) -> Dict[str, str]:
        """
        Get all SEO-related tags
        
        Returns:
            Dict[str, str]: SEO tags
        """
        seo_tags = [
            'title', 'description', 'keywords', 'author', 'robots',
            'canonical', 'og:title', 'og:description', 'og:type',
            'og:url', 'og:image', 'og:site_name', 'og:locale',
            'twitter:card', 'twitter:title', 'twitter:description',
            'twitter:image', 'twitter:site', 'twitter:creator'
        ]
        return {k: v for k, v in self.tags.items() if k in seo_tags}
    
    def get_score(self) -> int:
        """
        Calculate meta tags SEO score
        
        Returns:
            int: SEO score (0-100)
        """
        score = 0
        max_score = 100
        
        # Basic SEO tags (40 points)
        if self.get_title():
            score += 10
        if self.get_description():
            score += 10
        if self.get_keywords():
            score += 5
        if self.get_author():
            score += 5
        if self.get_robots():
            score += 5
        if self.get_viewport():
            score += 5
        
        # Open Graph tags (30 points)
        og_tags = self.get_og_tags()
        if og_tags.get('og:title'):
            score += 5
        if og_tags.get('og:description'):
            score += 5
        if og_tags.get('og:type'):
            score += 5
        if og_tags.get('og:url'):
            score += 5
        if og_tags.get('og:image'):
            score += 5
        if og_tags.get('og:site_name'):
            score += 5
        
        # Twitter Card tags (20 points)
        twitter_tags = self.get_twitter_tags()
        if twitter_tags.get('twitter:card'):
            score += 5
        if twitter_tags.get('twitter:title'):
            score += 5
        if twitter_tags.get('twitter:description'):
            score += 5
        if twitter_tags.get('twitter:image'):
            score += 5
        
        # Technical tags (10 points)
        if self.get_charset():
            score += 5
        if self.get_locale():
            score += 5
        
        return min(score, max_score)
    
    def get_issues(self) -> List[str]:
        """
        Get list of meta tag issues
        
        Returns:
            List[str]: List of issues
        """
        issues = []
        
        # Missing essential tags
        if not self.get_title():
            issues.append("Missing title tag")
        if not self.get_description():
            issues.append("Missing meta description")
        
        # Open Graph issues
        og_tags = self.get_og_tags()
        if not og_tags.get('og:title'):
            issues.append("Missing Open Graph title")
        if not og_tags.get('og:description'):
            issues.append("Missing Open Graph description")
        if not og_tags.get('og:type'):
            issues.append("Missing Open Graph type")
        if not og_tags.get('og:url'):
            issues.append("Missing Open Graph URL")
        
        # Twitter Card issues
        twitter_tags = self.get_twitter_tags()
        if not twitter_tags.get('twitter:card'):
            issues.append("Missing Twitter Card type")
        if not twitter_tags.get('twitter:title'):
            issues.append("Missing Twitter Card title")
        if not twitter_tags.get('twitter:description'):
            issues.append("Missing Twitter Card description")
        
        # Technical issues
        if not self.get_viewport():
            issues.append("Missing viewport meta tag")
        if not self.get_charset():
            issues.append("Missing character encoding")
        
        return issues
    
    def get_recommendations(self) -> List[str]:
        """
        Get meta tag improvement recommendations
        
        Returns:
            List[str]: List of recommendations
        """
        recommendations = []
        score = self.get_score()
        
        if score < 50:
            recommendations.append("Critical meta tag issues - immediate action required")
        elif score < 70:
            recommendations.append("Moderate meta tag issues - improvements recommended")
        elif score < 90:
            recommendations.append("Minor meta tag issues - fine-tuning recommended")
        else:
            recommendations.append("Excellent meta tags - maintain current practices")
        
        # Specific recommendations based on issues
        issues = self.get_issues()
        for issue in issues:
            if "Missing title" in issue:
                recommendations.append("Add a compelling title tag (30-60 characters)")
            elif "Missing meta description" in issue:
                recommendations.append("Add a descriptive meta description (120-160 characters)")
            elif "Missing Open Graph" in issue:
                recommendations.append("Add Open Graph tags for better social media sharing")
            elif "Missing Twitter Card" in issue:
                recommendations.append("Add Twitter Card tags for better Twitter sharing")
            elif "Missing viewport" in issue:
                recommendations.append("Add viewport meta tag for mobile optimization")
            elif "Missing character encoding" in issue:
                recommendations.append("Add character encoding meta tag")
        
        return recommendations
    
    def get_duplicates(self) -> Dict[str, List[str]]:
        """
        Find duplicate meta tags
        
        Returns:
            Dict[str, List[str]]: Duplicate tags grouped by name
        """
        duplicates = defaultdict(list)
        
        for name, value in self.tags.items():
            duplicates[name].append(value)
        
        # Filter only actual duplicates
        return {name: values for name, values in duplicates.items() if len(values) > 1}
    
    def get_missing_essential(self) -> List[str]:
        """
        Get list of missing essential meta tags
        
        Returns:
            List[str]: Missing essential tags
        """
        essential_tags = [
            'title', 'description', 'viewport', 'charset',
            'og:title', 'og:description', 'og:type', 'og:url',
            'twitter:card', 'twitter:title', 'twitter:description'
        ]
        
        missing = []
        for tag in essential_tags:
            if not self.has(tag):
                missing.append(tag)
        
        return missing
    
    def get_optimal_length_tags(self) -> Dict[str, bool]:
        """
        Check if meta tags have optimal length
        
        Returns:
            Dict[str, bool]: Tag name -> optimal length status
        """
        optimal_lengths = {
            'title': (30, 60),
            'description': (120, 160),
            'og:title': (30, 60),
            'og:description': (120, 160),
            'twitter:title': (30, 60),
            'twitter:description': (120, 160)
        }
        
        results = {}
        for tag, (min_len, max_len) in optimal_lengths.items():
            value = self.get(tag)
            if value:
                length = len(value)
                results[tag] = min_len <= length <= max_len
            else:
                results[tag] = False
        
        return results
    
    def to_dict(self) -> Dict[str, str]:
        """
        Convert to dictionary
        
        Returns:
            Dict[str, str]: Meta tags as dictionary
        """
        return dict(self.tags)
    
    def __len__(self) -> int:
        """Get number of meta tags"""
        return len(self.tags)
    
    def __contains__(self, name: str) -> bool:
        """Check if meta tag exists"""
        return name in self.tags
    
    def __iter__(self) -> Any:
        """Iterate over meta tags"""
        return iter(self.tags.items())
    
    def __eq__(self, other) -> bool:
        """Compare meta tags"""
        if not isinstance(other, MetaTags):
            return False
        return self.tags == other.tags
    
    def __hash__(self) -> int:
        """Hash meta tags"""
        return hash(tuple(sorted(self.tags.items())))
    
    def __str__(self) -> str:
        """String representation"""
        return f"MetaTags(count={len(self.tags)})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"MetaTags(tags={self.tags})"
    
    @classmethod
    def create(cls, tags: Dict[str, str]) -> "MetaTags":
        """
        Factory method to create meta tags
        
        Args:
            tags: Meta tags dictionary
            
        Returns:
            MetaTags: New meta tags instance
        """
        return cls(tags)
    
    @classmethod
    def create_empty(cls) -> "MetaTags":
        """
        Create empty meta tags
        
        Returns:
            MetaTags: Empty meta tags instance
        """
        return cls({})
    
    @classmethod
    def create_from_html(cls, html_content: str) -> "MetaTags":
        """
        Create meta tags from HTML content
        
        Args:
            html_content: HTML content
            
        Returns:
            MetaTags: Meta tags extracted from HTML
        """
        
        tags = {}
        
        # Extract meta tags using regex
        meta_pattern = r'<meta[^>]+(?:name|property)=["\']([^"\']+)["\'][^>]+content=["\']([^"\']+)["\'][^>]*>'
        matches = re.findall(meta_pattern, html_content, re.IGNORECASE)
        
        for name, content in matches:
            tags[name.lower()] = content
        
        return cls(tags) 