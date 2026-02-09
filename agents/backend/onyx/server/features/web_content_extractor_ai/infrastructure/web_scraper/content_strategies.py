"""
Content extraction strategies for web scraping.

Refactored to consolidate content extraction strategies into specialized classes.
"""

import json
import logging
from typing import Dict, Any
from bs4 import BeautifulSoup

import trafilatura
from readability import Document
from newspaper import Article

from .extraction_helpers import SafeExtractor

logger = logging.getLogger(__name__)


class TrafilaturaExtractor:
    """
    Trafilatura content extraction strategy.
    
    Single Responsibility: Extract content using Trafilatura.
    """
    
    def __init__(self):
        """Initialize Trafilatura extractor."""
        self._safe_extractor = SafeExtractor()
    
    def extract(self, html: str, url: str) -> Dict[str, Any]:
        """
        Extract content using Trafilatura.
        
        Args:
            html: HTML content
            url: URL of the page
        
        Returns:
            Dictionary with extracted content
        """
        return self._safe_extractor.extract(
            lambda: json.loads(trafilatura.extract(
                html,
                url=url,
                include_comments=False,
                include_tables=True,
                include_images=True,
                include_links=True,
                output_format='json'
            ) or '{}'),
            tool_name="Trafilatura"
        )


class ReadabilityExtractor:
    """
    Readability content extraction strategy.
    
    Single Responsibility: Extract content using Readability.
    """
    
    def __init__(self):
        """Initialize Readability extractor."""
        self._safe_extractor = SafeExtractor()
    
    def extract(self, html: str) -> Dict[str, Any]:
        """
        Extract content using Readability.
        
        Args:
            html: HTML content
        
        Returns:
            Dictionary with extracted content
        """
        return self._safe_extractor.extract(
            lambda: {
                "title": Document(html).title(),
                "content": Document(html).summary(),
                "short_title": Document(html).short_title()
            },
            tool_name="Readability"
        )


class NewspaperExtractor:
    """
    Newspaper3k content extraction strategy.
    
    Single Responsibility: Extract content using Newspaper3k.
    """
    
    def __init__(self):
        """Initialize Newspaper extractor."""
        self._safe_extractor = SafeExtractor()
    
    def extract(self, url: str, html: str) -> Dict[str, Any]:
        """
        Extract content using Newspaper3k.
        
        Args:
            url: URL of the page
            html: HTML content
        
        Returns:
            Dictionary with extracted content
        """
        return self._safe_extractor.extract(
            lambda: self._build_result(url, html),
            tool_name="Newspaper3k"
        )
    
    def _build_result(self, url: str, html: str) -> Dict[str, Any]:
        """Build Newspaper3k result."""
        article = Article(url)
        article.set_html(html)
        article.parse()
        
        return {
            "title": article.title,
            "text": article.text,
            "authors": article.authors,
            "publish_date": article.publish_date.isoformat() if article.publish_date else None,
            "top_image": article.top_image,
            "images": article.images,
            "movies": article.movies,
            "keywords": article.keywords,
            "summary": article.summary
        }


class BeautifulSoupExtractor:
    """
    BeautifulSoup content extraction strategy.
    
    Single Responsibility: Extract main content using BeautifulSoup heuristics.
    """
    
    def extract(self, soup: BeautifulSoup) -> str:
        """
        Extract main content using heuristics.
        
        Args:
            soup: BeautifulSoup object
        
        Returns:
            Main content text
        """
        # Try to find main content
        main_selectors = [
            'main',
            'article',
            '[role="main"]',
            '.content',
            '.post-content',
            '.entry-content',
            '#content',
            '.article-body',
            '.post-body'
        ]
        
        for selector in main_selectors:
            element = soup.select_one(selector)
            if element:
                # Remove unwanted elements
                for tag in element.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                    tag.decompose()
                text = element.get_text(separator=' ', strip=True)
                if len(text) > 200:  # Significant content
                    return text
        
        # Fallback: use entire body
        body = soup.find('body')
        if body:
            for tag in body.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()
            return body.get_text(separator=' ', strip=True)
        
        return ""

