"""
HTML element extraction helper functions for web scraping.

Refactored to consolidate functions into ElementExtractor class.
"""

from typing import List, Dict, Any
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup


class ElementExtractor:
    """
    HTML element extraction utilities.
    
    Responsibilities:
    - Extract links from HTML
    - Extract images from HTML
    - Normalize URLs
    - Check if URLs are external
    
    Single Responsibility: Handle all HTML element extraction operations.
    """
    
    def __init__(self, base_url: str = None):
        """
        Initialize element extractor.
        
        Args:
            base_url: Base URL for normalizing relative URLs
        """
        self.base_url = base_url
    
    def normalize_url(self, url: str) -> str:
        """
        Normalize a URL (relative to absolute).
        
        Args:
            url: URL to normalize
        
        Returns:
            Normalized absolute URL
        """
        if self.base_url:
            return urljoin(self.base_url, url)
        return url
    
    def is_external_url(self, url: str) -> bool:
        """
        Check if URL is external (different domain).
        
        Args:
            url: URL to check
        
        Returns:
            True if external, False otherwise
        """
        if not self.base_url:
            return False
        return urlparse(url).netloc != urlparse(self.base_url).netloc
    
    def extract_links(
        self,
        soup: BeautifulSoup,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Extract all links from HTML with normalization.
        
        Args:
            soup: BeautifulSoup object
            limit: Maximum number of links to extract
        
        Returns:
            List of link dictionaries with text, url, and external flag
        """
        links = []
        for link in soup.find_all('a', href=True, limit=limit):
            href = link.get('href', '')
            if href:
                normalized_url = self.normalize_url(href)
                links.append({
                    "text": link.get_text(strip=True),
                    "url": normalized_url,
                    "external": self.is_external_url(normalized_url)
                })
        return links
    
    def extract_images(
        self,
        soup: BeautifulSoup,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Extract all images from HTML with normalization.
        
        Args:
            soup: BeautifulSoup object
            limit: Maximum number of images to extract
        
        Returns:
            List of image dictionaries with src, alt, title, width, height
        """
        images = []
        for img in soup.find_all('img', src=True, limit=limit):
            src = img.get('src', '')
            if src:
                normalized_src = self.normalize_url(src)
                images.append({
                    "src": normalized_src,
                    "alt": img.get('alt', ''),
                    "title": img.get('title', ''),
                    "width": img.get('width', ''),
                    "height": img.get('height', '')
                })
        return images


# Backward compatibility functions
def normalize_url(url: str, base_url: str = None) -> str:
    """
    Normalize a URL (backward compatibility).
    
    Args:
        url: URL to normalize
        base_url: Base URL for relative URLs
    
    Returns:
        Normalized absolute URL
    """
    extractor = ElementExtractor(base_url=base_url)
    return extractor.normalize_url(url)


def is_external_url(url: str, base_url: str) -> bool:
    """
    Check if URL is external (backward compatibility).
    
    Args:
        url: URL to check
        base_url: Base URL for comparison
    
    Returns:
        True if external, False otherwise
    """
    extractor = ElementExtractor(base_url=base_url)
    return extractor.is_external_url(url)


def extract_links(
    soup: BeautifulSoup,
    base_url: str,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Extract all links (backward compatibility).
    
    Args:
        soup: BeautifulSoup object
        base_url: Base URL for normalizing relative URLs
        limit: Maximum number of links to extract
    
    Returns:
        List of link dictionaries
    """
    extractor = ElementExtractor(base_url=base_url)
    return extractor.extract_links(soup, limit)


def extract_images(
    soup: BeautifulSoup,
    base_url: str,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Extract all images (backward compatibility).
    
    Args:
        soup: BeautifulSoup object
        base_url: Base URL for normalizing relative URLs
        limit: Maximum number of images to extract
    
    Returns:
        List of image dictionaries
    """
    extractor = ElementExtractor(base_url=base_url)
    return extractor.extract_images(soup, limit)
