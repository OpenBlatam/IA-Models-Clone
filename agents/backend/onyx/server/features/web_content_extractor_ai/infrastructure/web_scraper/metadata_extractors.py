"""
Metadata extraction helper functions for web scraping.

Refactored to consolidate functions into MetadataExtractor class.
"""

from typing import Dict, List
from bs4 import BeautifulSoup
import re


class MetadataExtractor:
    """
    Metadata extraction utilities.
    
    Responsibilities:
    - Extract meta tags
    - Extract text from elements
    - Extract attributes from elements
    - Extract prefixed meta tags (OG, Twitter)
    - Extract keywords list
    
    Single Responsibility: Handle all metadata extraction operations.
    """
    
    def extract_meta_tag(
        self,
        soup: BeautifulSoup,
        name: str,
        attribute: str = "name",
        default: str = ""
    ) -> str:
        """
        Extract a single meta tag value.
        
        Args:
            soup: BeautifulSoup object
            name: Meta tag name or property
            attribute: Attribute to search by ("name" or "property")
            default: Default value if not found
        
        Returns:
            Meta tag content or default
        """
        tag = soup.find('meta', attrs={attribute: name})
        if tag:
            return tag.get('content', default)
        return default
    
    def extract_text_from_element(
        self,
        soup: BeautifulSoup,
        selector: str,
        default: str = ""
    ) -> str:
        """
        Extract text from an element using CSS selector.
        
        Args:
            soup: BeautifulSoup object
            selector: CSS selector
            default: Default value if not found
        
        Returns:
            Element text or default
        """
        element = soup.select_one(selector)
        if element:
            return element.get_text(strip=True)
        return default
    
    def extract_attribute_from_element(
        self,
        soup: BeautifulSoup,
        selector: str,
        attribute: str,
        default: str = ""
    ) -> str:
        """
        Extract attribute value from an element.
        
        Args:
            soup: BeautifulSoup object
            selector: CSS selector
            attribute: Attribute name to extract
            default: Default value if not found
        
        Returns:
            Attribute value or default
        """
        element = soup.select_one(selector)
        if element:
            return element.get(attribute, default)
        return default
    
    def extract_prefixed_meta_tags(
        self,
        soup: BeautifulSoup,
        prefix: str,
        attribute: str = "property"
    ) -> Dict[str, str]:
        """
        Extract meta tags with a specific prefix (e.g., "og:", "twitter:").
        
        Args:
            soup: BeautifulSoup object
            prefix: Prefix to match (e.g., "og:", "twitter:")
            attribute: Attribute to search by ("name" or "property")
        
        Returns:
            Dictionary of extracted tags (without prefix)
        """
        tags = {}
        pattern = re.compile(f'^{re.escape(prefix)}')
        meta_tags = soup.find_all('meta', attrs={attribute: pattern})
        
        for tag in meta_tags:
            key = tag.get(attribute, '').replace(prefix, '')
            content = tag.get('content', '')
            if key and content:
                tags[key] = content
        
        return tags
    
    def extract_keywords_list(
        self,
        soup: BeautifulSoup,
        default: List[str] = None
    ) -> List[str]:
        """
        Extract keywords from meta tag and split into list.
        
        Args:
            soup: BeautifulSoup object
            default: Default value if not found
        
        Returns:
            List of keywords
        """
        if default is None:
            default = []
        
        keywords_str = self.extract_meta_tag(soup, "keywords", "name")
        if keywords_str:
            return [k.strip() for k in keywords_str.split(',') if k.strip()]
        return default


# Backward compatibility functions
def extract_meta_tag(
    soup: BeautifulSoup,
    name: str,
    attribute: str = "name",
    default: str = ""
) -> str:
    """
    Extract a single meta tag (backward compatibility).
    
    Args:
        soup: BeautifulSoup object
        name: Meta tag name or property
        attribute: Attribute to search by ("name" or "property")
        default: Default value if not found
    
    Returns:
        Meta tag content or default
    """
    extractor = MetadataExtractor()
    return extractor.extract_meta_tag(soup, name, attribute, default)


def extract_text_from_element(
    soup: BeautifulSoup,
    selector: str,
    default: str = ""
) -> str:
    """
    Extract text from element (backward compatibility).
    
    Args:
        soup: BeautifulSoup object
        selector: CSS selector
        default: Default value if not found
    
    Returns:
        Element text or default
    """
    extractor = MetadataExtractor()
    return extractor.extract_text_from_element(soup, selector, default)


def extract_attribute_from_element(
    soup: BeautifulSoup,
    selector: str,
    attribute: str,
    default: str = ""
) -> str:
    """
    Extract attribute from element (backward compatibility).
    
    Args:
        soup: BeautifulSoup object
        selector: CSS selector
        attribute: Attribute name to extract
        default: Default value if not found
    
    Returns:
        Attribute value or default
    """
    extractor = MetadataExtractor()
    return extractor.extract_attribute_from_element(soup, selector, attribute, default)


def extract_prefixed_meta_tags(
    soup: BeautifulSoup,
    prefix: str,
    attribute: str = "property"
) -> Dict[str, str]:
    """
    Extract prefixed meta tags (backward compatibility).
    
    Args:
        soup: BeautifulSoup object
        prefix: Prefix to match (e.g., "og:", "twitter:")
        attribute: Attribute to search by ("name" or "property")
    
    Returns:
        Dictionary of extracted tags
    """
    extractor = MetadataExtractor()
    return extractor.extract_prefixed_meta_tags(soup, prefix, attribute)


def extract_keywords_list(
    soup: BeautifulSoup,
    default: List[str] = None
) -> List[str]:
    """
    Extract keywords list (backward compatibility).
    
    Args:
        soup: BeautifulSoup object
        default: Default value if not found
    
    Returns:
        List of keywords
    """
    extractor = MetadataExtractor()
    return extractor.extract_keywords_list(soup, default)
