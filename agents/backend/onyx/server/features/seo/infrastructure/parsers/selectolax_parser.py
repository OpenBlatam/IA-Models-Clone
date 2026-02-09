from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import selectolax
from shared.core.exceptions import HTMLParsingError
from shared.core.logging import get_logger
from domain.value_objects.meta_tags import MetaTags
from domain.value_objects.parsed_data import ParsedData
                    import json
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Selectolax HTML Parser
Ultra-fast HTML parsing with maximum performance
"""



logger = get_logger(__name__)


@dataclass
class ParsedHTMLData:
    """Parsed HTML data structure"""
    title: Optional[str]
    description: Optional[str]
    keywords: Optional[str]
    meta_tags: MetaTags
    links: List[str]
    headings: List[str]
    images: List[str]
    scripts: List[str]
    stylesheets: List[str]
    processing_time: float
    content_length: int


class SelectolaxParser:
    """
    Ultra-fast HTML parser using selectolax
    
    This parser provides maximum performance for HTML parsing
    with advanced features and error handling.
    """
    
    def __init__(self) -> Any:
        """Initialize parser"""
        self.parser = selectolax.HTMLParser
        self._parse_count = 0
        self._total_parse_time = 0.0
        self._error_count = 0
    
    def parse(self, html_content: str) -> ParsedHTMLData:
        """
        Parse HTML content with maximum performance
        
        Args:
            html_content: HTML content to parse
            
        Returns:
            ParsedHTMLData: Parsed HTML data
            
        Raises:
            HTMLParsingError: If parsing fails
        """
        start_time = time.time()
        
        try:
            # Parse HTML with selectolax
            tree = self.parser(html_content)
            
            # Extract data efficiently
            parsed_data = self._extract_data(tree, html_content)
            
            # Update metrics
            self._parse_count += 1
            parse_time = time.time() - start_time
            self._total_parse_time += parse_time
            
            logger.debug(
                "HTML parsing completed",
                content_length=len(html_content),
                parse_time=parse_time,
                links_count=len(parsed_data.links),
                meta_tags_count=len(parsed_data.meta_tags.tags)
            )
            
            return parsed_data
            
        except Exception as e:
            self._error_count += 1
            logger.error("HTML parsing failed", error=str(e), content_length=len(html_content))
            raise HTMLParsingError(f"Failed to parse HTML: {str(e)}")
    
    def _extract_data(self, tree, html_content: str) -> ParsedHTMLData:
        """
        Extract data from parsed HTML tree
        
        Args:
            tree: Parsed HTML tree
            html_content: Original HTML content
            
        Returns:
            ParsedHTMLData: Extracted data
        """
        # Extract title
        title_elem = tree.css_first('title')
        title = title_elem.text() if title_elem else None
        
        # Extract meta tags efficiently
        meta_tags = self._extract_meta_tags(tree)
        
        # Extract links efficiently
        links = self._extract_links(tree)
        
        # Extract headings
        headings = self._extract_headings(tree)
        
        # Extract images
        images = self._extract_images(tree)
        
        # Extract scripts
        scripts = self._extract_scripts(tree)
        
        # Extract stylesheets
        stylesheets = self._extract_stylesheets(tree)
        
        # Get description and keywords from meta tags
        description = meta_tags.get('description') or meta_tags.get('og:description')
        keywords = meta_tags.get('keywords')
        
        return ParsedHTMLData(
            title=title,
            description=description,
            keywords=keywords,
            meta_tags=MetaTags(meta_tags),
            links=links,
            headings=headings,
            images=images,
            scripts=scripts,
            stylesheets=stylesheets,
            processing_time=time.time(),
            content_length=len(html_content)
        )
    
    def _extract_meta_tags(self, tree) -> Dict[str, str]:
        """
        Extract meta tags efficiently
        
        Args:
            tree: Parsed HTML tree
            
        Returns:
            Dict[str, str]: Meta tags
        """
        meta_tags = {}
        
        for meta in tree.css('meta'):
            name = meta.attributes.get('name') or meta.attributes.get('property')
            content = meta.attributes.get('content')
            
            if name and content:
                meta_tags[name.lower()] = content
        
        return meta_tags
    
    def _extract_links(self, tree) -> List[str]:
        """
        Extract links efficiently
        
        Args:
            tree: Parsed HTML tree
            
        Returns:
            List[str]: Links
        """
        links = []
        
        for link in tree.css('a'):
            href = link.attributes.get('href')
            if href:
                links.append(href)
        
        return links
    
    def _extract_headings(self, tree) -> List[str]:
        """
        Extract headings efficiently
        
        Args:
            tree: Parsed HTML tree
            
        Returns:
            List[str]: Headings
        """
        headings = []
        
        for heading in tree.css('h1, h2, h3, h4, h5, h6'):
            text = heading.text()
            if text:
                headings.append(text)
        
        return headings
    
    def _extract_images(self, tree) -> List[str]:
        """
        Extract images efficiently
        
        Args:
            tree: Parsed HTML tree
            
        Returns:
            List[str]: Image URLs
        """
        images = []
        
        for img in tree.css('img'):
            src = img.attributes.get('src')
            if src:
                images.append(src)
        
        return images
    
    def _extract_scripts(self, tree) -> List[str]:
        """
        Extract scripts efficiently
        
        Args:
            tree: Parsed HTML tree
            
        Returns:
            List[str]: Script URLs
        """
        scripts = []
        
        for script in tree.css('script'):
            src = script.attributes.get('src')
            if src:
                scripts.append(src)
        
        return scripts
    
    def _extract_stylesheets(self, tree) -> List[str]:
        """
        Extract stylesheets efficiently
        
        Args:
            tree: Parsed HTML tree
            
        Returns:
            List[str]: Stylesheet URLs
        """
        stylesheets = []
        
        for link in tree.css('link[rel="stylesheet"]'):
            href = link.attributes.get('href')
            if href:
                stylesheets.append(href)
        
        return stylesheets
    
    def parse_for_seo(self, html_content: str) -> ParsedData:
        """
        Parse HTML specifically for SEO analysis
        
        Args:
            html_content: HTML content to parse
            
        Returns:
            ParsedData: SEO-specific parsed data
        """
        parsed_html = self.parse(html_content)
        
        return ParsedData(
            title=parsed_html.title,
            description=parsed_html.description,
            keywords=parsed_html.keywords,
            meta_tags=parsed_html.meta_tags,
            links=parsed_html.links,
            processing_time=parsed_html.processing_time
        )
    
    def extract_text_content(self, html_content: str) -> str:
        """
        Extract text content from HTML
        
        Args:
            html_content: HTML content
            
        Returns:
            str: Extracted text content
        """
        try:
            tree = self.parser(html_content)
            
            # Remove script and style elements
            for script in tree.css('script'):
                script.decompose()
            for style in tree.css('style'):
                style.decompose()
            
            # Get text content
            text = tree.text()
            
            # Clean up whitespace
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            return ' '.join(lines)
            
        except Exception as e:
            logger.warning("Failed to extract text content", error=str(e))
            return ""
    
    def extract_structured_data(self, html_content: str) -> List[Dict[str, Any]]:
        """
        Extract structured data (JSON-LD, Microdata, RDFa)
        
        Args:
            html_content: HTML content
            
        Returns:
            List[Dict[str, Any]]: Structured data
        """
        try:
            tree = self.parser(html_content)
            structured_data = []
            
            # Extract JSON-LD
            for script in tree.css('script[type="application/ld+json"]'):
                try:
                    data = json.loads(script.text())
                    structured_data.append({
                        "type": "json-ld",
                        "data": data
                    })
                except Exception as e:
                    logger.warning("Failed to parse JSON-LD", error=str(e))
            
            # Extract Microdata
            for item in tree.css('[itemtype]'):
                microdata = self._extract_microdata(item)
                if microdata:
                    structured_data.append({
                        "type": "microdata",
                        "data": microdata
                    })
            
            return structured_data
            
        except Exception as e:
            logger.warning("Failed to extract structured data", error=str(e))
            return []
    
    def _extract_microdata(self, element) -> Optional[Dict[str, Any]]:
        """
        Extract microdata from element
        
        Args:
            element: HTML element
            
        Returns:
            Optional[Dict[str, Any]]: Microdata
        """
        try:
            itemtype = element.attributes.get('itemtype')
            if not itemtype:
                return None
            
            microdata = {
                "itemtype": itemtype,
                "properties": {}
            }
            
            # Extract properties
            for prop in element.css('[itemprop]'):
                prop_name = prop.attributes.get('itemprop')
                prop_value = prop.attributes.get('content') or prop.text()
                
                if prop_name and prop_value:
                    microdata["properties"][prop_name] = prop_value
            
            return microdata
            
        except Exception as e:
            logger.warning("Failed to extract microdata", error=str(e))
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get parser metrics
        
        Returns:
            Dict[str, Any]: Parser metrics
        """
        avg_parse_time = (
            self._total_parse_time / self._parse_count 
            if self._parse_count > 0 else 0.0
        )
        
        error_rate = (
            self._error_count / self._parse_count * 100 
            if self._parse_count > 0 else 0.0
        )
        
        return {
            "parse_count": self._parse_count,
            "error_count": self._error_count,
            "success_rate": 100.0 - error_rate,
            "error_rate": error_rate,
            "average_parse_time": avg_parse_time,
            "total_parse_time": self._total_parse_time
        }
    
    def reset_metrics(self) -> None:
        """Reset parser metrics"""
        self._parse_count = 0
        self._error_count = 0
        self._total_parse_time = 0.0
        logger.info("HTML parser metrics reset")
    
    def is_healthy(self) -> bool:
        """
        Check if parser is healthy
        
        Returns:
            bool: True if healthy
        """
        return self._error_count < 100  # Allow some errors 