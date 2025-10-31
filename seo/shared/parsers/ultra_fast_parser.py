from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
from selectolax.parser import HTMLParser
import lxml.etree as etree
from lxml import html
import orjson
import structlog
        import time
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Ultra-Fast HTML Parser v10
Maximum Performance with Fastest Libraries
"""


# Ultra-fast imports

logger = structlog.get_logger(__name__)


@dataclass
class ParsedElement:
    """Ultra-optimized parsed element"""
    tag: str
    text: str
    attributes: Dict[str, str]
    xpath: str
    css_selector: str


@dataclass
class SEOData:
    """Ultra-optimized SEO data structure"""
    title: str
    description: str
    keywords: List[str]
    h1_tags: List[str]
    h2_tags: List[str]
    h3_tags: List[str]
    images: List[Dict[str, str]]
    links: List[Dict[str, str]]
    meta_tags: Dict[str, str]
    structured_data: List[Dict[str, Any]]
    canonical_url: str
    robots: str
    language: str
    charset: str
    word_count: int
    load_time: float


class UltraFastHTMLParser:
    """Ultra-fast HTML parser with maximum performance optimizations"""
    
    def __init__(self, enable_cache: bool = True):
        
    """__init__ function."""
self.enable_cache = enable_cache
        self.cache = {}
        
        # Pre-compiled regex patterns for maximum performance
        self.title_pattern = re.compile(r'<title[^>]*>(.*?)</title>', re.IGNORECASE | re.DOTALL)
        self.meta_pattern = re.compile(r'<meta[^>]+>', re.IGNORECASE)
        self.link_pattern = re.compile(r'<link[^>]+>', re.IGNORECASE)
        self.script_pattern = re.compile(r'<script[^>]*>(.*?)</script>', re.IGNORECASE | re.DOTALL)
        self.json_ld_pattern = re.compile(r'application/ld\+json')
        
        # Performance metrics
        self.parse_count = 0
        self.total_time = 0.0
        self.avg_parse_time = 0.0
    
    def parse_html(self, html_content: str, base_url: str = "") -> SEOData:
        """Parse HTML with maximum performance"""
        start_time = time.time()
        
        # Use selectolax for maximum speed
        parser = HTMLParser(html_content)
        
        # Extract basic SEO elements
        title = self._extract_title(parser)
        description = self._extract_description(parser)
        keywords = self._extract_keywords(parser)
        h1_tags = self._extract_headings(parser, "h1")
        h2_tags = self._extract_headings(parser, "h2")
        h3_tags = self._extract_headings(parser, "h3")
        images = self._extract_images(parser, base_url)
        links = self._extract_links(parser, base_url)
        meta_tags = self._extract_meta_tags(parser)
        structured_data = self._extract_structured_data(parser)
        canonical_url = self._extract_canonical(parser, base_url)
        robots = self._extract_robots(parser)
        language = self._extract_language(parser)
        charset = self._extract_charset(parser)
        word_count = self._count_words(parser)
        
        elapsed = time.time() - start_time
        
        seo_data = SEOData(
            title=title,
            description=description,
            keywords=keywords,
            h1_tags=h1_tags,
            h2_tags=h2_tags,
            h3_tags=h3_tags,
            images=images,
            links=links,
            meta_tags=meta_tags,
            structured_data=structured_data,
            canonical_url=canonical_url,
            robots=robots,
            language=language,
            charset=charset,
            word_count=word_count,
            load_time=elapsed
        )
        
        # Update metrics
        self._update_metrics(elapsed)
        
        logger.debug(
            "HTML parsing completed",
            title_length=len(title),
            description_length=len(description),
            h1_count=len(h1_tags),
            h2_count=len(h2_tags),
            h3_count=len(h3_tags),
            image_count=len(images),
            link_count=len(links),
            word_count=word_count,
            elapsed=elapsed
        )
        
        return seo_data
    
    def _extract_title(self, parser: HTMLParser) -> str:
        """Extract title with maximum performance"""
        title_elem = parser.css_first("title")
        if title_elem:
            return title_elem.text().strip()
        
        # Fallback to regex for maximum speed
        match = self.title_pattern.search(parser.html)
        if match:
            return re.sub(r'<[^>]+>', '', match.group(1)).strip()
        
        return ""
    
    def _extract_description(self, parser: HTMLParser) -> str:
        """Extract meta description with maximum performance"""
        desc_elem = parser.css_first('meta[name="description"]')
        if desc_elem:
            return desc_elem.attributes.get("content", "").strip()
        
        # Fallback to regex
        desc_match = re.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']', parser.html, re.IGNORECASE)
        if desc_match:
            return desc_match.group(1).strip()
        
        return ""
    
    def _extract_keywords(self, parser: HTMLParser) -> List[str]:
        """Extract meta keywords with maximum performance"""
        keywords_elem = parser.css_first('meta[name="keywords"]')
        if keywords_elem:
            keywords_str = keywords_elem.attributes.get("content", "")
            return [kw.strip() for kw in keywords_str.split(",") if kw.strip()]
        
        return []
    
    def _extract_headings(self, parser: HTMLParser, tag: str) -> List[str]:
        """Extract headings with maximum performance"""
        headings = parser.css(tag)
        return [h.text().strip() for h in headings if h.text().strip()]
    
    def _extract_images(self, parser: HTMLParser, base_url: str) -> List[Dict[str, str]]:
        """Extract images with maximum performance"""
        images = []
        img_elements = parser.css("img")
        
        for img in img_elements:
            attrs = img.attributes
            src = attrs.get("src", "")
            alt = attrs.get("alt", "")
            title = attrs.get("title", "")
            
            if src:
                # Resolve relative URLs
                if base_url and not src.startswith(("http://", "https://")):
                    src = urljoin(base_url, src)
                
                images.append({
                    "src": src,
                    "alt": alt,
                    "title": title,
                    "width": attrs.get("width", ""),
                    "height": attrs.get("height", "")
                })
        
        return images
    
    def _extract_links(self, parser: HTMLParser, base_url: str) -> List[Dict[str, str]]:
        """Extract links with maximum performance"""
        links = []
        link_elements = parser.css("a")
        
        for link in link_elements:
            attrs = link.attributes
            href = attrs.get("href", "")
            text = link.text().strip()
            
            if href:
                # Resolve relative URLs
                if base_url and not href.startswith(("http://", "https://", "mailto:", "tel:")):
                    href = urljoin(base_url, href)
                
                links.append({
                    "href": href,
                    "text": text,
                    "title": attrs.get("title", ""),
                    "rel": attrs.get("rel", "")
                })
        
        return links
    
    def _extract_meta_tags(self, parser: HTMLParser) -> Dict[str, str]:
        """Extract all meta tags with maximum performance"""
        meta_tags = {}
        meta_elements = parser.css("meta")
        
        for meta in meta_elements:
            attrs = meta.attributes
            name = attrs.get("name", attrs.get("property", ""))
            content = attrs.get("content", "")
            
            if name and content:
                meta_tags[name.lower()] = content
        
        return meta_tags
    
    def _extract_structured_data(self, parser: HTMLParser) -> List[Dict[str, Any]]:
        """Extract structured data with maximum performance"""
        structured_data = []
        script_elements = parser.css('script[type="application/ld+json"]')
        
        for script in script_elements:
            try:
                content = script.text()
                if content:
                    data = orjson.loads(content)
                    structured_data.append(data)
            except Exception as e:
                logger.warning("Failed to parse structured data", error=str(e))
        
        return structured_data
    
    def _extract_canonical(self, parser: HTMLParser, base_url: str) -> str:
        """Extract canonical URL with maximum performance"""
        canonical_elem = parser.css_first('link[rel="canonical"]')
        if canonical_elem:
            href = canonical_elem.attributes.get("href", "")
            if href and base_url and not href.startswith(("http://", "https://")):
                href = urljoin(base_url, href)
            return href
        
        return ""
    
    def _extract_robots(self, parser: HTMLParser) -> str:
        """Extract robots meta tag with maximum performance"""
        robots_elem = parser.css_first('meta[name="robots"]')
        if robots_elem:
            return robots_elem.attributes.get("content", "")
        
        return ""
    
    def _extract_language(self, parser: HTMLParser) -> str:
        """Extract language with maximum performance"""
        # Check html lang attribute
        html_elem = parser.css_first("html")
        if html_elem:
            lang = html_elem.attributes.get("lang", "")
            if lang:
                return lang
        
        # Check meta http-equiv
        lang_elem = parser.css_first('meta[http-equiv="content-language"]')
        if lang_elem:
            return lang_elem.attributes.get("content", "")
        
        return ""
    
    def _extract_charset(self, parser: HTMLParser) -> str:
        """Extract charset with maximum performance"""
        charset_elem = parser.css_first('meta[charset]')
        if charset_elem:
            return charset_elem.attributes.get("charset", "")
        
        # Check meta http-equiv
        charset_elem = parser.css_first('meta[http-equiv="content-type"]')
        if charset_elem:
            content = charset_elem.attributes.get("content", "")
            charset_match = re.search(r'charset=([^;]+)', content, re.IGNORECASE)
            if charset_match:
                return charset_match.group(1)
        
        return "utf-8"
    
    def _count_words(self, parser: HTMLParser) -> int:
        """Count words with maximum performance"""
        text_elements = parser.css("p, h1, h2, h3, h4, h5, h6, li, td, th, div, span")
        word_count = 0
        
        for elem in text_elements:
            text = elem.text()
            if text:
                # Simple word counting - split by whitespace
                words = text.split()
                word_count += len(words)
        
        return word_count
    
    def extract_text_content(self, html_content: str) -> str:
        """Extract clean text content with maximum performance"""
        parser = HTMLParser(html_content)
        
        # Remove script and style elements
        for script in parser.css("script, style"):
            script.decompose()
        
        # Get text content
        text = parser.text()
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_links_by_domain(self, html_content: str, base_url: str) -> Dict[str, List[str]]:
        """Extract links grouped by domain with maximum performance"""
        parser = HTMLParser(html_content)
        links_by_domain = {}
        
        link_elements = parser.css("a")
        
        for link in link_elements:
            href = link.attributes.get("href", "")
            if href and href.startswith(("http://", "https://")):
                try:
                    domain = urlparse(href).netloc
                    if domain not in links_by_domain:
                        links_by_domain[domain] = []
                    links_by_domain[domain].append(href)
                except Exception:
                    continue
        
        return links_by_domain
    
    def validate_html(self, html_content: str) -> Dict[str, Any]:
        """Validate HTML structure with maximum performance"""
        try:
            # Use lxml for validation
            doc = html.fromstring(html_content)
            
            # Check for common issues
            issues = []
            
            # Check for missing title
            if not doc.xpath("//title"):
                issues.append("Missing title tag")
            
            # Check for missing meta description
            if not doc.xpath('//meta[@name="description"]'):
                issues.append("Missing meta description")
            
            # Check for missing h1
            if not doc.xpath("//h1"):
                issues.append("Missing h1 tag")
            
            # Check for images without alt
            images_without_alt = doc.xpath("//img[not(@alt)]")
            if images_without_alt:
                issues.append(f"{len(images_without_alt)} images without alt text")
            
            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "issue_count": len(issues)
            }
            
        except Exception as e:
            return {
                "valid": False,
                "issues": [f"HTML parsing error: {str(e)}"],
                "issue_count": 1
            }
    
    def _update_metrics(self, elapsed: float):
        """Update performance metrics"""
        self.parse_count += 1
        self.total_time += elapsed
        self.avg_parse_time = self.total_time / self.parse_count
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "parse_count": self.parse_count,
            "total_time": self.total_time,
            "avg_parse_time": self.avg_parse_time
        }


# Global parser instance for maximum performance
_global_parser: Optional[UltraFastHTMLParser] = None


def get_html_parser() -> UltraFastHTMLParser:
    """Get global HTML parser instance"""
    global _global_parser
    if _global_parser is None:
        _global_parser = UltraFastHTMLParser()
    return _global_parser 