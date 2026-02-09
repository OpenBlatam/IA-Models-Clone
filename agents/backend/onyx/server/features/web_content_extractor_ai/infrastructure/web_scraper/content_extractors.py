"""
Content extraction utilities for web scraping.

Refactored to consolidate content extraction methods into specialized classes.
"""

from typing import List, Dict, Any
from urllib.parse import urljoin
from bs4 import BeautifulSoup


class TableExtractor:
    """
    Table extraction utilities.
    
    Single Responsibility: Extract tables from HTML.
    """
    
    def extract(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract tables from HTML.
        
        Args:
            soup: BeautifulSoup object
        
        Returns:
            List of table dictionaries
        """
        tables = []
        for table in soup.find_all('table'):
            try:
                rows = []
                headers = []
                
                # Extract headers
                thead = table.find('thead')
                if thead:
                    header_row = thead.find('tr')
                    if header_row:
                        headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
                
                # Extract rows
                tbody = table.find('tbody') or table
                for tr in tbody.find_all('tr'):
                    cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                    if cells:
                        rows.append(cells)
                
                if rows:
                    tables.append({
                        "headers": headers if headers else None,
                        "rows": rows,
                        "row_count": len(rows),
                        "column_count": len(headers) if headers else len(rows[0]) if rows else 0
                    })
            except Exception:
                continue
        
        return tables


class VideoExtractor:
    """
    Video extraction utilities.
    
    Single Responsibility: Extract videos from HTML.
    """
    
    def __init__(self, base_url: str = None):
        """
        Initialize video extractor.
        
        Args:
            base_url: Base URL for normalizing relative URLs
        """
        self.base_url = base_url
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL."""
        if self.base_url and url:
            return urljoin(self.base_url, url)
        return url
    
    def extract(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract videos from HTML.
        
        Args:
            soup: BeautifulSoup object
        
        Returns:
            List of video dictionaries
        """
        videos = []
        
        # HTML5 video tags
        for video in soup.find_all('video'):
            src = video.get('src', '')
            if not src:
                source = video.find('source')
                if source:
                    src = source.get('src', '')
            
            if src:
                videos.append({
                    "type": "html5",
                    "src": self._normalize_url(src),
                    "poster": self._normalize_url(video.get('poster', '')) if video.get('poster') else None,
                    "width": video.get('width'),
                    "height": video.get('height'),
                    "duration": video.get('duration')
                })
        
        # iframes (YouTube, Vimeo, etc.)
        for iframe in soup.find_all('iframe'):
            src = iframe.get('src', '')
            if src and any(domain in src.lower() for domain in ['youtube', 'vimeo', 'dailymotion', 'twitch']):
                videos.append({
                    "type": "embed",
                    "src": src,
                    "width": iframe.get('width'),
                    "height": iframe.get('height')
                })
        
        # Open Graph video
        og_video = soup.find('meta', property='og:video')
        if og_video:
            videos.append({
                "type": "og_video",
                "src": og_video.get('content', '')
            })
        
        return videos


class QuoteExtractor:
    """
    Quote extraction utilities.
    
    Single Responsibility: Extract quotes from HTML.
    """
    
    def extract(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract quotes from HTML.
        
        Args:
            soup: BeautifulSoup object
        
        Returns:
            List of quote dictionaries
        """
        quotes = []
        
        # Blockquotes
        for blockquote in soup.find_all('blockquote'):
            text = blockquote.get_text(strip=True)
            cite = blockquote.get('cite', '')
            author = None
            
            # Find author in footer or cite
            footer = blockquote.find('footer')
            if footer:
                author = footer.get_text(strip=True)
            cite_tag = blockquote.find('cite')
            if cite_tag:
                author = cite_tag.get_text(strip=True)
            
            if text:
                quotes.append({
                    "text": text,
                    "author": author,
                    "cite": cite
                })
        
        # Inline quotes with <q>
        for q in soup.find_all('q'):
            text = q.get_text(strip=True)
            cite = q.get('cite', '')
            if text:
                quotes.append({
                    "text": text,
                    "cite": cite,
                    "type": "inline"
                })
        
        return quotes


class CodeBlockExtractor:
    """
    Code block extraction utilities.
    
    Single Responsibility: Extract code blocks from HTML.
    """
    
    def extract(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract code blocks from HTML.
        
        Args:
            soup: BeautifulSoup object
        
        Returns:
            List of code block dictionaries
        """
        code_blocks = []
        
        for pre in soup.find_all('pre'):
            code = pre.find('code')
            if code:
                language = code.get('class', [])
                if language:
                    # Extract language from class like 'language-python'
                    lang = [cls.replace('language-', '') for cls in language if 'language-' in cls]
                    language = lang[0] if lang else None
                else:
                    language = None
                
                code_blocks.append({
                    "code": code.get_text(),
                    "language": language,
                    "line_count": len(code.get_text().split('\n'))
                })
        
        return code_blocks


class FormExtractor:
    """
    Form extraction utilities.
    
    Single Responsibility: Extract forms from HTML.
    """
    
    def extract(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract forms from HTML.
        
        Args:
            soup: BeautifulSoup object
        
        Returns:
            List of form dictionaries
        """
        forms = []
        
        for form in soup.find_all('form'):
            form_data = {
                "action": form.get('action', ''),
                "method": form.get('method', 'get').upper(),
                "fields": []
            }
            
            # Extract fields
            for input_field in form.find_all(['input', 'textarea', 'select']):
                field_data = {
                    "type": input_field.get('type', input_field.name),
                    "name": input_field.get('name', ''),
                    "label": ""
                }
                
                # Find associated label
                field_id = input_field.get('id', '')
                if field_id:
                    label = soup.find('label', {'for': field_id})
                    if label:
                        field_data["label"] = label.get_text(strip=True)
                
                # Parent label
                if not field_data["label"]:
                    parent_label = input_field.find_parent('label')
                    if parent_label:
                        field_data["label"] = parent_label.get_text(strip=True)
                
                form_data["fields"].append(field_data)
            
            if form_data["fields"]:
                forms.append(form_data)
        
        return forms


class FeedExtractor:
    """
    Feed extraction utilities.
    
    Single Responsibility: Extract RSS/Atom feeds from HTML.
    """
    
    def __init__(self, base_url: str = None):
        """
        Initialize feed extractor.
        
        Args:
            base_url: Base URL for normalizing relative URLs
        """
        self.base_url = base_url
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL."""
        if self.base_url and url:
            return urljoin(self.base_url, url)
        return url
    
    def extract(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract RSS/Atom feeds from HTML.
        
        Args:
            soup: BeautifulSoup object
        
        Returns:
            List of feed dictionaries
        """
        feeds = []
        
        # Find feed links
        for link in soup.find_all('link', type=['application/rss+xml', 'application/atom+xml', 'application/xml']):
            href = link.get('href', '')
            if href:
                feeds.append({
                    "type": link.get('type', ''),
                    "title": link.get('title', ''),
                    "url": self._normalize_url(href)
                })
        
        return feeds

