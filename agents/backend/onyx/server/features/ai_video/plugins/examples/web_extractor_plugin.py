from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlparse
import aiohttp
from bs4 import BeautifulSoup
import trafilatura
from newspaper import Article
import json
from ..base import BasePlugin, PluginMetadata
from ...core.types import ExtractedContent, ContentType
from ...core.exceptions import PluginError, ValidationError
    from ...plugins import PluginManager, ManagerConfig
from typing import Any, List, Dict, Optional
"""
Web Extractor Plugin - Example Plugin Implementation

This plugin demonstrates how to create a production-ready plugin that:
- Extracts content from web URLs
- Handles multiple content types
- Provides configuration options
- Includes error handling and retry logic
- Monitors performance
- Integrates with the plugin system
"""



logger = logging.getLogger(__name__)


class WebExtractorPlugin(BasePlugin):
    """
    Advanced web content extractor plugin.
    
    Features:
    - Multiple extraction methods (newspaper3k, trafilatura, BeautifulSoup)
    - Automatic content type detection
    - Configurable timeouts and retries
    - Performance monitoring
    - Error handling and recovery
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
super().__init__(config)
        
        # Plugin metadata
        self.name = "web_extractor"
        self.version = "1.0.0"
        self.description = "Advanced web content extractor with multiple extraction methods"
        self.author = "AI Video Team"
        self.category = "extractor"
        
        # Default configuration
        self.default_config = {
            "timeout": 30,
            "max_retries": 3,
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "extraction_methods": ["newspaper3k", "trafilatura", "beautifulsoup"],
            "enable_metadata": True,
            "enable_images": False,
            "max_content_length": 50000,
            "language_detection": True,
            "clean_html": True
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **(config or {})}
        
        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "total_extraction_time": 0.0,
            "average_extraction_time": 0.0
        }
        
        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"WebExtractorPlugin initialized with config: {self.config}")
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name=self.name,
            version=self.version,
            description=self.description,
            author=self.author,
            category=self.category,
            dependencies={
                "aiohttp": ">=3.8.0",
                "beautifulsoup4": ">=4.9.0",
                "newspaper3k": ">=0.2.8",
                "trafilatura": ">=5.0.0"
            },
            config_schema={
                "timeout": {"type": "integer", "default": 30, "min": 5, "max": 300},
                "max_retries": {"type": "integer", "default": 3, "min": 1, "max": 10},
                "user_agent": {"type": "string", "default": "Mozilla/5.0..."},
                "extraction_methods": {"type": "array", "items": {"type": "string"}},
                "enable_metadata": {"type": "boolean", "default": True},
                "enable_images": {"type": "boolean", "default": False},
                "max_content_length": {"type": "integer", "default": 50000},
                "language_detection": {"type": "boolean", "default": True},
                "clean_html": {"type": "boolean", "default": True}
            }
        )
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration."""
        try:
            # Check required fields
            if "timeout" in config and not isinstance(config["timeout"], int):
                raise ValidationError("timeout must be an integer")
            
            if "max_retries" in config and not isinstance(config["max_retries"], int):
                raise ValidationError("max_retries must be an integer")
            
            if "timeout" in config and (config["timeout"] < 5 or config["timeout"] > 300):
                raise ValidationError("timeout must be between 5 and 300 seconds")
            
            if "max_retries" in config and (config["max_retries"] < 1 or config["max_retries"] > 10):
                raise ValidationError("max_retries must be between 1 and 10")
            
            # Check extraction methods
            valid_methods = ["newspaper3k", "trafilatura", "beautifulsoup"]
            if "extraction_methods" in config:
                methods = config["extraction_methods"]
                if not isinstance(methods, list):
                    raise ValidationError("extraction_methods must be a list")
                
                for method in methods:
                    if method not in valid_methods:
                        raise ValidationError(f"Invalid extraction method: {method}")
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    async def initialize(self) -> Any:
        """Initialize the plugin."""
        logger.info("🚀 Initializing WebExtractorPlugin...")
        
        try:
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=self.config["timeout"])
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"User-Agent": self.config["user_agent"]}
            )
            
            # Validate configuration
            if not self.validate_config(self.config):
                raise ValidationError("Invalid plugin configuration")
            
            logger.info("✅ WebExtractorPlugin initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize WebExtractorPlugin: {e}")
            raise PluginError(f"Initialization failed: {e}")
    
    async def cleanup(self) -> Any:
        """Cleanup plugin resources."""
        logger.info("🧹 Cleaning up WebExtractorPlugin...")
        
        if self.session:
            await self.session.close()
            self.session = None
        
        logger.info("✅ WebExtractorPlugin cleanup complete")
    
    async def extract_content(self, url: str) -> ExtractedContent:
        """
        Extract content from a web URL.
        
        Args:
            url: URL to extract content from
            
        Returns:
            Extracted content with metadata
        """
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        logger.info(f"🔍 Extracting content from: {url}")
        
        try:
            # Validate URL
            if not self._is_valid_url(url):
                raise ValidationError(f"Invalid URL: {url}")
            
            # Fetch HTML content
            html_content = await self._fetch_html(url)
            
            # Extract content using multiple methods
            extracted_content = await self._extract_with_methods(html_content, url)
            
            # Update statistics
            extraction_time = time.time() - start_time
            self.stats["successful_extractions"] += 1
            self.stats["total_extraction_time"] += extraction_time
            self.stats["average_extraction_time"] = (
                self.stats["total_extraction_time"] / self.stats["successful_extractions"]
            )
            
            logger.info(f"✅ Content extracted successfully in {extraction_time:.2f}s")
            
            return extracted_content
            
        except Exception as e:
            self.stats["failed_extractions"] += 1
            logger.error(f"❌ Failed to extract content from {url}: {e}")
            raise PluginError(f"Content extraction failed: {e}")
    
    async async def _fetch_html(self, url: str) -> str:
        """Fetch HTML content from URL with retry logic."""
        for attempt in range(self.config["max_retries"]):
            try:
                if not self.session:
                    raise PluginError("HTTP session not initialized")
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        raise PluginError(f"HTTP {response.status}: {response.reason}")
                        
            except Exception as e:
                if attempt == self.config["max_retries"] - 1:
                    raise PluginError(f"Failed to fetch HTML after {self.config['max_retries']} attempts: {e}")
                
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
    
    async def _extract_with_methods(self, html_content: str, url: str) -> ExtractedContent:
        """Extract content using multiple methods and combine results."""
        extraction_results = {}
        
        for method in self.config["extraction_methods"]:
            try:
                if method == "newspaper3k":
                    result = await self._extract_with_newspaper(html_content, url)
                elif method == "trafilatura":
                    result = await self._extract_with_trafilatura(html_content, url)
                elif method == "beautifulsoup":
                    result = await self._extract_with_beautifulsoup(html_content, url)
                else:
                    continue
                
                if result and result.get("text"):
                    extraction_results[method] = result
                    
            except Exception as e:
                logger.warning(f"Extraction method '{method}' failed: {e}")
        
        # Combine results from different methods
        return self._combine_extraction_results(extraction_results, url)
    
    async def _extract_with_newspaper(self, html_content: str, url: str) -> Dict[str, Any]:
        """Extract content using newspaper3k."""
        try:
            article = Article(url)
            article.download(input_html=html_content)
            article.parse()
            
            return {
                "text": article.text,
                "title": article.title,
                "authors": article.authors,
                "publish_date": article.publish_date.isoformat() if article.publish_date else None,
                "top_image": article.top_image,
                "meta_description": article.meta_description,
                "meta_keywords": article.meta_keywords,
                "method": "newspaper3k"
            }
        except Exception as e:
            logger.warning(f"Newspaper3k extraction failed: {e}")
            return {}
    
    async def _extract_with_trafilatura(self, html_content: str, url: str) -> Dict[str, Any]:
        """Extract content using trafilatura."""
        try:
            extracted = trafilatura.extract(
                html_content,
                include_formatting=True,
                include_links=True,
                include_images=self.config["enable_images"],
                include_tables=True
            )
            
            metadata = trafilatura.extract_metadata(html_content)
            
            return {
                "text": extracted,
                "title": metadata.title if metadata else None,
                "authors": [metadata.author] if metadata and metadata.author else [],
                "publish_date": metadata.date.isoformat() if metadata and metadata.date else None,
                "method": "trafilatura"
            }
        except Exception as e:
            logger.warning(f"Trafilatura extraction failed: {e}")
            return {}
    
    async def _extract_with_beautifulsoup(self, html_content: str, url: str) -> Dict[str, Any]:
        """Extract content using BeautifulSoup."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Extract metadata
            title = soup.find('title')
            title_text = title.get_text() if title else None
            
            meta_description = soup.find('meta', attrs={'name': 'description'})
            description = meta_description.get('content') if meta_description else None
            
            return {
                "text": text,
                "title": title_text,
                "meta_description": description,
                "method": "beautifulsoup"
            }
        except Exception as e:
            logger.warning(f"BeautifulSoup extraction failed: {e}")
            return {}
    
    def _combine_extraction_results(self, results: Dict[str, Dict[str, Any]], url: str) -> ExtractedContent:
        """Combine results from different extraction methods."""
        if not results:
            raise PluginError("No content could be extracted from any method")
        
        # Find the best result (prefer newspaper3k, then trafilatura, then beautifulsoup)
        preferred_order = ["newspaper3k", "trafilatura", "beautifulsoup"]
        best_result = None
        
        for method in preferred_order:
            if method in results:
                best_result = results[method]
                break
        
        if not best_result:
            # Use the first available result
            best_result = next(iter(results.values()))
        
        # Create extracted content
        content = ExtractedContent(
            url=url,
            title=best_result.get("title", ""),
            text=best_result.get("text", ""),
            content_type=ContentType.ARTICLE,
            language=self._detect_language(best_result.get("text", "")),
            metadata={
                "extraction_method": best_result.get("method", "unknown"),
                "authors": best_result.get("authors", []),
                "publish_date": best_result.get("publish_date"),
                "meta_description": best_result.get("meta_description"),
                "meta_keywords": best_result.get("meta_keywords"),
                "top_image": best_result.get("top_image"),
                "extraction_stats": self.stats.copy()
            }
        )
        
        # Truncate content if too long
        if len(content.text) > self.config["max_content_length"]:
            content.text = content.text[:self.config["max_content_length"]] + "..."
            content.metadata["truncated"] = True
        
        return content
    
    def _detect_language(self, text: str) -> str:
        """Detect language of the text."""
        if not self.config["language_detection"] or not text:
            return "unknown"
        
        try:
            # Simple language detection based on common words
            # In production, you might want to use a proper language detection library
            text_lower = text.lower()
            
            # Spanish words
            spanish_words = ["el", "la", "de", "que", "y", "a", "en", "un", "es", "se", "no", "te", "lo", "le", "da", "su", "por", "son", "con", "para"]
            spanish_count = sum(1 for word in spanish_words if word in text_lower)
            
            # English words
            english_words = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at"]
            english_count = sum(1 for word in english_words if word in text_lower)
            
            if spanish_count > english_count:
                return "es"
            elif english_count > spanish_count:
                return "en"
            else:
                return "unknown"
                
        except Exception:
            return "unknown"
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get plugin statistics."""
        return {
            "plugin_name": self.name,
            "plugin_version": self.version,
            "total_requests": self.stats["total_requests"],
            "successful_extractions": self.stats["successful_extractions"],
            "failed_extractions": self.stats["failed_extractions"],
            "success_rate": (
                self.stats["successful_extractions"] / self.stats["total_requests"]
                if self.stats["total_requests"] > 0 else 0.0
            ),
            "average_extraction_time": self.stats["average_extraction_time"],
            "total_extraction_time": self.stats["total_extraction_time"]
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """Update plugin configuration."""
        try:
            # Validate new configuration
            test_config = {**self.config, **new_config}
            if not self.validate_config(test_config):
                return False
            
            # Update configuration
            self.config.update(new_config)
            logger.info(f"Configuration updated: {new_config}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False


# Example usage
async def example_usage():
    """Example of how to use the WebExtractorPlugin."""
    
    # Create plugin manager
    config = ManagerConfig(auto_discover=False)
    manager = PluginManager(config)
    await manager.start()
    
    # Create and load the plugin
    plugin_config = {
        "timeout": 30,
        "max_retries": 3,
        "extraction_methods": ["newspaper3k", "trafilatura"],
        "enable_metadata": True
    }
    
    plugin = WebExtractorPlugin(plugin_config)
    await plugin.initialize()
    
    try:
        # Extract content from a URL
        url = "https://example.com"
        content = await plugin.extract_content(url)
        
        print(f"Title: {content.title}")
        print(f"Text length: {len(content.text)} characters")
        print(f"Language: {content.language}")
        print(f"Metadata: {content.metadata}")
        
        # Get plugin statistics
        stats = plugin.get_stats()
        print(f"Plugin stats: {stats}")
        
    finally:
        await plugin.cleanup()
        await manager.shutdown()


match __name__:
    case "__main__":
    asyncio.run(example_usage()) 