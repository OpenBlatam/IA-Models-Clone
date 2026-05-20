from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import logging
from typing import Dict, Optional
from ...web_extract import WebContentExtractor, ExtractedContent
from ...plugins import BasePlugin
from datetime import datetime
from typing import Any, List, Dict, Optional
import asyncio
"""
Integrated Workflow - Extractors

Integrated extractor components that use available plugins.
"""


logger = logging.getLogger(__name__)


class IntegratedExtractor(WebContentExtractor):
    """Integrated extractor that uses available plugins."""
    
    def __init__(self, extractors: Dict[str, BasePlugin]):
        
    """__init__ function."""
super().__init__()
        self.extractors = extractors
        self.last_used = None
        self.extraction_stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'extractor_usage': {}
        }
    
    async def extract(self, url: str) -> Optional[ExtractedContent]:
        """Extract content using available plugins."""
        self.extraction_stats['total_extractions'] += 1
        
        # Try each extractor in order of preference
        for name, extractor in self.extractors.items():
            try:
                if hasattr(extractor, 'extract_content'):
                    logger.info(f"🔍 Trying extractor: {name}")
                    result = await extractor.extract_content(url)
                    
                    if result:
                        self.last_used = name
                        self.extraction_stats['successful_extractions'] += 1
                        self.extraction_stats['extractor_usage'][name] = \
                            self.extraction_stats['extractor_usage'].get(name, 0) + 1
                        
                        logger.info(f"✅ Content extracted successfully using: {name}")
                        return result
                        
            except Exception as e:
                logger.warning(f"❌ Extractor {name} failed: {e}")
                self.extraction_stats['failed_extractions'] += 1
                continue
        
        # Fallback to default extraction
        logger.info("🔄 Falling back to default extraction")
        try:
            result = await super().extract(url)
            if result:
                self.extraction_stats['successful_extractions'] += 1
                self.extraction_stats['extractor_usage']['default'] = \
                    self.extraction_stats['extractor_usage'].get('default', 0) + 1
                return result
        except Exception as e:
            logger.error(f"❌ Default extraction also failed: {e}")
            self.extraction_stats['failed_extractions'] += 1
        
        return None
    
    def get_last_used_extractor(self) -> Optional[str]:
        """Get the name of the last used extractor."""
        return self.last_used
    
    def get_extraction_stats(self) -> Dict:
        """Get extraction statistics."""
        return self.extraction_stats.copy()
    
    def get_available_extractors(self) -> list[str]:
        """Get list of available extractors."""
        return list(self.extractors.keys())
    
    def get_extractor_info(self, name: str) -> Optional[Dict]:
        """Get information about a specific extractor."""
        if name in self.extractors:
            extractor = self.extractors[name]
            metadata = extractor.get_metadata()
            return {
                'name': name,
                'version': metadata.version,
                'description': metadata.description,
                'author': metadata.author,
                'category': metadata.category,
                'usage_count': self.extraction_stats['extractor_usage'].get(name, 0)
            }
        return None


class FallbackExtractor(WebContentExtractor):
    """Fallback extractor for when plugins are not available."""
    
    def __init__(self) -> Any:
        super().__init__()
        self.fallback_methods = [
            self._extract_with_basic_parser,
            self._extract_with_metadata,
            self._extract_with_text_only
        ]
    
    async def extract(self, url: str) -> Optional[ExtractedContent]:
        """Extract content using fallback methods."""
        for method in self.fallback_methods:
            try:
                result = await method(url)
                if result and result.text.strip():
                    logger.info(f"✅ Fallback extraction successful with {method.__name__}")
                    return result
            except Exception as e:
                logger.warning(f"❌ Fallback method {method.__name__} failed: {e}")
                continue
        
        return None
    
    async def _extract_with_basic_parser(self, url: str) -> Optional[ExtractedContent]:
        """Basic content extraction."""
        # Implementation would use basic HTML parsing
        return await super().extract(url)
    
    async def _extract_with_metadata(self, url: str) -> Optional[ExtractedContent]:
        """Extract content with metadata focus."""
        # Implementation would focus on metadata extraction
        return await super().extract(url)
    
    async def _extract_with_text_only(self, url: str) -> Optional[ExtractedContent]:
        """Extract only text content."""
        # Implementation would extract only text
        return await super().extract(url)


class ExtractorManager:
    """Manager for extractor plugins."""
    
    def __init__(self) -> Any:
        self.extractors: Dict[str, BasePlugin] = {}
        self.extractor_priorities: Dict[str, int] = {}
        self.extraction_history: list[Dict] = []
    
    def add_extractor(self, name: str, extractor: BasePlugin, priority: int = 0):
        """Add an extractor with priority."""
        self.extractors[name] = extractor
        self.extractor_priorities[name] = priority
        logger.info(f"📥 Added extractor: {name} (priority: {priority})")
    
    def remove_extractor(self, name: str):
        """Remove an extractor."""
        if name in self.extractors:
            del self.extractors[name]
            del self.extractor_priorities[name]
            logger.info(f"🗑️ Removed extractor: {name}")
    
    def get_prioritized_extractors(self) -> Dict[str, BasePlugin]:
        """Get extractors sorted by priority."""
        sorted_extractors = sorted(
            self.extractors.items(),
            key=lambda x: self.extractor_priorities.get(x[0], 0),
            reverse=True
        )
        return dict(sorted_extractors)
    
    def update_priority(self, name: str, priority: int):
        """Update extractor priority."""
        if name in self.extractors:
            self.extractor_priorities[name] = priority
            logger.info(f"🔄 Updated priority for {name}: {priority}")
    
    def get_extractor_stats(self) -> Dict:
        """Get extractor statistics."""
        return {
            'total_extractors': len(self.extractors),
            'priorities': self.extractor_priorities.copy(),
            'extraction_history': self.extraction_history[-10:],  # Last 10 extractions
            'extractors': list(self.extractors.keys())
        }
    
    def record_extraction(self, extractor_name: str, url: str, success: bool, duration: float):
        """Record extraction attempt."""
        self.extraction_history.append({
            'extractor': extractor_name,
            'url': url,
            'success': success,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 100 records
        if len(self.extraction_history) > 100:
            self.extraction_history = self.extraction_history[-100:] 