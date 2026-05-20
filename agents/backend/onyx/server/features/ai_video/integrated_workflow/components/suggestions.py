from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import logging
from typing import Dict, Optional
from ...suggestions import SuggestionEngine, ContentSuggestions
from ...web_extract import ExtractedContent
from ...plugins import BasePlugin
from datetime import datetime
from typing import Any, List, Dict, Optional
import asyncio
"""
Integrated Workflow - Suggestions

Integrated suggestion engine components that use available plugins.
"""


logger = logging.getLogger(__name__)


class IntegratedSuggestionEngine(SuggestionEngine):
    """Integrated suggestion engine that uses available plugins."""
    
    def __init__(self, engines: Dict[str, BasePlugin]):
        
    """__init__ function."""
super().__init__()
        self.engines = engines
        self.last_used = None
        self.suggestion_stats = {
            'total_suggestions': 0,
            'successful_suggestions': 0,
            'failed_suggestions': 0,
            'engine_usage': {}
        }
    
    async def generate_suggestions(self, content: ExtractedContent) -> ContentSuggestions:
        """Generate suggestions using available plugins."""
        self.suggestion_stats['total_suggestions'] += 1
        
        # Use the first available suggestion engine
        for name, engine in self.engines.items():
            try:
                if hasattr(engine, 'generate_suggestions'):
                    logger.info(f"💡 Trying suggestion engine: {name}")
                    result = await engine.generate_suggestions(content)
                    
                    if result:
                        self.last_used = name
                        self.suggestion_stats['successful_suggestions'] += 1
                        self.suggestion_stats['engine_usage'][name] = \
                            self.suggestion_stats['engine_usage'].get(name, 0) + 1
                        
                        logger.info(f"✅ Suggestions generated successfully using: {name}")
                        return result
                        
            except Exception as e:
                logger.warning(f"❌ Suggestion engine {name} failed: {e}")
                self.suggestion_stats['failed_suggestions'] += 1
                continue
        
        # Fallback to default suggestions
        logger.info("🔄 Falling back to default suggestions")
        try:
            result = await super().generate_suggestions(content)
            if result:
                self.suggestion_stats['successful_suggestions'] += 1
                self.suggestion_stats['engine_usage']['default'] = \
                    self.suggestion_stats['engine_usage'].get('default', 0) + 1
                return result
        except Exception as e:
            logger.error(f"❌ Default suggestions also failed: {e}")
            self.suggestion_stats['failed_suggestions'] += 1
        
        # Return empty suggestions as last resort
        return ContentSuggestions(
            title_suggestions=[],
            description_suggestions=[],
            tag_suggestions=[],
            style_suggestions=[]
        )
    
    def get_last_used_engine(self) -> Optional[str]:
        """Get the name of the last used suggestion engine."""
        return self.last_used
    
    def get_suggestion_stats(self) -> Dict:
        """Get suggestion statistics."""
        return self.suggestion_stats.copy()
    
    def get_available_engines(self) -> list[str]:
        """Get list of available suggestion engines."""
        return list(self.engines.keys())
    
    def get_engine_info(self, name: str) -> Optional[Dict]:
        """Get information about a specific suggestion engine."""
        if name in self.engines:
            engine = self.engines[name]
            metadata = engine.get_metadata()
            return {
                'name': name,
                'version': metadata.version,
                'description': metadata.description,
                'author': metadata.author,
                'category': metadata.category,
                'usage_count': self.suggestion_stats['engine_usage'].get(name, 0)
            }
        return None


class FallbackSuggestionEngine(SuggestionEngine):
    """Fallback suggestion engine for when plugins are not available."""
    
    def __init__(self) -> Any:
        super().__init__()
        self.fallback_methods = [
            self._generate_basic_suggestions,
            self._generate_keyword_suggestions,
            self._generate_template_suggestions
        ]
    
    async def generate_suggestions(self, content: ExtractedContent) -> ContentSuggestions:
        """Generate suggestions using fallback methods."""
        for method in self.fallback_methods:
            try:
                result = await method(content)
                if result and (result.title_suggestions or result.description_suggestions):
                    logger.info(f"✅ Fallback suggestions successful with {method.__name__}")
                    return result
            except Exception as e:
                logger.warning(f"❌ Fallback method {method.__name__} failed: {e}")
                continue
        
        return await super().generate_suggestions(content)
    
    async def _generate_basic_suggestions(self, content: ExtractedContent) -> ContentSuggestions:
        """Generate basic suggestions from content."""
        title_suggestions = [content.title] if content.title else []
        description_suggestions = [content.text[:200] + "..."] if len(content.text) > 200 else [content.text]
        
        return ContentSuggestions(
            title_suggestions=title_suggestions,
            description_suggestions=description_suggestions,
            tag_suggestions=[],
            style_suggestions=[]
        )
    
    async def _generate_keyword_suggestions(self, content: ExtractedContent) -> ContentSuggestions:
        """Generate suggestions based on keywords."""
        # Simple keyword extraction
        words = content.text.lower().split()
        keywords = [word for word in words if len(word) > 3][:10]
        
        return ContentSuggestions(
            title_suggestions=[],
            description_suggestions=[],
            tag_suggestions=keywords,
            style_suggestions=[]
        )
    
    async def _generate_template_suggestions(self, content: ExtractedContent) -> ContentSuggestions:
        """Generate suggestions using templates."""
        templates = [
            "Amazing content about {topic}",
            "Discover {topic} in this video",
            "Everything you need to know about {topic}"
        ]
        
        # Extract topic from content
        topic = content.title or "this topic"f"
        
        title_suggestions = [template" for template in templates]
        
        return ContentSuggestions(
            title_suggestions=title_suggestions,
            description_suggestions=[],
            tag_suggestions=[],
            style_suggestions=[]
        )


class SuggestionEngineManager:
    """Manager for suggestion engine plugins."""
    
    def __init__(self) -> Any:
        self.engines: Dict[str, BasePlugin] = {}
        self.engine_priorities: Dict[str, int] = {}
        self.suggestion_history: list[Dict] = []
    
    def add_engine(self, name: str, engine: BasePlugin, priority: int = 0):
        """Add a suggestion engine with priority."""
        self.engines[name] = engine
        self.engine_priorities[name] = priority
        logger.info(f"💡 Added suggestion engine: {name} (priority: {priority})")
    
    def remove_engine(self, name: str):
        """Remove a suggestion engine."""
        if name in self.engines:
            del self.engines[name]
            del self.engine_priorities[name]
            logger.info(f"🗑️ Removed suggestion engine: {name}")
    
    def get_prioritized_engines(self) -> Dict[str, BasePlugin]:
        """Get engines sorted by priority."""
        sorted_engines = sorted(
            self.engines.items(),
            key=lambda x: self.engine_priorities.get(x[0], 0),
            reverse=True
        )
        return dict(sorted_engines)
    
    def update_priority(self, name: str, priority: int):
        """Update engine priority."""
        if name in self.engines:
            self.engine_priorities[name] = priority
            logger.info(f"🔄 Updated priority for {name}: {priority}")
    
    def get_engine_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            'total_engines': len(self.engines),
            'priorities': self.engine_priorities.copy(),
            'suggestion_history': self.suggestion_history[-10:],  # Last 10 suggestions
            'engines': list(self.engines.keys())
        }
    
    def record_suggestion(self, engine_name: str, content_length: int, success: bool, duration: float):
        """Record suggestion attempt."""
        self.suggestion_history.append({
            'engine': engine_name,
            'content_length': content_length,
            'success': success,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 100 records
        if len(self.suggestion_history) > 100:
            self.suggestion_history = self.suggestion_history[-100:] 