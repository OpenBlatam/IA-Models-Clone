from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from typing import List, Dict, Any, Optional
from core.config import get_config
from core.exceptions import ProcessingError, ValidationError, handle_async_exception
from core.types import NLPResult, Language
from cache_manager import cache
from async_processor import processor
import structlog
import asyncio
        from nlp_utils import analyze_nlp_sync
        from nlp_utils import get_nlp_stats as get_utils_stats
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
NLP Service for OS Content UGC Video Generator
Handles natural language processing business logic
"""


logger = structlog.get_logger("os_content.nlp_service")

class NLPService:
    """Natural Language Processing service"""
    
    def __init__(self) -> Any:
        self.config = get_config()
    
    @handle_async_exception
    async def analyze_text(self, text: str, language: str = "es") -> Dict[str, Any]:
        """Analyze text using NLP"""
        
        # Validate input
        if not text or not text.strip():
            raise ValidationError("Text cannot be empty", field="text")
        
        # Check cache first
        cache_key = f"nlp_analysis:{hash(text)}:{language}"
        cached_result = await cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Perform analysis
        result = await self._perform_nlp_analysis(text, language)
        
        # Cache result
        await cache.set(cache_key, result, ttl=self.config.cache.ttl)
        
        return result
    
    @handle_async_exception
    async def batch_analyze_texts(self, texts: List[str], language: str = "es") -> List[Dict[str, Any]]:
        """Batch analyze multiple texts"""
        
        if not texts:
            return []
        
        if len(texts) > 100:
            raise ValidationError("Maximum 100 texts per batch", field="texts")
        
        # Create tasks for batch processing
        tasks = []
        for text in texts:
            task = processor.submit_task(
                self.analyze_text,
                text,
                language,
                priority=processor.TaskPriority.NORMAL,
                timeout=30
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch NLP analysis failed for text {i}: {result}")
                processed_results.append({"error": str(result)})
            else:
                processed_results.append(result)
        
        return processed_results
    
    @handle_async_exception
    async def _perform_nlp_analysis(self, text: str, language: str) -> Dict[str, Any]:
        """Perform NLP analysis using the NLP utils"""
        
        
        # Run analysis in thread pool for CPU-intensive tasks
        result = await processor.run_in_thread(analyze_nlp_sync, text, language)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        return result
    
    @handle_async_exception
    async def generate_keywords(self, text: str, language: str = "es", max_keywords: int = 10) -> List[str]:
        """Generate keywords from text"""
        
        analysis = await self.analyze_text(text, language)
        
        # Extract keywords from entities and tokens
        keywords = []
        
        # Add entities as keywords
        if "entities" in analysis:
            for entity in analysis["entities"]:
                if entity.get("text") and len(keywords) < max_keywords:
                    keywords.append(entity["text"])
        
        # Add significant tokens as keywords
        if "tokens" in analysis:
            for token in analysis["tokens"]:
                if len(token) > 3 and len(keywords) < max_keywords:
                    keywords.append(token)
        
        return keywords[:max_keywords]
    
    @handle_async_exception
    async def get_sentiment(self, text: str, language: str = "es") -> Dict[str, Any]:
        """Get sentiment analysis for text"""
        
        analysis = await self.analyze_text(text, language)
        
        return {
            "text": text,
            "sentiment": analysis.get("sentiment", "neutral"),
            "confidence": analysis.get("confidence", 0.0),
            "language": language
        }
    
    @handle_async_exception
    async def extract_entities(self, text: str, language: str = "es") -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        
        analysis = await self.analyze_text(text, language)
        
        return analysis.get("entities", [])
    
    @handle_async_exception
    async def get_nlp_stats(self) -> Dict[str, Any]:
        """Get NLP service statistics"""
        
        
        utils_stats = await get_utils_stats()
        
        return {
            "service": "nlp",
            "cache_stats": utils_stats.get("cache_stats", {}),
            "model_cache_size": utils_stats.get("model_cache_size", 0),
            "pipeline_cache_size": utils_stats.get("pipeline_cache_size", 0),
            "supported_languages": utils_stats.get("supported_languages", []),
            "gpu_available": utils_stats.get("gpu_available", False),
            "gpu_memory_allocated": utils_stats.get("gpu_memory_allocated", 0)
        }

# Global NLP service instance
nlp_service = NLPService() 