"""
Application Services - Orchestrate domain logic and infrastructure
"""

import hashlib
import time
from typing import Dict, Any, List
from functools import lru_cache

from ..domain.interfaces import (
    IAnalysisRepository,
    ICacheService,
    IMLService,
    IEventBus
)
from ..domain.entities import ContentAnalysis
from ..domain.value_objects import AnalysisResult, SimilarityResult, QualityResult
from ..domain.events import AnalysisCompletedEvent
from ..core.logging_config import get_logger
from .dtos import AnalysisRequest, SimilarityRequest, QualityRequest, BatchRequest

logger = get_logger(__name__)


class AnalysisService:
    """
    Application service for content analysis
    Orchestrates domain logic and infrastructure
    """
    
    def __init__(
        self,
        repository: IAnalysisRepository,
        cache_service: ICacheService,
        ml_service: IMLService,
        event_bus: IEventBus
    ):
        self.repository = repository
        self.cache_service = cache_service
        self.ml_service = ml_service
        self.event_bus = event_bus
        logger.debug("AnalysisService initialized")
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate content hash"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _extract_words(self, content: str) -> List[str]:
        """Extract words from content"""
        import re
        words = re.findall(r'\b\w+\b', content.lower())
        return words
    
    def _calculate_redundancy_score(self, content: str) -> float:
        """Calculate redundancy score (domain logic)"""
        words = self._extract_words(content)
        if not words:
            return 0.0
        
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        total_words = len(words)
        repeated_words = sum(1 for count in word_freq.values() if count > 1)
        
        # Calculate redundancy score (0-1)
        if total_words == 0:
            return 0.0
        
        redundancy_ratio = repeated_words / len(word_freq) if word_freq else 0.0
        return min(redundancy_ratio, 1.0)
    
    async def analyze_content(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Analyze content for redundancy
        
        Orchestration:
        1. Check cache
        2. Calculate analysis (domain logic)
        3. Save to repository
        4. Cache result
        5. Publish event
        """
        content_hash = self._calculate_content_hash(request.content)
        cache_key = f"analysis:{content_hash}"
        
        # Check cache
        cached_result = await self.cache_service.get(cache_key)
        if cached_result:
            logger.debug("Returning cached analysis result")
            return AnalysisResult(**cached_result)
        
        # Extract words
        words = self._extract_words(request.content)
        word_count = len(words)
        unique_words = len(set(words))
        
        # Calculate redundancy (domain logic)
        redundancy_score = self._calculate_redundancy_score(request.content)
        
        # Create value object
        result = AnalysisResult(
            content_hash=content_hash,
            word_count=word_count,
            character_count=len(request.content),
            unique_words=unique_words,
            redundancy_score=redundancy_score,
            timestamp=time.time()
        )
        
        # Create entity (for repository)
        entity = ContentAnalysis(
            content=request.content,
            content_hash=content_hash,
            word_count=word_count,
            character_count=len(request.content),
            unique_words=unique_words,
            redundancy_score=redundancy_score
        )
        
        # Save to repository
        try:
            await self.repository.save_analysis(entity)
        except Exception as e:
            logger.warning(f"Failed to save analysis to repository: {e}")
        
        # Cache result
        await self.cache_service.set(cache_key, result.to_dict(), ttl=3600)
        
        # Publish event
        try:
            event = AnalysisCompletedEvent(
                aggregate_id=content_hash,
                content_hash=content_hash,
                redundancy_score=redundancy_score,
                word_count=word_count
            )
            await self.event_bus.publish(event.event_type, event.to_dict())
        except Exception as e:
            logger.warning(f"Failed to publish event: {e}")
        
        return result
    
    async def detect_similarity(self, request: SimilarityRequest) -> SimilarityResult:
        """
        Detect similarity between two texts
        
        TODO: Implement similarity detection with ML service
        """
        # Placeholder implementation
        words1 = set(self._extract_words(request.text1))
        words2 = set(self._extract_words(request.text2))
        
        common_words = list(words1.intersection(words2))
        all_words = words1.union(words2)
        
        similarity_score = len(common_words) / len(all_words) if all_words else 0.0
        is_similar = similarity_score >= request.threshold
        
        return SimilarityResult(
            similarity_score=similarity_score,
            is_similar=is_similar,
            common_words=common_words[:10],  # Limit to 10
            differences=[],
            threshold=request.threshold,
            timestamp=time.time()
        )


class BatchService:
    """
    Application service for batch processing
    """
    
    def __init__(self, analysis_service: AnalysisService):
        self.analysis_service = analysis_service
        logger.debug("BatchService initialized")
    
    async def process_batch(self, request: BatchRequest) -> Dict[str, Any]:
        """
        Process multiple items in batch
        """
        results = []
        errors = []
        
        for item in request.items:
            try:
                analysis_request = AnalysisRequest(content=item, threshold=request.threshold)
                result = await self.analysis_service.analyze_content(analysis_request)
                results.append(result)
            except Exception as e:
                errors.append({"item": item[:50], "error": str(e)})
                logger.error(f"Batch item failed: {e}")
        
        return {
            "total_items": len(request.items),
            "processed_items": len(results),
            "failed_items": len(errors),
            "results": [r.to_dict() for r in results],
            "errors": errors,
            "timestamp": time.time()
        }


class QualityService:
    """
    Application service for quality assessment
    """
    
    def __init__(self, ml_service: IMLService, cache_service: ICacheService):
        self.ml_service = ml_service
        self.cache_service = cache_service
        logger.debug("QualityService initialized")
    
    async def assess_quality(self, request: QualityRequest) -> QualityResult:
        """
        Assess content quality
        """
        content_hash = hashlib.sha256(request.content.encode()).hexdigest()
        cache_key = f"quality:{content_hash}"
        
        # Check cache
        cached = await self.cache_service.get(cache_key)
        if cached:
            return QualityResult(**cached)
        
        # Detect language
        try:
            language = await self.ml_service.detect_language(request.content)
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            language = "unknown"
        
        # Analyze sentiment (if requested)
        sentiment_score = None
        if request.include_sentiment:
            try:
                sentiment_data = await self.ml_service.analyze_sentiment(request.content)
                sentiment_score = sentiment_data.get("score", None)
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
        
        # Calculate readability (simplified)
        readability_score = self._calculate_readability(request.content)
        
        # Calculate quality score
        quality_score = (readability_score + (sentiment_score or 0.5)) / 2 if sentiment_score else readability_score
        
        # Generate suggestions
        suggestions = []
        if readability_score < 0.6:
            suggestions.append("Improve readability")
        if quality_score < 0.7:
            suggestions.append("Consider improving overall quality")
        
        result = QualityResult(
            quality_score=quality_score,
            readability_score=readability_score,
            sentiment_score=sentiment_score,
            language=language,
            suggestions=suggestions,
            timestamp=time.time()
        )
        
        # Cache result
        await self.cache_service.set(cache_key, result.to_dict(), ttl=3600)
        
        return result
    
    def _calculate_readability(self, content: str) -> float:
        """Calculate readability score (simplified)"""
        # Flesch Reading Ease approximation
        sentences = content.split('.')
        words = content.split()
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Simplified formula
        readability = 1.0 - min(1.0, (avg_sentence_length / 20.0 + avg_word_length / 10.0) / 2.0)
        return max(0.0, min(1.0, readability))
