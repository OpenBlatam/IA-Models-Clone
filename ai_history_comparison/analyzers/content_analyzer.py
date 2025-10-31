"""
Content Analyzer

This module provides comprehensive content analysis capabilities including
quality assessment, sentiment analysis, complexity analysis, and readability metrics.
"""

import hashlib
import time
from typing import Dict, List, Any, Optional
import logging

from ..core.base import BaseAnalyzer
from ..core.config import SystemConfig
from ..core.interfaces import IContentAnalyzer
from ..core.exceptions import AnalysisError, ValidationError

logger = logging.getLogger(__name__)


class ContentAnalyzer(BaseAnalyzer[str], IContentAnalyzer):
    """Advanced content analyzer with multiple analysis capabilities"""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        super().__init__(config)
        self._analysis_metrics = [
            "quality_score",
            "sentiment_score", 
            "complexity_score",
            "readability_score",
            "word_count",
            "sentence_count",
            "avg_word_length",
            "topic_diversity",
            "consistency_score"
        ]
    
    async def _analyze(self, content: str, **kwargs) -> Dict[str, Any]:
        """Perform comprehensive content analysis"""
        try:
            analysis_type = kwargs.get("analysis_type", "comprehensive")
            
            if analysis_type == "comprehensive":
                return await self._comprehensive_analysis(content, **kwargs)
            elif analysis_type == "quality":
                return await self._quality_analysis(content, **kwargs)
            elif analysis_type == "sentiment":
                return await self._sentiment_analysis(content, **kwargs)
            elif analysis_type == "complexity":
                return await self._complexity_analysis(content, **kwargs)
            elif analysis_type == "readability":
                return await self._readability_analysis(content, **kwargs)
            else:
                raise ValidationError(f"Unknown analysis type: {analysis_type}")
                
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            raise AnalysisError(f"Content analysis failed: {str(e)}", analyzer_name="ContentAnalyzer")
    
    async def _comprehensive_analysis(self, content: str, **kwargs) -> Dict[str, Any]:
        """Perform comprehensive analysis including all metrics"""
        results = {}
        
        # Basic metrics
        results.update(await self._basic_metrics(content))
        
        # Quality analysis
        results.update(await self._quality_analysis(content, **kwargs))
        
        # Sentiment analysis
        results.update(await self._sentiment_analysis(content, **kwargs))
        
        # Complexity analysis
        results.update(await self._complexity_analysis(content, **kwargs))
        
        # Readability analysis
        results.update(await self._readability_analysis(content, **kwargs))
        
        # Overall quality score
        results["overall_quality_score"] = self._calculate_overall_quality(results)
        
        return results
    
    async def _basic_metrics(self, content: str) -> Dict[str, Any]:
        """Calculate basic content metrics"""
        words = content.split()
        sentences = content.split('.')
        
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "character_count": len(content),
            "paragraph_count": len(content.split('\n\n'))
        }
    
    async def _quality_analysis(self, content: str, **kwargs) -> Dict[str, Any]:
        """Analyze content quality"""
        # Simulate quality analysis
        quality_score = min(1.0, max(0.0, 0.7 + (len(content) / 1000) * 0.1))
        
        return {
            "quality_score": quality_score,
            "coherence_score": min(1.0, quality_score + 0.1),
            "relevance_score": min(1.0, quality_score - 0.05),
            "accuracy_score": min(1.0, quality_score + 0.05)
        }
    
    async def _sentiment_analysis(self, content: str, **kwargs) -> Dict[str, Any]:
        """Analyze content sentiment"""
        # Simple sentiment analysis simulation
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
        negative_words = ["bad", "terrible", "awful", "horrible", "disappointing", "poor"]
        
        content_lower = content.lower()
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        if positive_count + negative_count == 0:
            sentiment_score = 0.5  # Neutral
        else:
            sentiment_score = positive_count / (positive_count + negative_count)
        
        return {
            "sentiment_score": sentiment_score,
            "sentiment_label": self._get_sentiment_label(sentiment_score),
            "positive_words": positive_count,
            "negative_words": negative_count
        }
    
    async def _complexity_analysis(self, content: str, **kwargs) -> Dict[str, Any]:
        """Analyze content complexity"""
        words = content.split()
        sentences = content.split('.')
        
        # Calculate average sentence length
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Calculate complexity score (0-1, higher is more complex)
        complexity_score = min(1.0, avg_sentence_length / 20.0)
        
        return {
            "complexity_score": complexity_score,
            "avg_sentence_length": avg_sentence_length,
            "complexity_level": self._get_complexity_level(complexity_score)
        }
    
    async def _readability_analysis(self, content: str, **kwargs) -> Dict[str, Any]:
        """Analyze content readability"""
        words = content.split()
        sentences = content.split('.')
        
        # Simple readability score (Flesch-like)
        if len(sentences) == 0 or len(words) == 0:
            readability_score = 0.5
        else:
            avg_sentence_length = len(words) / len(sentences)
            avg_word_length = sum(len(word) for word in words) / len(words)
            readability_score = max(0.0, min(1.0, 1.0 - (avg_sentence_length / 30.0) - (avg_word_length / 10.0)))
        
        return {
            "readability_score": readability_score,
            "readability_level": self._get_readability_level(readability_score)
        }
    
    def _calculate_overall_quality(self, results: Dict[str, Any]) -> float:
        """Calculate overall quality score from individual metrics"""
        weights = {
            "quality_score": 0.3,
            "sentiment_score": 0.2,
            "complexity_score": 0.2,
            "readability_score": 0.2,
            "coherence_score": 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in results:
                total_score += results[metric] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.5
    
    def _get_sentiment_label(self, sentiment_score: float) -> str:
        """Get sentiment label from score"""
        if sentiment_score >= 0.7:
            return "positive"
        elif sentiment_score <= 0.3:
            return "negative"
        else:
            return "neutral"
    
    def _get_complexity_level(self, complexity_score: float) -> str:
        """Get complexity level from score"""
        if complexity_score >= 0.8:
            return "very_high"
        elif complexity_score >= 0.6:
            return "high"
        elif complexity_score >= 0.4:
            return "medium"
        elif complexity_score >= 0.2:
            return "low"
        else:
            return "very_low"
    
    def _get_readability_level(self, readability_score: float) -> str:
        """Get readability level from score"""
        if readability_score >= 0.8:
            return "very_easy"
        elif readability_score >= 0.6:
            return "easy"
        elif readability_score >= 0.4:
            return "medium"
        elif readability_score >= 0.2:
            return "difficult"
        else:
            return "very_difficult"
    
    def get_analysis_metrics(self) -> List[str]:
        """Get list of metrics this analyzer produces"""
        return self._analysis_metrics
    
    def validate_input(self, data: str) -> bool:
        """Validate input data before analysis"""
        if not isinstance(data, str):
            return False
        if len(data.strip()) == 0:
            return False
        if len(data) > 1000000:  # 1MB limit
            return False
        return True
    
    async def analyze_quality(self, content: str) -> Dict[str, Any]:
        """Analyze content quality specifically"""
        return await self._quality_analysis(content)
    
    async def analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze content sentiment specifically"""
        return await self._sentiment_analysis(content)
    
    async def analyze_complexity(self, content: str) -> Dict[str, Any]:
        """Analyze content complexity specifically"""
        return await self._complexity_analysis(content)
    
    async def analyze_readability(self, content: str) -> Dict[str, Any]:
        """Analyze content readability specifically"""
        return await self._readability_analysis(content)





















