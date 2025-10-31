"""
🚀 Refactored LinkedIn Posts Optimization Core
============================================

A clean, maintainable implementation focusing on core functionality:
- Content optimization
- Engagement prediction
- Performance monitoring
- Clean architecture patterns
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Types of LinkedIn content."""
    POST = "post"
    ARTICLE = "article"
    VIDEO = "video"
    IMAGE = "image"
    POLL = "poll"


class OptimizationStrategy(Enum):
    """Content optimization strategies."""
    ENGAGEMENT = "engagement"
    REACH = "reach"
    CLICKS = "clicks"
    SHARES = "shares"
    COMMENTS = "comments"


@dataclass
class ContentMetrics:
    """Metrics for content performance."""
    views: int = 0
    likes: int = 0
    shares: int = 0
    comments: int = 0
    clicks: int = 0
    engagement_rate: float = 0.0
    
    def calculate_engagement_rate(self) -> float:
        """Calculate engagement rate based on interactions."""
        total_interactions = self.likes + self.shares + self.comments + self.clicks
        if self.views > 0:
            self.engagement_rate = (total_interactions / self.views) * 100
        return self.engagement_rate


@dataclass
class ContentData:
    """LinkedIn content data structure."""
    id: str
    content: str
    content_type: ContentType
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    media_urls: List[str] = field(default_factory=list)
    posted_at: Optional[datetime] = None
    metrics: Optional[ContentMetrics] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ContentMetrics()


@dataclass
class OptimizationResult:
    """Result of content optimization."""
    original_content: ContentData
    optimized_content: ContentData
    optimization_score: float
    improvements: List[str]
    predicted_engagement_increase: float
    timestamp: datetime = field(default_factory=datetime.now)


class ContentAnalyzer(Protocol):
    """Protocol for content analysis."""
    
    async def analyze_content(self, content: ContentData) -> Dict[str, Any]:
        """Analyze content and return insights."""
        ...


class ContentOptimizer(Protocol):
    """Protocol for content optimization."""
    
    async def optimize_content(
        self, 
        content: ContentData, 
        strategy: OptimizationStrategy
    ) -> OptimizationResult:
        """Optimize content based on strategy."""
        ...


class EngagementPredictor(Protocol):
    """Protocol for engagement prediction."""
    
    async def predict_engagement(self, content: ContentData) -> float:
        """Predict engagement rate for content."""
        ...


class BaseContentAnalyzer:
    """Base implementation of content analyzer."""
    
    async def analyze_content(self, content: ContentData) -> Dict[str, Any]:
        """Analyze content and return insights."""
        analysis = {
            "content_length": len(content.content),
            "hashtag_count": len(content.hashtags),
            "mention_count": len(content.mentions),
            "link_count": len(content.links),
            "media_count": len(content.media_urls),
            "readability_score": self._calculate_readability(content.content),
            "sentiment_score": self._calculate_sentiment(content.content),
            "optimal_posting_time": self._get_optimal_posting_time(),
            "recommended_hashtags": self._get_recommended_hashtags(content),
        }
        
        logger.info(f"Content analysis completed for content {content.id}")
        return analysis
    
    def _calculate_readability(self, content: str) -> float:
        """Calculate content readability score."""
        # Simple Flesch Reading Ease approximation
        sentences = content.split('.')
        words = content.split()
        syllables = sum(len(word) // 3 for word in words)
        
        if len(sentences) > 0 and len(words) > 0:
            return 206.835 - (1.015 * (len(words) / len(sentences))) - (84.6 * (syllables / len(words)))
        return 0.0
    
    def _calculate_sentiment(self, content: str) -> float:
        """Calculate content sentiment score (-1 to 1)."""
        # Simple sentiment analysis based on positive/negative words
        positive_words = {'great', 'amazing', 'excellent', 'good', 'best', 'love', 'awesome'}
        negative_words = {'bad', 'terrible', 'worst', 'hate', 'awful', 'poor'}
        
        words = content.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_words = len(words)
        if total_words > 0:
            return (positive_count - negative_count) / total_words
        return 0.0
    
    def _get_optimal_posting_time(self) -> str:
        """Get optimal posting time recommendation."""
        # Based on LinkedIn engagement research
        return "Tuesday-Thursday, 8-10 AM or 12-2 PM"
    
    def _get_recommended_hashtags(self, content: ContentData) -> List[str]:
        """Get recommended hashtags based on content."""
        # Industry-specific hashtag recommendations
        industry_hashtags = {
            "tech": ["#technology", "#innovation", "#digitaltransformation"],
            "business": ["#business", "#leadership", "#strategy"],
            "marketing": ["#marketing", "#digitalmarketing", "#growth"],
            "sales": ["#sales", "#b2b", "#networking"],
        }
        
        # Simple keyword-based industry detection
        content_lower = content.content.lower()
        if any(word in content_lower for word in ["tech", "software", "ai", "digital"]):
            return industry_hashtags["tech"]
        elif any(word in content_lower for word in ["business", "strategy", "leadership"]):
            return industry_hashtags["business"]
        elif any(word in content_lower for word in ["marketing", "growth", "campaign"]):
            return industry_hashtags["marketing"]
        elif any(word in content_lower for word in ["sales", "b2b", "networking"]):
            return industry_hashtags["sales"]
        
        return ["#linkedin", "#professional", "#networking"]


class BaseContentOptimizer:
    """Base implementation of content optimizer."""
    
    def __init__(self, analyzer: ContentAnalyzer):
        self.analyzer = analyzer
    
    async def optimize_content(
        self, 
        content: ContentData, 
        strategy: OptimizationStrategy
    ) -> OptimizationResult:
        """Optimize content based on strategy."""
        # Analyze current content
        analysis = await self.analyzer.analyze_content(content)
        
        # Create optimized version
        optimized_content = self._create_optimized_content(content, analysis, strategy)
        
        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(content, optimized_content)
        
        # Identify improvements
        improvements = self._identify_improvements(content, optimized_content, analysis)
        
        # Predict engagement increase
        predicted_increase = self._predict_engagement_increase(optimization_score)
        
        result = OptimizationResult(
            original_content=content,
            optimized_content=optimized_content,
            optimization_score=optimization_score,
            improvements=improvements,
            predicted_engagement_increase=predicted_increase
        )
        
        logger.info(f"Content optimization completed for content {content.id}")
        return result
    
    def _create_optimized_content(
        self, 
        content: ContentData, 
        analysis: Dict[str, Any], 
        strategy: OptimizationStrategy
    ) -> ContentData:
        """Create optimized content based on analysis and strategy."""
        optimized_content = ContentData(
            id=f"{content.id}_optimized",
            content=content.content,
            content_type=content.content_type,
            hashtags=content.hashtags.copy(),
            mentions=content.mentions.copy(),
            links=content.links.copy(),
            media_urls=content.media_urls.copy(),
            posted_at=content.posted_at
        )
        
        # Apply strategy-specific optimizations
        if strategy == OptimizationStrategy.ENGAGEMENT:
            optimized_content = self._optimize_for_engagement(optimized_content, analysis)
        elif strategy == OptimizationStrategy.REACH:
            optimized_content = self._optimize_for_reach(optimized_content, analysis)
        elif strategy == OptimizationStrategy.CLICKS:
            optimized_content = self._optimize_for_clicks(optimized_content, analysis)
        
        return optimized_content
    
    def _optimize_for_engagement(self, content: ContentData, analysis: Dict[str, Any]) -> ContentData:
        """Optimize content for maximum engagement."""
        # Add recommended hashtags if not present
        recommended_hashtags = analysis.get("recommended_hashtags", [])
        for hashtag in recommended_hashtags:
            if hashtag not in content.hashtags:
                content.hashtags.append(hashtag)
        
        # Improve content structure
        if analysis.get("content_length", 0) < 100:
            content.content += "\n\nWhat are your thoughts on this? Share your experience below! 👇"
        
        return content
    
    def _optimize_for_reach(self, content: ContentData, analysis: Dict[str, Any]) -> ContentData:
        """Optimize content for maximum reach."""
        # Add trending hashtags
        trending_hashtags = ["#linkedin", "#networking", "#professional"]
        for hashtag in trending_hashtags:
            if hashtag not in content.hashtags:
                content.hashtags.append(hashtag)
        
        return content
    
    def _optimize_for_clicks(self, content: ContentData, analysis: Dict[str, Any]) -> ContentData:
        """Optimize content for maximum clicks."""
        # Add call-to-action
        if "click" not in content.content.lower():
            content.content += "\n\n🔗 Click the link in the comments for more details!"
        
        return content
    
    def _calculate_optimization_score(self, original: ContentData, optimized: ContentData) -> float:
        """Calculate optimization score (0-100)."""
        score = 0.0
        
        # Hashtag optimization
        if len(optimized.hashtags) > len(original.hashtags):
            score += 20
        
        # Content enhancement
        if len(optimized.content) > len(original.content):
            score += 30
        
        # Structure improvement
        if "\n\n" in optimized.content:
            score += 25
        
        # Call-to-action
        if any(word in optimized.content.lower() for word in ["click", "link", "comment"]):
            score += 25
        
        return min(score, 100.0)
    
    def _identify_improvements(
        self, 
        original: ContentData, 
        optimized: ContentData, 
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Identify specific improvements made."""
        improvements = []
        
        if len(optimized.hashtags) > len(original.hashtags):
            improvements.append("Added relevant hashtags for better discoverability")
        
        if len(optimized.content) > len(original.content):
            improvements.append("Enhanced content with engaging elements")
        
        if analysis.get("readability_score", 0) < 60:
            improvements.append("Improved readability with better structure")
        
        if analysis.get("sentiment_score", 0) < 0:
            improvements.append("Enhanced positive sentiment")
        
        return improvements
    
    def _predict_engagement_increase(self, optimization_score: float) -> float:
        """Predict engagement increase based on optimization score."""
        # Simple linear prediction model
        return (optimization_score / 100) * 0.5  # 0-50% increase


class BaseEngagementPredictor:
    """Base implementation of engagement predictor."""
    
    def __init__(self, analyzer: ContentAnalyzer):
        self.analyzer = analyzer
    
    async def predict_engagement(self, content: ContentData) -> float:
        """Predict engagement rate for content."""
        analysis = await self.analyzer.analyze_content(content)
        
        # Base engagement rate
        base_rate = 2.0  # 2% base engagement
        
        # Content length factor
        length_factor = min(len(content.content) / 100, 2.0)
        
        # Hashtag factor
        hashtag_factor = min(len(content.hashtags) * 0.5, 3.0)
        
        # Media factor
        media_factor = len(content.media_urls) * 0.5 + 1.0
        
        # Readability factor
        readability_factor = max(analysis.get("readability_score", 50) / 100, 0.5)
        
        # Sentiment factor
        sentiment_factor = max(analysis.get("sentiment_score", 0) + 1, 0.5)
        
        # Calculate predicted engagement
        predicted_rate = base_rate * length_factor * hashtag_factor * media_factor * readability_factor * sentiment_factor
        
        logger.info(f"Engagement prediction: {predicted_rate:.2f}% for content {content.id}")
        return min(predicted_rate, 15.0)  # Cap at 15%


class LinkedInOptimizationService:
    """Main service for LinkedIn content optimization."""
    
    def __init__(
        self,
        analyzer: Optional[ContentAnalyzer] = None,
        optimizer: Optional[ContentOptimizer] = None,
        predictor: Optional[EngagementPredictor] = None
    ):
        self.analyzer = analyzer or BaseContentAnalyzer()
        self.optimizer = optimizer or BaseContentOptimizer(self.analyzer)
        self.predictor = predictor or BaseEngagementPredictor(self.analyzer)
    
    async def optimize_linkedin_post(
        self, 
        content: str, 
        strategy: OptimizationStrategy = OptimizationStrategy.ENGAGEMENT
    ) -> OptimizationResult:
        """Optimize a LinkedIn post."""
        # Create content data
        content_data = ContentData(
            id=str(hash(content))[:8],
            content=content,
            content_type=ContentType.POST
        )
        
        # Optimize content
        result = await self.optimizer.optimize_content(content_data, strategy)
        
        return result
    
    async def predict_post_engagement(self, content: str) -> float:
        """Predict engagement for a LinkedIn post."""
        content_data = ContentData(
            id=str(hash(content))[:8],
            content=content,
            content_type=ContentType.POST
        )
        
        return await self.predictor.predict_engagement(content_data)
    
    async def get_content_insights(self, content: str) -> Dict[str, Any]:
        """Get comprehensive insights for content."""
        content_data = ContentData(
            id=str(hash(content))[:8],
            content=content,
            content_type=ContentType.POST
        )
        
        return await self.analyzer.analyze_content(content_data)


# Factory function for easy service creation
def create_linkedin_optimization_service() -> LinkedInOptimizationService:
    """Create a LinkedIn optimization service with default implementations."""
    return LinkedInOptimizationService()


# Example usage
async def main():
    """Example usage of the LinkedIn optimization service."""
    service = create_linkedin_optimization_service()
    
    # Sample content
    sample_content = """
    Just finished an amazing project using React and TypeScript! 
    The development experience was incredible, and the final result exceeded expectations.
    #react #typescript #webdevelopment
    """
    
    # Optimize content
    result = await service.optimize_linkedin_post(sample_content, OptimizationStrategy.ENGAGEMENT)
    
    print(f"Optimization Score: {result.optimization_score:.1f}%")
    print(f"Predicted Engagement Increase: {result.predicted_engagement_increase:.1f}%")
    print("\nImprovements:")
    for improvement in result.improvements:
        print(f"- {improvement}")
    
    # Predict engagement
    engagement = await service.predict_post_engagement(sample_content)
    print(f"\nPredicted Engagement Rate: {engagement:.2f}%")


if __name__ == "__main__":
    asyncio.run(main())






