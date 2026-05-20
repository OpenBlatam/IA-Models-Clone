"""
🚀 Optimized LinkedIn Posts Optimization System
==============================================

A unified, high-performance LinkedIn content optimization system that:
- Consolidates the best features from all previous versions
- Implements advanced ML with efficient PyTorch models
- Uses modern async patterns for high performance
- Includes comprehensive error handling and monitoring
- Optimizes memory usage and reduces dependencies
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import pickle
from functools import lru_cache
import hashlib

# Core ML imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel, pipeline
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available, using fallback models")

try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available, using fallback models")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Performance monitoring
class PerformanceMonitor:
    """Monitor system performance and response times."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
    
    def start_operation(self, operation_name: str):
        """Start timing an operation."""
        self.metrics[operation_name] = {'start': time.time()}
    
    def end_operation(self, operation_name: str) -> float:
        """End timing an operation and return duration."""
        if operation_name in self.metrics:
            duration = time.time() - self.metrics[operation_name]['start']
            self.metrics[operation_name]['duration'] = duration
            self.metrics[operation_name]['end'] = time.time()
            return duration
        return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            'total_uptime': time.time() - self.start_time,
            'operations': {}
        }
        
        for op_name, op_data in self.metrics.items():
            if 'duration' in op_data:
                stats['operations'][op_name] = {
                    'duration': op_data['duration'],
                    'start': op_data['start'],
                    'end': op_data['end']
                }
        
        return stats

# Enums
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

class ErrorLevel(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Data structures
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'views': self.views,
            'likes': self.likes,
            'shares': self.shares,
            'comments': self.comments,
            'clicks': self.clicks,
            'engagement_rate': self.engagement_rate
        }

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
    
    def get_content_hash(self) -> str:
        """Generate content hash for caching."""
        content_str = f"{self.content}{self.content_type.value}{''.join(self.hashtags)}"
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'content': self.content,
            'content_type': self.content_type.value,
            'hashtags': self.hashtags,
            'mentions': self.mentions,
            'links': self.links,
            'media_urls': self.media_urls,
            'posted_at': self.posted_at.isoformat() if self.posted_at else None,
            'metrics': self.metrics.to_dict() if self.metrics else None
        }

@dataclass
class OptimizationResult:
    """Result of content optimization."""
    original_content: ContentData
    optimized_content: ContentData
    optimization_score: float
    improvements: List[str]
    predicted_engagement_increase: float
    confidence_score: float
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'original_content': self.original_content.to_dict(),
            'optimized_content': self.optimized_content.to_dict(),
            'optimization_score': self.optimization_score,
            'improvements': self.improvements,
            'predicted_engagement_increase': self.predicted_engagement_increase,
            'confidence_score': self.confidence_score,
            'processing_time': self.processing_time,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class OptimizationError:
    """Error information for optimization failures."""
    message: str
    error_type: str
    level: ErrorLevel
    timestamp: datetime = field(default_factory=datetime.now)
    details: Optional[Dict[str, Any]] = None

# Core interfaces
class ContentAnalyzer(ABC):
    """Abstract base class for content analysis."""
    
    @abstractmethod
    async def analyze_content(self, content: ContentData) -> Dict[str, Any]:
        """Analyze content and return insights."""
        pass

class ContentOptimizer(ABC):
    """Abstract base class for content optimization."""
    
    @abstractmethod
    async def optimize_content(
        self, 
        content: ContentData, 
        strategy: OptimizationStrategy
    ) -> ContentData:
        """Optimize content based on strategy."""
        pass

class EngagementPredictor(ABC):
    """Abstract base class for engagement prediction."""
    
    @abstractmethod
    async def predict_engagement(
        self, 
        content: ContentData
    ) -> Tuple[float, float]:
        """Predict engagement score and confidence."""
        pass

# Fallback implementations for when ML libraries aren't available
class FallbackContentAnalyzer(ContentAnalyzer):
    """Fallback content analyzer when ML libraries aren't available."""
    
    async def analyze_content(self, content: ContentData) -> Dict[str, Any]:
        """Basic content analysis using heuristics."""
        analysis = {
            'sentiment': 'neutral',
            'readability_score': 0.7,
            'keyword_density': {},
            'content_length_score': min(len(content.content) / 1000, 1.0),
            'hashtag_score': min(len(content.hashtags) / 5, 1.0),
            'mention_score': min(len(content.mentions) / 3, 1.0)
        }
        
        # Simple sentiment analysis
        positive_words = ['amazing', 'great', 'excellent', 'wonderful', 'fantastic']
        negative_words = ['terrible', 'awful', 'horrible', 'bad', 'worst']
        
        content_lower = content.content.lower()
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        if positive_count > negative_count:
            analysis['sentiment'] = 'positive'
        elif negative_count > positive_count:
            analysis['sentiment'] = 'negative'
        
        return analysis

class FallbackContentOptimizer(ContentOptimizer):
    """Fallback content optimizer when ML libraries aren't available."""
    
    async def optimize_content(
        self, 
        content: ContentData, 
        strategy: OptimizationStrategy
    ) -> ContentData:
        """Basic content optimization using heuristics."""
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
        
        # Add strategy-specific optimizations
        if strategy == OptimizationStrategy.ENGAGEMENT:
            if not optimized_content.content.endswith('?'):
                optimized_content.content += " What do you think?"
        
        elif strategy == OptimizationStrategy.CLICKS:
            if optimized_content.links:
                optimized_content.content += f"\n\nCheck out the link above for more details!"
        
        elif strategy == OptimizationStrategy.SHARES:
            optimized_content.content += "\n\nShare this if you found it helpful!"
        
        return optimized_content

class FallbackEngagementPredictor(EngagementPredictor):
    """Fallback engagement predictor when ML libraries aren't available."""
    
    async def predict_engagement(
        self, 
        content: ContentData
    ) -> Tuple[float, float]:
        """Basic engagement prediction using heuristics."""
        base_score = 0.5
        
        # Content length factor
        length_factor = min(len(content.content) / 500, 1.0)
        
        # Hashtag factor
        hashtag_factor = min(len(content.hashtags) / 3, 1.0)
        
        # Mention factor
        mention_factor = min(len(content.mentions) / 2, 1.0)
        
        # Calculate final score
        engagement_score = base_score * (0.4 + 0.2 * length_factor + 0.2 * hashtag_factor + 0.2 * mention_factor)
        confidence_score = 0.6  # Lower confidence for fallback model
        
        return min(engagement_score, 1.0), confidence_score

# ML-powered implementations
if TORCH_AVAILABLE:
    class TransformerContentAnalyzer(ContentAnalyzer):
        """Advanced content analysis using transformer models."""
        
        def __init__(self):
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = AutoModel.from_pretrained("distilbert-base-uncased")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis", 
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        
        async def analyze_content(self, content: ContentData) -> Dict[str, Any]:
            """Analyze content using transformer models."""
            try:
                # Sentiment analysis
                sentiment_result = self.sentiment_analyzer(content.content[:512])
                
                # Content encoding
                inputs = self.tokenizer(
                    content.content[:512], 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                
                # Extract features
                analysis = {
                    'sentiment': sentiment_result[0]['label'],
                    'sentiment_score': sentiment_result[0]['score'],
                    'content_embedding': embeddings.cpu().numpy().tolist(),
                    'readability_score': self._calculate_readability(content.content),
                    'keyword_density': self._extract_keywords(content.content),
                    'content_length_score': min(len(content.content) / 1000, 1.0),
                    'hashtag_score': min(len(content.hashtags) / 5, 1.0),
                    'mention_score': min(len(content.mentions) / 3, 1.0)
                }
                
                return analysis
                
            except Exception as e:
                logger.error(f"Error in transformer analysis: {e}")
                # Fallback to basic analysis
                fallback = FallbackContentAnalyzer()
                return await fallback.analyze_content(content)
        
        def _calculate_readability(self, text: str) -> float:
            """Calculate text readability score."""
            sentences = text.split('.')
            words = text.split()
            syllables = sum(self._count_syllables(word) for word in words)
            
            if len(sentences) == 0 or len(words) == 0:
                return 0.0
            
            # Flesch Reading Ease
            flesch_score = 206.835 - (1.015 * len(words) / len(sentences)) - (84.6 * syllables / len(words))
            return max(0.0, min(100.0, flesch_score)) / 100.0
        
        def _count_syllables(self, word: str) -> int:
            """Count syllables in a word."""
            word = word.lower()
            count = 0
            vowels = "aeiouy"
            on_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not on_vowel:
                    count += 1
                on_vowel = is_vowel
            
            if word.endswith('e'):
                count -= 1
            return max(1, count)
        
        def _extract_keywords(self, text: str) -> Dict[str, float]:
            """Extract keywords and their frequency."""
            words = text.lower().split()
            word_freq = {}
            
            for word in words:
                if len(word) > 3:  # Filter short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Normalize frequencies
            max_freq = max(word_freq.values()) if word_freq else 1
            return {word: freq / max_freq for word, freq in word_freq.items()}

if SKLEARN_AVAILABLE:
    class MLContentOptimizer(ContentOptimizer):
        """ML-based content optimizer."""
        
        def __init__(self):
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.optimization_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
            self.is_trained = False
        
        async def optimize_content(
            self, 
            content: ContentData, 
            strategy: OptimizationStrategy
        ) -> ContentData:
            """Optimize content using ML models."""
            try:
                if not self.is_trained:
                    await self._train_model()
                
                # Create optimized content
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
                    optimized_content = await self._optimize_for_engagement(optimized_content)
                elif strategy == OptimizationStrategy.REACH:
                    optimized_content = await self._optimize_for_reach(optimized_content)
                elif strategy == OptimizationStrategy.CLICKS:
                    optimized_content = await self._optimize_for_clicks(optimized_content)
                elif strategy == OptimizationStrategy.SHARES:
                    optimized_content = await self._optimize_for_shares(optimized_content)
                elif strategy == OptimizationStrategy.COMMENTS:
                    optimized_content = await self._optimize_for_comments(optimized_content)
                
                return optimized_content
                
            except Exception as e:
                logger.error(f"Error in ML optimization: {e}")
                # Fallback to basic optimization
                fallback = FallbackContentOptimizer()
                return await fallback.optimize_content(content, strategy)
        
        async def _train_model(self):
            """Train the optimization model with sample data."""
            # This would typically use real training data
            # For now, we'll create synthetic data
            sample_texts = [
                "Great project completed! #coding #development",
                "Excited to share our new feature! #innovation #tech",
                "Team collaboration leads to success! #teamwork #leadership"
            ]
            
            # Fit vectorizer
            self.vectorizer.fit(sample_texts)
            self.is_trained = True
        
        async def _optimize_for_engagement(self, content: ContentData) -> ContentData:
            """Optimize content for maximum engagement."""
            if not content.content.endswith('?'):
                content.content += " What are your thoughts?"
            
            # Add engaging hashtags if not present
            engaging_hashtags = ['#engagement', '#discussion', '#community']
            for tag in engaging_hashtags:
                if tag not in content.hashtags:
                    content.hashtags.append(tag)
            
            return content
        
        async def _optimize_for_reach(self, content: ContentData) -> ContentData:
            """Optimize content for maximum reach."""
            # Add trending hashtags
            trending_tags = ['#trending', '#viral', '#popular']
            for tag in trending_tags:
                if tag not in content.hashtags:
                    content.hashtags.append(tag)
            
            return content
        
        async def _optimize_for_clicks(self, content: ContentData) -> ContentData:
            """Optimize content for maximum clicks."""
            if content.links:
                content.content += "\n\n🔗 Click the link above for more details!"
            
            return content
        
        async def _optimize_for_shares(self, content: ContentData) -> ContentData:
            """Optimize content for maximum shares."""
            content.content += "\n\n📤 Share this if you found it helpful!"
            
            return content
        
        async def _optimize_for_comments(self, content: ContentData) -> ContentData:
            """Optimize content for maximum comments."""
            if not content.content.endswith('?'):
                content.content += " What's your experience with this?"
            
            return content

if TORCH_AVAILABLE:
    class PyTorchEngagementPredictor(EngagementPredictor):
        """PyTorch-based engagement predictor."""
        
        def __init__(self):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self._create_model()
            self.model.to(self.device)
        
        def _create_model(self) -> nn.Module:
            """Create a simple neural network for engagement prediction."""
            model = nn.Sequential(
                nn.Linear(100, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            return model
        
        async def predict_engagement(
            self, 
            content: ContentData
        ) -> Tuple[float, float]:
            """Predict engagement using PyTorch model."""
            try:
                # Extract features
                features = self._extract_features(content)
                
                # Convert to tensor
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Make prediction
                with torch.no_grad():
                    prediction = self.model(features_tensor)
                    engagement_score = prediction.item()
                
                # Calculate confidence based on feature quality
                confidence_score = self._calculate_confidence(features)
                
                return engagement_score, confidence_score
                
            except Exception as e:
                logger.error(f"Error in PyTorch prediction: {e}")
                # Fallback to basic prediction
                fallback = FallbackEngagementPredictor()
                return await fallback.predict_engagement(content)
        
        def _extract_features(self, content: ContentData) -> List[float]:
            """Extract numerical features from content."""
            features = [
                len(content.content) / 1000,  # Normalized length
                len(content.hashtags) / 10,   # Normalized hashtag count
                len(content.mentions) / 5,    # Normalized mention count
                len(content.links) / 3,       # Normalized link count
                len(content.media_urls) / 2,  # Normalized media count
                # Add more features as needed
            ]
            
            # Pad to 100 features
            while len(features) < 100:
                features.append(0.0)
            
            return features[:100]
        
        def _calculate_confidence(self, features: List[float]) -> float:
            """Calculate confidence score based on feature quality."""
            # Simple confidence calculation
            non_zero_features = sum(1 for f in features if f > 0)
            return min(non_zero_features / len(features), 1.0)

# Main service class
class LinkedInOptimizationService:
    """Main service for LinkedIn content optimization."""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.cache = {}
        self.error_log = []
        
        # Initialize components based on available libraries
        if TORCH_AVAILABLE:
            self.analyzer = TransformerContentAnalyzer()
        else:
            self.analyzer = FallbackContentAnalyzer()
        
        if SKLEARN_AVAILABLE:
            self.optimizer = MLContentOptimizer()
        else:
            self.optimizer = FallbackContentOptimizer()
        
        if TORCH_AVAILABLE:
            self.predictor = PyTorchEngagementPredictor()
        else:
            self.predictor = FallbackEngagementPredictor()
    
    async def optimize_linkedin_post(
        self, 
        content: Union[str, ContentData], 
        strategy: OptimizationStrategy = OptimizationStrategy.ENGAGEMENT
    ) -> OptimizationResult:
        """Optimize a LinkedIn post for maximum performance."""
        start_time = time.time()
        
        try:
            # Convert string to ContentData if needed
            if isinstance(content, str):
                content_data = ContentData(
                    id=f"post_{int(time.time())}",
                    content=content,
                    content_type=ContentType.POST
                )
            else:
                content_data = content
            
            # Check cache
            cache_key = f"{content_data.get_content_hash()}_{strategy.value}"
            if cache_key in self.cache:
                cached_result = self.cache[cache_key]
                cached_result.processing_time = 0.001  # Cache hit time
                return cached_result
            
            # Analyze content
            self.monitor.start_operation('content_analysis')
            analysis = await self.analyzer.analyze_content(content_data)
            analysis_time = self.monitor.end_operation('content_analysis')
            
            # Optimize content
            self.monitor.start_operation('content_optimization')
            optimized_content = await self.optimizer.optimize_content(content_data, strategy)
            optimization_time = self.monitor.end_operation('content_optimization')
            
            # Predict engagement
            self.monitor.start_operation('engagement_prediction')
            original_engagement, original_confidence = await self.predictor.predict_engagement(content_data)
            optimized_engagement, optimized_confidence = await self.predictor.predict_engagement(optimized_content)
            prediction_time = self.monitor.end_operation('engagement_prediction')
            
            # Calculate improvements
            engagement_increase = (optimized_engagement - original_engagement) * 100
            optimization_score = min(engagement_increase, 100.0)
            
            improvements = self._generate_improvements(analysis, strategy)
            
            # Create result
            result = OptimizationResult(
                original_content=content_data,
                optimized_content=optimized_content,
                optimization_score=optimization_score,
                improvements=improvements,
                predicted_engagement_increase=engagement_increase,
                confidence_score=optimized_confidence,
                processing_time=time.time() - start_time
            )
            
            # Cache result
            self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            error = OptimizationError(
                message=str(e),
                error_type=type(e).__name__,
                level=ErrorLevel.HIGH,
                details={'strategy': strategy.value}
            )
            self.error_log.append(error)
            
            logger.error(f"Error in optimization: {e}")
            raise
    
    def _generate_improvements(self, analysis: Dict[str, Any], strategy: OptimizationStrategy) -> List[str]:
        """Generate improvement suggestions based on analysis."""
        improvements = []
        
        if analysis.get('readability_score', 0) < 0.6:
            improvements.append("Consider simplifying language for better readability")
        
        if analysis.get('hashtag_score', 0) < 0.4:
            improvements.append("Add more relevant hashtags to increase discoverability")
        
        if analysis.get('mention_score', 0) < 0.3:
            improvements.append("Consider mentioning relevant people or companies")
        
        if strategy == OptimizationStrategy.ENGAGEMENT:
            improvements.append("Added engaging question to encourage interaction")
        elif strategy == OptimizationStrategy.REACH:
            improvements.append("Added trending hashtags for better visibility")
        elif strategy == OptimizationStrategy.CLICKS:
            improvements.append("Enhanced call-to-action for link clicks")
        
        return improvements
    
    async def batch_optimize(
        self, 
        contents: List[Union[str, ContentData]], 
        strategy: OptimizationStrategy = OptimizationStrategy.ENGAGEMENT
    ) -> List[OptimizationResult]:
        """Optimize multiple posts in batch."""
        tasks = [
            self.optimize_linkedin_post(content, strategy) 
            for content in contents
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors
        valid_results = []
        for result in results:
            if isinstance(result, OptimizationResult):
                valid_results.append(result)
            else:
                logger.error(f"Batch optimization error: {result}")
        
        return valid_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get system performance statistics."""
        return self.monitor.get_stats()
    
    def get_error_log(self) -> List[OptimizationError]:
        """Get error log."""
        return self.error_log.copy()
    
    def clear_cache(self):
        """Clear optimization cache."""
        self.cache.clear()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'analyzer': 'available' if hasattr(self.analyzer, 'analyze_content') else 'unavailable',
                'optimizer': 'available' if hasattr(self.optimizer, 'optimize_content') else 'unavailable',
                'predictor': 'available' if hasattr(self.predictor, 'predict_engagement') else 'unavailable'
            },
            'performance': self.get_performance_stats(),
            'errors': len(self.error_log)
        }
        
        # Check for critical errors
        critical_errors = [e for e in self.error_log if e.level == ErrorLevel.CRITICAL]
        if critical_errors:
            health_status['status'] = 'degraded'
            health_status['critical_errors'] = len(critical_errors)
        
        return health_status

# Factory function
def create_linkedin_optimization_service() -> LinkedInOptimizationService:
    """Create and return a LinkedIn optimization service instance."""
    return LinkedInOptimizationService()

# Utility functions
async def demo_optimization():
    """Demonstrate the optimization system."""
    service = create_linkedin_optimization_service()
    
    # Test content
    test_content = "Just finished an amazing project! #coding #development"
    
    print("🚀 LinkedIn Posts Optimization Demo")
    print("=" * 50)
    
    # Test different strategies
    strategies = [
        OptimizationStrategy.ENGAGEMENT,
        OptimizationStrategy.REACH,
        OptimizationStrategy.CLICKS
    ]
    
    for strategy in strategies:
        print(f"\n📊 Testing {strategy.value.upper()} strategy...")
        
        try:
            result = await service.optimize_linkedin_post(test_content, strategy)
            
            print(f"✅ Optimization Score: {result.optimization_score:.1f}%")
            print(f"📈 Engagement Increase: {result.predicted_engagement_increase:.1f}%")
            print(f"🎯 Confidence: {result.confidence_score:.1f}")
            print(f"⚡ Processing Time: {result.processing_time:.3f}s")
            print(f"💡 Improvements: {', '.join(result.improvements)}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Health check
    print(f"\n🏥 System Health Check:")
    health = await service.health_check()
    print(f"Status: {health['status']}")
    print(f"Errors: {health['errors']}")

if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_optimization())

