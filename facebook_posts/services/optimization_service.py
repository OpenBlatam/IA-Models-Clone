"""
Advanced Optimization Service for Facebook Posts API
AI-powered content optimization, A/B testing, and performance enhancement
"""

import asyncio
import json
import time
import random
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog

from ..core.config import get_settings
from ..core.models import FacebookPost, PostStatus, ContentType, AudienceType, OptimizationLevel
from ..infrastructure.cache import get_cache_manager
from ..infrastructure.database import get_db_manager, PostRepository
from ..infrastructure.monitoring import get_monitor, timed
from ..services.ai_service import get_ai_service
from ..services.analytics_service import get_analytics_service

logger = structlog.get_logger(__name__)


class OptimizationStrategy(Enum):
    """Optimization strategy types"""
    ENGAGEMENT = "engagement"
    REACH = "reach"
    CLICKS = "clicks"
    SHARES = "shares"
    COMMENTS = "comments"
    SENTIMENT = "sentiment"
    READABILITY = "readability"
    CREATIVITY = "creativity"


class TestType(Enum):
    """A/B test types"""
    HEADLINE = "headline"
    CONTENT = "content"
    HASHTAGS = "hashtags"
    POSTING_TIME = "posting_time"
    AUDIENCE = "audience"
    TONE = "tone"
    LENGTH = "length"


@dataclass
class OptimizationResult:
    """Result of content optimization"""
    original_content: str
    optimized_content: str
    optimization_type: str
    expected_improvement: float
    confidence_score: float
    changes_made: List[str]
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTest:
    """A/B test configuration"""
    test_id: str
    test_type: TestType
    original_variant: Dict[str, Any]
    test_variant: Dict[str, Any]
    target_metric: str
    test_duration_days: int
    created_at: datetime
    status: str = "active"  # active, completed, paused
    results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceInsight:
    """Performance insight data"""
    metric: str
    current_value: float
    optimal_value: float
    improvement_potential: float
    recommendation: str
    implementation: str
    expected_impact: str


class ContentOptimizer:
    """Advanced content optimization engine"""
    
    def __init__(self):
        self.ai_service = get_ai_service()
        self.analytics_service = get_analytics_service()
        self.cache_manager = get_cache_manager()
        self.monitor = get_monitor()
        self.optimization_rules = self._load_optimization_rules()
    
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """Load optimization rules and patterns"""
        return {
            "engagement": {
                "question_patterns": ["What do you think?", "Share your experience", "What's your opinion?"],
                "cta_patterns": ["Learn more", "Discover", "Find out", "Get started"],
                "emoji_usage": ["🚀", "💡", "🎯", "⭐", "🔥"],
                "optimal_length": (100, 200),
                "hashtag_count": (3, 5)
            },
            "reach": {
                "trending_topics": ["AI", "technology", "innovation", "business"],
                "optimal_posting_times": ["9:00", "13:00", "19:00"],
                "hashtag_strategy": "trending",
                "content_type_preference": ["educational", "entertainment"]
            },
            "clicks": {
                "link_placement": "middle",
                "link_text": "Learn more", "Discover", "Find out",
                "urgency_words": ["now", "today", "limited", "exclusive"],
                "benefit_focused": True
            },
            "shares": {
                "emotional_triggers": ["inspire", "motivate", "surprise", "educate"],
                "quote_format": True,
                "visual_elements": True,
                "storytelling": True
            },
            "comments": {
                "question_types": ["open_ended", "personal", "controversial"],
                "debate_starters": ["agree or disagree", "what's your take", "your thoughts"],
                "community_focused": True
            }
        }
    
    @timed("content_optimization")
    async def optimize_content(
        self,
        content: str,
        strategy: OptimizationStrategy,
        target_audience: AudienceType,
        current_metrics: Optional[Dict[str, float]] = None
    ) -> OptimizationResult:
        """Optimize content based on strategy"""
        start_time = time.time()
        
        try:
            # Get optimization rules for strategy
            rules = self.optimization_rules.get(strategy.value, {})
            
            # Analyze current content
            analysis = await self.ai_service.analyze_content(content)
            
            # Generate optimized content
            optimized_content = await self._apply_optimization_rules(
                content, rules, strategy, target_audience, analysis
            )
            
            # Calculate expected improvement
            expected_improvement = await self._calculate_improvement_potential(
                content, optimized_content, strategy, analysis
            )
            
            # Generate changes summary
            changes_made = self._generate_changes_summary(content, optimized_content)
            
            processing_time = time.time() - start_time
            
            result = OptimizationResult(
                original_content=content,
                optimized_content=optimized_content,
                optimization_type=strategy.value,
                expected_improvement=expected_improvement,
                confidence_score=analysis.quality_score,
                changes_made=changes_made,
                processing_time=processing_time,
                metadata={
                    "strategy": strategy.value,
                    "target_audience": target_audience.value,
                    "original_analysis": analysis.__dict__,
                    "rules_applied": list(rules.keys())
                }
            )
            
            # Cache result
            cache_key = f"optimization:{hash(content)}:{strategy.value}:{target_audience.value}"
            await self.cache_manager.cache.set(cache_key, result.__dict__, ttl=3600)
            
            logger.info(
                "Content optimization completed",
                strategy=strategy.value,
                expected_improvement=expected_improvement,
                processing_time=processing_time
            )
            
            return result
            
        except Exception as e:
            logger.error("Content optimization failed", error=str(e), exc_info=True)
            raise
    
    async def _apply_optimization_rules(
        self,
        content: str,
        rules: Dict[str, Any],
        strategy: OptimizationStrategy,
        target_audience: AudienceType,
        analysis: Any
    ) -> str:
        """Apply optimization rules to content"""
        optimized_content = content
        
        if strategy == OptimizationStrategy.ENGAGEMENT:
            optimized_content = await self._optimize_for_engagement(optimized_content, rules, analysis)
        elif strategy == OptimizationStrategy.REACH:
            optimized_content = await self._optimize_for_reach(optimized_content, rules, target_audience)
        elif strategy == OptimizationStrategy.CLICKS:
            optimized_content = await self._optimize_for_clicks(optimized_content, rules)
        elif strategy == OptimizationStrategy.SHARES:
            optimized_content = await self._optimize_for_shares(optimized_content, rules)
        elif strategy == OptimizationStrategy.COMMENTS:
            optimized_content = await self._optimize_for_comments(optimized_content, rules)
        elif strategy == OptimizationStrategy.SENTIMENT:
            optimized_content = await self._optimize_for_sentiment(optimized_content, analysis)
        elif strategy == OptimizationStrategy.READABILITY:
            optimized_content = await self._optimize_for_readability(optimized_content, analysis)
        elif strategy == OptimizationStrategy.CREATIVITY:
            optimized_content = await self._optimize_for_creativity(optimized_content, analysis)
        
        return optimized_content
    
    async def _optimize_for_engagement(self, content: str, rules: Dict[str, Any], analysis: Any) -> str:
        """Optimize content for engagement"""
        optimized = content
        
        # Add question if engagement score is low
        if analysis.engagement_score < 0.6 and "?" not in content:
            questions = rules.get("question_patterns", [])
            if questions:
                optimized += f"\n\n{random.choice(questions)}"
        
        # Add call-to-action if missing
        if not any(cta in content.lower() for cta in ["learn", "discover", "find", "get"]):
            ctas = rules.get("cta_patterns", [])
            if ctas:
                optimized += f"\n\n{random.choice(ctas)}"
        
        # Add emojis if engagement is low
        if analysis.engagement_score < 0.5:
            emojis = rules.get("emoji_usage", [])
            if emojis and not any(emoji in content for emoji in emojis):
                optimized = f"{random.choice(emojis)} {optimized}"
        
        return optimized
    
    async def _optimize_for_reach(self, content: str, rules: Dict[str, Any], target_audience: AudienceType) -> str:
        """Optimize content for reach"""
        optimized = content
        
        # Add trending hashtags
        trending_topics = rules.get("trending_topics", [])
        if trending_topics:
            hashtags = [f"#{topic}" for topic in trending_topics[:3]]
            if not any(hashtag in content for hashtag in hashtags):
                optimized += f"\n\n{' '.join(hashtags)}"
        
        return optimized
    
    async def _optimize_for_clicks(self, content: str, rules: Dict[str, Any]) -> str:
        """Optimize content for clicks"""
        optimized = content
        
        # Add urgency words
        urgency_words = rules.get("urgency_words", [])
        if urgency_words and not any(word in content.lower() for word in urgency_words):
            optimized = f"{random.choice(urgency_words).title()} - {optimized}"
        
        return optimized
    
    async def _optimize_for_shares(self, content: str, rules: Dict[str, Any]) -> str:
        """Optimize content for shares"""
        optimized = content
        
        # Add emotional triggers
        emotional_triggers = rules.get("emotional_triggers", [])
        if emotional_triggers:
            trigger = random.choice(emotional_triggers)
            if trigger not in content.lower():
                optimized = f"This will {trigger} you: {optimized}"
        
        return optimized
    
    async def _optimize_for_comments(self, content: str, rules: Dict[str, Any]) -> str:
        """Optimize content for comments"""
        optimized = content
        
        # Add debate starters
        debate_starters = rules.get("debate_starters", [])
        if debate_starters and "?" not in content:
            starter = random.choice(debate_starters)
            optimized += f"\n\n{starter}?"
        
        return optimized
    
    async def _optimize_for_sentiment(self, content: str, analysis: Any) -> str:
        """Optimize content for sentiment"""
        if analysis.sentiment_score < 0.3:
            # Make more positive
            positive_words = ["amazing", "incredible", "fantastic", "wonderful", "excellent"]
            negative_words = ["bad", "terrible", "awful", "horrible", "disappointing"]
            
            optimized = content
            for neg_word in negative_words:
                if neg_word in content.lower():
                    optimized = optimized.replace(neg_word, random.choice(positive_words))
            
            return optimized
        
        return content
    
    async def _optimize_for_readability(self, content: str, analysis: Any) -> str:
        """Optimize content for readability"""
        if analysis.readability_score < 0.6:
            # Simplify sentences
            sentences = content.split(". ")
            simplified_sentences = []
            
            for sentence in sentences:
                if len(sentence.split()) > 20:  # Long sentence
                    # Split into shorter sentences
                    words = sentence.split()
                    mid_point = len(words) // 2
                    simplified_sentences.append(" ".join(words[:mid_point]) + ".")
                    simplified_sentences.append(" ".join(words[mid_point:]))
                else:
                    simplified_sentences.append(sentence)
            
            return ". ".join(simplified_sentences)
        
        return content
    
    async def _optimize_for_creativity(self, content: str, analysis: Any) -> str:
        """Optimize content for creativity"""
        if analysis.creativity_score < 0.6:
            # Add creative elements
            creative_elements = ["✨", "🎨", "💫", "🌟", "🎭"]
            if not any(element in content for element in creative_elements):
                content = f"{random.choice(creative_elements)} {content}"
            
            # Add wordplay or alliteration
            words = content.split()
            if len(words) >= 3:
                # Simple alliteration
                first_letters = [word[0].lower() for word in words if word.isalpha()]
                if len(set(first_letters)) > len(first_letters) * 0.7:  # Not much alliteration
                    # Add some alliteration
                    content = f"Brilliant {content.lower()}"
        
        return content
    
    async def _calculate_improvement_potential(
        self,
        original_content: str,
        optimized_content: str,
        strategy: OptimizationStrategy,
        analysis: Any
    ) -> float:
        """Calculate expected improvement potential"""
        base_improvement = 0.1  # 10% base improvement
        
        # Strategy-specific improvements
        strategy_improvements = {
            OptimizationStrategy.ENGAGEMENT: 0.2,
            OptimizationStrategy.REACH: 0.15,
            OptimizationStrategy.CLICKS: 0.25,
            OptimizationStrategy.SHARES: 0.18,
            OptimizationStrategy.COMMENTS: 0.22,
            OptimizationStrategy.SENTIMENT: 0.12,
            OptimizationStrategy.READABILITY: 0.08,
            OptimizationStrategy.CREATIVITY: 0.14
        }
        
        strategy_improvement = strategy_improvements.get(strategy, 0.1)
        
        # Adjust based on current performance
        if strategy == OptimizationStrategy.ENGAGEMENT and analysis.engagement_score < 0.5:
            strategy_improvement *= 1.5
        elif strategy == OptimizationStrategy.SENTIMENT and analysis.sentiment_score < 0.3:
            strategy_improvement *= 1.3
        
        return min(base_improvement + strategy_improvement, 0.5)  # Cap at 50%
    
    def _generate_changes_summary(self, original: str, optimized: str) -> List[str]:
        """Generate summary of changes made"""
        changes = []
        
        if len(optimized) > len(original):
            changes.append("Added engaging elements")
        
        if "?" in optimized and "?" not in original:
            changes.append("Added question to encourage engagement")
        
        if any(emoji in optimized for emoji in ["🚀", "💡", "🎯", "⭐", "🔥"]) and not any(emoji in original for emoji in ["🚀", "💡", "🎯", "⭐", "🔥"]):
            changes.append("Added emojis for visual appeal")
        
        if len(optimized.split(". ")) > len(original.split(". ")):
            changes.append("Improved readability with shorter sentences")
        
        if not changes:
            changes.append("Minor content refinements")
        
        return changes


class ABTestManager:
    """A/B testing manager for content optimization"""
    
    def __init__(self):
        self.cache_manager = get_cache_manager()
        self.analytics_service = get_analytics_service()
        self.db_manager = get_db_manager()
        self.active_tests: Dict[str, ABTest] = {}
        self.test_results: Dict[str, Dict[str, Any]] = {}
    
    async def create_ab_test(
        self,
        test_type: TestType,
        original_variant: Dict[str, Any],
        test_variant: Dict[str, Any],
        target_metric: str,
        test_duration_days: int = 7
    ) -> ABTest:
        """Create a new A/B test"""
        test_id = f"test_{int(time.time())}_{random.randint(1000, 9999)}"
        
        test = ABTest(
            test_id=test_id,
            test_type=test_type,
            original_variant=original_variant,
            test_variant=test_variant,
            target_metric=target_metric,
            test_duration_days=test_duration_days,
            created_at=datetime.now()
        )
        
        self.active_tests[test_id] = test
        
        # Cache test configuration
        await self.cache_manager.cache.set(f"ab_test:{test_id}", test.__dict__, ttl=test_duration_days * 86400)
        
        logger.info("A/B test created", test_id=test_id, test_type=test_type.value)
        
        return test
    
    async def get_test_variant(self, test_id: str, user_id: str) -> Dict[str, Any]:
        """Get test variant for user (A/B split)"""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.active_tests[test_id]
        
        # Simple 50/50 split based on user ID hash
        user_hash = hash(user_id) % 2
        
        if user_hash == 0:
            return test.original_variant
        else:
            return test.test_variant
    
    async def record_test_metric(self, test_id: str, variant: str, metric_value: float):
        """Record metric for A/B test"""
        if test_id not in self.test_results:
            self.test_results[test_id] = {"original": [], "test": []}
        
        self.test_results[test_id][variant].append({
            "value": metric_value,
            "timestamp": datetime.now()
        })
        
        # Update cache
        await self.cache_manager.cache.set(
            f"ab_test_results:{test_id}",
            self.test_results[test_id],
            ttl=86400
        )
    
    async def analyze_test_results(self, test_id: str) -> Dict[str, Any]:
        """Analyze A/B test results"""
        if test_id not in self.test_results:
            return {"error": "No test results found"}
        
        results = self.test_results[test_id]
        
        if not results["original"] or not results["test"]:
            return {"error": "Insufficient test data"}
        
        # Calculate statistics
        original_values = [r["value"] for r in results["original"]]
        test_values = [r["value"] for r in results["test"]]
        
        original_mean = sum(original_values) / len(original_values)
        test_mean = sum(test_values) / len(test_values)
        
        improvement = ((test_mean - original_mean) / original_mean) * 100 if original_mean > 0 else 0
        
        # Simple statistical significance (t-test approximation)
        significance = self._calculate_significance(original_values, test_values)
        
        return {
            "test_id": test_id,
            "original_variant": {
                "mean": original_mean,
                "count": len(original_values)
            },
            "test_variant": {
                "mean": test_mean,
                "count": len(test_values)
            },
            "improvement_percentage": improvement,
            "statistical_significance": significance,
            "recommendation": "test_variant" if improvement > 5 and significance > 0.95 else "original_variant"
        }
    
    def _calculate_significance(self, original: List[float], test: List[float]) -> float:
        """Calculate statistical significance (simplified)"""
        if len(original) < 30 or len(test) < 30:
            return 0.8  # Lower confidence for small samples
        
        # Simple approximation
        original_mean = sum(original) / len(original)
        test_mean = sum(test) / len(test)
        
        if abs(test_mean - original_mean) / original_mean > 0.1:  # 10% difference
            return 0.95
        elif abs(test_mean - original_mean) / original_mean > 0.05:  # 5% difference
            return 0.85
        else:
            return 0.7


class PerformanceOptimizer:
    """Performance optimization and insights"""
    
    def __init__(self):
        self.analytics_service = get_analytics_service()
        self.cache_manager = get_cache_manager()
        self.optimization_rules = self._load_performance_rules()
    
    def _load_performance_rules(self) -> Dict[str, Any]:
        """Load performance optimization rules"""
        return {
            "posting_times": {
                "professionals": ["9:00", "13:00", "17:00"],
                "general": ["8:00", "12:00", "19:00"],
                "students": ["10:00", "14:00", "20:00"]
            },
            "content_length": {
                "optimal": (100, 200),
                "minimum": 50,
                "maximum": 300
            },
            "hashtag_count": {
                "optimal": 3,
                "minimum": 1,
                "maximum": 5
            },
            "engagement_thresholds": {
                "excellent": 0.8,
                "good": 0.6,
                "average": 0.4,
                "poor": 0.2
            }
        }
    
    async def generate_performance_insights(self, post_id: str) -> List[PerformanceInsight]:
        """Generate performance insights for a post"""
        try:
            # Get post metrics
            metrics = await self.analytics_service.get_post_analytics(post_id)
            
            if not metrics:
                return []
            
            insights = []
            
            # Engagement insight
            if metrics.engagement_rate < self.optimization_rules["engagement_thresholds"]["good"]:
                insights.append(PerformanceInsight(
                    metric="engagement_rate",
                    current_value=metrics.engagement_rate,
                    optimal_value=self.optimization_rules["engagement_thresholds"]["good"],
                    improvement_potential=0.2,
                    recommendation="Add questions or call-to-action to increase engagement",
                    implementation="Include 'What do you think?' or 'Share your experience'",
                    expected_impact="20-30% increase in engagement"
                ))
            
            # Reach insight
            if metrics.reach < metrics.impressions * 0.8:
                insights.append(PerformanceInsight(
                    metric="reach",
                    current_value=metrics.reach,
                    optimal_value=metrics.impressions * 0.9,
                    improvement_potential=0.15,
                    recommendation="Optimize posting time for better reach",
                    implementation="Post during peak hours (9 AM, 1 PM, 7 PM)",
                    expected_impact="15-25% increase in reach"
                ))
            
            # Clicks insight
            if metrics.clicks < metrics.views * 0.05:
                insights.append(PerformanceInsight(
                    metric="clicks",
                    current_value=metrics.clicks,
                    optimal_value=metrics.views * 0.1,
                    improvement_potential=0.3,
                    recommendation="Add compelling call-to-action",
                    implementation="Include 'Learn more' or 'Discover' with link",
                    expected_impact="30-50% increase in clicks"
                ))
            
            return insights
            
        except Exception as e:
            logger.error("Failed to generate performance insights", post_id=post_id, error=str(e))
            return []
    
    async def get_optimization_recommendations(self, audience_type: AudienceType) -> List[Dict[str, Any]]:
        """Get optimization recommendations for audience type"""
        try:
            # Get audience insights
            audience_insight = await self.analytics_service.get_audience_insights(audience_type.value)
            
            recommendations = []
            
            # Posting time recommendations
            optimal_times = self.optimization_rules["posting_times"].get(audience_type.value, ["9:00", "13:00", "19:00"])
            recommendations.append({
                "type": "posting_time",
                "title": "Optimal Posting Times",
                "description": f"Post during these times for {audience_type.value} audience",
                "recommendation": optimal_times,
                "impact": "20-30% increase in reach"
            })
            
            # Content type recommendations
            if audience_insight.preferred_content_types:
                recommendations.append({
                    "type": "content_type",
                    "title": "Preferred Content Types",
                    "description": f"Focus on these content types for {audience_type.value} audience",
                    "recommendation": audience_insight.preferred_content_types,
                    "impact": "15-25% increase in engagement"
                })
            
            # Hashtag recommendations
            recommendations.append({
                "type": "hashtags",
                "title": "Hashtag Strategy",
                "description": "Optimal hashtag usage",
                "recommendation": f"Use {self.optimization_rules['hashtag_count']['optimal']} relevant hashtags",
                "impact": "10-20% increase in discoverability"
            })
            
            return recommendations
            
        except Exception as e:
            logger.error("Failed to get optimization recommendations", audience_type=audience_type.value, error=str(e))
            return []


class OptimizationService:
    """Main optimization service orchestrator"""
    
    def __init__(self):
        self.content_optimizer = ContentOptimizer()
        self.ab_test_manager = ABTestManager()
        self.performance_optimizer = PerformanceOptimizer()
        self.cache_manager = get_cache_manager()
        self.monitor = get_monitor()
    
    async def optimize_content(
        self,
        content: str,
        strategy: OptimizationStrategy,
        target_audience: AudienceType,
        current_metrics: Optional[Dict[str, float]] = None
    ) -> OptimizationResult:
        """Optimize content using advanced optimization engine"""
        return await self.content_optimizer.optimize_content(
            content, strategy, target_audience, current_metrics
        )
    
    async def create_ab_test(
        self,
        test_type: TestType,
        original_variant: Dict[str, Any],
        test_variant: Dict[str, Any],
        target_metric: str,
        test_duration_days: int = 7
    ) -> ABTest:
        """Create A/B test for content optimization"""
        return await self.ab_test_manager.create_ab_test(
            test_type, original_variant, test_variant, target_metric, test_duration_days
        )
    
    async def get_test_variant(self, test_id: str, user_id: str) -> Dict[str, Any]:
        """Get test variant for user"""
        return await self.ab_test_manager.get_test_variant(test_id, user_id)
    
    async def record_test_metric(self, test_id: str, variant: str, metric_value: float):
        """Record metric for A/B test"""
        await self.ab_test_manager.record_test_metric(test_id, variant, metric_value)
    
    async def analyze_test_results(self, test_id: str) -> Dict[str, Any]:
        """Analyze A/B test results"""
        return await self.ab_test_manager.analyze_test_results(test_id)
    
    async def generate_performance_insights(self, post_id: str) -> List[PerformanceInsight]:
        """Generate performance insights"""
        return await self.performance_optimizer.generate_performance_insights(post_id)
    
    async def get_optimization_recommendations(self, audience_type: AudienceType) -> List[Dict[str, Any]]:
        """Get optimization recommendations"""
        return await self.performance_optimizer.get_optimization_recommendations(audience_type)


# Global optimization service instance
_optimization_service: Optional[OptimizationService] = None


def get_optimization_service() -> OptimizationService:
    """Get global optimization service instance"""
    global _optimization_service
    
    if _optimization_service is None:
        _optimization_service = OptimizationService()
    
    return _optimization_service


# Export all classes and functions
__all__ = [
    # Enums
    'OptimizationStrategy',
    'TestType',
    
    # Data classes
    'OptimizationResult',
    'ABTest',
    'PerformanceInsight',
    
    # Services
    'ContentOptimizer',
    'ABTestManager',
    'PerformanceOptimizer',
    'OptimizationService',
    
    # Utility functions
    'get_optimization_service',
]






























