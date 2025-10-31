"""
Content Optimization Engine
===========================

Advanced content optimization and A/B testing capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import re

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from ..schemas import CopywritingRequest, CopywritingVariant, FeedbackRequest
from ..services import CopywritingRecord, FeedbackRecord
from ..utils import calculate_confidence_score, extract_keywords, calculate_readability_score

logger = logging.getLogger(__name__)


class OptimizationStrategy(str, Enum):
    """Content optimization strategies"""
    A_B_TESTING = "a_b_testing"
    KEYWORD_OPTIMIZATION = "keyword_optimization"
    READABILITY_IMPROVEMENT = "readability_improvement"
    ENGAGEMENT_BOOST = "engagement_boost"
    CONVERSION_OPTIMIZATION = "conversion_optimization"
    TONE_ADJUSTMENT = "tone_adjustment"


class OptimizationGoal(str, Enum):
    """Optimization goals"""
    CLICK_THROUGH_RATE = "ctr"
    CONVERSION_RATE = "conversion"
    ENGAGEMENT_TIME = "engagement"
    SHARE_RATE = "shares"
    FEEDBACK_SCORE = "feedback"
    READABILITY = "readability"


@dataclass
class OptimizationResult:
    """Result of content optimization"""
    original_variant: CopywritingVariant
    optimized_variant: CopywritingVariant
    optimization_strategy: OptimizationStrategy
    improvement_score: float
    changes_made: List[str]
    confidence_boost: float


@dataclass
class ABTestResult:
    """A/B test result"""
    test_id: str
    variant_a: CopywritingVariant
    variant_b: CopywritingVariant
    winner: str  # "A", "B", or "tie"
    confidence_level: float
    sample_size: int
    metrics: Dict[str, float]
    statistical_significance: bool


@dataclass
class ContentInsight:
    """Content insight and recommendation"""
    insight_type: str
    description: str
    impact_score: float
    recommendation: str
    implementation_difficulty: str  # "easy", "medium", "hard"
    expected_improvement: float


class ContentOptimizer:
    """Advanced content optimization engine"""
    
    def __init__(self):
        self.optimization_rules = self._load_optimization_rules()
        self.ab_tests: Dict[str, ABTestResult] = {}
        self.performance_data: Dict[str, List[float]] = {}
    
    async def optimize_content(
        self,
        variant: CopywritingVariant,
        request: CopywritingRequest,
        strategy: OptimizationStrategy,
        goal: OptimizationGoal = OptimizationGoal.CONVERSION_RATE
    ) -> OptimizationResult:
        """Optimize content using specified strategy"""
        
        logger.info(f"Optimizing content with strategy: {strategy}")
        
        # Create a copy for optimization
        optimized_content = variant.content
        changes_made = []
        
        if strategy == OptimizationStrategy.KEYWORD_OPTIMIZATION:
            optimized_content, changes = await self._optimize_keywords(
                optimized_content, request
            )
            changes_made.extend(changes)
        
        elif strategy == OptimizationStrategy.READABILITY_IMPROVEMENT:
            optimized_content, changes = await self._improve_readability(
                optimized_content, request
            )
            changes_made.extend(changes)
        
        elif strategy == OptimizationStrategy.ENGAGEMENT_BOOST:
            optimized_content, changes = await self._boost_engagement(
                optimized_content, request
            )
            changes_made.extend(changes)
        
        elif strategy == OptimizationStrategy.CONVERSION_OPTIMIZATION:
            optimized_content, changes = await self._optimize_conversion(
                optimized_content, request
            )
            changes_made.extend(changes)
        
        elif strategy == OptimizationStrategy.TONE_ADJUSTMENT:
            optimized_content, changes = await self._adjust_tone(
                optimized_content, request
            )
            changes_made.extend(changes)
        
        # Create optimized variant
        optimized_variant = CopywritingVariant(
            title=variant.title,
            content=optimized_content,
            word_count=len(optimized_content.split()),
            cta=variant.cta,
            confidence_score=self._calculate_optimized_confidence(
                optimized_content, request
            )
        )
        
        # Calculate improvement
        improvement_score = self._calculate_improvement_score(
            variant, optimized_variant, goal
        )
        
        confidence_boost = optimized_variant.confidence_score - variant.confidence_score
        
        return OptimizationResult(
            original_variant=variant,
            optimized_variant=optimized_variant,
            optimization_strategy=strategy,
            improvement_score=improvement_score,
            changes_made=changes_made,
            confidence_boost=confidence_boost
        )
    
    async def run_ab_test(
        self,
        variant_a: CopywritingVariant,
        variant_b: CopywritingVariant,
        test_duration_hours: int = 24,
        target_metric: OptimizationGoal = OptimizationGoal.CONVERSION_RATE
    ) -> ABTestResult:
        """Run A/B test between two variants"""
        
        test_id = f"ab_test_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting A/B test {test_id}")
        
        # Simulate A/B test (in real implementation, this would track actual user interactions)
        await asyncio.sleep(0.1)  # Simulate test duration
        
        # Mock performance data
        performance_a = await self._simulate_performance(variant_a, target_metric)
        performance_b = await self._simulate_performance(variant_b, target_metric)
        
        # Calculate winner
        if performance_a > performance_b * 1.05:  # 5% improvement threshold
            winner = "A"
            confidence_level = 0.85
        elif performance_b > performance_a * 1.05:
            winner = "B"
            confidence_level = 0.85
        else:
            winner = "tie"
            confidence_level = 0.5
        
        # Statistical significance (simplified)
        sample_size = 1000  # Mock sample size
        statistical_significance = confidence_level > 0.8
        
        result = ABTestResult(
            test_id=test_id,
            variant_a=variant_a,
            variant_b=variant_b,
            winner=winner,
            confidence_level=confidence_level,
            sample_size=sample_size,
            metrics={
                f"{target_metric}_a": performance_a,
                f"{target_metric}_b": performance_b,
                "improvement_percentage": abs(performance_b - performance_a) / performance_a * 100
            },
            statistical_significance=statistical_significance
        )
        
        self.ab_tests[test_id] = result
        return result
    
    async def get_content_insights(
        self,
        variant: CopywritingVariant,
        request: CopywritingRequest,
        historical_data: Optional[List[Dict]] = None
    ) -> List[ContentInsight]:
        """Get content insights and recommendations"""
        
        insights = []
        
        # Readability analysis
        readability_score = calculate_readability_score(variant.content)
        if readability_score < 0.6:
            insights.append(ContentInsight(
                insight_type="readability",
                description=f"Content readability score is {readability_score:.2f}, below optimal range",
                impact_score=0.8,
                recommendation="Simplify sentence structure and use shorter words",
                implementation_difficulty="easy",
                expected_improvement=0.15
            ))
        
        # Keyword density analysis
        keywords = extract_keywords(variant.content)
        if len(keywords) < 3:
            insights.append(ContentInsight(
                insight_type="keywords",
                description="Content has low keyword density",
                impact_score=0.7,
                recommendation="Include more relevant keywords naturally",
                implementation_difficulty="medium",
                expected_improvement=0.12
            ))
        
        # CTA analysis
        if request.include_cta and not variant.cta:
            insights.append(ContentInsight(
                insight_type="cta",
                description="Content lacks a clear call-to-action",
                impact_score=0.9,
                recommendation="Add a compelling call-to-action",
                implementation_difficulty="easy",
                expected_improvement=0.25
            ))
        
        # Word count analysis
        if request.word_count:
            word_ratio = variant.word_count / request.word_count
            if word_ratio < 0.8 or word_ratio > 1.2:
                insights.append(ContentInsight(
                    insight_type="length",
                    description=f"Content length is {word_ratio:.1f}x target length",
                    impact_score=0.6,
                    recommendation="Adjust content length to match target",
                    implementation_difficulty="medium",
                    expected_improvement=0.08
                ))
        
        # Tone consistency analysis
        tone_consistency = self._analyze_tone_consistency(variant.content, request.tone)
        if tone_consistency < 0.7:
            insights.append(ContentInsight(
                insight_type="tone",
                description="Content tone doesn't match requested tone",
                impact_score=0.7,
                recommendation="Adjust language to match target tone",
                implementation_difficulty="medium",
                expected_improvement=0.10
            ))
        
        # Historical performance insights
        if historical_data:
            historical_insights = await self._analyze_historical_performance(
                variant, historical_data
            )
            insights.extend(historical_insights)
        
        # Sort by impact score
        insights.sort(key=lambda x: x.impact_score, reverse=True)
        
        return insights
    
    async def batch_optimize(
        self,
        variants: List[CopywritingVariant],
        request: CopywritingRequest,
        strategies: List[OptimizationStrategy]
    ) -> List[OptimizationResult]:
        """Optimize multiple variants with multiple strategies"""
        
        results = []
        
        for variant in variants:
            for strategy in strategies:
                try:
                    result = await self.optimize_content(variant, request, strategy)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to optimize variant with {strategy}: {e}")
        
        # Sort by improvement score
        results.sort(key=lambda x: x.improvement_score, reverse=True)
        
        return results
    
    # Optimization strategies implementation
    async def _optimize_keywords(
        self, 
        content: str, 
        request: CopywritingRequest
    ) -> Tuple[str, List[str]]:
        """Optimize keyword usage"""
        changes = []
        
        # Extract topic keywords
        topic_keywords = extract_keywords(request.topic)
        
        # Check keyword density
        content_lower = content.lower()
        for keyword in topic_keywords:
            if keyword.lower() not in content_lower:
                # Add keyword naturally
                content = self._insert_keyword_naturally(content, keyword)
                changes.append(f"Added keyword: {keyword}")
        
        return content, changes
    
    async def _improve_readability(
        self, 
        content: str, 
        request: CopywritingRequest
    ) -> Tuple[str, List[str]]:
        """Improve content readability"""
        changes = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        
        # Shorten long sentences
        improved_sentences = []
        for sentence in sentences:
            if len(sentence.split()) > 20:  # Long sentence
                # Split into shorter sentences
                shorter_sentences = self._split_long_sentence(sentence)
                improved_sentences.extend(shorter_sentences)
                changes.append("Split long sentence for better readability")
            else:
                improved_sentences.append(sentence)
        
        content = '. '.join(improved_sentences)
        
        # Replace complex words with simpler ones
        content = self._simplify_vocabulary(content)
        if content != content:  # If changes were made
            changes.append("Simplified vocabulary")
        
        return content, changes
    
    async def _boost_engagement(
        self, 
        content: str, 
        request: CopywritingRequest
    ) -> Tuple[str, List[str]]:
        """Boost content engagement"""
        changes = []
        
        # Add engaging elements
        if not any(char in content for char in ['?', '!']):
            # Add a question or exclamation
            content = self._add_engaging_element(content, request.tone)
            changes.append("Added engaging element")
        
        # Improve opening
        if not self._has_strong_opening(content):
            content = self._strengthen_opening(content)
            changes.append("Strengthened opening")
        
        return content, changes
    
    async def _optimize_conversion(
        self, 
        content: str, 
        request: CopywritingRequest
    ) -> Tuple[str, List[str]]:
        """Optimize for conversion"""
        changes = []
        
        # Add urgency if appropriate
        if request.purpose == "sales" and "now" not in content.lower():
            content = self._add_urgency(content)
            changes.append("Added urgency element")
        
        # Strengthen CTA
        if request.include_cta:
            content = self._strengthen_cta(content)
            changes.append("Strengthened call-to-action")
        
        # Add social proof elements
        content = self._add_social_proof(content)
        changes.append("Added social proof element")
        
        return content, changes
    
    async def _adjust_tone(
        self, 
        content: str, 
        request: CopywritingRequest
    ) -> Tuple[str, List[str]]:
        """Adjust content tone"""
        changes = []
        
        # Tone-specific adjustments
        if request.tone == "professional":
            content = self._make_more_professional(content)
            changes.append("Adjusted to professional tone")
        elif request.tone == "casual":
            content = self._make_more_casual(content)
            changes.append("Adjusted to casual tone")
        elif request.tone == "urgent":
            content = self._make_more_urgent(content)
            changes.append("Adjusted to urgent tone")
        
        return content, changes
    
    # Helper methods
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """Load optimization rules"""
        return {
            "readability": {
                "max_sentence_length": 20,
                "min_sentence_length": 5,
                "target_readability_score": 0.7
            },
            "keywords": {
                "min_keyword_density": 0.02,
                "max_keyword_density": 0.05
            },
            "engagement": {
                "min_questions": 1,
                "min_exclamations": 1
            }
        }
    
    def _calculate_optimized_confidence(
        self, 
        content: str, 
        request: CopywritingRequest
    ) -> float:
        """Calculate confidence score for optimized content"""
        base_score = 0.7
        
        # Readability bonus
        readability = calculate_readability_score(content)
        readability_bonus = readability * 0.2
        
        # Keyword bonus
        keywords = extract_keywords(content)
        keyword_bonus = min(len(keywords) / 10, 0.1)
        
        # Length bonus
        if request.word_count:
            word_ratio = len(content.split()) / request.word_count
            length_bonus = 0.1 if 0.9 <= word_ratio <= 1.1 else 0.0
        else:
            length_bonus = 0.1
        
        return min(base_score + readability_bonus + keyword_bonus + length_bonus, 1.0)
    
    def _calculate_improvement_score(
        self, 
        original: CopywritingVariant, 
        optimized: CopywritingVariant, 
        goal: OptimizationGoal
    ) -> float:
        """Calculate improvement score"""
        if goal == OptimizationGoal.FEEDBACK_SCORE:
            return optimized.confidence_score - original.confidence_score
        elif goal == OptimizationGoal.READABILITY:
            original_readability = calculate_readability_score(original.content)
            optimized_readability = calculate_readability_score(optimized.content)
            return optimized_readability - original_readability
        else:
            # Generic improvement based on confidence score
            return optimized.confidence_score - original.confidence_score
    
    async def _simulate_performance(
        self, 
        variant: CopywritingVariant, 
        metric: OptimizationGoal
    ) -> float:
        """Simulate performance for A/B testing"""
        # Mock performance simulation
        base_performance = 0.5
        
        # Adjust based on confidence score
        confidence_factor = variant.confidence_score
        
        # Adjust based on content length
        length_factor = min(variant.word_count / 500, 1.0)
        
        # Add some randomness
        random_factor = np.random.normal(1.0, 0.1)
        
        return base_performance * confidence_factor * length_factor * random_factor
    
    def _analyze_tone_consistency(self, content: str, target_tone: str) -> float:
        """Analyze tone consistency"""
        # Simple tone analysis (would be more sophisticated in production)
        tone_indicators = {
            "professional": ["therefore", "furthermore", "consequently"],
            "casual": ["hey", "awesome", "cool", "great"],
            "urgent": ["now", "immediately", "urgent", "limited time"]
        }
        
        content_lower = content.lower()
        indicators = tone_indicators.get(target_tone, [])
        
        if not indicators:
            return 0.5
        
        matches = sum(1 for indicator in indicators if indicator in content_lower)
        return matches / len(indicators)
    
    async def _analyze_historical_performance(
        self, 
        variant: CopywritingVariant, 
        historical_data: List[Dict]
    ) -> List[ContentInsight]:
        """Analyze historical performance data"""
        insights = []
        
        # Analyze patterns in historical data
        if historical_data:
            avg_performance = np.mean([d.get('performance', 0.5) for d in historical_data])
            
            if avg_performance < 0.6:
                insights.append(ContentInsight(
                    insight_type="historical_performance",
                    description="Historical performance is below average",
                    impact_score=0.8,
                    recommendation="Consider different approach or strategy",
                    implementation_difficulty="hard",
                    expected_improvement=0.20
                ))
        
        return insights
    
    # Content modification helpers
    def _insert_keyword_naturally(self, content: str, keyword: str) -> str:
        """Insert keyword naturally into content"""
        # Simple implementation - would be more sophisticated in production
        sentences = content.split('.')
        if sentences:
            # Insert in the middle sentence
            middle_idx = len(sentences) // 2
            sentences[middle_idx] += f" {keyword}"
            return '. '.join(sentences)
        return content
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split long sentence into shorter ones"""
        # Simple implementation
        words = sentence.split()
        if len(words) > 20:
            mid_point = len(words) // 2
            return [' '.join(words[:mid_point]), ' '.join(words[mid_point:])]
        return [sentence]
    
    def _simplify_vocabulary(self, content: str) -> str:
        """Simplify vocabulary"""
        # Simple word replacement
        replacements = {
            "utilize": "use",
            "facilitate": "help",
            "implement": "do",
            "comprehensive": "complete"
        }
        
        for complex_word, simple_word in replacements.items():
            content = content.replace(complex_word, simple_word)
        
        return content
    
    def _add_engaging_element(self, content: str, tone: str) -> str:
        """Add engaging element based on tone"""
        if tone == "professional":
            return content + " What do you think?"
        elif tone == "casual":
            return content + " Pretty cool, right?"
        else:
            return content + " Ready to get started?"
    
    def _has_strong_opening(self, content: str) -> bool:
        """Check if content has strong opening"""
        first_sentence = content.split('.')[0]
        strong_openings = ["imagine", "what if", "discover", "unlock", "transform"]
        return any(opening in first_sentence.lower() for opening in strong_openings)
    
    def _strengthen_opening(self, content: str) -> str:
        """Strengthen content opening"""
        sentences = content.split('.')
        if sentences:
            first_sentence = sentences[0]
            strengthened = f"Discover how {first_sentence.lower()}"
            sentences[0] = strengthened
            return '. '.join(sentences)
        return content
    
    def _add_urgency(self, content: str) -> str:
        """Add urgency to content"""
        return content + " Limited time offer - act now!"
    
    def _strengthen_cta(self, content: str) -> str:
        """Strengthen call-to-action"""
        return content + " Don't wait - take action today!"
    
    def _add_social_proof(self, content: str) -> str:
        """Add social proof element"""
        return content + " Join thousands of satisfied customers."
    
    def _make_more_professional(self, content: str) -> str:
        """Make content more professional"""
        return content.replace("awesome", "excellent").replace("cool", "impressive")
    
    def _make_more_casual(self, content: str) -> str:
        """Make content more casual"""
        return content.replace("excellent", "awesome").replace("impressive", "cool")
    
    def _make_more_urgent(self, content: str) -> str:
        """Make content more urgent"""
        return content.replace("soon", "immediately").replace("later", "now")


# Global content optimizer instance
content_optimizer = ContentOptimizer()






























