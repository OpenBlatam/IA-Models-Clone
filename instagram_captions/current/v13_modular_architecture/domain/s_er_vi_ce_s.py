from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import List, Dict, Any, Optional, Tuple
from .entities import (
import re
import math
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Instagram Captions API v13.0 - Domain Services

Domain services for complex business logic that doesn't belong to entities.
Pure business logic without external dependencies.
"""

    CaptionRequest, CaptionResponse, CaptionStyle, Content, Hashtags,
    QualityMetrics, PerformanceMetrics, QualityLevel, RequestId
)


class QualityAssessmentService:
    """Service for assessing caption quality using business rules."""
    
    @staticmethod
    def assess_caption_quality(
        caption: str, 
        content: Content, 
        style: CaptionStyle,
        hashtags: Hashtags
    ) -> QualityMetrics:
        """Assess overall quality of generated caption."""
        
        # Base quality calculation
        content_score = QualityAssessmentService._assess_content_quality(caption, content)
        style_score = QualityAssessmentService._assess_style_consistency(caption, style)
        hashtag_score = QualityAssessmentService._assess_hashtag_quality(hashtags)
        readability_score = QualityAssessmentService._assess_readability(caption)
        
        # Weighted average
        overall_score = (
            content_score * 0.4 +
            style_score * 0.3 +
            hashtag_score * 0.2 +
            readability_score * 0.1
        )
        
        # Engagement prediction
        engagement_prediction = QualityAssessmentService._predict_engagement(
            caption, hashtags, style
        )
        
        # Virality score
        virality_score = QualityAssessmentService._calculate_virality_potential(
            caption, hashtags, style
        )
        
        return QualityMetrics(
            score=min(overall_score, 100.0),
            engagement_prediction=engagement_prediction,
            virality_score=virality_score,
            readability_score=readability_score
        )
    
    @staticmethod
    def _assess_content_quality(caption: str, content: Content) -> float:
        """Assess how well caption matches content."""
        base_score = 70.0
        
        # Length optimization
        caption_len = len(caption)
        if 40 <= caption_len <= 200:
            base_score += 15
        elif 20 <= caption_len <= 250:
            base_score += 10
        elif caption_len < 20:
            base_score -= 20
        
        # Word count optimization
        word_count = len(caption.split())
        if 8 <= word_count <= 30:
            base_score += 10
        
        # Content relevance (simplified)
        content_words = set(content.description.lower().split())
        caption_words = set(caption.lower().split())
        relevance = len(content_words.intersection(caption_words)) / max(len(content_words), 1)
        base_score += relevance * 20
        
        return min(base_score, 100.0)
    
    @staticmethod
    def _assess_style_consistency(caption: str, style: CaptionStyle) -> float:
        """Assess style consistency."""
        base_score = 75.0
        caption_lower = caption.lower()
        
        style_indicators = {
            CaptionStyle.CASUAL: ['amazing', 'love', 'awesome', 'cool', 'nice'],
            CaptionStyle.PROFESSIONAL: ['quality', 'professional', 'excellence', 'results'],
            CaptionStyle.LUXURY: ['luxury', 'premium', 'exclusive', 'finest', 'elegant'],
            CaptionStyle.EDUCATIONAL: ['learn', 'discover', 'understand', 'knowledge'],
            CaptionStyle.STORYTELLING: ['story', 'journey', 'experience', 'moment'],
            CaptionStyle.INSPIRATIONAL: ['inspire', 'motivate', 'achieve', 'success'],
            CaptionStyle.CALL_TO_ACTION: ['click', 'follow', 'subscribe', 'join', 'buy']
        }
        
        indicators = style_indicators.get(style, [])
        matches = sum(1 for word in indicators if word in caption_lower)
        base_score += min(matches * 5, 25)
        
        return min(base_score, 100.0)
    
    @staticmethod
    def _assess_hashtag_quality(hashtags: Hashtags) -> float:
        """Assess hashtag quality and relevance."""
        base_score = 70.0
        
        # Count optimization
        count = hashtags.count
        if 15 <= count <= 25:
            base_score += 20
        elif 10 <= count <= 30:
            base_score += 15
        elif count < 5:
            base_score -= 30
        
        # Diversity check (simplified)
        unique_chars = len(set(''.join(hashtags.tags).lower()))
        diversity_bonus = min(unique_chars / 50, 1.0) * 10
        base_score += diversity_bonus
        
        return min(base_score, 100.0)
    
    @staticmethod
    def _assess_readability(caption: str) -> float:
        """Assess caption readability."""
        base_score = 75.0
        
        # Sentence count
        sentences = len([s for s in caption.split('.') if s.strip()])
        if 1 <= sentences <= 3:
            base_score += 15
        
        # Average word length
        words = caption.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if 4 <= avg_word_length <= 7:
                base_score += 10
        
        return min(base_score, 100.0)
    
    @staticmethod
    def _predict_engagement(caption: str, hashtags: Hashtags, style: CaptionStyle) -> float:
        """Predict engagement potential."""
        base_engagement = 60.0
        caption_lower = caption.lower()
        
        # Question boost
        if '?' in caption:
            base_engagement += 15
        
        # Emoji presence
        emoji_count = sum(1 for char in caption if ord(char) > 127)
        base_engagement += min(emoji_count * 3, 20)
        
        # Call-to-action boost
        cta_words = ['click', 'link', 'follow', 'subscribe', 'comment', 'share']
        if any(word in caption_lower for word in cta_words):
            base_engagement += 10
        
        # Style-specific boosts
        style_boosts = {
            CaptionStyle.CALL_TO_ACTION: 15,
            CaptionStyle.PLAYFUL: 10,
            CaptionStyle.INSPIRATIONAL: 8
        }
        base_engagement += style_boosts.get(style, 0)
        
        return min(base_engagement, 100.0)
    
    @staticmethod
    def _calculate_virality_potential(caption: str, hashtags: Hashtags, style: CaptionStyle) -> float:
        """Calculate virality potential."""
        base_virality = 50.0
        caption_lower = caption.lower()
        
        # Trending words
        trending_words = ['viral', 'trending', 'amazing', 'incredible', 'shocking']
        trend_matches = sum(1 for word in trending_words if word in caption_lower)
        base_virality += trend_matches * 8
        
        # Hashtag virality
        viral_hashtags = ['#viral', '#trending', '#fyp', '#explore', '#amazing']
        hashtag_matches = sum(1 for tag in hashtags.tags if tag.lower() in viral_hashtags)
        base_virality += hashtag_matches * 5
        
        # Emotional words
        emotional_words = ['love', 'hate', 'amazing', 'incredible', 'shocking', 'beautiful']
        emotion_matches = sum(1 for word in emotional_words if word in caption_lower)
        base_virality += emotion_matches * 3
        
        return min(base_virality, 100.0)


class HashtagOptimizationService:
    """Service for optimizing hashtag generation and selection."""
    
    @staticmethod
    def generate_optimized_hashtags(
        content: Content,
        style: CaptionStyle,
        target_count: int
    ) -> Hashtags:
        """Generate optimized hashtags for content and style."""
        
        # Base hashtags for style
        style_hashtags = HashtagOptimizationService._get_style_hashtags(style)
        
        # Content-derived hashtags
        content_hashtags = HashtagOptimizationService._extract_content_hashtags(content)
        
        # Popular general hashtags
        popular_hashtags = HashtagOptimizationService._get_popular_hashtags()
        
        # Trending hashtags
        trending_hashtags = HashtagOptimizationService._get_trending_hashtags()
        
        # Combine and optimize
        all_hashtags = (
            style_hashtags[:8] +
            content_hashtags[:6] +
            popular_hashtags[:4] +
            trending_hashtags[:2]
        )
        
        # Remove duplicates while preserving order
        unique_hashtags = []
        seen = set()
        for tag in all_hashtags:
            if tag.lower() not in seen:
                unique_hashtags.append(tag)
                seen.add(tag.lower())
        
        # Fill to target count if needed
        while len(unique_hashtags) < target_count:
            filler_tags = HashtagOptimizationService._get_filler_hashtags()
            for tag in filler_tags:
                if tag.lower() not in seen and len(unique_hashtags) < target_count:
                    unique_hashtags.append(tag)
                    seen.add(tag.lower())
        
        return Hashtags(unique_hashtags[:target_count])
    
    @staticmethod
    def _get_style_hashtags(style: CaptionStyle) -> List[str]:
        """Get hashtags specific to caption style."""
        style_mapping = {
            CaptionStyle.CASUAL: [
                "#lifestyle", "#daily", "#vibes", "#mood", "#authentic", 
                "#real", "#casual", "#everyday"
            ],
            CaptionStyle.PROFESSIONAL: [
                "#business", "#professional", "#quality", "#excellence", 
                "#success", "#leadership", "#corporate", "#expert"
            ],
            CaptionStyle.LUXURY: [
                "#luxury", "#premium", "#exclusive", "#highend", 
                "#sophisticated", "#elegant", "#prestige", "#luxury"
            ],
            CaptionStyle.EDUCATIONAL: [
                "#education", "#learning", "#knowledge", "#tips", 
                "#howto", "#tutorial", "#guide", "#learn"
            ],
            CaptionStyle.STORYTELLING: [
                "#story", "#journey", "#experience", "#narrative", 
                "#adventure", "#memories", "#moments", "#tale"
            ],
            CaptionStyle.INSPIRATIONAL: [
                "#inspiration", "#motivation", "#success", "#goals", 
                "#mindset", "#believe", "#achieve", "#inspire"
            ],
            CaptionStyle.CALL_TO_ACTION: [
                "#action", "#now", "#limited", "#exclusive", 
                "#follow", "#subscribe", "#join", "#click"
            ]
        }
        
        return style_mapping.get(style, ["#content", "#post", "#share", "#like"])
    
    @staticmethod
    def _extract_content_hashtags(content: Content) -> List[str]:
        """Extract relevant hashtags from content description."""
        words = content.description.lower().split()
        hashtags = []
        
        # Convert meaningful words to hashtags
        for word in words:
            # Clean word
            clean_word = re.sub(r'[^a-zA-Z0-9]', '', word)
            
            # Only use words longer than 3 characters
            if len(clean_word) > 3 and clean_word.isalpha():
                hashtags.append(f"#{clean_word}")
        
        return hashtags[:10]  # Limit content-derived hashtags
    
    @staticmethod
    def _get_popular_hashtags() -> List[str]:
        """Get popular general hashtags."""
        return [
            "#instagood", "#photooftheday", "#love", "#beautiful", 
            "#amazing", "#follow", "#like", "#share", "#instagram", "#post"
        ]
    
    @staticmethod
    def _get_trending_hashtags() -> List[str]:
        """Get currently trending hashtags."""
        return [
            "#viral", "#trending", "#explore", "#fyp", "#discover",
            "#featured", "#popular", "#hot", "#new", "#fresh"
        ]
    
    @staticmethod
    def _get_filler_hashtags() -> List[str]:
        """Get filler hashtags to reach target count."""
        return [
            "#content", "#creative", "#digital", "#social", "#media",
            "#online", "#community", "#engagement", "#growth", "#success"
        ]


class CaptionGenerationService:
    """Service for caption generation business logic."""
    
    @staticmethod
    async def validate_request(request: CaptionRequest) -> List[str]:
        """Validate caption request and return list of validation errors."""
        errors = []
        
        # Content validation
        if not request.content.is_valid:
            errors.append("Content description is too short")
        
        # Hashtag count validation
        if not 5 <= request.hashtag_count <= 50:
            errors.append("Hashtag count must be between 5 and 50")
        
        # Custom instructions validation
        if request.custom_instructions and len(request.custom_instructions) > 500:
            errors.append("Custom instructions too long (max 500 characters)")
        
        return errors
    
    @staticmethod
    def determine_processing_strategy(request: CaptionRequest) -> str:
        """Determine the best processing strategy for the request."""
        
        # Priority-based strategy
        if request.priority == "urgent":
            return "ultra_fast"
        elif request.priority == "high":
            return "fast"
        
        # Content complexity-based strategy
        if request.content.word_count > 20:
            return "comprehensive"
        elif request.content.word_count < 5:
            return "simple"
        
        # Feature-based strategy
        if request.enable_advanced_analysis:
            return "advanced"
        
        return "standard"
    
    @staticmethod
    def calculate_processing_timeout(request: CaptionRequest) -> float:
        """Calculate appropriate timeout for request processing."""
        base_timeout = 5.0  # 5 seconds base
        
        # Adjust based on features
        if request.enable_advanced_analysis:
            base_timeout += 2.0
        if request.enable_sentiment_analysis:
            base_timeout += 1.0
        if request.enable_seo_optimization:
            base_timeout += 1.5
        
        # Adjust based on content complexity
        if request.content.word_count > 20:
            base_timeout += 1.0
        
        return min(base_timeout, 15.0)  # Max 15 seconds


class BatchOptimizationService:
    """Service for optimizing batch processing operations."""
    
    @staticmethod
    def optimize_batch_execution(requests: List[CaptionRequest]) -> Dict[str, Any]:
        """Optimize batch execution strategy."""
        
        # Analyze request characteristics
        total_requests = len(requests)
        complex_requests = sum(1 for req in requests if req.enable_advanced_analysis)
        urgent_requests = sum(1 for req in requests if req.priority in ["urgent", "high"])
        
        # Determine optimal strategy
        if urgent_requests > total_requests * 0.5:
            strategy = "priority_first"
            max_concurrent = min(total_requests, 20)
        elif complex_requests > total_requests * 0.3:
            strategy = "balanced"
            max_concurrent = min(total_requests, 10)
        else:
            strategy = "throughput_optimized"
            max_concurrent = min(total_requests, 50)
        
        # Calculate estimated time
        if strategy == "priority_first":
            estimated_time = total_requests * 0.015  # 15ms per request
        elif strategy == "balanced":
            estimated_time = total_requests * 0.025  # 25ms per request
        else:
            estimated_time = total_requests * 0.020  # 20ms per request
        
        return {
            "strategy": strategy,
            "max_concurrent": max_concurrent,
            "estimated_time": estimated_time,
            "total_requests": total_requests,
            "complex_requests": complex_requests,
            "urgent_requests": urgent_requests
        } 