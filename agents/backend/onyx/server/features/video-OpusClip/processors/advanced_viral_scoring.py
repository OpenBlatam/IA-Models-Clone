"""
Advanced Viral Scoring System

Multi-factor viral potential scoring based on historical data, trends, and engagement patterns.
This is a key differentiator for creating truly viral content.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any, Tuple, Union
import asyncio
import numpy as np
import json
import time
from dataclasses import dataclass
from enum import Enum
import structlog
from pathlib import Path
import requests
import aiohttp
from datetime import datetime, timedelta
import pickle
import hashlib

from ..models.video_models import VideoClipRequest, VideoClipResponse
from ..error_handling import ErrorHandler, ProcessingError, ValidationError

logger = structlog.get_logger("advanced_viral_scoring")
error_handler = ErrorHandler()

class ViralFactor(Enum):
    """Factors that contribute to viral potential."""
    ENGAGEMENT_INTENSITY = "engagement_intensity"
    CONTENT_NOVELTY = "content_novelty"
    EMOTIONAL_IMPACT = "emotional_impact"
    TREND_RELEVANCE = "trend_relevance"
    AUDIENCE_ALIGNMENT = "audience_alignment"
    TIMING_OPTIMALITY = "timing_optimality"
    SHAREABILITY = "shareability"
    CONTROVERSY_LEVEL = "controversy_level"

class TrendSource(Enum):
    """Sources for trend data."""
    TIKTOK_TRENDS = "tiktok_trends"
    YOUTUBE_TRENDS = "youtube_trends"
    TWITTER_TRENDS = "twitter_trends"
    INSTAGRAM_TRENDS = "instagram_trends"
    GOOGLE_TRENDS = "google_trends"

@dataclass
class ViralScore:
    """Comprehensive viral potential score."""
    overall_score: float
    factor_scores: Dict[ViralFactor, float]
    confidence: float
    recommendations: List[str]
    trend_alignment: Dict[str, float]
    audience_potential: Dict[str, float]
    optimal_timing: Optional[datetime]
    metadata: Dict[str, Any]

@dataclass
class TrendData:
    """Trend data from external sources."""
    keyword: str
    trend_score: float
    source: TrendSource
    timestamp: datetime
    growth_rate: float
    peak_time: Optional[datetime]
    related_keywords: List[str]

@dataclass
class HistoricalViralData:
    """Historical viral content data for analysis."""
    content_id: str
    platform: str
    viral_score: float
    actual_views: int
    engagement_rate: float
    shares: int
    comments: int
    likes: int
    duration: float
    content_type: str
    keywords: List[str]
    timestamp: datetime

class TrendAnalyzer:
    """Analyzes current trends across platforms."""
    
    def __init__(self):
        self.trend_cache = {}
        self.cache_duration = 3600  # 1 hour
        self.api_keys = {}
        self._load_api_keys()
    
    def _load_api_keys(self):
        """Load API keys for trend analysis."""
        try:
            # Load from environment or config
            self.api_keys = {
                "google_trends": None,  # Would load from env
                "tiktok_api": None,     # Would load from env
                "youtube_api": None,    # Would load from env
                "twitter_api": None,    # Would load from env
            }
        except Exception as e:
            logger.warning(f"Failed to load API keys: {e}")
    
    async def get_trending_keywords(self, platform: str, limit: int = 50) -> List[TrendData]:
        """Get trending keywords for a platform."""
        try:
            cache_key = f"{platform}_{limit}_{int(time.time() // self.cache_duration)}"
            
            if cache_key in self.trend_cache:
                return self.trend_cache[cache_key]
            
            trends = []
            
            if platform == "tiktok":
                trends = await self._get_tiktok_trends(limit)
            elif platform == "youtube":
                trends = await self._get_youtube_trends(limit)
            elif platform == "twitter":
                trends = await self._get_twitter_trends(limit)
            elif platform == "instagram":
                trends = await self._get_instagram_trends(limit)
            else:
                trends = await self._get_google_trends(limit)
            
            # Cache results
            self.trend_cache[cache_key] = trends
            
            return trends
            
        except Exception as e:
            logger.error(f"Trend analysis failed for {platform}: {e}")
            return []
    
    async def _get_tiktok_trends(self, limit: int) -> List[TrendData]:
        """Get TikTok trending keywords."""
        try:
            # Placeholder implementation - would use TikTok API
            trends = []
            
            # Simulate trending keywords
            trending_keywords = [
                "AI", "viral", "trending", "fyp", "dance", "challenge",
                "comedy", "lifehack", "tutorial", "reaction", "transformation"
            ]
            
            for i, keyword in enumerate(trending_keywords[:limit]):
                trend = TrendData(
                    keyword=keyword,
                    trend_score=0.9 - (i * 0.05),  # Decreasing trend score
                    source=TrendSource.TIKTOK_TRENDS,
                    timestamp=datetime.now(),
                    growth_rate=0.1 + (i * 0.02),
                    peak_time=datetime.now() + timedelta(hours=i),
                    related_keywords=self._get_related_keywords(keyword)
                )
                trends.append(trend)
            
            return trends
            
        except Exception as e:
            logger.error(f"TikTok trends failed: {e}")
            return []
    
    async def _get_youtube_trends(self, limit: int) -> List[TrendData]:
        """Get YouTube trending keywords."""
        try:
            # Placeholder implementation - would use YouTube API
            trends = []
            
            trending_keywords = [
                "tutorial", "review", "reaction", "gaming", "tech",
                "lifestyle", "cooking", "fitness", "travel", "education"
            ]
            
            for i, keyword in enumerate(trending_keywords[:limit]):
                trend = TrendData(
                    keyword=keyword,
                    trend_score=0.85 - (i * 0.04),
                    source=TrendSource.YOUTUBE_TRENDS,
                    timestamp=datetime.now(),
                    growth_rate=0.08 + (i * 0.015),
                    peak_time=datetime.now() + timedelta(hours=i*2),
                    related_keywords=self._get_related_keywords(keyword)
                )
                trends.append(trend)
            
            return trends
            
        except Exception as e:
            logger.error(f"YouTube trends failed: {e}")
            return []
    
    async def _get_twitter_trends(self, limit: int) -> List[TrendData]:
        """Get Twitter trending keywords."""
        try:
            # Placeholder implementation - would use Twitter API
            trends = []
            
            trending_keywords = [
                "breaking", "news", "politics", "tech", "sports",
                "entertainment", "business", "science", "health", "environment"
            ]
            
            for i, keyword in enumerate(trending_keywords[:limit]):
                trend = TrendData(
                    keyword=keyword,
                    trend_score=0.8 - (i * 0.03),
                    source=TrendSource.TWITTER_TRENDS,
                    timestamp=datetime.now(),
                    growth_rate=0.12 + (i * 0.01),
                    peak_time=datetime.now() + timedelta(hours=i),
                    related_keywords=self._get_related_keywords(keyword)
                )
                trends.append(trend)
            
            return trends
            
        except Exception as e:
            logger.error(f"Twitter trends failed: {e}")
            return []
    
    async def _get_instagram_trends(self, limit: int) -> List[TrendData]:
        """Get Instagram trending keywords."""
        try:
            # Placeholder implementation - would use Instagram API
            trends = []
            
            trending_keywords = [
                "lifestyle", "fashion", "beauty", "food", "travel",
                "fitness", "art", "photography", "inspiration", "motivation"
            ]
            
            for i, keyword in enumerate(trending_keywords[:limit]):
                trend = TrendData(
                    keyword=keyword,
                    trend_score=0.88 - (i * 0.04),
                    source=TrendSource.INSTAGRAM_TRENDS,
                    timestamp=datetime.now(),
                    growth_rate=0.09 + (i * 0.02),
                    peak_time=datetime.now() + timedelta(hours=i*1.5),
                    related_keywords=self._get_related_keywords(keyword)
                )
                trends.append(trend)
            
            return trends
            
        except Exception as e:
            logger.error(f"Instagram trends failed: {e}")
            return []
    
    async def _get_google_trends(self, limit: int) -> List[TrendData]:
        """Get Google Trends data."""
        try:
            # Placeholder implementation - would use Google Trends API
            trends = []
            
            trending_keywords = [
                "artificial intelligence", "climate change", "renewable energy",
                "cryptocurrency", "sustainable living", "mental health",
                "remote work", "electric vehicles", "space exploration", "quantum computing"
            ]
            
            for i, keyword in enumerate(trending_keywords[:limit]):
                trend = TrendData(
                    keyword=keyword,
                    trend_score=0.75 - (i * 0.03),
                    source=TrendSource.GOOGLE_TRENDS,
                    timestamp=datetime.now(),
                    growth_rate=0.06 + (i * 0.01),
                    peak_time=datetime.now() + timedelta(days=i),
                    related_keywords=self._get_related_keywords(keyword)
                )
                trends.append(trend)
            
            return trends
            
        except Exception as e:
            logger.error(f"Google trends failed: {e}")
            return []
    
    def _get_related_keywords(self, keyword: str) -> List[str]:
        """Get related keywords for a trend."""
        # Simple keyword expansion - would use more sophisticated NLP
        related_map = {
            "AI": ["artificial intelligence", "machine learning", "automation", "technology"],
            "viral": ["trending", "popular", "famous", "buzz"],
            "tutorial": ["how to", "guide", "tips", "tricks"],
            "comedy": ["funny", "humor", "joke", "laugh"],
            "dance": ["music", "choreography", "performance", "rhythm"]
        }
        
        return related_map.get(keyword.lower(), [keyword])

class HistoricalAnalyzer:
    """Analyzes historical viral content data."""
    
    def __init__(self):
        self.viral_data = []
        self.patterns = {}
        self._load_historical_data()
    
    def _load_historical_data(self):
        """Load historical viral content data."""
        try:
            # Load from database or file
            data_file = Path("/tmp/viral_data.pkl")
            if data_file.exists():
                with open(data_file, "rb") as f:
                    self.viral_data = pickle.load(f)
            else:
                # Generate sample data
                self.viral_data = self._generate_sample_data()
                self._save_data()
            
            # Analyze patterns
            self._analyze_patterns()
            
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            self.viral_data = []
    
    def _generate_sample_data(self) -> List[HistoricalViralData]:
        """Generate sample historical data."""
        data = []
        
        # Generate sample viral content data
        for i in range(1000):
            content = HistoricalViralData(
                content_id=f"content_{i}",
                platform=np.random.choice(["tiktok", "youtube", "instagram", "twitter"]),
                viral_score=np.random.uniform(0.1, 1.0),
                actual_views=np.random.randint(1000, 10000000),
                engagement_rate=np.random.uniform(0.01, 0.15),
                shares=np.random.randint(10, 100000),
                comments=np.random.randint(5, 50000),
                likes=np.random.randint(50, 500000),
                duration=np.random.uniform(5, 60),
                content_type=np.random.choice(["tutorial", "comedy", "dance", "reaction", "lifestyle"]),
                keywords=np.random.choice([
                    ["AI", "tech", "innovation"],
                    ["comedy", "funny", "humor"],
                    ["dance", "music", "choreography"],
                    ["tutorial", "how to", "guide"],
                    ["lifestyle", "fashion", "beauty"]
                ]),
                timestamp=datetime.now() - timedelta(days=np.random.randint(0, 365))
            )
            data.append(content)
        
        return data
    
    def _save_data(self):
        """Save historical data."""
        try:
            data_file = Path("/tmp/viral_data.pkl")
            with open(data_file, "wb") as f:
                pickle.dump(self.viral_data, f)
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
    
    def _analyze_patterns(self):
        """Analyze patterns in historical data."""
        try:
            # Analyze by platform
            platform_patterns = {}
            for platform in ["tiktok", "youtube", "instagram", "twitter"]:
                platform_data = [d for d in self.viral_data if d.platform == platform]
                if platform_data:
                    platform_patterns[platform] = {
                        "avg_viral_score": np.mean([d.viral_score for d in platform_data]),
                        "avg_engagement": np.mean([d.engagement_rate for d in platform_data]),
                        "optimal_duration": np.median([d.duration for d in platform_data]),
                        "top_keywords": self._get_top_keywords(platform_data)
                    }
            
            # Analyze by content type
            content_patterns = {}
            for content_type in ["tutorial", "comedy", "dance", "reaction", "lifestyle"]:
                type_data = [d for d in self.viral_data if d.content_type == content_type]
                if type_data:
                    content_patterns[content_type] = {
                        "avg_viral_score": np.mean([d.viral_score for d in type_data]),
                        "avg_engagement": np.mean([d.engagement_rate for d in type_data]),
                        "optimal_duration": np.median([d.duration for d in type_data])
                    }
            
            self.patterns = {
                "platforms": platform_patterns,
                "content_types": content_patterns
            }
            
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
    
    def _get_top_keywords(self, data: List[HistoricalViralData]) -> List[str]:
        """Get top keywords from data."""
        keyword_counts = {}
        for item in data:
            for keyword in item.keywords:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        return sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]

class AudienceAnalyzer:
    """Analyzes audience characteristics and preferences."""
    
    def __init__(self):
        self.audience_profiles = {}
        self._load_audience_data()
    
    def _load_audience_data(self):
        """Load audience data."""
        try:
            # Placeholder - would load from database
            self.audience_profiles = {
                "tiktok": {
                    "age_groups": {"13-17": 0.25, "18-24": 0.35, "25-34": 0.25, "35+": 0.15},
                    "interests": ["dance", "comedy", "lifestyle", "music", "challenges"],
                    "optimal_posting_times": ["19:00", "20:00", "21:00"],
                    "preferred_duration": 15,
                    "engagement_patterns": {"likes": 0.05, "shares": 0.01, "comments": 0.002}
                },
                "youtube": {
                    "age_groups": {"18-24": 0.20, "25-34": 0.30, "35-44": 0.25, "45+": 0.25},
                    "interests": ["tutorials", "reviews", "gaming", "tech", "education"],
                    "optimal_posting_times": ["14:00", "15:00", "16:00"],
                    "preferred_duration": 45,
                    "engagement_patterns": {"likes": 0.03, "shares": 0.005, "comments": 0.001}
                },
                "instagram": {
                    "age_groups": {"18-24": 0.30, "25-34": 0.35, "35-44": 0.25, "45+": 0.10},
                    "interests": ["lifestyle", "fashion", "beauty", "travel", "food"],
                    "optimal_posting_times": ["12:00", "13:00", "17:00"],
                    "preferred_duration": 30,
                    "engagement_patterns": {"likes": 0.04, "shares": 0.008, "comments": 0.0015}
                }
            }
        except Exception as e:
            logger.error(f"Failed to load audience data: {e}")
    
    async def analyze_audience_potential(self, 
                                       content_keywords: List[str], 
                                       platform: str,
                                       duration: float) -> Dict[str, float]:
        """Analyze audience potential for content."""
        try:
            if platform not in self.audience_profiles:
                return {"potential": 0.5, "alignment": 0.5}
            
            profile = self.audience_profiles[platform]
            
            # Calculate interest alignment
            interest_alignment = 0.0
            for keyword in content_keywords:
                for interest in profile["interests"]:
                    if keyword.lower() in interest.lower() or interest.lower() in keyword.lower():
                        interest_alignment += 1
                        break
            
            interest_alignment = min(interest_alignment / len(content_keywords), 1.0)
            
            # Calculate duration alignment
            preferred_duration = profile["preferred_duration"]
            duration_diff = abs(duration - preferred_duration) / preferred_duration
            duration_alignment = max(0, 1 - duration_diff)
            
            # Calculate overall potential
            potential = (interest_alignment * 0.7 + duration_alignment * 0.3)
            
            return {
                "potential": potential,
                "alignment": interest_alignment,
                "duration_score": duration_alignment,
                "audience_size": self._estimate_audience_size(platform, interest_alignment)
            }
            
        except Exception as e:
            logger.error(f"Audience analysis failed: {e}")
            return {"potential": 0.5, "alignment": 0.5}
    
    def _estimate_audience_size(self, platform: str, alignment: float) -> int:
        """Estimate potential audience size."""
        base_sizes = {
            "tiktok": 1000000,
            "youtube": 2000000,
            "instagram": 1500000,
            "twitter": 500000
        }
        
        base_size = base_sizes.get(platform, 1000000)
        return int(base_size * alignment)

class AdvancedViralScorer:
    """Main advanced viral scoring system."""
    
    def __init__(self):
        self.trend_analyzer = TrendAnalyzer()
        self.historical_analyzer = HistoricalAnalyzer()
        self.audience_analyzer = AudienceAnalyzer()
        self.scoring_weights = {
            ViralFactor.ENGAGEMENT_INTENSITY: 0.25,
            ViralFactor.CONTENT_NOVELTY: 0.15,
            ViralFactor.EMOTIONAL_IMPACT: 0.20,
            ViralFactor.TREND_RELEVANCE: 0.15,
            ViralFactor.AUDIENCE_ALIGNMENT: 0.15,
            ViralFactor.TIMING_OPTIMALITY: 0.05,
            ViralFactor.SHAREABILITY: 0.03,
            ViralFactor.CONTROVERSY_LEVEL: 0.02
        }
    
    async def calculate_viral_score(self, 
                                  content_data: Dict[str, Any],
                                  platform: str = "tiktok") -> ViralScore:
        """Calculate comprehensive viral score."""
        try:
            logger.info(f"Calculating viral score for {platform}")
            
            # Extract content information
            keywords = content_data.get("keywords", [])
            duration = content_data.get("duration", 15.0)
            engagement_scores = content_data.get("engagement_scores", [])
            content_text = content_data.get("content_text", "")
            
            # Calculate individual factor scores
            factor_scores = {}
            
            # Engagement intensity
            factor_scores[ViralFactor.ENGAGEMENT_INTENSITY] = await self._calculate_engagement_intensity(engagement_scores)
            
            # Content novelty
            factor_scores[ViralFactor.CONTENT_NOVELTY] = await self._calculate_content_novelty(content_text, keywords)
            
            # Emotional impact
            factor_scores[ViralFactor.EMOTIONAL_IMPACT] = await self._calculate_emotional_impact(content_text, engagement_scores)
            
            # Trend relevance
            factor_scores[ViralFactor.TREND_RELEVANCE] = await self._calculate_trend_relevance(keywords, platform)
            
            # Audience alignment
            factor_scores[ViralFactor.AUDIENCE_ALIGNMENT] = await self._calculate_audience_alignment(keywords, platform, duration)
            
            # Timing optimality
            factor_scores[ViralFactor.TIMING_OPTIMALITY] = await self._calculate_timing_optimality(platform)
            
            # Shareability
            factor_scores[ViralFactor.SHAREABILITY] = await self._calculate_shareability(content_text, keywords)
            
            # Controversy level
            factor_scores[ViralFactor.CONTROVERSY_LEVEL] = await self._calculate_controversy_level(content_text)
            
            # Calculate overall score
            overall_score = sum(
                factor_scores[factor] * weight 
                for factor, weight in self.scoring_weights.items()
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(factor_scores)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(factor_scores, platform)
            
            # Analyze trend alignment
            trend_alignment = await self._analyze_trend_alignment(keywords, platform)
            
            # Analyze audience potential
            audience_potential = await self.audience_analyzer.analyze_audience_potential(keywords, platform, duration)
            
            # Calculate optimal timing
            optimal_timing = await self._calculate_optimal_timing(platform)
            
            return ViralScore(
                overall_score=min(overall_score, 1.0),
                factor_scores=factor_scores,
                confidence=confidence,
                recommendations=recommendations,
                trend_alignment=trend_alignment,
                audience_potential=audience_potential,
                optimal_timing=optimal_timing,
                metadata={
                    "platform": platform,
                    "calculation_time": time.time(),
                    "weights_used": self.scoring_weights
                }
            )
            
        except Exception as e:
            logger.error(f"Viral score calculation failed: {e}")
            raise ProcessingError(f"Viral score calculation failed: {e}")
    
    async def _calculate_engagement_intensity(self, engagement_scores: List[Dict]) -> float:
        """Calculate engagement intensity score."""
        try:
            if not engagement_scores:
                return 0.5
            
            scores = [score.get("score", 0) for score in engagement_scores]
            
            # Calculate intensity based on score distribution
            mean_score = np.mean(scores)
            score_variance = np.var(scores)
            max_score = np.max(scores)
            
            # Higher intensity for higher mean, lower variance, and high peaks
            intensity = (mean_score * 0.5 + (1 - min(score_variance, 1)) * 0.3 + max_score * 0.2)
            
            return min(intensity, 1.0)
            
        except Exception as e:
            logger.error(f"Engagement intensity calculation failed: {e}")
            return 0.5
    
    async def _calculate_content_novelty(self, content_text: str, keywords: List[str]) -> float:
        """Calculate content novelty score."""
        try:
            # Simple novelty calculation - would use more sophisticated NLP
            novelty_score = 0.5
            
            # Check for novel keywords
            novel_keywords = ["AI", "revolutionary", "breakthrough", "first", "never seen", "exclusive"]
            for keyword in novel_keywords:
                if keyword.lower() in content_text.lower():
                    novelty_score += 0.1
            
            # Check for question patterns (often more engaging)
            if "?" in content_text:
                novelty_score += 0.1
            
            # Check for exclamation patterns (often more engaging)
            if "!" in content_text:
                novelty_score += 0.05
            
            return min(novelty_score, 1.0)
            
        except Exception as e:
            logger.error(f"Content novelty calculation failed: {e}")
            return 0.5
    
    async def _calculate_emotional_impact(self, content_text: str, engagement_scores: List[Dict]) -> float:
        """Calculate emotional impact score."""
        try:
            emotional_score = 0.5
            
            # Check for emotional keywords
            emotional_keywords = {
                "amazing": 0.1, "incredible": 0.1, "shocking": 0.15, "unbelievable": 0.1,
                "heartbreaking": 0.2, "inspiring": 0.15, "motivating": 0.1, "touching": 0.1
            }
            
            for keyword, impact in emotional_keywords.items():
                if keyword.lower() in content_text.lower():
                    emotional_score += impact
            
            # Use engagement scores if available
            if engagement_scores:
                emotional_scores = [s.get("metadata", {}).get("emotion", 0) for s in engagement_scores]
                if emotional_scores:
                    emotional_score += np.mean(emotional_scores) * 0.3
            
            return min(emotional_score, 1.0)
            
        except Exception as e:
            logger.error(f"Emotional impact calculation failed: {e}")
            return 0.5
    
    async def _calculate_trend_relevance(self, keywords: List[str], platform: str) -> float:
        """Calculate trend relevance score."""
        try:
            # Get trending keywords
            trends = await self.trend_analyzer.get_trending_keywords(platform, 20)
            
            if not trends:
                return 0.5
            
            # Calculate relevance based on keyword overlap
            trend_keywords = [trend.keyword.lower() for trend in trends]
            content_keywords_lower = [kw.lower() for kw in keywords]
            
            matches = 0
            total_weight = 0
            
            for i, trend in enumerate(trends):
                weight = trend.trend_score
                total_weight += weight
                
                if trend.keyword.lower() in content_keywords_lower:
                    matches += weight
                
                # Check related keywords
                for related in trend.related_keywords:
                    if related.lower() in content_keywords_lower:
                        matches += weight * 0.5
            
            relevance = matches / total_weight if total_weight > 0 else 0.5
            
            return min(relevance, 1.0)
            
        except Exception as e:
            logger.error(f"Trend relevance calculation failed: {e}")
            return 0.5
    
    async def _calculate_audience_alignment(self, keywords: List[str], platform: str, duration: float) -> float:
        """Calculate audience alignment score."""
        try:
            audience_analysis = await self.audience_analyzer.analyze_audience_potential(keywords, platform, duration)
            return audience_analysis.get("potential", 0.5)
            
        except Exception as e:
            logger.error(f"Audience alignment calculation failed: {e}")
            return 0.5
    
    async def _calculate_timing_optimality(self, platform: str) -> float:
        """Calculate timing optimality score."""
        try:
            current_hour = datetime.now().hour
            
            optimal_times = {
                "tiktok": [19, 20, 21],
                "youtube": [14, 15, 16],
                "instagram": [12, 13, 17],
                "twitter": [9, 12, 15]
            }
            
            platform_times = optimal_times.get(platform, [12, 18, 20])
            
            # Calculate how close current time is to optimal times
            min_distance = min(abs(current_hour - optimal_time) for optimal_time in platform_times)
            timing_score = max(0, 1 - (min_distance / 12))  # Normalize to 0-1
            
            return timing_score
            
        except Exception as e:
            logger.error(f"Timing optimality calculation failed: {e}")
            return 0.5
    
    async def _calculate_shareability(self, content_text: str, keywords: List[str]) -> float:
        """Calculate shareability score."""
        try:
            shareability_score = 0.5
            
            # Check for shareable content indicators
            shareable_indicators = [
                "share", "tell your friends", "spread the word", "viral",
                "everyone needs to see", "you won't believe", "mind blown"
            ]
            
            for indicator in shareable_indicators:
                if indicator.lower() in content_text.lower():
                    shareability_score += 0.1
            
            # Check for question patterns (encourages engagement)
            if "?" in content_text:
                shareability_score += 0.1
            
            # Check for call-to-action patterns
            cta_patterns = ["like if", "comment if", "follow for", "subscribe for"]
            for pattern in cta_patterns:
                if pattern.lower() in content_text.lower():
                    shareability_score += 0.05
            
            return min(shareability_score, 1.0)
            
        except Exception as e:
            logger.error(f"Shareability calculation failed: {e}")
            return 0.5
    
    async def _calculate_controversy_level(self, content_text: str) -> float:
        """Calculate controversy level (moderate controversy can be good for virality)."""
        try:
            controversy_score = 0.0
            
            # Check for controversial keywords
            controversial_keywords = [
                "controversial", "debate", "argument", "disagree", "against",
                "shocking truth", "they don't want you to know", "hidden"
            ]
            
            for keyword in controversial_keywords:
                if keyword.lower() in content_text.lower():
                    controversy_score += 0.1
            
            # Moderate controversy is often good for virality
            if 0.2 <= controversy_score <= 0.6:
                return controversy_score
            elif controversy_score > 0.6:
                return 0.6  # Cap at moderate controversy
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Controversy level calculation failed: {e}")
            return 0.0
    
    def _calculate_confidence(self, factor_scores: Dict[ViralFactor, float]) -> float:
        """Calculate confidence in the viral score."""
        try:
            # Confidence based on score distribution and consistency
            scores = list(factor_scores.values())
            
            # Higher confidence for more consistent scores
            variance = np.var(scores)
            consistency = 1 - min(variance, 1)
            
            # Higher confidence for scores closer to 0.5 (more balanced)
            balance = 1 - np.mean([abs(score - 0.5) for score in scores])
            
            confidence = (consistency * 0.6 + balance * 0.4)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    async def _generate_recommendations(self, 
                                      factor_scores: Dict[ViralFactor, float], 
                                      platform: str) -> List[str]:
        """Generate recommendations to improve viral potential."""
        try:
            recommendations = []
            
            # Engagement intensity recommendations
            if factor_scores[ViralFactor.ENGAGEMENT_INTENSITY] < 0.6:
                recommendations.append("Increase engagement by adding more dynamic visuals or faster pacing")
            
            # Content novelty recommendations
            if factor_scores[ViralFactor.CONTENT_NOVELTY] < 0.5:
                recommendations.append("Add unique elements or unexpected twists to make content more novel")
            
            # Emotional impact recommendations
            if factor_scores[ViralFactor.EMOTIONAL_IMPACT] < 0.6:
                recommendations.append("Add emotional elements like storytelling or personal experiences")
            
            # Trend relevance recommendations
            if factor_scores[ViralFactor.TREND_RELEVANCE] < 0.5:
                recommendations.append(f"Incorporate trending keywords and topics for {platform}")
            
            # Audience alignment recommendations
            if factor_scores[ViralFactor.AUDIENCE_ALIGNMENT] < 0.6:
                recommendations.append(f"Better align content with {platform} audience preferences")
            
            # Timing recommendations
            if factor_scores[ViralFactor.TIMING_OPTIMALITY] < 0.5:
                recommendations.append("Post at optimal times for better visibility")
            
            # Shareability recommendations
            if factor_scores[ViralFactor.SHAREABILITY] < 0.5:
                recommendations.append("Add call-to-action elements to encourage sharing")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return ["Focus on creating engaging and shareable content"]
    
    async def _analyze_trend_alignment(self, keywords: List[str], platform: str) -> Dict[str, float]:
        """Analyze trend alignment across platforms."""
        try:
            trend_alignment = {}
            
            for platform_name in ["tiktok", "youtube", "instagram", "twitter"]:
                trends = await self.trend_analyzer.get_trending_keywords(platform_name, 10)
                
                if trends:
                    trend_keywords = [trend.keyword.lower() for trend in trends]
                    content_keywords_lower = [kw.lower() for kw in keywords]
                    
                    matches = sum(1 for kw in content_keywords_lower if kw in trend_keywords)
                    alignment = matches / len(keywords) if keywords else 0
                    
                    trend_alignment[platform_name] = min(alignment, 1.0)
                else:
                    trend_alignment[platform_name] = 0.5
            
            return trend_alignment
            
        except Exception as e:
            logger.error(f"Trend alignment analysis failed: {e}")
            return {platform: 0.5 for platform in ["tiktok", "youtube", "instagram", "twitter"]}
    
    async def _calculate_optimal_timing(self, platform: str) -> Optional[datetime]:
        """Calculate optimal posting time."""
        try:
            optimal_hours = {
                "tiktok": [19, 20, 21],
                "youtube": [14, 15, 16],
                "instagram": [12, 13, 17],
                "twitter": [9, 12, 15]
            }
            
            platform_hours = optimal_hours.get(platform, [12, 18, 20])
            optimal_hour = platform_hours[0]  # Use first optimal hour
            
            # Calculate next optimal time
            now = datetime.now()
            optimal_time = now.replace(hour=optimal_hour, minute=0, second=0, microsecond=0)
            
            # If optimal time has passed today, use tomorrow
            if optimal_time <= now:
                optimal_time += timedelta(days=1)
            
            return optimal_time
            
        except Exception as e:
            logger.error(f"Optimal timing calculation failed: {e}")
            return None

# Export the main class
__all__ = ["AdvancedViralScorer", "TrendAnalyzer", "HistoricalAnalyzer", "AudienceAnalyzer", "ViralScore"]


