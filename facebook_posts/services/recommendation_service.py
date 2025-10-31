"""
Advanced Recommendation Service for Facebook Posts API
Intelligent content recommendations, personalization, and smart suggestions
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
from ..services.ml_service import get_ml_service
from ..services.optimization_service import get_optimization_service

logger = structlog.get_logger(__name__)


class RecommendationType(Enum):
    """Recommendation types"""
    CONTENT = "content"
    TIMING = "timing"
    HASHTAGS = "hashtags"
    AUDIENCE = "audience"
    TOPICS = "topics"
    OPTIMIZATION = "optimization"
    TRENDING = "trending"
    PERSONALIZATION = "personalization"


class RecommendationPriority(Enum):
    """Recommendation priority levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Recommendation:
    """Content recommendation"""
    id: str
    type: RecommendationType
    priority: RecommendationPriority
    title: str
    description: str
    suggestion: str
    expected_impact: str
    confidence: float
    implementation: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class PersonalizedRecommendation:
    """Personalized recommendation for user"""
    user_id: str
    recommendations: List[Recommendation]
    personalization_score: float
    based_on: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ContentSuggestion:
    """Content suggestion"""
    topic: str
    content_type: ContentType
    audience_type: AudienceType
    suggested_content: str
    confidence: float
    expected_engagement: float
    tags: List[str] = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContentRecommender:
    """Content recommendation engine"""
    
    def __init__(self):
        self.ai_service = get_ai_service()
        self.analytics_service = get_analytics_service()
        self.ml_service = get_ml_service()
        self.cache_manager = get_cache_manager()
        self.monitor = get_monitor()
        self.trending_topics = self._load_trending_topics()
        self.content_templates = self._load_content_templates()
    
    def _load_trending_topics(self) -> Dict[str, List[str]]:
        """Load trending topics by category"""
        return {
            "technology": [
                "Artificial Intelligence", "Machine Learning", "Blockchain", "Cloud Computing",
                "Cybersecurity", "Data Science", "IoT", "5G", "Quantum Computing", "AR/VR"
            ],
            "business": [
                "Digital Transformation", "Remote Work", "Sustainability", "E-commerce",
                "Startup Culture", "Leadership", "Innovation", "Marketing", "Sales", "Finance"
            ],
            "lifestyle": [
                "Health & Wellness", "Mental Health", "Productivity", "Work-Life Balance",
                "Travel", "Food & Cooking", "Fitness", "Mindfulness", "Self-Care", "Hobbies"
            ],
            "education": [
                "Online Learning", "Skill Development", "Career Growth", "Professional Development",
                "Certifications", "Training", "Mentorship", "Networking", "Industry Trends", "Best Practices"
            ]
        }
    
    def _load_content_templates(self) -> Dict[str, List[str]]:
        """Load content templates by type"""
        return {
            "educational": [
                "Did you know that {topic}? Here's what you need to understand...",
                "5 key insights about {topic} that will change your perspective...",
                "The future of {topic} is here. Here's what it means for you...",
                "Breaking down {topic}: A comprehensive guide...",
                "Why {topic} matters more than you think..."
            ],
            "entertainment": [
                "You won't believe what happened with {topic}...",
                "The most surprising facts about {topic}...",
                "This {topic} story will make your day...",
                "Fun fact: {topic} is more interesting than you think...",
                "The hilarious truth about {topic}..."
            ],
            "promotional": [
                "Discover how {topic} can transform your business...",
                "Ready to revolutionize {topic}? Here's how...",
                "The {topic} solution you've been waiting for...",
                "Transform your approach to {topic} with these insights...",
                "Why {topic} is the key to your success..."
            ],
            "news": [
                "Breaking: {topic} makes headlines again...",
                "Latest update on {topic}: What you need to know...",
                "Industry experts weigh in on {topic}...",
                "The {topic} development that's changing everything...",
                "What the {topic} announcement means for the future..."
            ]
        }
    
    @timed("content_recommendation")
    async def recommend_content(
        self,
        user_id: str,
        audience_type: AudienceType,
        content_type: Optional[ContentType] = None,
        limit: int = 5
    ) -> List[ContentSuggestion]:
        """Recommend content topics and suggestions"""
        try:
            # Get user preferences and history
            user_preferences = await self._get_user_preferences(user_id)
            user_history = await self._get_user_content_history(user_id)
            
            # Get trending topics for audience
            trending_topics = self._get_trending_topics_for_audience(audience_type)
            
            # Generate content suggestions
            suggestions = []
            
            for topic in trending_topics[:limit * 2]:  # Get more topics to filter
                # Skip if user has recently posted about this topic
                if self._is_topic_recently_used(topic, user_history):
                    continue
                
                # Determine content type if not specified
                suggested_content_type = content_type or self._recommend_content_type(topic, audience_type)
                
                # Generate content using AI
                suggested_content = await self._generate_content_suggestion(
                    topic, suggested_content_type, audience_type
                )
                
                # Predict engagement
                engagement_prediction = await self.ml_service.predict_engagement(
                    suggested_content, {
                        "content_type": suggested_content_type.value,
                        "audience_type": audience_type.value
                    }
                )
                
                # Generate tags and hashtags
                tags = self._generate_tags(topic, suggested_content_type)
                hashtags = self._generate_hashtags(topic, suggested_content_type)
                
                suggestion = ContentSuggestion(
                    topic=topic,
                    content_type=suggested_content_type,
                    audience_type=audience_type,
                    suggested_content=suggested_content,
                    confidence=engagement_prediction.confidence,
                    expected_engagement=engagement_prediction.predicted_value,
                    tags=tags,
                    hashtags=hashtags,
                    metadata={
                        "user_preferences": user_preferences,
                        "trending_score": self._calculate_trending_score(topic),
                        "personalization_score": self._calculate_personalization_score(topic, user_preferences)
                    }
                )
                
                suggestions.append(suggestion)
            
            # Sort by expected engagement and confidence
            suggestions.sort(key=lambda x: x.expected_engagement * x.confidence, reverse=True)
            
            return suggestions[:limit]
            
        except Exception as e:
            logger.error("Content recommendation failed", user_id=user_id, error=str(e))
            return []
    
    async def _get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences from analytics"""
        try:
            # Get user's content history and preferences
            cache_key = f"user_preferences:{user_id}"
            cached_preferences = await self.cache_manager.cache.get(cache_key)
            
            if cached_preferences:
                return cached_preferences
            
            # Mock user preferences (in real implementation, this would come from user data)
            preferences = {
                "preferred_topics": ["technology", "business", "innovation"],
                "preferred_content_types": ["educational", "news"],
                "posting_frequency": "daily",
                "optimal_posting_times": ["9:00", "13:00", "17:00"],
                "engagement_goals": ["reach", "engagement"],
                "content_length_preference": "medium"
            }
            
            # Cache preferences
            await self.cache_manager.cache.set(cache_key, preferences, ttl=3600)
            
            return preferences
            
        except Exception as e:
            logger.error("Failed to get user preferences", user_id=user_id, error=str(e))
            return {}
    
    async def _get_user_content_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's content history"""
        try:
            # Get user's recent posts
            db_manager = get_db_manager()
            post_repo = PostRepository(db_manager)
            
            # Mock user history (in real implementation, this would query user's posts)
            history = [
                {"topic": "AI in Business", "created_at": datetime.now() - timedelta(days=1)},
                {"topic": "Digital Transformation", "created_at": datetime.now() - timedelta(days=3)},
                {"topic": "Remote Work", "created_at": datetime.now() - timedelta(days=5)}
            ]
            
            return history
            
        except Exception as e:
            logger.error("Failed to get user content history", user_id=user_id, error=str(e))
            return []
    
    def _get_trending_topics_for_audience(self, audience_type: AudienceType) -> List[str]:
        """Get trending topics for specific audience"""
        audience_topic_mapping = {
            AudienceType.PROFESSIONALS: self.trending_topics["technology"] + self.trending_topics["business"],
            AudienceType.GENERAL: self.trending_topics["lifestyle"] + self.trending_topics["education"],
            AudienceType.STUDENTS: self.trending_topics["education"] + self.trending_topics["technology"]
        }
        
        return audience_topic_mapping.get(audience_type, self.trending_topics["technology"])
    
    def _is_topic_recently_used(self, topic: str, user_history: List[Dict[str, Any]]) -> bool:
        """Check if topic was recently used by user"""
        recent_threshold = datetime.now() - timedelta(days=7)
        
        for post in user_history:
            if post["topic"].lower() in topic.lower() and post["created_at"] > recent_threshold:
                return True
        
        return False
    
    def _recommend_content_type(self, topic: str, audience_type: AudienceType) -> ContentType:
        """Recommend content type based on topic and audience"""
        # Simple heuristic-based recommendation
        if any(word in topic.lower() for word in ["ai", "technology", "innovation", "future"]):
            return ContentType.EDUCATIONAL
        elif any(word in topic.lower() for word in ["breaking", "news", "update", "announcement"]):
            return ContentType.NEWS
        elif any(word in topic.lower() for word in ["fun", "amazing", "surprising", "incredible"]):
            return ContentType.ENTERTAINMENT
        else:
            return ContentType.EDUCATIONAL
    
    async def _generate_content_suggestion(
        self,
        topic: str,
        content_type: ContentType,
        audience_type: AudienceType
    ) -> str:
        """Generate content suggestion using AI"""
        try:
            # Get content template
            templates = self.content_templates.get(content_type.value, self.content_templates["educational"])
            template = random.choice(templates)
            
            # Format template with topic
            content = template.format(topic=topic)
            
            # Enhance with AI if available
            try:
                from ..core.models import PostRequest
                request = PostRequest(
                    topic=topic,
                    content_type=content_type,
                    audience_type=audience_type,
                    tone="professional" if audience_type == AudienceType.PROFESSIONALS else "friendly"
                )
                
                ai_result = await self.ai_service.generate_content(request)
                if ai_result and ai_result.content:
                    content = ai_result.content
                    
            except Exception as e:
                logger.warning("AI content generation failed, using template", error=str(e))
            
            return content
            
        except Exception as e:
            logger.error("Content suggestion generation failed", error=str(e))
            return f"Interesting insights about {topic} that you should know..."
    
    def _generate_tags(self, topic: str, content_type: ContentType) -> List[str]:
        """Generate relevant tags for content"""
        base_tags = [content_type.value, topic.lower().replace(" ", "_")]
        
        # Add topic-specific tags
        topic_words = topic.lower().split()
        base_tags.extend(topic_words[:3])  # Add first 3 words as tags
        
        return base_tags[:5]  # Limit to 5 tags
    
    def _generate_hashtags(self, topic: str, content_type: ContentType) -> List[str]:
        """Generate relevant hashtags for content"""
        hashtags = []
        
        # Add content type hashtag
        hashtags.append(f"#{content_type.value}")
        
        # Add topic hashtags
        topic_words = topic.lower().split()
        for word in topic_words[:3]:
            hashtags.append(f"#{word}")
        
        # Add trending hashtags
        trending_hashtags = ["#innovation", "#technology", "#business", "#future", "#growth"]
        hashtags.extend(random.sample(trending_hashtags, 2))
        
        return hashtags[:5]  # Limit to 5 hashtags
    
    def _calculate_trending_score(self, topic: str) -> float:
        """Calculate trending score for topic"""
        # Simple trending score calculation
        trending_keywords = ["ai", "artificial intelligence", "machine learning", "blockchain", "crypto"]
        
        topic_lower = topic.lower()
        score = 0.5  # Base score
        
        for keyword in trending_keywords:
            if keyword in topic_lower:
                score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_personalization_score(self, topic: str, user_preferences: Dict[str, Any]) -> float:
        """Calculate personalization score for topic"""
        if not user_preferences:
            return 0.5
        
        preferred_topics = user_preferences.get("preferred_topics", [])
        topic_lower = topic.lower()
        
        score = 0.5  # Base score
        
        for preferred_topic in preferred_topics:
            if preferred_topic.lower() in topic_lower:
                score += 0.3
        
        return min(score, 1.0)


class TimingRecommender:
    """Optimal timing recommendation engine"""
    
    def __init__(self):
        self.analytics_service = get_analytics_service()
        self.cache_manager = get_cache_manager()
        self.optimal_times = self._load_optimal_times()
    
    def _load_optimal_times(self) -> Dict[str, Dict[str, List[str]]]:
        """Load optimal posting times by audience and content type"""
        return {
            "professionals": {
                "educational": ["9:00", "13:00", "17:00"],
                "news": ["8:00", "12:00", "18:00"],
                "entertainment": ["12:00", "19:00", "21:00"],
                "promotional": ["10:00", "14:00", "16:00"]
            },
            "general": {
                "educational": ["10:00", "14:00", "20:00"],
                "news": ["7:00", "12:00", "19:00"],
                "entertainment": ["12:00", "18:00", "22:00"],
                "promotional": ["11:00", "15:00", "19:00"]
            },
            "students": {
                "educational": ["10:00", "15:00", "21:00"],
                "news": ["8:00", "13:00", "20:00"],
                "entertainment": ["12:00", "19:00", "23:00"],
                "promotional": ["11:00", "16:00", "20:00"]
            }
        }
    
    @timed("timing_recommendation")
    async def recommend_timing(
        self,
        audience_type: AudienceType,
        content_type: ContentType,
        user_id: Optional[str] = None
    ) -> List[Recommendation]:
        """Recommend optimal posting times"""
        try:
            recommendations = []
            
            # Get optimal times for audience and content type
            audience_key = audience_type.value
            content_key = content_type.value
            
            optimal_times = self.optimal_times.get(audience_key, {}).get(content_key, ["9:00", "13:00", "19:00"])
            
            # Create timing recommendations
            for i, time_slot in enumerate(optimal_times):
                priority = RecommendationPriority.HIGH if i == 0 else RecommendationPriority.MEDIUM
                
                recommendation = Recommendation(
                    id=f"timing_{time_slot}_{audience_key}_{content_key}",
                    type=RecommendationType.TIMING,
                    priority=priority,
                    title=f"Optimal Posting Time: {time_slot}",
                    description=f"Post at {time_slot} for maximum engagement with {audience_key} audience",
                    suggestion=f"Schedule your {content_key} content for {time_slot}",
                    expected_impact="20-30% increase in engagement",
                    confidence=0.8,
                    implementation=f"Use scheduling tools to post at {time_slot}",
                    metadata={
                        "time_slot": time_slot,
                        "audience_type": audience_key,
                        "content_type": content_key,
                        "rank": i + 1
                    }
                )
                
                recommendations.append(recommendation)
            
            # Add personalized timing if user_id provided
            if user_id:
                personalized_timing = await self._get_personalized_timing(user_id, audience_type, content_type)
                if personalized_timing:
                    recommendations.extend(personalized_timing)
            
            return recommendations
            
        except Exception as e:
            logger.error("Timing recommendation failed", error=str(e))
            return []
    
    async def _get_personalized_timing(self, user_id: str, audience_type: AudienceType, content_type: ContentType) -> List[Recommendation]:
        """Get personalized timing recommendations based on user history"""
        try:
            # Get user's posting history and engagement data
            user_history = await self._get_user_posting_history(user_id)
            
            if not user_history:
                return []
            
            # Analyze best performing times
            best_times = self._analyze_best_times(user_history)
            
            recommendations = []
            for time_slot, engagement_rate in best_times.items():
                if engagement_rate > 0.7:  # High engagement threshold
                    recommendation = Recommendation(
                        id=f"personalized_timing_{time_slot}_{user_id}",
                        type=RecommendationType.TIMING,
                        priority=RecommendationPriority.HIGH,
                        title=f"Your Best Time: {time_slot}",
                        description=f"Your posts perform best at {time_slot} with {engagement_rate:.1%} engagement",
                        suggestion=f"Continue posting at {time_slot} for optimal results",
                        expected_impact="Maintain high engagement levels",
                        confidence=0.9,
                        implementation=f"Keep your current posting schedule around {time_slot}",
                        metadata={
                            "time_slot": time_slot,
                            "engagement_rate": engagement_rate,
                            "personalized": True
                        }
                    )
                    
                    recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error("Personalized timing recommendation failed", user_id=user_id, error=str(e))
            return []
    
    async def _get_user_posting_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's posting history with engagement data"""
        # Mock user posting history (in real implementation, this would query user's posts)
        return [
            {"time": "9:00", "engagement_rate": 0.8, "content_type": "educational"},
            {"time": "13:00", "engagement_rate": 0.75, "content_type": "news"},
            {"time": "17:00", "engagement_rate": 0.9, "content_type": "educational"},
            {"time": "21:00", "engagement_rate": 0.6, "content_type": "entertainment"}
        ]
    
    def _analyze_best_times(self, user_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze best performing times from user history"""
        time_performance = {}
        
        for post in user_history:
            time_slot = post["time"]
            engagement_rate = post["engagement_rate"]
            
            if time_slot in time_performance:
                time_performance[time_slot] = (time_performance[time_slot] + engagement_rate) / 2
            else:
                time_performance[time_slot] = engagement_rate
        
        # Sort by engagement rate
        return dict(sorted(time_performance.items(), key=lambda x: x[1], reverse=True))


class HashtagRecommender:
    """Hashtag recommendation engine"""
    
    def __init__(self):
        self.cache_manager = get_cache_manager()
        self.trending_hashtags = self._load_trending_hashtags()
        self.hashtag_categories = self._load_hashtag_categories()
    
    def _load_trending_hashtags(self) -> Dict[str, List[str]]:
        """Load trending hashtags by category"""
        return {
            "technology": ["#AI", "#MachineLearning", "#Blockchain", "#CloudComputing", "#Cybersecurity"],
            "business": ["#DigitalTransformation", "#Innovation", "#Leadership", "#Startup", "#Entrepreneurship"],
            "lifestyle": ["#Wellness", "#Productivity", "#Mindfulness", "#SelfCare", "#WorkLifeBalance"],
            "education": ["#Learning", "#SkillDevelopment", "#CareerGrowth", "#ProfessionalDevelopment", "#Training"]
        }
    
    def _load_hashtag_categories(self) -> Dict[str, List[str]]:
        """Load hashtag categories and their purposes"""
        return {
            "trending": ["#Innovation", "#Future", "#Technology", "#Business", "#Growth"],
            "niche": ["#AI", "#MachineLearning", "#Blockchain", "#CloudComputing", "#Cybersecurity"],
            "community": ["#TechCommunity", "#BusinessLeaders", "#Innovators", "#Entrepreneurs", "#Professionals"],
            "engagement": ["#WhatDoYouThink", "#ShareYourThoughts", "#YourExperience", "#Opinions", "#Discussion"]
        }
    
    @timed("hashtag_recommendation")
    async def recommend_hashtags(
        self,
        content: str,
        audience_type: AudienceType,
        content_type: ContentType,
        limit: int = 5
    ) -> List[Recommendation]:
        """Recommend hashtags for content"""
        try:
            recommendations = []
            
            # Extract keywords from content
            keywords = self._extract_keywords(content)
            
            # Get relevant hashtags
            relevant_hashtags = self._get_relevant_hashtags(keywords, audience_type, content_type)
            
            # Create hashtag recommendations
            for i, hashtag in enumerate(relevant_hashtags[:limit]):
                priority = RecommendationPriority.HIGH if i < 2 else RecommendationPriority.MEDIUM
                
                recommendation = Recommendation(
                    id=f"hashtag_{hashtag}_{audience_type.value}_{content_type.value}",
                    type=RecommendationType.HASHTAGS,
                    priority=priority,
                    title=f"Recommended Hashtag: {hashtag}",
                    description=f"Use {hashtag} to increase discoverability and reach",
                    suggestion=f"Add {hashtag} to your post for better visibility",
                    expected_impact="15-25% increase in reach",
                    confidence=0.8,
                    implementation=f"Include {hashtag} in your post text",
                    metadata={
                        "hashtag": hashtag,
                        "audience_type": audience_type.value,
                        "content_type": content_type.value,
                        "relevance_score": self._calculate_hashtag_relevance(hashtag, keywords)
                    }
                )
                
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error("Hashtag recommendation failed", error=str(e))
            return []
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content"""
        # Simple keyword extraction
        words = content.lower().split()
        
        # Filter out common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should"}
        
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        return keywords[:10]  # Limit to 10 keywords
    
    def _get_relevant_hashtags(self, keywords: List[str], audience_type: AudienceType, content_type: ContentType) -> List[str]:
        """Get relevant hashtags based on keywords, audience, and content type"""
        relevant_hashtags = []
        
        # Get audience-specific hashtags
        audience_hashtags = self.trending_hashtags.get(audience_type.value, self.trending_hashtags["technology"])
        relevant_hashtags.extend(audience_hashtags)
        
        # Get content-type specific hashtags
        if content_type == ContentType.EDUCATIONAL:
            relevant_hashtags.extend(["#Learning", "#Education", "#Knowledge", "#Insights"])
        elif content_type == ContentType.NEWS:
            relevant_hashtags.extend(["#News", "#Update", "#Breaking", "#Latest"])
        elif content_type == ContentType.ENTERTAINMENT:
            relevant_hashtags.extend(["#Fun", "#Entertainment", "#Interesting", "#Amazing"])
        elif content_type == ContentType.PROMOTIONAL:
            relevant_hashtags.extend(["#Product", "#Service", "#Solution", "#Offer"])
        
        # Add trending hashtags
        relevant_hashtags.extend(self.hashtag_categories["trending"])
        
        # Add engagement hashtags
        relevant_hashtags.extend(self.hashtag_categories["engagement"])
        
        # Remove duplicates and limit
        return list(set(relevant_hashtags))[:15]
    
    def _calculate_hashtag_relevance(self, hashtag: str, keywords: List[str]) -> float:
        """Calculate relevance score for hashtag"""
        hashtag_clean = hashtag.lower().replace("#", "")
        
        # Check if hashtag matches any keywords
        for keyword in keywords:
            if keyword in hashtag_clean or hashtag_clean in keyword:
                return 0.9
        
        # Check for partial matches
        for keyword in keywords:
            if any(word in hashtag_clean for word in keyword.split()):
                return 0.7
        
        return 0.5  # Base relevance score


class RecommendationService:
    """Main recommendation service orchestrator"""
    
    def __init__(self):
        self.content_recommender = ContentRecommender()
        self.timing_recommender = TimingRecommender()
        self.hashtag_recommender = HashtagRecommender()
        self.cache_manager = get_cache_manager()
        self.monitor = get_monitor()
    
    @timed("comprehensive_recommendations")
    async def get_comprehensive_recommendations(
        self,
        user_id: str,
        audience_type: AudienceType,
        content_type: Optional[ContentType] = None,
        limit: int = 10
    ) -> PersonalizedRecommendation:
        """Get comprehensive personalized recommendations"""
        try:
            recommendations = []
            
            # Get content recommendations
            content_suggestions = await self.content_recommender.recommend_content(
                user_id, audience_type, content_type, limit=3
            )
            
            # Convert content suggestions to recommendations
            for suggestion in content_suggestions:
                recommendation = Recommendation(
                    id=f"content_{suggestion.topic}_{user_id}",
                    type=RecommendationType.CONTENT,
                    priority=RecommendationPriority.HIGH,
                    title=f"Content Idea: {suggestion.topic}",
                    description=f"Create {suggestion.content_type.value} content about {suggestion.topic}",
                    suggestion=suggestion.suggested_content,
                    expected_impact=f"{suggestion.expected_engagement:.1%} expected engagement",
                    confidence=suggestion.confidence,
                    implementation="Use the suggested content as a starting point",
                    metadata={
                        "topic": suggestion.topic,
                        "content_type": suggestion.content_type.value,
                        "expected_engagement": suggestion.expected_engagement,
                        "tags": suggestion.tags,
                        "hashtags": suggestion.hashtags
                    }
                )
                recommendations.append(recommendation)
            
            # Get timing recommendations
            if content_type:
                timing_recommendations = await self.timing_recommender.recommend_timing(
                    audience_type, content_type, user_id
                )
                recommendations.extend(timing_recommendations)
            
            # Get hashtag recommendations for top content suggestion
            if content_suggestions:
                top_suggestion = content_suggestions[0]
                hashtag_recommendations = await self.hashtag_recommender.recommend_hashtags(
                    top_suggestion.suggested_content, audience_type, top_suggestion.content_type
                )
                recommendations.extend(hashtag_recommendations)
            
            # Calculate personalization score
            personalization_score = self._calculate_personalization_score(recommendations, user_id)
            
            # Get recommendation basis
            based_on = self._get_recommendation_basis(user_id, audience_type, content_type)
            
            return PersonalizedRecommendation(
                user_id=user_id,
                recommendations=recommendations[:limit],
                personalization_score=personalization_score,
                based_on=based_on
            )
            
        except Exception as e:
            logger.error("Comprehensive recommendations failed", user_id=user_id, error=str(e))
            return PersonalizedRecommendation(
                user_id=user_id,
                recommendations=[],
                personalization_score=0.0,
                based_on=["error"]
            )
    
    def _calculate_personalization_score(self, recommendations: List[Recommendation], user_id: str) -> float:
        """Calculate personalization score for recommendations"""
        if not recommendations:
            return 0.0
        
        # Calculate average confidence and priority scores
        confidence_scores = [rec.confidence for rec in recommendations]
        priority_scores = [0.9 if rec.priority == RecommendationPriority.HIGH else 0.7 if rec.priority == RecommendationPriority.MEDIUM else 0.5 for rec in recommendations]
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        avg_priority = sum(priority_scores) / len(priority_scores)
        
        return (avg_confidence + avg_priority) / 2
    
    def _get_recommendation_basis(self, user_id: str, audience_type: AudienceType, content_type: Optional[ContentType]) -> List[str]:
        """Get basis for recommendations"""
        basis = [
            f"Audience: {audience_type.value}",
            "Trending topics analysis",
            "Engagement prediction models"
        ]
        
        if content_type:
            basis.append(f"Content type: {content_type.value}")
        
        basis.extend([
            "User posting history",
            "Optimal timing analysis",
            "Hashtag performance data"
        ])
        
        return basis


# Global recommendation service instance
_recommendation_service: Optional[RecommendationService] = None


def get_recommendation_service() -> RecommendationService:
    """Get global recommendation service instance"""
    global _recommendation_service
    
    if _recommendation_service is None:
        _recommendation_service = RecommendationService()
    
    return _recommendation_service


# Export all classes and functions
__all__ = [
    # Enums
    'RecommendationType',
    'RecommendationPriority',
    
    # Data classes
    'Recommendation',
    'PersonalizedRecommendation',
    'ContentSuggestion',
    
    # Services
    'ContentRecommender',
    'TimingRecommender',
    'HashtagRecommender',
    'RecommendationService',
    
    # Utility functions
    'get_recommendation_service',
]






























