"""
Intelligent Recommendation Engine
================================

Advanced recommendation system with collaborative filtering,
content-based filtering, and hybrid approaches for document classification.
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
import statistics
import math

logger = logging.getLogger(__name__)

class RecommendationType(Enum):
    """Recommendation types"""
    DOCUMENT_TYPE = "document_type"
    TEMPLATE = "template"
    CONTENT_STYLE = "content_style"
    WORKFLOW = "workflow"
    INTEGRATION = "integration"
    OPTIMIZATION = "optimization"
    FEATURE = "feature"
    SERVICE = "service"

class RecommendationConfidence(Enum):
    """Recommendation confidence levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class FilteringMethod(Enum):
    """Filtering methods"""
    COLLABORATIVE = "collaborative"
    CONTENT_BASED = "content_based"
    HYBRID = "hybrid"
    KNOWLEDGE_BASED = "knowledge_based"
    DEMOGRAPHIC = "demographic"
    CONTEXTUAL = "contextual"

@dataclass
class UserProfile:
    """User profile for recommendations"""
    user_id: str
    preferences: Dict[str, Any]
    behavior_history: List[Dict[str, Any]]
    demographics: Dict[str, Any]
    expertise_level: str
    industry: str
    created_at: datetime
    last_updated: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Item:
    """Item for recommendation"""
    item_id: str
    item_type: RecommendationType
    name: str
    description: str
    features: Dict[str, Any]
    tags: List[str]
    popularity_score: float
    quality_score: float
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Interaction:
    """User-item interaction"""
    user_id: str
    item_id: str
    interaction_type: str  # view, click, use, rate, share
    rating: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Recommendation:
    """Recommendation result"""
    id: str
    user_id: str
    item_id: str
    item_type: RecommendationType
    score: float
    confidence: RecommendationConfidence
    reasoning: str
    filtering_method: FilteringMethod
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RecommendationSet:
    """Set of recommendations"""
    id: str
    user_id: str
    recommendations: List[Recommendation]
    total_count: int
    generated_at: datetime
    algorithm_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class IntelligentRecommender:
    """
    Intelligent recommendation engine
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize recommendation engine
        
        Args:
            data_dir: Directory for recommendation data
        """
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.user_profiles: Dict[str, UserProfile] = {}
        self.items: Dict[str, Item] = {}
        self.interactions: List[Interaction] = []
        self.recommendations: Dict[str, RecommendationSet] = {}
        
        # Recommendation algorithms
        self.collaborative_filter = CollaborativeFilter()
        self.content_filter = ContentBasedFilter()
        self.hybrid_filter = HybridFilter()
        
        # Initialize default items
        self._initialize_default_items()
        
        # Recommendation parameters
        self.min_interactions = 5
        self.max_recommendations = 20
        self.cold_start_threshold = 3
    
    def _initialize_default_items(self):
        """Initialize default items for recommendation"""
        default_items = [
            {
                "item_id": "novel_template_1",
                "item_type": RecommendationType.TEMPLATE,
                "name": "Classic Novel Template",
                "description": "Traditional three-act structure for novels",
                "features": {"genre": "fiction", "structure": "three_act", "length": "long"},
                "tags": ["novel", "fiction", "classic", "three_act"],
                "popularity_score": 0.8,
                "quality_score": 0.9
            },
            {
                "item_id": "contract_template_1",
                "item_type": RecommendationType.TEMPLATE,
                "name": "Business Contract Template",
                "description": "Standard business contract with legal clauses",
                "features": {"type": "legal", "industry": "business", "complexity": "high"},
                "tags": ["contract", "legal", "business", "professional"],
                "popularity_score": 0.9,
                "quality_score": 0.85
            },
            {
                "item_id": "design_template_1",
                "item_type": RecommendationType.TEMPLATE,
                "name": "Modern Design Template",
                "description": "Contemporary design with clean aesthetics",
                "features": {"style": "modern", "color_scheme": "minimal", "layout": "grid"},
                "tags": ["design", "modern", "minimal", "clean"],
                "popularity_score": 0.7,
                "quality_score": 0.8
            },
            {
                "item_id": "article_style_formal",
                "item_type": RecommendationType.CONTENT_STYLE,
                "name": "Formal Writing Style",
                "description": "Professional and academic writing style",
                "features": {"tone": "formal", "voice": "third_person", "complexity": "high"},
                "tags": ["formal", "professional", "academic", "serious"],
                "popularity_score": 0.6,
                "quality_score": 0.75
            },
            {
                "item_id": "workflow_automation",
                "item_type": RecommendationType.WORKFLOW,
                "name": "Document Processing Workflow",
                "description": "Automated workflow for document processing",
                "features": {"automation": "high", "steps": 5, "efficiency": "high"},
                "tags": ["automation", "workflow", "efficiency", "processing"],
                "popularity_score": 0.8,
                "quality_score": 0.9
            }
        ]
        
        for item_data in default_items:
            item = Item(
                item_id=item_data["item_id"],
                item_type=item_data["item_type"],
                name=item_data["name"],
                description=item_data["description"],
                features=item_data["features"],
                tags=item_data["tags"],
                popularity_score=item_data["popularity_score"],
                quality_score=item_data["quality_score"],
                created_at=datetime.now(),
                metadata={}
            )
            self.items[item.item_id] = item
    
    async def create_user_profile(self, user_id: str, preferences: Dict[str, Any], demographics: Dict[str, Any]) -> UserProfile:
        """
        Create user profile
        
        Args:
            user_id: Unique user identifier
            preferences: User preferences
            demographics: User demographics
            
        Returns:
            Created user profile
        """
        profile = UserProfile(
            user_id=user_id,
            preferences=preferences,
            behavior_history=[],
            demographics=demographics,
            expertise_level=demographics.get("expertise_level", "beginner"),
            industry=demographics.get("industry", "general"),
            created_at=datetime.now(),
            last_updated=datetime.now(),
            metadata={}
        )
        
        self.user_profiles[user_id] = profile
        
        logger.info(f"Created user profile for {user_id}")
        
        return profile
    
    async def record_interaction(self, user_id: str, item_id: str, interaction_type: str, rating: Optional[float] = None, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Record user-item interaction
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            interaction_type: Type of interaction
            rating: Optional rating
            context: Optional context
            
        Returns:
            Interaction ID
        """
        if context is None:
            context = {}
        
        interaction = Interaction(
            user_id=user_id,
            item_id=item_id,
            interaction_type=interaction_type,
            rating=rating,
            context=context
        )
        
        self.interactions.append(interaction)
        
        # Update user profile
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            profile.behavior_history.append({
                "item_id": item_id,
                "interaction_type": interaction_type,
                "rating": rating,
                "timestamp": interaction.timestamp,
                "context": context
            })
            profile.last_updated = datetime.now()
        
        logger.info(f"Recorded interaction: {user_id} -> {item_id} ({interaction_type})")
        
        return str(uuid.uuid4())
    
    async def get_recommendations(self, user_id: str, item_type: Optional[RecommendationType] = None, limit: int = 10) -> RecommendationSet:
        """
        Get recommendations for user
        
        Args:
            user_id: User identifier
            item_type: Optional item type filter
            limit: Maximum number of recommendations
            
        Returns:
            Set of recommendations
        """
        if user_id not in self.user_profiles:
            raise ValueError(f"User profile not found: {user_id}")
        
        user_profile = self.user_profiles[user_id]
        user_interactions = [i for i in self.interactions if i.user_id == user_id]
        
        # Determine if user is cold start
        is_cold_start = len(user_interactions) < self.cold_start_threshold
        
        # Select recommendation method
        if is_cold_start:
            recommendations = await self._get_cold_start_recommendations(user_profile, item_type, limit)
            algorithm_used = "cold_start"
        else:
            # Use hybrid approach for experienced users
            recommendations = await self._get_hybrid_recommendations(user_profile, user_interactions, item_type, limit)
            algorithm_used = "hybrid"
        
        # Create recommendation set
        recommendation_set = RecommendationSet(
            id=str(uuid.uuid4()),
            user_id=user_id,
            recommendations=recommendations,
            total_count=len(recommendations),
            generated_at=datetime.now(),
            algorithm_used=algorithm_used,
            metadata={
                "is_cold_start": is_cold_start,
                "user_interactions": len(user_interactions),
                "item_type_filter": item_type.value if item_type else None
            }
        )
        
        self.recommendations[recommendation_set.id] = recommendation_set
        
        logger.info(f"Generated {len(recommendations)} recommendations for {user_id}")
        
        return recommendation_set
    
    async def _get_cold_start_recommendations(self, user_profile: UserProfile, item_type: Optional[RecommendationType], limit: int) -> List[Recommendation]:
        """Get recommendations for cold start users"""
        recommendations = []
        
        # Filter items by type if specified
        candidate_items = list(self.items.values())
        if item_type:
            candidate_items = [item for item in candidate_items if item.item_type == item_type]
        
        # Sort by popularity and quality
        candidate_items.sort(key=lambda x: (x.popularity_score + x.quality_score) / 2, reverse=True)
        
        # Create recommendations
        for i, item in enumerate(candidate_items[:limit]):
            score = (item.popularity_score + item.quality_score) / 2
            confidence = self._calculate_confidence(score, "cold_start")
            
            recommendation = Recommendation(
                id=str(uuid.uuid4()),
                user_id=user_profile.user_id,
                item_id=item.item_id,
                item_type=item.item_type,
                score=score,
                confidence=confidence,
                reasoning=f"Popular and high-quality {item.item_type.value} based on overall ratings",
                filtering_method=FilteringMethod.KNOWLEDGE_BASED,
                created_at=datetime.now(),
                metadata={"cold_start": True}
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    async def _get_hybrid_recommendations(self, user_profile: UserProfile, user_interactions: List[Interaction], item_type: Optional[RecommendationType], limit: int) -> List[Recommendation]:
        """Get hybrid recommendations for experienced users"""
        # Get collaborative filtering recommendations
        collab_recs = await self.collaborative_filter.get_recommendations(
            user_profile, user_interactions, self.items, item_type, limit // 2
        )
        
        # Get content-based recommendations
        content_recs = await self.content_filter.get_recommendations(
            user_profile, user_interactions, self.items, item_type, limit // 2
        )
        
        # Combine and rank recommendations
        all_recommendations = collab_recs + content_recs
        
        # Remove duplicates and sort by score
        unique_recommendations = {}
        for rec in all_recommendations:
            if rec.item_id not in unique_recommendations or rec.score > unique_recommendations[rec.item_id].score:
                unique_recommendations[rec.item_id] = rec
        
        # Sort by score and return top recommendations
        sorted_recommendations = sorted(unique_recommendations.values(), key=lambda x: x.score, reverse=True)
        
        return sorted_recommendations[:limit]
    
    def _calculate_confidence(self, score: float, method: str) -> RecommendationConfidence:
        """Calculate recommendation confidence"""
        if method == "cold_start":
            if score >= 0.8:
                return RecommendationConfidence.HIGH
            elif score >= 0.6:
                return RecommendationConfidence.MEDIUM
            else:
                return RecommendationConfidence.LOW
        else:
            if score >= 0.9:
                return RecommendationConfidence.VERY_HIGH
            elif score >= 0.7:
                return RecommendationConfidence.HIGH
            elif score >= 0.5:
                return RecommendationConfidence.MEDIUM
            elif score >= 0.3:
                return RecommendationConfidence.LOW
            else:
                return RecommendationConfidence.VERY_LOW
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID"""
        return self.user_profiles.get(user_id)
    
    def get_item(self, item_id: str) -> Optional[Item]:
        """Get item by ID"""
        return self.items.get(item_id)
    
    def get_recommendation_set(self, set_id: str) -> Optional[RecommendationSet]:
        """Get recommendation set by ID"""
        return self.recommendations.get(set_id)
    
    def get_recommender_statistics(self) -> Dict[str, Any]:
        """Get recommender statistics"""
        total_users = len(self.user_profiles)
        total_items = len(self.items)
        total_interactions = len(self.interactions)
        total_recommendations = sum(len(rec_set.recommendations) for rec_set in self.recommendations.values())
        
        # Interaction types
        interaction_types = Counter(interaction.interaction_type for interaction in self.interactions)
        
        # Item types
        item_types = Counter(item.item_type for item in self.items.values())
        
        # User expertise levels
        expertise_levels = Counter(profile.expertise_level for profile in self.user_profiles.values())
        
        return {
            "total_users": total_users,
            "total_items": total_items,
            "total_interactions": total_interactions,
            "total_recommendations": total_recommendations,
            "interaction_types": dict(interaction_types),
            "item_types": {item_type.value: count for item_type, count in item_types.items()},
            "expertise_levels": dict(expertise_levels),
            "average_interactions_per_user": total_interactions / max(total_users, 1),
            "recommendation_sets": len(self.recommendations)
        }

class CollaborativeFilter:
    """Collaborative filtering recommendation algorithm"""
    
    async def get_recommendations(self, user_profile: UserProfile, user_interactions: List[Interaction], items: Dict[str, Item], item_type: Optional[RecommendationType], limit: int) -> List[Recommendation]:
        """Get collaborative filtering recommendations"""
        recommendations = []
        
        # Find similar users based on interactions
        similar_users = self._find_similar_users(user_profile.user_id, user_interactions)
        
        # Get items liked by similar users
        candidate_items = self._get_items_from_similar_users(similar_users, items, item_type)
        
        # Calculate scores and create recommendations
        for item_id, score in candidate_items.items():
            if item_id in items:
                item = items[item_id]
                confidence = self._calculate_collaborative_confidence(score, len(similar_users))
                
                recommendation = Recommendation(
                    id=str(uuid.uuid4()),
                    user_id=user_profile.user_id,
                    item_id=item_id,
                    item_type=item.item_type,
                    score=score,
                    confidence=confidence,
                    reasoning=f"Recommended by {len(similar_users)} similar users",
                    filtering_method=FilteringMethod.COLLABORATIVE,
                    created_at=datetime.now()
                )
                
                recommendations.append(recommendation)
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x.score, reverse=True)
        return recommendations[:limit]
    
    def _find_similar_users(self, user_id: str, user_interactions: List[Interaction]) -> List[str]:
        """Find users similar to the given user"""
        # Simplified similarity calculation
        # In practice, you'd use more sophisticated algorithms like cosine similarity
        
        user_items = {interaction.item_id for interaction in user_interactions}
        
        # Find other users who interacted with similar items
        similar_users = []
        all_interactions = []  # This would come from the main recommender
        
        # Group interactions by user
        user_item_sets = defaultdict(set)
        for interaction in all_interactions:
            if interaction.user_id != user_id:
                user_item_sets[interaction.user_id].add(interaction.item_id)
        
        # Calculate similarity
        for other_user, other_items in user_item_sets.items():
            if len(other_items) > 0:
                similarity = len(user_items.intersection(other_items)) / len(user_items.union(other_items))
                if similarity > 0.1:  # Threshold for similarity
                    similar_users.append(other_user)
        
        return similar_users
    
    def _get_items_from_similar_users(self, similar_users: List[str], items: Dict[str, Item], item_type: Optional[RecommendationType]) -> Dict[str, float]:
        """Get items recommended by similar users"""
        # Simplified implementation
        # In practice, you'd analyze interactions from similar users
        
        candidate_items = {}
        
        # For demo purposes, recommend popular items
        for item in items.values():
            if item_type is None or item.item_type == item_type:
                # Simulate score based on popularity
                score = item.popularity_score * 0.8  # Collaborative boost
                candidate_items[item.item_id] = score
        
        return candidate_items
    
    def _calculate_collaborative_confidence(self, score: float, similar_users_count: int) -> RecommendationConfidence:
        """Calculate confidence for collaborative filtering"""
        confidence_factor = min(1.0, similar_users_count / 10)  # More users = higher confidence
        adjusted_score = score * confidence_factor
        
        if adjusted_score >= 0.8:
            return RecommendationConfidence.HIGH
        elif adjusted_score >= 0.6:
            return RecommendationConfidence.MEDIUM
        else:
            return RecommendationConfidence.LOW

class ContentBasedFilter:
    """Content-based filtering recommendation algorithm"""
    
    async def get_recommendations(self, user_profile: UserProfile, user_interactions: List[Interaction], items: Dict[str, Item], item_type: Optional[RecommendationType], limit: int) -> List[Recommendation]:
        """Get content-based recommendations"""
        recommendations = []
        
        # Build user preference profile
        user_preferences = self._build_user_preferences(user_interactions, items)
        
        # Find items similar to user preferences
        candidate_items = self._find_similar_items(user_preferences, items, item_type)
        
        # Create recommendations
        for item_id, score in candidate_items.items():
            if item_id in items:
                item = items[item_id]
                confidence = self._calculate_content_confidence(score)
                
                recommendation = Recommendation(
                    id=str(uuid.uuid4()),
                    user_id=user_profile.user_id,
                    item_id=item_id,
                    item_type=item.item_type,
                    score=score,
                    confidence=confidence,
                    reasoning="Similar to your previous preferences and interactions",
                    filtering_method=FilteringMethod.CONTENT_BASED,
                    created_at=datetime.now()
                )
                
                recommendations.append(recommendation)
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x.score, reverse=True)
        return recommendations[:limit]
    
    def _build_user_preferences(self, user_interactions: List[Interaction], items: Dict[str, Item]) -> Dict[str, float]:
        """Build user preference profile from interactions"""
        preferences = defaultdict(float)
        
        for interaction in user_interactions:
            if interaction.item_id in items:
                item = items[interaction.item_id]
                weight = 1.0
                
                # Weight by interaction type
                if interaction.interaction_type == "rate" and interaction.rating:
                    weight = interaction.rating / 5.0  # Normalize rating
                elif interaction.interaction_type == "use":
                    weight = 1.5
                elif interaction.interaction_type == "share":
                    weight = 2.0
                
                # Add item features to preferences
                for feature, value in item.features.items():
                    if isinstance(value, str):
                        preferences[f"{feature}_{value}"] += weight
                    else:
                        preferences[feature] += weight * value
        
        return dict(preferences)
    
    def _find_similar_items(self, user_preferences: Dict[str, float], items: Dict[str, Item], item_type: Optional[RecommendationType]) -> Dict[str, float]:
        """Find items similar to user preferences"""
        candidate_items = {}
        
        for item in items.values():
            if item_type is None or item.item_type == item_type:
                similarity = self._calculate_item_similarity(user_preferences, item)
                if similarity > 0.1:  # Threshold for similarity
                    candidate_items[item.item_id] = similarity
        
        return candidate_items
    
    def _calculate_item_similarity(self, user_preferences: Dict[str, float], item: Item) -> float:
        """Calculate similarity between user preferences and item"""
        similarity = 0.0
        total_weight = 0.0
        
        for feature, value in item.features.items():
            if isinstance(value, str):
                feature_key = f"{feature}_{value}"
                if feature_key in user_preferences:
                    similarity += user_preferences[feature_key]
                    total_weight += 1.0
            else:
                if feature in user_preferences:
                    similarity += user_preferences[feature] * value
                    total_weight += 1.0
        
        # Also consider tags
        for tag in item.tags:
            if tag in user_preferences:
                similarity += user_preferences[tag] * 0.5
                total_weight += 0.5
        
        if total_weight > 0:
            return similarity / total_weight
        
        return 0.0
    
    def _calculate_content_confidence(self, score: float) -> RecommendationConfidence:
        """Calculate confidence for content-based filtering"""
        if score >= 0.8:
            return RecommendationConfidence.HIGH
        elif score >= 0.6:
            return RecommendationConfidence.MEDIUM
        else:
            return RecommendationConfidence.LOW

class HybridFilter:
    """Hybrid filtering recommendation algorithm"""
    
    async def get_recommendations(self, user_profile: UserProfile, user_interactions: List[Interaction], items: Dict[str, Item], item_type: Optional[RecommendationType], limit: int) -> List[Recommendation]:
        """Get hybrid recommendations combining multiple methods"""
        # This would combine collaborative and content-based filtering
        # For now, return empty list as it's handled by the main recommender
        return []

# Example usage
if __name__ == "__main__":
    # Initialize recommender
    recommender = IntelligentRecommender()
    
    # Create user profile
    user_profile = await recommender.create_user_profile(
        "user_001",
        {"preferred_genres": ["fiction", "business"], "writing_style": "formal"},
        {"expertise_level": "intermediate", "industry": "publishing"}
    )
    
    # Record some interactions
    await recommender.record_interaction("user_001", "novel_template_1", "view", rating=4.5)
    await recommender.record_interaction("user_001", "contract_template_1", "use", rating=4.0)
    await recommender.record_interaction("user_001", "article_style_formal", "rate", rating=5.0)
    
    # Get recommendations
    recommendations = await recommender.get_recommendations("user_001", limit=5)
    
    print("Recommendations Generated:")
    print(f"Total Recommendations: {recommendations.total_count}")
    print(f"Algorithm Used: {recommendations.algorithm_used}")
    print(f"Is Cold Start: {recommendations.metadata['is_cold_start']}")
    
    print("\nTop Recommendations:")
    for i, rec in enumerate(recommendations.recommendations[:3], 1):
        item = recommender.get_item(rec.item_id)
        print(f"{i}. {item.name if item else rec.item_id}")
        print(f"   Score: {rec.score:.3f}")
        print(f"   Confidence: {rec.confidence.value}")
        print(f"   Reasoning: {rec.reasoning}")
        print()
    
    # Get statistics
    stats = recommender.get_recommender_statistics()
    print("Recommender Statistics:")
    print(f"Total Users: {stats['total_users']}")
    print(f"Total Items: {stats['total_items']}")
    print(f"Total Interactions: {stats['total_interactions']}")
    print(f"Average Interactions per User: {stats['average_interactions_per_user']:.2f}")
    
    print("\nIntelligent Recommender initialized successfully")

























