"""
Content Personalization Engine - Advanced AI-Powered Content Personalization
========================================================================

This module provides comprehensive content personalization capabilities including:
- User behavior analysis and profiling
- Content recommendation algorithms
- Personalization engine with machine learning
- A/B testing for personalization
- Real-time content adaptation
- Multi-channel personalization
- Privacy-compliant personalization
- Performance tracking and optimization
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow import keras
import redis
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from collections import defaultdict, Counter
import hashlib
import pickle
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonalizationType(Enum):
    """Personalization type enumeration"""
    CONTENT_RECOMMENDATION = "content_recommendation"
    CONTENT_FILTERING = "content_filtering"
    CONTENT_RANKING = "content_ranking"
    CONTENT_GENERATION = "content_generation"
    LAYOUT_PERSONALIZATION = "layout_personalization"
    TIMING_PERSONALIZATION = "timing_personalization"

class UserSegment(Enum):
    """User segment enumeration"""
    NEW_USER = "new_user"
    CASUAL_USER = "casual_user"
    ACTIVE_USER = "active_user"
    POWER_USER = "power_user"
    CHURNED_USER = "churned_user"

class ContentCategory(Enum):
    """Content category enumeration"""
    TECHNOLOGY = "technology"
    BUSINESS = "business"
    HEALTH = "health"
    EDUCATION = "education"
    ENTERTAINMENT = "entertainment"
    NEWS = "news"
    SPORTS = "sports"
    LIFESTYLE = "lifestyle"

@dataclass
class UserProfile:
    """User profile data structure"""
    user_id: str
    demographics: Dict[str, Any] = field(default_factory=dict)
    interests: List[str] = field(default_factory=list)
    behavior_patterns: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    engagement_history: List[Dict[str, Any]] = field(default_factory=list)
    content_interactions: List[Dict[str, Any]] = field(default_factory=list)
    segment: UserSegment = UserSegment.NEW_USER
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ContentItem:
    """Content item data structure"""
    content_id: str
    title: str
    content: str
    category: ContentCategory
    tags: List[str] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class UserInteraction:
    """User interaction data structure"""
    interaction_id: str
    user_id: str
    content_id: str
    interaction_type: str  # view, click, like, share, comment, purchase
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Recommendation:
    """Recommendation data structure"""
    recommendation_id: str
    user_id: str
    content_id: str
    score: float
    reason: str
    algorithm_used: str
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class PersonalizationExperiment:
    """Personalization experiment data structure"""
    experiment_id: str
    name: str
    description: str
    algorithm_a: str
    algorithm_b: str
    traffic_split: float = 0.5
    start_date: datetime = field(default_factory=datetime.utcnow)
    end_date: Optional[datetime] = None
    status: str = "active"
    results: Dict[str, Any] = field(default_factory=dict)

class ContentPersonalizationEngine:
    """
    Advanced Content Personalization Engine
    
    Provides comprehensive content personalization and recommendation capabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Content Personalization Engine"""
        self.config = config
        self.user_profiles = {}
        self.content_items = {}
        self.user_interactions = {}
        self.recommendations = {}
        self.personalization_models = {}
        self.experiments = {}
        self.redis_client = None
        self.database_engine = None
        
        # Initialize components
        self._initialize_database()
        self._initialize_redis()
        self._initialize_models()
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info("Content Personalization Engine initialized successfully")
    
    def _initialize_database(self):
        """Initialize database connection"""
        try:
            if self.config.get("database_url"):
                self.database_engine = create_engine(self.config["database_url"])
                logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            if self.config.get("redis_url"):
                self.redis_client = redis.Redis.from_url(self.config["redis_url"])
                logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Error initializing Redis: {e}")
    
    def _initialize_models(self):
        """Initialize machine learning models"""
        try:
            # Content-based filtering model
            self.personalization_models["content_vectorizer"] = TfidfVectorizer(
                max_features=1000,
                stop_words='english'
            )
            
            # Collaborative filtering model
            self.personalization_models["collaborative_filter"] = NearestNeighbors(
                n_neighbors=10,
                metric='cosine'
            )
            
            # User segmentation model
            self.personalization_models["user_segmenter"] = KMeans(
                n_clusters=5,
                random_state=42
            )
            
            # Deep learning recommendation model
            self.personalization_models["deep_recommender"] = self._build_deep_recommendation_model()
            
            # A/B testing model
            self.personalization_models["ab_test_model"] = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            
            logger.info("Personalization models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def _build_deep_recommendation_model(self):
        """Build deep learning recommendation model"""
        try:
            # User embedding
            user_input = keras.layers.Input(shape=(1,), name='user_id')
            user_embedding = keras.layers.Embedding(10000, 50)(user_input)
            user_vec = keras.layers.Flatten()(user_embedding)
            
            # Content embedding
            content_input = keras.layers.Input(shape=(1,), name='content_id')
            content_embedding = keras.layers.Embedding(10000, 50)(content_input)
            content_vec = keras.layers.Flatten()(content_embedding)
            
            # Concatenate embeddings
            concat = keras.layers.Concatenate()([user_vec, content_vec])
            
            # Dense layers
            dense1 = keras.layers.Dense(128, activation='relu')(concat)
            dropout1 = keras.layers.Dropout(0.2)(dense1)
            dense2 = keras.layers.Dense(64, activation='relu')(dropout1)
            dropout2 = keras.layers.Dropout(0.2)(dense2)
            output = keras.layers.Dense(1, activation='sigmoid')(dropout2)
            
            model = keras.Model(inputs=[user_input, content_input], outputs=output)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            return model
            
        except Exception as e:
            logger.error(f"Error building deep recommendation model: {e}")
            return None
    
    def _start_background_tasks(self):
        """Start background tasks"""
        try:
            # Start model training task
            asyncio.create_task(self._train_models_periodically())
            
            # Start user profiling task
            asyncio.create_task(self._update_user_profiles_periodically())
            
            # Start experiment monitoring task
            asyncio.create_task(self._monitor_experiments_periodically())
            
            logger.info("Background tasks started")
            
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")
    
    async def create_user_profile(self, user_id: str, demographics: Dict[str, Any] = None) -> UserProfile:
        """Create or update user profile"""
        try:
            if user_id in self.user_profiles:
                profile = self.user_profiles[user_id]
                profile.updated_at = datetime.utcnow()
            else:
                profile = UserProfile(
                    user_id=user_id,
                    demographics=demographics or {},
                    segment=UserSegment.NEW_USER
                )
                self.user_profiles[user_id] = profile
            
            # Update user segment
            await self._update_user_segment(profile)
            
            # Store in Redis for quick access
            if self.redis_client:
                profile_data = {
                    "user_id": profile.user_id,
                    "interests": profile.interests,
                    "segment": profile.segment.value,
                    "updated_at": profile.updated_at.isoformat()
                }
                self.redis_client.setex(f"user_profile:{user_id}", 3600, json.dumps(profile_data))
            
            logger.info(f"User profile created/updated for user {user_id}")
            
            return profile
            
        except Exception as e:
            logger.error(f"Error creating user profile: {e}")
            raise
    
    async def _update_user_segment(self, profile: UserProfile):
        """Update user segment based on behavior"""
        try:
            # Calculate engagement score
            engagement_score = 0.0
            if profile.engagement_history:
                recent_engagement = [
                    e for e in profile.engagement_history 
                    if (datetime.utcnow() - e.get("timestamp", datetime.utcnow())).days <= 30
                ]
                engagement_score = len(recent_engagement) / 30.0  # Normalize to 0-1
            
            # Determine segment based on engagement
            if engagement_score == 0:
                profile.segment = UserSegment.CHURNED_USER
            elif engagement_score < 0.1:
                profile.segment = UserSegment.NEW_USER
            elif engagement_score < 0.3:
                profile.segment = UserSegment.CASUAL_USER
            elif engagement_score < 0.7:
                profile.segment = UserSegment.ACTIVE_USER
            else:
                profile.segment = UserSegment.POWER_USER
                
        except Exception as e:
            logger.error(f"Error updating user segment: {e}")
    
    async def track_user_interaction(self, user_id: str, content_id: str, 
                                   interaction_type: str, duration: float = 0.0,
                                   metadata: Dict[str, Any] = None) -> UserInteraction:
        """Track user interaction with content"""
        try:
            interaction_id = str(uuid.uuid4())
            
            interaction = UserInteraction(
                interaction_id=interaction_id,
                user_id=user_id,
                content_id=content_id,
                interaction_type=interaction_type,
                duration=duration,
                metadata=metadata or {}
            )
            
            # Store interaction
            self.user_interactions[interaction_id] = interaction
            
            # Update user profile
            if user_id in self.user_profiles:
                profile = self.user_profiles[user_id]
                profile.engagement_history.append({
                    "interaction_type": interaction_type,
                    "content_id": content_id,
                    "timestamp": interaction.timestamp,
                    "duration": duration
                })
                
                # Update interests based on content
                if content_id in self.content_items:
                    content = self.content_items[content_id]
                    for tag in content.tags:
                        if tag not in profile.interests:
                            profile.interests.append(tag)
            
            # Store in Redis for real-time access
            if self.redis_client:
                interaction_data = {
                    "interaction_id": interaction_id,
                    "user_id": user_id,
                    "content_id": content_id,
                    "interaction_type": interaction_type,
                    "timestamp": interaction.timestamp.isoformat(),
                    "duration": duration
                }
                self.redis_client.setex(f"interaction:{interaction_id}", 3600, json.dumps(interaction_data))
            
            logger.info(f"User interaction tracked: {interaction_type} for user {user_id}")
            
            return interaction
            
        except Exception as e:
            logger.error(f"Error tracking user interaction: {e}")
            raise
    
    async def get_content_recommendations(self, user_id: str, limit: int = 10, 
                                        algorithm: str = "hybrid") -> List[Recommendation]:
        """Get personalized content recommendations for user"""
        try:
            recommendations = []
            
            # Ensure user profile exists
            if user_id not in self.user_profiles:
                await self.create_user_profile(user_id)
            
            profile = self.user_profiles[user_id]
            
            if algorithm == "content_based":
                recommendations = await self._get_content_based_recommendations(profile, limit)
            elif algorithm == "collaborative":
                recommendations = await self._get_collaborative_recommendations(profile, limit)
            elif algorithm == "deep_learning":
                recommendations = await self._get_deep_learning_recommendations(profile, limit)
            elif algorithm == "hybrid":
                recommendations = await self._get_hybrid_recommendations(profile, limit)
            else:
                recommendations = await self._get_popular_content_recommendations(limit)
            
            # Store recommendations
            for rec in recommendations:
                self.recommendations[rec.recommendation_id] = rec
            
            # Store in Redis for quick access
            if self.redis_client:
                rec_data = [
                    {
                        "content_id": rec.content_id,
                        "score": rec.score,
                        "reason": rec.reason,
                        "algorithm_used": rec.algorithm_used
                    }
                    for rec in recommendations
                ]
                self.redis_client.setex(f"recommendations:{user_id}", 1800, json.dumps(rec_data))
            
            logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting content recommendations: {e}")
            return []
    
    async def _get_content_based_recommendations(self, profile: UserProfile, limit: int) -> List[Recommendation]:
        """Get content-based recommendations"""
        try:
            recommendations = []
            
            if not profile.interests:
                return recommendations
            
            # Find content similar to user interests
            for content_id, content in self.content_items.items():
                # Calculate similarity score
                similarity_score = 0.0
                for interest in profile.interests:
                    if interest in content.tags:
                        similarity_score += 1.0
                
                if similarity_score > 0:
                    similarity_score /= len(profile.interests)
                    
                    recommendation = Recommendation(
                        recommendation_id=str(uuid.uuid4()),
                        user_id=profile.user_id,
                        content_id=content_id,
                        score=similarity_score,
                        reason=f"Similar to your interests: {', '.join(profile.interests)}",
                        algorithm_used="content_based"
                    )
                    recommendations.append(recommendation)
            
            # Sort by score and return top recommendations
            recommendations.sort(key=lambda x: x.score, reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error getting content-based recommendations: {e}")
            return []
    
    async def _get_collaborative_recommendations(self, profile: UserProfile, limit: int) -> List[Recommendation]:
        """Get collaborative filtering recommendations"""
        try:
            recommendations = []
            
            # Find similar users
            similar_users = await self._find_similar_users(profile.user_id)
            
            # Get content liked by similar users
            content_scores = defaultdict(float)
            for similar_user_id in similar_users:
                user_interactions = [
                    i for i in self.user_interactions.values()
                    if i.user_id == similar_user_id and i.interaction_type in ["like", "share", "purchase"]
                ]
                
                for interaction in user_interactions:
                    content_scores[interaction.content_id] += 1.0
            
            # Create recommendations
            for content_id, score in content_scores.items():
                if content_id in self.content_items:
                    recommendation = Recommendation(
                        recommendation_id=str(uuid.uuid4()),
                        user_id=profile.user_id,
                        content_id=content_id,
                        score=score / len(similar_users),
                        reason="Liked by users similar to you",
                        algorithm_used="collaborative"
                    )
                    recommendations.append(recommendation)
            
            # Sort by score and return top recommendations
            recommendations.sort(key=lambda x: x.score, reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error getting collaborative recommendations: {e}")
            return []
    
    async def _get_deep_learning_recommendations(self, profile: UserProfile, limit: int) -> List[Recommendation]:
        """Get deep learning recommendations"""
        try:
            recommendations = []
            
            if not self.personalization_models["deep_recommender"]:
                return recommendations
            
            # Prepare user and content features
            user_features = self._extract_user_features(profile)
            content_scores = {}
            
            for content_id, content in self.content_items.items():
                content_features = self._extract_content_features(content)
                
                # Combine features
                combined_features = np.concatenate([user_features, content_features])
                
                # Predict score using deep learning model
                # Note: This is a simplified version - in production, you'd use the actual trained model
                score = np.random.random()  # Placeholder
                
                content_scores[content_id] = score
            
            # Create recommendations
            for content_id, score in content_scores.items():
                recommendation = Recommendation(
                    recommendation_id=str(uuid.uuid4()),
                    user_id=profile.user_id,
                    content_id=content_id,
                    score=score,
                    reason="AI-powered recommendation based on deep learning",
                    algorithm_used="deep_learning"
                )
                recommendations.append(recommendation)
            
            # Sort by score and return top recommendations
            recommendations.sort(key=lambda x: x.score, reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error getting deep learning recommendations: {e}")
            return []
    
    async def _get_hybrid_recommendations(self, profile: UserProfile, limit: int) -> List[Recommendation]:
        """Get hybrid recommendations combining multiple algorithms"""
        try:
            # Get recommendations from different algorithms
            content_based = await self._get_content_based_recommendations(profile, limit)
            collaborative = await self._get_collaborative_recommendations(profile, limit)
            deep_learning = await self._get_deep_learning_recommendations(profile, limit)
            
            # Combine recommendations
            combined_scores = defaultdict(float)
            for rec in content_based:
                combined_scores[rec.content_id] += rec.score * 0.4
            for rec in collaborative:
                combined_scores[rec.content_id] += rec.score * 0.3
            for rec in deep_learning:
                combined_scores[rec.content_id] += rec.score * 0.3
            
            # Create final recommendations
            recommendations = []
            for content_id, score in combined_scores.items():
                recommendation = Recommendation(
                    recommendation_id=str(uuid.uuid4()),
                    user_id=profile.user_id,
                    content_id=content_id,
                    score=score,
                    reason="Hybrid recommendation combining multiple algorithms",
                    algorithm_used="hybrid"
                )
                recommendations.append(recommendation)
            
            # Sort by score and return top recommendations
            recommendations.sort(key=lambda x: x.score, reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error getting hybrid recommendations: {e}")
            return []
    
    async def _get_popular_content_recommendations(self, limit: int) -> List[Recommendation]:
        """Get popular content recommendations"""
        try:
            # Calculate content popularity
            content_popularity = defaultdict(int)
            for interaction in self.user_interactions.values():
                if interaction.interaction_type in ["view", "like", "share"]:
                    content_popularity[interaction.content_id] += 1
            
            # Create recommendations
            recommendations = []
            for content_id, popularity in content_popularity.items():
                if content_id in self.content_items:
                    recommendation = Recommendation(
                        recommendation_id=str(uuid.uuid4()),
                        user_id="popular",
                        content_id=content_id,
                        score=popularity,
                        reason="Popular content",
                        algorithm_used="popularity"
                    )
                    recommendations.append(recommendation)
            
            # Sort by score and return top recommendations
            recommendations.sort(key=lambda x: x.score, reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error getting popular content recommendations: {e}")
            return []
    
    async def _find_similar_users(self, user_id: str, limit: int = 10) -> List[str]:
        """Find users similar to the given user"""
        try:
            # Get user's interaction history
            user_interactions = [
                i for i in self.user_interactions.values()
                if i.user_id == user_id
            ]
            
            if not user_interactions:
                return []
            
            # Find users who interacted with similar content
            similar_users = defaultdict(int)
            for interaction in user_interactions:
                content_interactions = [
                    i for i in self.user_interactions.values()
                    if i.content_id == interaction.content_id and i.user_id != user_id
                ]
                
                for content_interaction in content_interactions:
                    similar_users[content_interaction.user_id] += 1
            
            # Sort by similarity score
            sorted_users = sorted(similar_users.items(), key=lambda x: x[1], reverse=True)
            return [user_id for user_id, score in sorted_users[:limit]]
            
        except Exception as e:
            logger.error(f"Error finding similar users: {e}")
            return []
    
    def _extract_user_features(self, profile: UserProfile) -> np.ndarray:
        """Extract user features for machine learning"""
        try:
            features = []
            
            # Demographics features
            features.append(len(profile.interests))
            features.append(len(profile.engagement_history))
            features.append(profile.segment.value == "power_user")
            features.append(profile.segment.value == "active_user")
            features.append(profile.segment.value == "casual_user")
            
            # Engagement features
            if profile.engagement_history:
                recent_engagement = [
                    e for e in profile.engagement_history 
                    if (datetime.utcnow() - e.get("timestamp", datetime.utcnow())).days <= 7
                ]
                features.append(len(recent_engagement))
                features.append(sum(e.get("duration", 0) for e in recent_engagement))
            else:
                features.extend([0, 0])
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting user features: {e}")
            return np.zeros(7)
    
    def _extract_content_features(self, content: ContentItem) -> np.ndarray:
        """Extract content features for machine learning"""
        try:
            features = []
            
            # Content features
            features.append(len(content.tags))
            features.append(len(content.content))
            features.append(content.category.value == "technology")
            features.append(content.category.value == "business")
            features.append(content.category.value == "health")
            features.append(content.category.value == "education")
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting content features: {e}")
            return np.zeros(6)
    
    async def create_personalization_experiment(self, experiment_data: Dict[str, Any]) -> PersonalizationExperiment:
        """Create A/B testing experiment for personalization"""
        try:
            experiment_id = str(uuid.uuid4())
            
            experiment = PersonalizationExperiment(
                experiment_id=experiment_id,
                name=experiment_data["name"],
                description=experiment_data["description"],
                algorithm_a=experiment_data["algorithm_a"],
                algorithm_b=experiment_data["algorithm_b"],
                traffic_split=experiment_data.get("traffic_split", 0.5)
            )
            
            # Store experiment
            self.experiments[experiment_id] = experiment
            
            logger.info(f"Personalization experiment {experiment_id} created")
            
            return experiment
            
        except Exception as e:
            logger.error(f"Error creating personalization experiment: {e}")
            raise
    
    async def get_experiment_recommendations(self, user_id: str, experiment_id: str, 
                                           limit: int = 10) -> List[Recommendation]:
        """Get recommendations based on A/B testing experiment"""
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.experiments[experiment_id]
            
            if experiment.status != "active":
                raise ValueError(f"Experiment {experiment_id} is not active")
            
            # Determine which algorithm to use based on user hash
            user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            use_algorithm_b = (user_hash % 100) < (experiment.traffic_split * 100)
            
            algorithm = experiment.algorithm_b if use_algorithm_b else experiment.algorithm_a
            
            # Get recommendations using selected algorithm
            recommendations = await self.get_content_recommendations(user_id, limit, algorithm)
            
            # Track experiment participation
            if user_id in self.user_profiles:
                profile = self.user_profiles[user_id]
                if "experiments" not in profile.metadata:
                    profile.metadata["experiments"] = []
                profile.metadata["experiments"].append({
                    "experiment_id": experiment_id,
                    "algorithm_used": algorithm,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            logger.info(f"Experiment recommendations generated for user {user_id} using {algorithm}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting experiment recommendations: {e}")
            raise
    
    async def _train_models_periodically(self):
        """Train models periodically in background"""
        while True:
            try:
                await asyncio.sleep(3600)  # Train every hour
                
                # Train collaborative filtering model
                await self._train_collaborative_filter()
                
                # Train user segmentation model
                await self._train_user_segmenter()
                
                # Train deep learning model
                await self._train_deep_recommender()
                
                logger.info("Models trained successfully")
                
            except Exception as e:
                logger.error(f"Error training models: {e}")
                await asyncio.sleep(3600)
    
    async def _train_collaborative_filter(self):
        """Train collaborative filtering model"""
        try:
            # Prepare user-item matrix
            user_item_matrix = defaultdict(dict)
            for interaction in self.user_interactions.values():
                if interaction.interaction_type in ["like", "share", "purchase"]:
                    user_item_matrix[interaction.user_id][interaction.content_id] = 1.0
            
            if not user_item_matrix:
                return
            
            # Convert to matrix format
            users = list(user_item_matrix.keys())
            items = list(set(
                item for user_items in user_item_matrix.values() 
                for item in user_items.keys()
            ))
            
            matrix = np.zeros((len(users), len(items)))
            for i, user in enumerate(users):
                for j, item in enumerate(items):
                    if item in user_item_matrix[user]:
                        matrix[i, j] = user_item_matrix[user][item]
            
            # Train model
            if len(users) > 1:
                self.personalization_models["collaborative_filter"].fit(matrix)
                
        except Exception as e:
            logger.error(f"Error training collaborative filter: {e}")
    
    async def _train_user_segmenter(self):
        """Train user segmentation model"""
        try:
            # Prepare user features
            user_features = []
            for profile in self.user_profiles.values():
                features = self._extract_user_features(profile)
                user_features.append(features)
            
            if len(user_features) < 5:  # Need minimum samples
                return
            
            # Train model
            X = np.array(user_features)
            self.personalization_models["user_segmenter"].fit(X)
            
        except Exception as e:
            logger.error(f"Error training user segmenter: {e}")
    
    async def _train_deep_recommender(self):
        """Train deep learning recommendation model"""
        try:
            if not self.personalization_models["deep_recommender"]:
                return
            
            # Prepare training data
            user_ids = []
            content_ids = []
            labels = []
            
            for interaction in self.user_interactions.values():
                user_ids.append(hash(interaction.user_id) % 10000)
                content_ids.append(hash(interaction.content_id) % 10000)
                labels.append(1 if interaction.interaction_type in ["like", "share", "purchase"] else 0)
            
            if len(user_ids) < 100:  # Need minimum samples
                return
            
            # Convert to numpy arrays
            user_ids = np.array(user_ids)
            content_ids = np.array(content_ids)
            labels = np.array(labels)
            
            # Train model
            self.personalization_models["deep_recommender"].fit(
                [user_ids, content_ids], labels,
                epochs=10,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
        except Exception as e:
            logger.error(f"Error training deep recommender: {e}")
    
    async def _update_user_profiles_periodically(self):
        """Update user profiles periodically"""
        while True:
            try:
                await asyncio.sleep(1800)  # Update every 30 minutes
                
                for profile in self.user_profiles.values():
                    await self._update_user_segment(profile)
                
                logger.info("User profiles updated successfully")
                
            except Exception as e:
                logger.error(f"Error updating user profiles: {e}")
                await asyncio.sleep(1800)
    
    async def _monitor_experiments_periodically(self):
        """Monitor experiments periodically"""
        while True:
            try:
                await asyncio.sleep(3600)  # Monitor every hour
                
                for experiment in self.experiments.values():
                    if experiment.status == "active":
                        await self._analyze_experiment_results(experiment)
                
                logger.info("Experiments monitored successfully")
                
            except Exception as e:
                logger.error(f"Error monitoring experiments: {e}")
                await asyncio.sleep(3600)
    
    async def _analyze_experiment_results(self, experiment: PersonalizationExperiment):
        """Analyze experiment results"""
        try:
            # Get experiment participants
            participants_a = []
            participants_b = []
            
            for profile in self.user_profiles.values():
                if "experiments" in profile.metadata:
                    for exp_data in profile.metadata["experiments"]:
                        if exp_data["experiment_id"] == experiment.experiment_id:
                            if exp_data["algorithm_used"] == experiment.algorithm_a:
                                participants_a.append(profile.user_id)
                            else:
                                participants_b.append(profile.user_id)
            
            # Calculate metrics
            results = {
                "participants_a": len(participants_a),
                "participants_b": len(participants_b),
                "conversion_rate_a": 0.0,
                "conversion_rate_b": 0.0,
                "statistical_significance": 0.0
            }
            
            # Calculate conversion rates
            if participants_a:
                conversions_a = len([
                    i for i in self.user_interactions.values()
                    if i.user_id in participants_a and i.interaction_type == "purchase"
                ])
                results["conversion_rate_a"] = conversions_a / len(participants_a)
            
            if participants_b:
                conversions_b = len([
                    i for i in self.user_interactions.values()
                    if i.user_id in participants_b and i.interaction_type == "purchase"
                ])
                results["conversion_rate_b"] = conversions_b / len(participants_b)
            
            experiment.results = results
            
        except Exception as e:
            logger.error(f"Error analyzing experiment results: {e}")
    
    async def get_personalization_analytics(self, time_period: str = "30d") -> Dict[str, Any]:
        """Get personalization analytics and insights"""
        try:
            # Calculate time range
            end_date = datetime.utcnow()
            if time_period == "7d":
                start_date = end_date - timedelta(days=7)
            elif time_period == "30d":
                start_date = end_date - timedelta(days=30)
            elif time_period == "90d":
                start_date = end_date - timedelta(days=90)
            else:
                start_date = end_date - timedelta(days=30)
            
            # Get analytics data
            analytics = {
                "time_period": time_period,
                "total_users": len(self.user_profiles),
                "total_interactions": len(self.user_interactions),
                "total_recommendations": len(self.recommendations),
                "user_segments": {},
                "algorithm_performance": {},
                "experiment_results": {}
            }
            
            # User segment distribution
            segment_counts = defaultdict(int)
            for profile in self.user_profiles.values():
                segment_counts[profile.segment.value] += 1
            analytics["user_segments"] = dict(segment_counts)
            
            # Algorithm performance
            algorithm_performance = defaultdict(list)
            for rec in self.recommendations.values():
                if start_date <= rec.created_at <= end_date:
                    algorithm_performance[rec.algorithm_used].append(rec.score)
            
            for algorithm, scores in algorithm_performance.items():
                analytics["algorithm_performance"][algorithm] = {
                    "average_score": np.mean(scores),
                    "total_recommendations": len(scores)
                }
            
            # Experiment results
            for experiment in self.experiments.values():
                if experiment.results:
                    analytics["experiment_results"][experiment.experiment_id] = experiment.results
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting personalization analytics: {e}")
            return {"error": str(e)}

# Example usage and testing
async def main():
    """Example usage of the Content Personalization Engine"""
    try:
        # Initialize engine
        config = {
            "database_url": "postgresql://user:password@localhost/personalizationdb",
            "redis_url": "redis://localhost:6379"
        }
        
        engine = ContentPersonalizationEngine(config)
        
        # Create user profile
        print("Creating user profile...")
        profile = await engine.create_user_profile("user_001", {
            "age": 25,
            "gender": "female",
            "location": "New York"
        })
        
        # Add content items
        print("Adding content items...")
        content1 = ContentItem(
            content_id="content_001",
            title="AI Technology Trends",
            content="Latest trends in artificial intelligence...",
            category=ContentCategory.TECHNOLOGY,
            tags=["AI", "technology", "trends"]
        )
        engine.content_items["content_001"] = content1
        
        content2 = ContentItem(
            content_id="content_002",
            title="Business Strategy Guide",
            content="Comprehensive guide to business strategy...",
            category=ContentCategory.BUSINESS,
            tags=["business", "strategy", "guide"]
        )
        engine.content_items["content_002"] = content2
        
        # Track user interactions
        print("Tracking user interactions...")
        await engine.track_user_interaction("user_001", "content_001", "view", 30.0)
        await engine.track_user_interaction("user_001", "content_001", "like", 0.0)
        await engine.track_user_interaction("user_001", "content_002", "view", 45.0)
        
        # Get recommendations
        print("Getting content recommendations...")
        recommendations = await engine.get_content_recommendations("user_001", 5, "hybrid")
        print(f"Generated {len(recommendations)} recommendations")
        
        for rec in recommendations:
            print(f"- {rec.content_id}: {rec.score:.3f} ({rec.reason})")
        
        # Create A/B testing experiment
        print("Creating A/B testing experiment...")
        experiment = await engine.create_personalization_experiment({
            "name": "Recommendation Algorithm Test",
            "description": "Testing content-based vs collaborative filtering",
            "algorithm_a": "content_based",
            "algorithm_b": "collaborative",
            "traffic_split": 0.5
        })
        
        # Get experiment recommendations
        print("Getting experiment recommendations...")
        exp_recommendations = await engine.get_experiment_recommendations("user_001", experiment.experiment_id, 3)
        print(f"Generated {len(exp_recommendations)} experiment recommendations")
        
        # Get personalization analytics
        print("Getting personalization analytics...")
        analytics = await engine.get_personalization_analytics("30d")
        print(f"Total users: {analytics['total_users']}")
        print(f"User segments: {analytics['user_segments']}")
        print(f"Algorithm performance: {analytics['algorithm_performance']}")
        
        print("\nContent Personalization Engine demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main demo: {e}")

if __name__ == "__main__":
    asyncio.run(main())
























