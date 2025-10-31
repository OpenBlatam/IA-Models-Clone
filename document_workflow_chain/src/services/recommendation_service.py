"""
Recommendation Service - Advanced Implementation
==============================================

Advanced recommendation service with collaborative filtering, content-based filtering, and hybrid approaches.
"""

from __future__ import annotations
import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')

from .analytics_service import analytics_service
from .ai_service import ai_service

logger = logging.getLogger(__name__)


class RecommendationType(str, Enum):
    """Recommendation type enumeration"""
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    CONTENT_BASED = "content_based"
    HYBRID = "hybrid"
    POPULARITY_BASED = "popularity_based"
    DEMOGRAPHIC = "demographic"
    KNOWLEDGE_BASED = "knowledge_based"


class RecommendationAlgorithm(str, Enum):
    """Recommendation algorithm enumeration"""
    USER_BASED_CF = "user_based_cf"
    ITEM_BASED_CF = "item_based_cf"
    MATRIX_FACTORIZATION = "matrix_factorization"
    CONTENT_SIMILARITY = "content_similarity"
    TFIDF_SIMILARITY = "tfidf_similarity"
    POPULARITY = "popularity"
    DEMOGRAPHIC_FILTERING = "demographic_filtering"


class RecommendationService:
    """Advanced recommendation service with multiple algorithms"""
    
    def __init__(self):
        self.recommendation_models = {}
        self.user_interactions = defaultdict(list)
        self.item_features = {}
        self.user_profiles = {}
        self.popularity_scores = {}
        self.recommendation_cache = {}
        
        self.recommendation_stats = {
            "total_recommendations": 0,
            "recommendations_by_type": {rec_type.value: 0 for rec_type in RecommendationType},
            "recommendations_by_algorithm": {algo.value: 0 for algo in RecommendationAlgorithm},
            "total_users": 0,
            "total_items": 0,
            "interaction_count": 0
        }
        
        # Model storage
        self.cf_models = {}
        self.content_models = {}
        self.hybrid_models = {}
    
    async def add_user_interaction(
        self,
        user_id: str,
        item_id: str,
        interaction_type: str,
        rating: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ):
        """Add user interaction for recommendation learning"""
        try:
            interaction = {
                "user_id": user_id,
                "item_id": item_id,
                "interaction_type": interaction_type,
                "rating": rating,
                "timestamp": timestamp or datetime.utcnow()
            }
            
            self.user_interactions[user_id].append(interaction)
            self.recommendation_stats["interaction_count"] += 1
            
            # Update user profile
            await self._update_user_profile(user_id, interaction)
            
            # Update popularity scores
            await self._update_popularity_scores(item_id, interaction_type)
            
            # Track analytics
            await analytics_service.track_event(
                "user_interaction_added",
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "interaction_type": interaction_type,
                    "rating": rating
                }
            )
            
            logger.info(f"User interaction added: {user_id} -> {item_id} ({interaction_type})")
        
        except Exception as e:
            logger.error(f"Failed to add user interaction: {e}")
    
    async def add_item_features(
        self,
        item_id: str,
        features: Dict[str, Any],
        content: Optional[str] = None
    ):
        """Add item features for content-based recommendations"""
        try:
            item_data = {
                "item_id": item_id,
                "features": features,
                "content": content,
                "created_at": datetime.utcnow()
            }
            
            self.item_features[item_id] = item_data
            self.recommendation_stats["total_items"] = len(self.item_features)
            
            logger.info(f"Item features added: {item_id}")
        
        except Exception as e:
            logger.error(f"Failed to add item features: {e}")
    
    async def create_recommendation_model(
        self,
        name: str,
        recommendation_type: RecommendationType,
        algorithm: RecommendationAlgorithm,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new recommendation model"""
        try:
            model_id = f"rec_model_{len(self.recommendation_models) + 1}"
            
            model_config = {
                "id": model_id,
                "name": name,
                "type": recommendation_type.value,
                "algorithm": algorithm.value,
                "parameters": parameters or {},
                "created_at": datetime.utcnow().isoformat(),
                "trained_at": None,
                "status": "created",
                "performance_metrics": {}
            }
            
            self.recommendation_models[model_id] = model_config
            
            logger.info(f"Recommendation model created: {model_id} - {name}")
            return model_id
        
        except Exception as e:
            logger.error(f"Failed to create recommendation model: {e}")
            raise
    
    async def train_recommendation_model(self, model_id: str) -> Dict[str, Any]:
        """Train recommendation model"""
        try:
            if model_id not in self.recommendation_models:
                raise ValueError(f"Model not found: {model_id}")
            
            model = self.recommendation_models[model_id]
            model["status"] = "training"
            
            # Train based on type and algorithm
            if model["type"] == RecommendationType.COLLABORATIVE_FILTERING.value:
                result = await self._train_collaborative_filtering(model)
            elif model["type"] == RecommendationType.CONTENT_BASED.value:
                result = await self._train_content_based(model)
            elif model["type"] == RecommendationType.HYBRID.value:
                result = await self._train_hybrid(model)
            elif model["type"] == RecommendationType.POPULARITY_BASED.value:
                result = await self._train_popularity_based(model)
            else:
                raise ValueError(f"Unsupported recommendation type: {model['type']}")
            
            model["status"] = "trained"
            model["trained_at"] = datetime.utcnow().isoformat()
            model["performance_metrics"] = result.get("metrics", {})
            
            logger.info(f"Recommendation model trained: {model_id} - {model['name']}")
            return result
        
        except Exception as e:
            logger.error(f"Failed to train recommendation model: {e}")
            if model_id in self.recommendation_models:
                self.recommendation_models[model_id]["status"] = "failed"
            raise
    
    async def _train_collaborative_filtering(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """Train collaborative filtering model"""
        try:
            algorithm = model["algorithm"]
            
            if algorithm == RecommendationAlgorithm.USER_BASED_CF.value:
                return await self._train_user_based_cf(model)
            elif algorithm == RecommendationAlgorithm.ITEM_BASED_CF.value:
                return await self._train_item_based_cf(model)
            elif algorithm == RecommendationAlgorithm.MATRIX_FACTORIZATION.value:
                return await self._train_matrix_factorization(model)
            else:
                raise ValueError(f"Unsupported CF algorithm: {algorithm}")
        
        except Exception as e:
            logger.error(f"Failed to train collaborative filtering: {e}")
            raise
    
    async def _train_user_based_cf(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """Train user-based collaborative filtering"""
        try:
            # Create user-item matrix
            user_item_matrix = self._create_user_item_matrix()
            
            # Calculate user similarities
            user_similarities = cosine_similarity(user_item_matrix)
            
            # Store model
            self.cf_models[model["id"]] = {
                "type": "user_based_cf",
                "user_item_matrix": user_item_matrix,
                "user_similarities": user_similarities,
                "user_ids": list(self.user_interactions.keys())
            }
            
            return {
                "status": "trained",
                "algorithm": "user_based_cf",
                "users_count": len(self.user_interactions),
                "items_count": len(self.item_features),
                "metrics": {
                    "sparsity": self._calculate_sparsity(user_item_matrix),
                    "avg_interactions_per_user": len(self.user_interactions) / max(1, len(self.user_interactions))
                }
            }
        
        except Exception as e:
            logger.error(f"Failed to train user-based CF: {e}")
            raise
    
    async def _train_item_based_cf(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """Train item-based collaborative filtering"""
        try:
            # Create user-item matrix
            user_item_matrix = self._create_user_item_matrix()
            
            # Calculate item similarities
            item_similarities = cosine_similarity(user_item_matrix.T)
            
            # Store model
            self.cf_models[model["id"]] = {
                "type": "item_based_cf",
                "user_item_matrix": user_item_matrix,
                "item_similarities": item_similarities,
                "item_ids": list(self.item_features.keys())
            }
            
            return {
                "status": "trained",
                "algorithm": "item_based_cf",
                "users_count": len(self.user_interactions),
                "items_count": len(self.item_features),
                "metrics": {
                    "sparsity": self._calculate_sparsity(user_item_matrix),
                    "avg_interactions_per_item": len(self.user_interactions) / max(1, len(self.item_features))
                }
            }
        
        except Exception as e:
            logger.error(f"Failed to train item-based CF: {e}")
            raise
    
    async def _train_matrix_factorization(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """Train matrix factorization model"""
        try:
            # Create user-item matrix
            user_item_matrix = self._create_user_item_matrix()
            
            # Apply SVD
            n_components = model["parameters"].get("n_components", 50)
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            user_factors = svd.fit_transform(user_item_matrix)
            item_factors = svd.components_.T
            
            # Store model
            self.cf_models[model["id"]] = {
                "type": "matrix_factorization",
                "user_factors": user_factors,
                "item_factors": item_factors,
                "svd": svd,
                "user_ids": list(self.user_interactions.keys()),
                "item_ids": list(self.item_features.keys())
            }
            
            return {
                "status": "trained",
                "algorithm": "matrix_factorization",
                "users_count": len(self.user_interactions),
                "items_count": len(self.item_features),
                "n_components": n_components,
                "explained_variance_ratio": svd.explained_variance_ratio_.sum(),
                "metrics": {
                    "sparsity": self._calculate_sparsity(user_item_matrix)
                }
            }
        
        except Exception as e:
            logger.error(f"Failed to train matrix factorization: {e}")
            raise
    
    async def _train_content_based(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """Train content-based recommendation model"""
        try:
            algorithm = model["algorithm"]
            
            if algorithm == RecommendationAlgorithm.TFIDF_SIMILARITY.value:
                return await self._train_tfidf_similarity(model)
            elif algorithm == RecommendationAlgorithm.CONTENT_SIMILARITY.value:
                return await self._train_content_similarity(model)
            else:
                raise ValueError(f"Unsupported content-based algorithm: {algorithm}")
        
        except Exception as e:
            logger.error(f"Failed to train content-based model: {e}")
            raise
    
    async def _train_tfidf_similarity(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """Train TF-IDF based content similarity"""
        try:
            # Prepare content data
            item_contents = []
            item_ids = []
            
            for item_id, item_data in self.item_features.items():
                if item_data.get("content"):
                    item_contents.append(item_data["content"])
                    item_ids.append(item_id)
            
            if not item_contents:
                raise ValueError("No content data available for TF-IDF training")
            
            # Create TF-IDF matrix
            vectorizer = TfidfVectorizer(
                max_features=model["parameters"].get("max_features", 1000),
                stop_words='english'
            )
            tfidf_matrix = vectorizer.fit_transform(item_contents)
            
            # Calculate content similarities
            content_similarities = cosine_similarity(tfidf_matrix)
            
            # Store model
            self.content_models[model["id"]] = {
                "type": "tfidf_similarity",
                "vectorizer": vectorizer,
                "tfidf_matrix": tfidf_matrix,
                "content_similarities": content_similarities,
                "item_ids": item_ids
            }
            
            return {
                "status": "trained",
                "algorithm": "tfidf_similarity",
                "items_count": len(item_ids),
                "vocabulary_size": len(vectorizer.vocabulary_),
                "metrics": {
                    "avg_content_length": np.mean([len(content) for content in item_contents])
                }
            }
        
        except Exception as e:
            logger.error(f"Failed to train TF-IDF similarity: {e}")
            raise
    
    async def _train_content_similarity(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """Train feature-based content similarity"""
        try:
            # Prepare feature data
            feature_vectors = []
            item_ids = []
            
            for item_id, item_data in self.item_features.items():
                features = item_data.get("features", {})
                if features:
                    # Convert features to vector (simplified)
                    feature_vector = list(features.values())
                    feature_vectors.append(feature_vector)
                    item_ids.append(item_id)
            
            if not feature_vectors:
                raise ValueError("No feature data available for content similarity training")
            
            # Calculate content similarities
            feature_matrix = np.array(feature_vectors)
            content_similarities = cosine_similarity(feature_matrix)
            
            # Store model
            self.content_models[model["id"]] = {
                "type": "content_similarity",
                "feature_matrix": feature_matrix,
                "content_similarities": content_similarities,
                "item_ids": item_ids
            }
            
            return {
                "status": "trained",
                "algorithm": "content_similarity",
                "items_count": len(item_ids),
                "feature_dimensions": len(feature_vectors[0]) if feature_vectors else 0,
                "metrics": {
                    "avg_similarity": np.mean(content_similarities)
                }
            }
        
        except Exception as e:
            logger.error(f"Failed to train content similarity: {e}")
            raise
    
    async def _train_hybrid(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """Train hybrid recommendation model"""
        try:
            # Combine collaborative filtering and content-based approaches
            cf_weight = model["parameters"].get("cf_weight", 0.6)
            content_weight = model["parameters"].get("content_weight", 0.4)
            
            # Train both components
            cf_result = await self._train_collaborative_filtering(model)
            content_result = await self._train_content_based(model)
            
            # Store hybrid model
            self.hybrid_models[model["id"]] = {
                "type": "hybrid",
                "cf_weight": cf_weight,
                "content_weight": content_weight,
                "cf_model_id": model["id"] + "_cf",
                "content_model_id": model["id"] + "_content"
            }
            
            return {
                "status": "trained",
                "algorithm": "hybrid",
                "cf_weight": cf_weight,
                "content_weight": content_weight,
                "cf_metrics": cf_result.get("metrics", {}),
                "content_metrics": content_result.get("metrics", {}),
                "metrics": {
                    "combined_score": cf_weight * cf_result.get("metrics", {}).get("sparsity", 0) + 
                                    content_weight * content_result.get("metrics", {}).get("avg_similarity", 0)
                }
            }
        
        except Exception as e:
            logger.error(f"Failed to train hybrid model: {e}")
            raise
    
    async def _train_popularity_based(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """Train popularity-based recommendation model"""
        try:
            # Calculate popularity scores
            popularity_scores = {}
            
            for item_id, interactions in self._get_item_interactions().items():
                score = 0
                for interaction in interactions:
                    if interaction["interaction_type"] == "view":
                        score += 1
                    elif interaction["interaction_type"] == "like":
                        score += 2
                    elif interaction["interaction_type"] == "purchase":
                        score += 5
                    elif interaction["interaction_type"] == "rating" and interaction.get("rating"):
                        score += interaction["rating"]
                
                popularity_scores[item_id] = score
            
            # Store model
            self.popularity_scores = popularity_scores
            
            return {
                "status": "trained",
                "algorithm": "popularity",
                "items_count": len(popularity_scores),
                "metrics": {
                    "avg_popularity": np.mean(list(popularity_scores.values())) if popularity_scores else 0,
                    "max_popularity": max(popularity_scores.values()) if popularity_scores else 0
                }
            }
        
        except Exception as e:
            logger.error(f"Failed to train popularity-based model: {e}")
            raise
    
    async def get_recommendations(
        self,
        user_id: str,
        model_id: str,
        num_recommendations: int = 10,
        exclude_interacted: bool = True
    ) -> Dict[str, Any]:
        """Get recommendations for a user"""
        try:
            if model_id not in self.recommendation_models:
                raise ValueError(f"Model not found: {model_id}")
            
            model = self.recommendation_models[model_id]
            
            if model["status"] != "trained":
                raise ValueError(f"Model is not trained: {model_id}")
            
            # Get recommendations based on model type
            if model["type"] == RecommendationType.COLLABORATIVE_FILTERING.value:
                recommendations = await self._get_cf_recommendations(
                    user_id, model_id, num_recommendations, exclude_interacted
                )
            elif model["type"] == RecommendationType.CONTENT_BASED.value:
                recommendations = await self._get_content_based_recommendations(
                    user_id, model_id, num_recommendations, exclude_interacted
                )
            elif model["type"] == RecommendationType.HYBRID.value:
                recommendations = await self._get_hybrid_recommendations(
                    user_id, model_id, num_recommendations, exclude_interacted
                )
            elif model["type"] == RecommendationType.POPULARITY_BASED.value:
                recommendations = await self._get_popularity_recommendations(
                    user_id, num_recommendations, exclude_interacted
                )
            else:
                raise ValueError(f"Unsupported recommendation type: {model['type']}")
            
            # Update statistics
            self.recommendation_stats["total_recommendations"] += len(recommendations)
            self.recommendation_stats["recommendations_by_type"][model["type"]] += len(recommendations)
            self.recommendation_stats["recommendations_by_algorithm"][model["algorithm"]] += len(recommendations)
            
            # Cache recommendations
            cache_key = f"{user_id}_{model_id}_{num_recommendations}"
            self.recommendation_cache[cache_key] = {
                "recommendations": recommendations,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Track analytics
            await analytics_service.track_event(
                "recommendations_generated",
                {
                    "user_id": user_id,
                    "model_id": model_id,
                    "model_type": model["type"],
                    "algorithm": model["algorithm"],
                    "num_recommendations": len(recommendations)
                }
            )
            
            return {
                "user_id": user_id,
                "model_id": model_id,
                "recommendations": recommendations,
                "num_recommendations": len(recommendations),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            raise
    
    async def _get_cf_recommendations(
        self,
        user_id: str,
        model_id: str,
        num_recommendations: int,
        exclude_interacted: bool
    ) -> List[Dict[str, Any]]:
        """Get collaborative filtering recommendations"""
        try:
            if model_id not in self.cf_models:
                return []
            
            cf_model = self.cf_models[model_id]
            algorithm = cf_model["type"]
            
            if algorithm == "user_based_cf":
                return await self._get_user_based_recommendations(
                    user_id, cf_model, num_recommendations, exclude_interacted
                )
            elif algorithm == "item_based_cf":
                return await self._get_item_based_recommendations(
                    user_id, cf_model, num_recommendations, exclude_interacted
                )
            elif algorithm == "matrix_factorization":
                return await self._get_matrix_factorization_recommendations(
                    user_id, cf_model, num_recommendations, exclude_interacted
                )
            else:
                return []
        
        except Exception as e:
            logger.error(f"Failed to get CF recommendations: {e}")
            return []
    
    async def _get_user_based_recommendations(
        self,
        user_id: str,
        cf_model: Dict[str, Any],
        num_recommendations: int,
        exclude_interacted: bool
    ) -> List[Dict[str, Any]]:
        """Get user-based collaborative filtering recommendations"""
        try:
            user_item_matrix = cf_model["user_item_matrix"]
            user_similarities = cf_model["user_similarities"]
            user_ids = cf_model["user_ids"]
            
            if user_id not in user_ids:
                return []
            
            user_idx = user_ids.index(user_id)
            user_ratings = user_item_matrix[user_idx]
            
            # Find similar users
            user_similarity_scores = user_similarities[user_idx]
            similar_users = np.argsort(user_similarity_scores)[::-1][1:11]  # Top 10 similar users
            
            # Calculate recommendation scores
            recommendation_scores = {}
            for similar_user_idx in similar_users:
                similarity = user_similarity_scores[similar_user_idx]
                similar_user_ratings = user_item_matrix[similar_user_idx]
                
                for item_idx, rating in enumerate(similar_user_ratings):
                    if rating > 0 and user_ratings[item_idx] == 0:  # Item not rated by user
                        item_id = list(self.item_features.keys())[item_idx]
                        if item_id not in recommendation_scores:
                            recommendation_scores[item_id] = 0
                        recommendation_scores[item_id] += similarity * rating
            
            # Sort and return top recommendations
            sorted_items = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for item_id, score in sorted_items[:num_recommendations]:
                if exclude_interacted and self._has_user_interacted(user_id, item_id):
                    continue
                
                recommendations.append({
                    "item_id": item_id,
                    "score": score,
                    "reason": "user_based_cf"
                })
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Failed to get user-based recommendations: {e}")
            return []
    
    async def _get_item_based_recommendations(
        self,
        user_id: str,
        cf_model: Dict[str, Any],
        num_recommendations: int,
        exclude_interacted: bool
    ) -> List[Dict[str, Any]]:
        """Get item-based collaborative filtering recommendations"""
        try:
            user_item_matrix = cf_model["user_item_matrix"]
            item_similarities = cf_model["item_similarities"]
            item_ids = cf_model["item_ids"]
            
            if user_id not in self.user_interactions:
                return []
            
            user_ratings = {}
            for interaction in self.user_interactions[user_id]:
                if interaction.get("rating"):
                    user_ratings[interaction["item_id"]] = interaction["rating"]
            
            # Calculate recommendation scores
            recommendation_scores = {}
            for rated_item_id, rating in user_ratings.items():
                if rated_item_id in item_ids:
                    rated_item_idx = item_ids.index(rated_item_id)
                    item_similarity_scores = item_similarities[rated_item_idx]
                    
                    for similar_item_idx, similarity in enumerate(item_similarity_scores):
                        similar_item_id = item_ids[similar_item_idx]
                        if similar_item_id not in user_ratings:  # Item not rated by user
                            if similar_item_id not in recommendation_scores:
                                recommendation_scores[similar_item_id] = 0
                            recommendation_scores[similar_item_id] += similarity * rating
            
            # Sort and return top recommendations
            sorted_items = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for item_id, score in sorted_items[:num_recommendations]:
                if exclude_interacted and self._has_user_interacted(user_id, item_id):
                    continue
                
                recommendations.append({
                    "item_id": item_id,
                    "score": score,
                    "reason": "item_based_cf"
                })
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Failed to get item-based recommendations: {e}")
            return []
    
    async def _get_matrix_factorization_recommendations(
        self,
        user_id: str,
        cf_model: Dict[str, Any],
        num_recommendations: int,
        exclude_interacted: bool
    ) -> List[Dict[str, Any]]:
        """Get matrix factorization recommendations"""
        try:
            user_factors = cf_model["user_factors"]
            item_factors = cf_model["item_factors"]
            user_ids = cf_model["user_ids"]
            item_ids = cf_model["item_ids"]
            
            if user_id not in user_ids:
                return []
            
            user_idx = user_ids.index(user_id)
            user_factor = user_factors[user_idx]
            
            # Calculate recommendation scores
            recommendation_scores = {}
            for item_idx, item_factor in enumerate(item_factors):
                item_id = item_ids[item_idx]
                score = np.dot(user_factor, item_factor)
                recommendation_scores[item_id] = score
            
            # Sort and return top recommendations
            sorted_items = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for item_id, score in sorted_items[:num_recommendations]:
                if exclude_interacted and self._has_user_interacted(user_id, item_id):
                    continue
                
                recommendations.append({
                    "item_id": item_id,
                    "score": score,
                    "reason": "matrix_factorization"
                })
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Failed to get matrix factorization recommendations: {e}")
            return []
    
    async def _get_content_based_recommendations(
        self,
        user_id: str,
        model_id: str,
        num_recommendations: int,
        exclude_interacted: bool
    ) -> List[Dict[str, Any]]:
        """Get content-based recommendations"""
        try:
            if model_id not in self.content_models:
                return []
            
            content_model = self.content_models[model_id]
            algorithm = content_model["type"]
            
            if algorithm == "tfidf_similarity":
                return await self._get_tfidf_recommendations(
                    user_id, content_model, num_recommendations, exclude_interacted
                )
            elif algorithm == "content_similarity":
                return await self._get_content_similarity_recommendations(
                    user_id, content_model, num_recommendations, exclude_interacted
                )
            else:
                return []
        
        except Exception as e:
            logger.error(f"Failed to get content-based recommendations: {e}")
            return []
    
    async def _get_tfidf_recommendations(
        self,
        user_id: str,
        content_model: Dict[str, Any],
        num_recommendations: int,
        exclude_interacted: bool
    ) -> List[Dict[str, Any]]:
        """Get TF-IDF based recommendations"""
        try:
            # Get user's liked items
            user_liked_items = []
            for interaction in self.user_interactions.get(user_id, []):
                if interaction["interaction_type"] in ["like", "purchase", "rating"]:
                    if interaction.get("rating", 0) >= 3:  # Consider ratings >= 3 as positive
                        user_liked_items.append(interaction["item_id"])
            
            if not user_liked_items:
                return []
            
            # Calculate average content similarity
            content_similarities = content_model["content_similarities"]
            item_ids = content_model["item_ids"]
            
            recommendation_scores = {}
            for liked_item_id in user_liked_items:
                if liked_item_id in item_ids:
                    liked_item_idx = item_ids.index(liked_item_id)
                    item_similarity_scores = content_similarities[liked_item_idx]
                    
                    for similar_item_idx, similarity in enumerate(item_similarity_scores):
                        similar_item_id = item_ids[similar_item_idx]
                        if similar_item_id not in user_liked_items:  # Item not liked by user
                            if similar_item_id not in recommendation_scores:
                                recommendation_scores[similar_item_id] = []
                            recommendation_scores[similar_item_id].append(similarity)
            
            # Average similarity scores
            for item_id, similarities in recommendation_scores.items():
                recommendation_scores[item_id] = np.mean(similarities)
            
            # Sort and return top recommendations
            sorted_items = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for item_id, score in sorted_items[:num_recommendations]:
                if exclude_interacted and self._has_user_interacted(user_id, item_id):
                    continue
                
                recommendations.append({
                    "item_id": item_id,
                    "score": score,
                    "reason": "tfidf_similarity"
                })
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Failed to get TF-IDF recommendations: {e}")
            return []
    
    async def _get_content_similarity_recommendations(
        self,
        user_id: str,
        content_model: Dict[str, Any],
        num_recommendations: int,
        exclude_interacted: bool
    ) -> List[Dict[str, Any]]:
        """Get feature-based content similarity recommendations"""
        try:
            # Similar to TF-IDF but using feature similarities
            user_liked_items = []
            for interaction in self.user_interactions.get(user_id, []):
                if interaction["interaction_type"] in ["like", "purchase", "rating"]:
                    if interaction.get("rating", 0) >= 3:
                        user_liked_items.append(interaction["item_id"])
            
            if not user_liked_items:
                return []
            
            content_similarities = content_model["content_similarities"]
            item_ids = content_model["item_ids"]
            
            recommendation_scores = {}
            for liked_item_id in user_liked_items:
                if liked_item_id in item_ids:
                    liked_item_idx = item_ids.index(liked_item_id)
                    item_similarity_scores = content_similarities[liked_item_idx]
                    
                    for similar_item_idx, similarity in enumerate(item_similarity_scores):
                        similar_item_id = item_ids[similar_item_idx]
                        if similar_item_id not in user_liked_items:
                            if similar_item_id not in recommendation_scores:
                                recommendation_scores[similar_item_id] = []
                            recommendation_scores[similar_item_id].append(similarity)
            
            # Average similarity scores
            for item_id, similarities in recommendation_scores.items():
                recommendation_scores[item_id] = np.mean(similarities)
            
            # Sort and return top recommendations
            sorted_items = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for item_id, score in sorted_items[:num_recommendations]:
                if exclude_interacted and self._has_user_interacted(user_id, item_id):
                    continue
                
                recommendations.append({
                    "item_id": item_id,
                    "score": score,
                    "reason": "content_similarity"
                })
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Failed to get content similarity recommendations: {e}")
            return []
    
    async def _get_hybrid_recommendations(
        self,
        user_id: str,
        model_id: str,
        num_recommendations: int,
        exclude_interacted: bool
    ) -> List[Dict[str, Any]]:
        """Get hybrid recommendations"""
        try:
            if model_id not in self.hybrid_models:
                return []
            
            hybrid_model = self.hybrid_models[model_id]
            cf_weight = hybrid_model["cf_weight"]
            content_weight = hybrid_model["content_weight"]
            
            # Get CF recommendations
            cf_recommendations = await self._get_cf_recommendations(
                user_id, hybrid_model["cf_model_id"], num_recommendations * 2, exclude_interacted
            )
            
            # Get content-based recommendations
            content_recommendations = await self._get_content_based_recommendations(
                user_id, hybrid_model["content_model_id"], num_recommendations * 2, exclude_interacted
            )
            
            # Combine recommendations
            combined_scores = {}
            
            # Add CF scores
            for rec in cf_recommendations:
                item_id = rec["item_id"]
                combined_scores[item_id] = cf_weight * rec["score"]
            
            # Add content-based scores
            for rec in content_recommendations:
                item_id = rec["item_id"]
                if item_id in combined_scores:
                    combined_scores[item_id] += content_weight * rec["score"]
                else:
                    combined_scores[item_id] = content_weight * rec["score"]
            
            # Sort and return top recommendations
            sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for item_id, score in sorted_items[:num_recommendations]:
                recommendations.append({
                    "item_id": item_id,
                    "score": score,
                    "reason": "hybrid"
                })
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Failed to get hybrid recommendations: {e}")
            return []
    
    async def _get_popularity_recommendations(
        self,
        user_id: str,
        num_recommendations: int,
        exclude_interacted: bool
    ) -> List[Dict[str, Any]]:
        """Get popularity-based recommendations"""
        try:
            # Sort items by popularity
            sorted_items = sorted(
                self.popularity_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            recommendations = []
            for item_id, score in sorted_items[:num_recommendations]:
                if exclude_interacted and self._has_user_interacted(user_id, item_id):
                    continue
                
                recommendations.append({
                    "item_id": item_id,
                    "score": score,
                    "reason": "popularity"
                })
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Failed to get popularity recommendations: {e}")
            return []
    
    def _create_user_item_matrix(self) -> np.ndarray:
        """Create user-item interaction matrix"""
        try:
            user_ids = list(self.user_interactions.keys())
            item_ids = list(self.item_features.keys())
            
            if not user_ids or not item_ids:
                return np.array([])
            
            matrix = np.zeros((len(user_ids), len(item_ids)))
            
            for user_idx, user_id in enumerate(user_ids):
                for interaction in self.user_interactions[user_id]:
                    if interaction["item_id"] in item_ids:
                        item_idx = item_ids.index(interaction["item_id"])
                        rating = interaction.get("rating", 1)  # Default rating of 1
                        matrix[user_idx, item_idx] = rating
            
            return matrix
        
        except Exception as e:
            logger.error(f"Failed to create user-item matrix: {e}")
            return np.array([])
    
    def _calculate_sparsity(self, matrix: np.ndarray) -> float:
        """Calculate sparsity of user-item matrix"""
        try:
            if matrix.size == 0:
                return 1.0
            
            non_zero_elements = np.count_nonzero(matrix)
            total_elements = matrix.size
            return 1.0 - (non_zero_elements / total_elements)
        
        except Exception as e:
            logger.error(f"Failed to calculate sparsity: {e}")
            return 1.0
    
    def _get_item_interactions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get interactions grouped by item"""
        try:
            item_interactions = defaultdict(list)
            
            for user_id, interactions in self.user_interactions.items():
                for interaction in interactions:
                    item_interactions[interaction["item_id"]].append(interaction)
            
            return dict(item_interactions)
        
        except Exception as e:
            logger.error(f"Failed to get item interactions: {e}")
            return {}
    
    def _has_user_interacted(self, user_id: str, item_id: str) -> bool:
        """Check if user has interacted with item"""
        try:
            for interaction in self.user_interactions.get(user_id, []):
                if interaction["item_id"] == item_id:
                    return True
            return False
        
        except Exception as e:
            logger.error(f"Failed to check user interaction: {e}")
            return False
    
    async def _update_user_profile(self, user_id: str, interaction: Dict[str, Any]):
        """Update user profile based on interaction"""
        try:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = {
                    "interaction_count": 0,
                    "preferred_categories": defaultdict(int),
                    "avg_rating": 0.0,
                    "last_interaction": None
                }
            
            profile = self.user_profiles[user_id]
            profile["interaction_count"] += 1
            profile["last_interaction"] = interaction["timestamp"]
            
            # Update category preferences
            item_id = interaction["item_id"]
            if item_id in self.item_features:
                item_features = self.item_features[item_id].get("features", {})
                category = item_features.get("category", "unknown")
                profile["preferred_categories"][category] += 1
            
            # Update average rating
            if interaction.get("rating"):
                current_avg = profile["avg_rating"]
                interaction_count = profile["interaction_count"]
                new_avg = ((current_avg * (interaction_count - 1)) + interaction["rating"]) / interaction_count
                profile["avg_rating"] = new_avg
        
        except Exception as e:
            logger.error(f"Failed to update user profile: {e}")
    
    async def _update_popularity_scores(self, item_id: str, interaction_type: str):
        """Update popularity scores for items"""
        try:
            if item_id not in self.popularity_scores:
                self.popularity_scores[item_id] = 0
            
            # Weight different interaction types
            weights = {
                "view": 1,
                "like": 2,
                "purchase": 5,
                "rating": 3
            }
            
            weight = weights.get(interaction_type, 1)
            self.popularity_scores[item_id] += weight
        
        except Exception as e:
            logger.error(f"Failed to update popularity scores: {e}")
    
    async def get_recommendation_stats(self) -> Dict[str, Any]:
        """Get recommendation service statistics"""
        try:
            return {
                "total_recommendations": self.recommendation_stats["total_recommendations"],
                "recommendations_by_type": self.recommendation_stats["recommendations_by_type"],
                "recommendations_by_algorithm": self.recommendation_stats["recommendations_by_algorithm"],
                "total_users": len(self.user_interactions),
                "total_items": len(self.item_features),
                "interaction_count": self.recommendation_stats["interaction_count"],
                "cached_recommendations": len(self.recommendation_cache),
                "models_count": len(self.recommendation_models),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get recommendation stats: {e}")
            return {"error": str(e)}


# Global recommendation service instance
recommendation_service = RecommendationService()

