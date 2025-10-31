"""
Recommendation System Engine for Export IA
Advanced recommendation algorithms with collaborative filtering, content-based, and hybrid approaches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import time
import json
import random
from pathlib import Path
from collections import defaultdict, deque
import copy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
import implicit
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization
import lightfm
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
import surprise
from surprise import SVD, SVDpp, NMF, KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate

logger = logging.getLogger(__name__)

@dataclass
class RecommendationConfig:
    """Configuration for recommendation system"""
    # Algorithm types
    algorithm_type: str = "hybrid"  # collaborative, content_based, hybrid, deep_learning
    
    # Collaborative filtering parameters
    cf_method: str = "matrix_factorization"  # matrix_factorization, neighborhood, deep_cf
    cf_factors: int = 50
    cf_iterations: int = 20
    cf_regularization: float = 0.01
    cf_learning_rate: float = 0.01
    
    # Content-based parameters
    cb_similarity_metric: str = "cosine"  # cosine, euclidean, manhattan, jaccard
    cb_feature_weighting: str = "tfidf"  # tfidf, binary, frequency
    cb_similarity_threshold: float = 0.1
    
    # Hybrid parameters
    hybrid_weights: Dict[str, float] = None  # {"collaborative": 0.6, "content": 0.4}
    hybrid_fusion_method: str = "weighted"  # weighted, switching, mixed
    
    # Deep learning parameters
    dl_model_type: str = "neural_cf"  # neural_cf, autoencoder, deep_fm, wide_deep
    dl_hidden_layers: List[int] = None  # [64, 32, 16]
    dl_dropout: float = 0.2
    dl_activation: str = "relu"  # relu, tanh, sigmoid
    dl_optimizer: str = "adam"  # adam, sgd, rmsprop
    dl_learning_rate: float = 0.001
    dl_batch_size: int = 32
    dl_epochs: int = 100
    
    # Matrix factorization parameters
    mf_algorithm: str = "als"  # als, bpr, lmf, svd, svdpp
    mf_factors: int = 50
    mf_iterations: int = 20
    mf_regularization: float = 0.01
    mf_alpha: float = 1.0
    
    # Neighborhood parameters
    neighborhood_method: str = "user_based"  # user_based, item_based
    neighborhood_similarity: str = "cosine"  # cosine, pearson, euclidean
    neighborhood_k: int = 20
    neighborhood_min_support: int = 5
    
    # Evaluation parameters
    evaluation_metrics: List[str] = None  # precision, recall, f1, ndcg, map, rmse, mae
    evaluation_k: int = 10
    evaluation_cv_folds: int = 5
    evaluation_test_size: float = 0.2
    
    # Data parameters
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    max_user_interactions: int = 1000
    max_item_interactions: int = 1000
    
    # Performance parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    enable_caching: bool = True
    cache_size: int = 1000

class CollaborativeFiltering:
    """Collaborative Filtering recommendation system"""
    
    def __init__(self, config: RecommendationConfig):
        self.config = config
        self.model = None
        self.user_item_matrix = None
        self.user_similarities = None
        self.item_similarities = None
        
    def fit(self, user_item_matrix: csr_matrix, user_features: np.ndarray = None, 
            item_features: np.ndarray = None):
        """Fit collaborative filtering model"""
        
        self.user_item_matrix = user_item_matrix
        
        if self.config.cf_method == "matrix_factorization":
            self._fit_matrix_factorization()
        elif self.config.cf_method == "neighborhood":
            self._fit_neighborhood()
        elif self.config.cf_method == "deep_cf":
            self._fit_deep_cf(user_features, item_features)
        else:
            raise ValueError(f"Unsupported CF method: {self.config.cf_method}")
            
    def _fit_matrix_factorization(self):
        """Fit matrix factorization model"""
        
        if self.config.mf_algorithm == "als":
            self.model = AlternatingLeastSquares(
                factors=self.config.mf_factors,
                iterations=self.config.mf_iterations,
                regularization=self.config.mf_regularization,
                alpha=self.config.mf_alpha
            )
        elif self.config.mf_algorithm == "bpr":
            self.model = BayesianPersonalizedRanking(
                factors=self.config.mf_factors,
                iterations=self.config.mf_iterations,
                regularization=self.config.mf_regularization
            )
        elif self.config.mf_algorithm == "lmf":
            self.model = LogisticMatrixFactorization(
                factors=self.config.mf_factors,
                iterations=self.config.mf_iterations,
                regularization=self.config.mf_regularization
            )
        else:
            raise ValueError(f"Unsupported MF algorithm: {self.config.mf_algorithm}")
            
        self.model.fit(self.user_item_matrix)
        
    def _fit_neighborhood(self):
        """Fit neighborhood-based model"""
        
        if self.config.neighborhood_method == "user_based":
            self.user_similarities = cosine_similarity(self.user_item_matrix)
        elif self.config.neighborhood_method == "item_based":
            self.item_similarities = cosine_similarity(self.user_item_matrix.T)
        else:
            raise ValueError(f"Unsupported neighborhood method: {self.config.neighborhood_method}")
            
    def _fit_deep_cf(self, user_features: np.ndarray, item_features: np.ndarray):
        """Fit deep collaborative filtering model"""
        
        self.model = NeuralCollaborativeFiltering(
            num_users=self.user_item_matrix.shape[0],
            num_items=self.user_item_matrix.shape[1],
            user_features_dim=user_features.shape[1] if user_features is not None else 0,
            item_features_dim=item_features.shape[1] if item_features is not None else 0,
            config=self.config
        )
        
        # Train the model
        self._train_deep_cf(user_features, item_features)
        
    def _train_deep_cf(self, user_features: np.ndarray, item_features: np.ndarray):
        """Train deep collaborative filtering model"""
        
        # Create training data
        user_indices, item_indices = self.user_item_matrix.nonzero()
        ratings = self.user_item_matrix.data
        
        # Convert to tensors
        user_tensor = torch.LongTensor(user_indices)
        item_tensor = torch.LongTensor(item_indices)
        rating_tensor = torch.FloatTensor(ratings)
        
        if user_features is not None:
            user_features_tensor = torch.FloatTensor(user_features[user_indices])
        else:
            user_features_tensor = None
            
        if item_features is not None:
            item_features_tensor = torch.FloatTensor(item_features[item_indices])
        else:
            item_features_tensor = None
            
        # Create dataset
        dataset = torch.utils.data.TensorDataset(
            user_tensor, item_tensor, rating_tensor
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.dl_batch_size, shuffle=True
        )
        
        # Train model
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.dl_learning_rate)
        criterion = nn.MSELoss()
        
        for epoch in range(self.config.dl_epochs):
            total_loss = 0.0
            
            for batch in dataloader:
                user_batch, item_batch, rating_batch = batch
                
                optimizer.zero_grad()
                
                # Forward pass
                if user_features_tensor is not None and item_features_tensor is not None:
                    user_feat_batch = user_features_tensor[user_batch]
                    item_feat_batch = item_features_tensor[item_batch]
                    predictions = self.model(user_batch, item_batch, user_feat_batch, item_feat_batch)
                else:
                    predictions = self.model(user_batch, item_batch)
                    
                loss = criterion(predictions, rating_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss = {total_loss:.4f}")
                
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair"""
        
        if self.config.cf_method == "matrix_factorization":
            return self.model.predict(user_id, item_id)
        elif self.config.cf_method == "neighborhood":
            return self._predict_neighborhood(user_id, item_id)
        elif self.config.cf_method == "deep_cf":
            return self._predict_deep_cf(user_id, item_id)
        else:
            raise ValueError(f"Unsupported CF method: {self.config.cf_method}")
            
    def _predict_neighborhood(self, user_id: int, item_id: int) -> float:
        """Predict using neighborhood method"""
        
        if self.config.neighborhood_method == "user_based":
            # Find similar users
            user_similarities = self.user_similarities[user_id]
            similar_users = np.argsort(user_similarities)[::-1][:self.config.neighborhood_k]
            
            # Calculate weighted average
            weighted_sum = 0.0
            similarity_sum = 0.0
            
            for similar_user in similar_users:
                if similar_user != user_id and self.user_item_matrix[similar_user, item_id] > 0:
                    similarity = user_similarities[similar_user]
                    rating = self.user_item_matrix[similar_user, item_id]
                    weighted_sum += similarity * rating
                    similarity_sum += abs(similarity)
                    
            if similarity_sum > 0:
                return weighted_sum / similarity_sum
            else:
                return 0.0
                
        elif self.config.neighborhood_method == "item_based":
            # Find similar items
            item_similarities = self.item_similarities[item_id]
            similar_items = np.argsort(item_similarities)[::-1][:self.config.neighborhood_k]
            
            # Calculate weighted average
            weighted_sum = 0.0
            similarity_sum = 0.0
            
            for similar_item in similar_items:
                if similar_item != item_id and self.user_item_matrix[user_id, similar_item] > 0:
                    similarity = item_similarities[similar_item]
                    rating = self.user_item_matrix[user_id, similar_item]
                    weighted_sum += similarity * rating
                    similarity_sum += abs(similarity)
                    
            if similarity_sum > 0:
                return weighted_sum / similarity_sum
            else:
                return 0.0
                
    def _predict_deep_cf(self, user_id: int, item_id: int) -> float:
        """Predict using deep collaborative filtering"""
        
        self.model.eval()
        
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id])
            item_tensor = torch.LongTensor([item_id])
            
            prediction = self.model(user_tensor, item_tensor)
            return prediction.item()
            
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Get recommendations for user"""
        
        if self.config.cf_method == "matrix_factorization":
            recommendations = self.model.recommend(
                user_id, self.user_item_matrix, N=n_recommendations
            )
            return [(item_id, score) for item_id, score in zip(recommendations[0], recommendations[1])]
        else:
            # For other methods, predict for all items
            predictions = []
            for item_id in range(self.user_item_matrix.shape[1]):
                if self.user_item_matrix[user_id, item_id] == 0:  # Not rated
                    score = self.predict(user_id, item_id)
                    predictions.append((item_id, score))
                    
            # Sort by score and return top N
            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions[:n_recommendations]

class ContentBasedFiltering:
    """Content-based filtering recommendation system"""
    
    def __init__(self, config: RecommendationConfig):
        self.config = config
        self.item_features = None
        self.item_similarities = None
        self.user_profiles = None
        
    def fit(self, item_features: np.ndarray, user_item_matrix: csr_matrix):
        """Fit content-based model"""
        
        self.item_features = item_features
        
        # Calculate item similarities
        if self.config.cb_similarity_metric == "cosine":
            self.item_similarities = cosine_similarity(item_features)
        elif self.config.cb_similarity_metric == "euclidean":
            distances = pdist(item_features, metric='euclidean')
            self.item_similarities = 1 / (1 + squareform(distances))
        elif self.config.cb_similarity_metric == "manhattan":
            distances = pdist(item_features, metric='manhattan')
            self.item_similarities = 1 / (1 + squareform(distances))
        elif self.config.cb_similarity_metric == "jaccard":
            # Convert to binary for Jaccard similarity
            binary_features = (item_features > 0).astype(int)
            self.item_similarities = cosine_similarity(binary_features)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.config.cb_similarity_metric}")
            
        # Build user profiles
        self._build_user_profiles(user_item_matrix)
        
    def _build_user_profiles(self, user_item_matrix: csr_matrix):
        """Build user profiles based on their item preferences"""
        
        num_users = user_item_matrix.shape[0]
        num_features = self.item_features.shape[1]
        
        self.user_profiles = np.zeros((num_users, num_features))
        
        for user_id in range(num_users):
            # Get items rated by user
            user_items = user_item_matrix[user_id].indices
            user_ratings = user_item_matrix[user_id].data
            
            if len(user_items) > 0:
                # Weighted average of item features
                weighted_features = np.zeros(num_features)
                total_weight = 0.0
                
                for item_idx, rating in zip(user_items, user_ratings):
                    weighted_features += rating * self.item_features[item_idx]
                    total_weight += rating
                    
                if total_weight > 0:
                    self.user_profiles[user_id] = weighted_features / total_weight
                    
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair"""
        
        user_profile = self.user_profiles[user_id]
        item_features = self.item_features[item_id]
        
        # Calculate similarity between user profile and item features
        if self.config.cb_similarity_metric == "cosine":
            similarity = cosine_similarity([user_profile], [item_features])[0][0]
        else:
            # Use precomputed similarities
            similarity = self.item_similarities[item_id, item_id]  # This would need proper implementation
            
        return similarity
        
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Get recommendations for user"""
        
        user_profile = self.user_profiles[user_id]
        
        # Calculate similarities with all items
        similarities = cosine_similarity([user_profile], self.item_features)[0]
        
        # Get top recommendations
        recommendations = []
        for item_id, similarity in enumerate(similarities):
            if similarity > self.config.cb_similarity_threshold:
                recommendations.append((item_id, similarity))
                
        # Sort by similarity and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]

class NeuralCollaborativeFiltering(nn.Module):
    """Neural Collaborative Filtering model"""
    
    def __init__(self, num_users: int, num_items: int, user_features_dim: int = 0, 
                 item_features_dim: int = 0, config: RecommendationConfig = None):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.user_features_dim = user_features_dim
        self.item_features_dim = item_features_dim
        self.config = config or RecommendationConfig()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, self.config.dl_hidden_layers[0])
        self.item_embedding = nn.Embedding(num_items, self.config.dl_hidden_layers[0])
        
        # Feature layers
        if user_features_dim > 0:
            self.user_feature_layer = nn.Linear(user_features_dim, self.config.dl_hidden_layers[0])
        if item_features_dim > 0:
            self.item_feature_layer = nn.Linear(item_features_dim, self.config.dl_hidden_layers[0])
            
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        input_dim = self.config.dl_hidden_layers[0] * 2
        
        if user_features_dim > 0:
            input_dim += self.config.dl_hidden_layers[0]
        if item_features_dim > 0:
            input_dim += self.config.dl_hidden_layers[0]
            
        for i in range(len(self.config.dl_hidden_layers) - 1):
            self.hidden_layers.append(
                nn.Linear(input_dim, self.config.dl_hidden_layers[i + 1])
            )
            input_dim = self.config.dl_hidden_layers[i + 1]
            
        # Output layer
        self.output_layer = nn.Linear(input_dim, 1)
        
        # Dropout
        self.dropout = nn.Dropout(self.config.dl_dropout)
        
        # Activation
        if self.config.dl_activation == "relu":
            self.activation = nn.ReLU()
        elif self.config.dl_activation == "tanh":
            self.activation = nn.Tanh()
        elif self.config.dl_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()
            
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor, 
                user_features: torch.Tensor = None, item_features: torch.Tensor = None):
        """Forward pass"""
        
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Concatenate embeddings
        x = torch.cat([user_emb, item_emb], dim=1)
        
        # Add feature layers if available
        if user_features is not None:
            user_feat = self.user_feature_layer(user_features)
            x = torch.cat([x, user_feat], dim=1)
            
        if item_features is not None:
            item_feat = self.item_feature_layer(item_features)
            x = torch.cat([x, item_feat], dim=1)
            
        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = self.activation(x)
            x = self.dropout(x)
            x = layer(x)
            
        # Output layer
        x = self.output_layer(x)
        
        return x.squeeze()

class HybridRecommendationSystem:
    """Hybrid recommendation system combining multiple approaches"""
    
    def __init__(self, config: RecommendationConfig):
        self.config = config
        self.cf_model = None
        self.cb_model = None
        self.weights = config.hybrid_weights or {"collaborative": 0.6, "content": 0.4}
        
    def fit(self, user_item_matrix: csr_matrix, item_features: np.ndarray = None, 
            user_features: np.ndarray = None):
        """Fit hybrid model"""
        
        # Fit collaborative filtering
        self.cf_model = CollaborativeFiltering(self.config)
        self.cf_model.fit(user_item_matrix, user_features, item_features)
        
        # Fit content-based filtering if item features available
        if item_features is not None:
            self.cb_model = ContentBasedFiltering(self.config)
            self.cb_model.fit(item_features, user_item_matrix)
            
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating using hybrid approach"""
        
        cf_prediction = self.cf_model.predict(user_id, item_id)
        
        if self.cb_model is not None:
            cb_prediction = self.cb_model.predict(user_id, item_id)
            
            if self.config.hybrid_fusion_method == "weighted":
                return (self.weights["collaborative"] * cf_prediction + 
                       self.weights["content"] * cb_prediction)
            elif self.config.hybrid_fusion_method == "switching":
                # Use CF if user has enough interactions, otherwise use CB
                user_interactions = self.cf_model.user_item_matrix[user_id].nnz
                if user_interactions >= 10:
                    return cf_prediction
                else:
                    return cb_prediction
            elif self.config.hybrid_fusion_method == "mixed":
                # Use both predictions with different weights
                return 0.7 * cf_prediction + 0.3 * cb_prediction
        else:
            return cf_prediction
            
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Get hybrid recommendations"""
        
        cf_recommendations = self.cf_model.recommend(user_id, n_recommendations)
        
        if self.cb_model is not None:
            cb_recommendations = self.cb_model.recommend(user_id, n_recommendations)
            
            # Combine recommendations
            combined_recommendations = {}
            
            for item_id, score in cf_recommendations:
                combined_recommendations[item_id] = self.weights["collaborative"] * score
                
            for item_id, score in cb_recommendations:
                if item_id in combined_recommendations:
                    combined_recommendations[item_id] += self.weights["content"] * score
                else:
                    combined_recommendations[item_id] = self.weights["content"] * score
                    
            # Sort and return top N
            recommendations = [(item_id, score) for item_id, score in combined_recommendations.items()]
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:n_recommendations]
        else:
            return cf_recommendations

class RecommendationSystemEngine:
    """Main Recommendation System Engine"""
    
    def __init__(self, config: RecommendationConfig):
        self.config = config
        self.model = None
        self.user_item_matrix = None
        self.item_features = None
        self.user_features = None
        
        # Results storage
        self.results = defaultdict(list)
        self.performance_metrics = defaultdict(list)
        
    def fit(self, user_item_matrix: csr_matrix, item_features: np.ndarray = None, 
            user_features: np.ndarray = None):
        """Fit recommendation model"""
        
        self.user_item_matrix = user_item_matrix
        self.item_features = item_features
        self.user_features = user_features
        
        # Initialize model based on algorithm type
        if self.config.algorithm_type == "collaborative":
            self.model = CollaborativeFiltering(self.config)
        elif self.config.algorithm_type == "content_based":
            if item_features is None:
                raise ValueError("Item features required for content-based filtering")
            self.model = ContentBasedFiltering(self.config)
        elif self.config.algorithm_type == "hybrid":
            self.model = HybridRecommendationSystem(self.config)
        else:
            raise ValueError(f"Unsupported algorithm type: {self.config.algorithm_type}")
            
        # Fit the model
        self.model.fit(user_item_matrix, item_features, user_features)
        
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair"""
        
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        return self.model.predict(user_id, item_id)
        
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Get recommendations for user"""
        
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        return self.model.recommend(user_id, n_recommendations)
        
    def evaluate(self, test_data: List[Tuple[int, int, float]]) -> Dict[str, float]:
        """Evaluate recommendation model"""
        
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        metrics = {}
        
        # Calculate predictions
        predictions = []
        actuals = []
        
        for user_id, item_id, actual_rating in test_data:
            predicted_rating = self.predict(user_id, item_id)
            predictions.append(predicted_rating)
            actuals.append(actual_rating)
            
        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # RMSE
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        metrics['rmse'] = rmse
        
        # MAE
        mae = np.mean(np.abs(predictions - actuals))
        metrics['mae'] = mae
        
        # Precision@K
        if 'precision' in (self.config.evaluation_metrics or []):
            precision = self._calculate_precision_at_k(test_data)
            metrics['precision_at_k'] = precision
            
        # Recall@K
        if 'recall' in (self.config.evaluation_metrics or []):
            recall = self._calculate_recall_at_k(test_data)
            metrics['recall_at_k'] = recall
            
        # NDCG@K
        if 'ndcg' in (self.config.evaluation_metrics or []):
            ndcg = self._calculate_ndcg_at_k(test_data)
            metrics['ndcg_at_k'] = ndcg
            
        return metrics
        
    def _calculate_precision_at_k(self, test_data: List[Tuple[int, int, float]]) -> float:
        """Calculate Precision@K"""
        
        precision_scores = []
        
        for user_id, _, _ in test_data:
            # Get recommendations
            recommendations = self.recommend(user_id, self.config.evaluation_k)
            recommended_items = [item_id for item_id, _ in recommendations]
            
            # Get actual items for user
            actual_items = [item_id for u_id, item_id, _ in test_data if u_id == user_id]
            
            # Calculate precision
            if len(recommended_items) > 0:
                precision = len(set(recommended_items) & set(actual_items)) / len(recommended_items)
                precision_scores.append(precision)
                
        return np.mean(precision_scores) if precision_scores else 0.0
        
    def _calculate_recall_at_k(self, test_data: List[Tuple[int, int, float]]) -> float:
        """Calculate Recall@K"""
        
        recall_scores = []
        
        for user_id, _, _ in test_data:
            # Get recommendations
            recommendations = self.recommend(user_id, self.config.evaluation_k)
            recommended_items = [item_id for item_id, _ in recommendations]
            
            # Get actual items for user
            actual_items = [item_id for u_id, item_id, _ in test_data if u_id == user_id]
            
            # Calculate recall
            if len(actual_items) > 0:
                recall = len(set(recommended_items) & set(actual_items)) / len(actual_items)
                recall_scores.append(recall)
                
        return np.mean(recall_scores) if recall_scores else 0.0
        
    def _calculate_ndcg_at_k(self, test_data: List[Tuple[int, int, float]]) -> float:
        """Calculate NDCG@K"""
        
        ndcg_scores = []
        
        for user_id, _, _ in test_data:
            # Get recommendations
            recommendations = self.recommend(user_id, self.config.evaluation_k)
            recommended_items = [item_id for item_id, _ in recommendations]
            
            # Get actual items for user
            actual_items = [item_id for u_id, item_id, _ in test_data if u_id == user_id]
            
            # Calculate NDCG
            if len(recommended_items) > 0 and len(actual_items) > 0:
                # DCG
                dcg = 0.0
                for i, item_id in enumerate(recommended_items):
                    if item_id in actual_items:
                        dcg += 1.0 / np.log2(i + 2)
                        
                # IDCG
                idcg = 0.0
                for i in range(min(len(actual_items), len(recommended_items))):
                    idcg += 1.0 / np.log2(i + 2)
                    
                # NDCG
                if idcg > 0:
                    ndcg = dcg / idcg
                    ndcg_scores.append(ndcg)
                    
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        
        metrics = {
            'algorithm_type': self.config.algorithm_type,
            'model_fitted': self.model is not None,
            'total_users': self.user_item_matrix.shape[0] if self.user_item_matrix is not None else 0,
            'total_items': self.user_item_matrix.shape[1] if self.user_item_matrix is not None else 0,
            'total_interactions': self.user_item_matrix.nnz if self.user_item_matrix is not None else 0
        }
        
        return metrics
        
    def save_model(self, filepath: str):
        """Save model"""
        
        model_data = {
            'config': self.config.__dict__,
            'user_item_matrix': self.user_item_matrix,
            'item_features': self.item_features,
            'user_features': self.user_features,
            'performance_metrics': self.get_performance_metrics()
        }
        
        torch.save(model_data, filepath)
        
    def load_model(self, filepath: str):
        """Load model"""
        
        model_data = torch.load(filepath)
        
        self.config = RecommendationConfig(**model_data['config'])
        self.user_item_matrix = model_data['user_item_matrix']
        self.item_features = model_data['item_features']
        self.user_features = model_data['user_features']
        
        # Re-fit model
        self.fit(self.user_item_matrix, self.item_features, self.user_features)

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test recommendation system
    print("Testing Recommendation System Engine...")
    
    # Create dummy data
    num_users = 100
    num_items = 50
    num_interactions = 500
    
    # Create random user-item matrix
    user_indices = np.random.randint(0, num_users, num_interactions)
    item_indices = np.random.randint(0, num_items, num_interactions)
    ratings = np.random.randint(1, 6, num_interactions)
    
    user_item_matrix = csr_matrix((ratings, (user_indices, item_indices)), 
                                 shape=(num_users, num_items))
    
    # Create dummy item features
    item_features = np.random.randn(num_items, 20)
    
    # Create config
    config = RecommendationConfig(
        algorithm_type="hybrid",
        cf_method="matrix_factorization",
        mf_algorithm="als",
        mf_factors=20,
        mf_iterations=10,
        evaluation_metrics=["precision", "recall", "rmse", "mae"]
    )
    
    # Create engine
    rec_engine = RecommendationSystemEngine(config)
    
    # Fit model
    print("Fitting recommendation model...")
    rec_engine.fit(user_item_matrix, item_features)
    
    # Test predictions
    print("Testing predictions...")
    prediction = rec_engine.predict(0, 0)
    print(f"Prediction for user 0, item 0: {prediction:.4f}")
    
    # Test recommendations
    print("Testing recommendations...")
    recommendations = rec_engine.recommend(0, 5)
    print(f"Recommendations for user 0: {recommendations}")
    
    # Test evaluation
    print("Testing evaluation...")
    test_data = [(0, 1, 4.0), (1, 2, 3.0), (2, 3, 5.0)]
    metrics = rec_engine.evaluate(test_data)
    print(f"Evaluation metrics: {metrics}")
    
    # Get performance metrics
    performance = rec_engine.get_performance_metrics()
    print(f"Performance metrics: {performance}")
    
    print("\nRecommendation system engine initialized successfully!")
























