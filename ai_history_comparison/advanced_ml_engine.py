"""
AI History Comparison System - Advanced ML Engine

This module provides advanced machine learning capabilities for enhanced content analysis,
including deep learning models, advanced clustering, anomaly detection, and predictive analytics.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import pickle
from pathlib import Path

# Advanced ML libraries
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Deep learning (optional)
try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Advanced NLP
try:
    import spacy
    from spacy import displacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

logger = logging.getLogger(__name__)

@dataclass
class MLModelConfig:
    """Configuration for ML models"""
    model_type: str
    parameters: Dict[str, Any]
    training_data_size: int
    accuracy_score: float
    last_trained: datetime
    version: str

@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection"""
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    anomaly_type: str
    explanation: str
    recommendations: List[str]

@dataclass
class AdvancedClusteringResult:
    """Result of advanced clustering"""
    clusters: Dict[int, List[str]]
    cluster_centers: Dict[int, Dict[str, float]]
    silhouette_score: float
    calinski_harabasz_score: float
    optimal_clusters: int
    algorithm_used: str
    feature_importance: Dict[str, float]

@dataclass
class PredictiveModelResult:
    """Result of predictive modeling"""
    predictions: List[float]
    confidence_intervals: List[Tuple[float, float]]
    model_accuracy: float
    feature_importance: Dict[str, float]
    trend_direction: str
    next_prediction: float
    model_type: str

class AdvancedMLEngine:
    """
    Advanced Machine Learning Engine for AI History Analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the advanced ML engine"""
        self.config = config or {}
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.vectorizers: Dict[str, Any] = {}
        self.model_configs: Dict[str, MLModelConfig] = {}
        
        # Initialize transformers if available
        self.transformer_model = None
        self.transformer_tokenizer = None
        if HAS_TRANSFORMERS:
            self._initialize_transformers()
        
        # Initialize spaCy if available
        self.nlp = None
        if HAS_SPACY:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
        
        # Model storage
        self.model_storage_path = Path("models")
        self.model_storage_path.mkdir(exist_ok=True)
        
        logger.info("Advanced ML Engine initialized successfully")

    def _initialize_transformers(self):
        """Initialize transformer models for advanced NLP"""
        try:
            # Use a lightweight model for embeddings
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.transformer_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.transformer_model = AutoModel.from_pretrained(model_name)
            logger.info("Transformer models initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize transformers: {e}")
            self.transformer_model = None
            self.transformer_tokenizer = None

    def detect_anomalies(self, entries: List[Dict[str, Any]], 
                        method: str = "isolation_forest") -> List[AnomalyDetectionResult]:
        """
        Detect anomalies in content entries using advanced ML methods
        
        Args:
            entries: List of content entries with metrics
            method: Anomaly detection method ('isolation_forest', 'dbscan', 'statistical')
        
        Returns:
            List of anomaly detection results
        """
        try:
            # Prepare features
            features = self._extract_features_for_anomaly_detection(entries)
            
            if method == "isolation_forest":
                return self._detect_anomalies_isolation_forest(features, entries)
            elif method == "dbscan":
                return self._detect_anomalies_dbscan(features, entries)
            elif method == "statistical":
                return self._detect_anomalies_statistical(features, entries)
            else:
                raise ValueError(f"Unknown anomaly detection method: {method}")
                
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return []

    def _extract_features_for_anomaly_detection(self, entries: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features for anomaly detection"""
        features = []
        
        for entry in entries:
            feature_vector = [
                entry.get('readability_score', 0),
                entry.get('sentiment_score', 0),
                entry.get('word_count', 0),
                entry.get('sentence_count', 0),
                entry.get('avg_word_length', 0),
                entry.get('complexity_score', 0),
                entry.get('topic_diversity', 0),
                entry.get('consistency_score', 0)
            ]
            features.append(feature_vector)
        
        return np.array(features)

    def _detect_anomalies_isolation_forest(self, features: np.ndarray, 
                                         entries: List[Dict[str, Any]]) -> List[AnomalyDetectionResult]:
        """Detect anomalies using Isolation Forest"""
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Train Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(features_scaled)
        anomaly_scores = iso_forest.decision_function(features_scaled)
        
        results = []
        for i, (entry, label, score) in enumerate(zip(entries, anomaly_labels, anomaly_scores)):
            is_anomaly = label == -1
            confidence = abs(score)
            
            # Determine anomaly type
            anomaly_type = self._classify_anomaly_type(entry, features[i])
            
            # Generate explanation and recommendations
            explanation, recommendations = self._generate_anomaly_explanation(entry, anomaly_type)
            
            results.append(AnomalyDetectionResult(
                is_anomaly=is_anomaly,
                anomaly_score=float(score),
                confidence=float(confidence),
                anomaly_type=anomaly_type,
                explanation=explanation,
                recommendations=recommendations
            ))
        
        return results

    def _detect_anomalies_dbscan(self, features: np.ndarray, 
                                entries: List[Dict[str, Any]]) -> List[AnomalyDetectionResult]:
        """Detect anomalies using DBSCAN clustering"""
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = dbscan.fit_predict(features_scaled)
        
        results = []
        for i, (entry, label) in enumerate(zip(entries, cluster_labels)):
            is_anomaly = label == -1  # DBSCAN marks outliers as -1
            
            # Calculate anomaly score based on distance to nearest cluster
            if is_anomaly:
                # Calculate distance to nearest cluster center
                non_outlier_features = features_scaled[cluster_labels != -1]
                if len(non_outlier_features) > 0:
                    distances = np.linalg.norm(non_outlier_features - features_scaled[i], axis=1)
                    anomaly_score = -np.min(distances)  # Negative for outliers
                else:
                    anomaly_score = -1.0
            else:
                anomaly_score = 0.0
            
            anomaly_type = self._classify_anomaly_type(entry, features[i])
            explanation, recommendations = self._generate_anomaly_explanation(entry, anomaly_type)
            
            results.append(AnomalyDetectionResult(
                is_anomaly=is_anomaly,
                anomaly_score=float(anomaly_score),
                confidence=float(abs(anomaly_score)),
                anomaly_type=anomaly_type,
                explanation=explanation,
                recommendations=recommendations
            ))
        
        return results

    def _detect_anomalies_statistical(self, features: np.ndarray, 
                                     entries: List[Dict[str, Any]]) -> List[AnomalyDetectionResult]:
        """Detect anomalies using statistical methods"""
        results = []
        
        for i, (entry, feature_vector) in enumerate(zip(entries, features)):
            # Calculate z-scores for each feature
            z_scores = []
            for j, feature_value in enumerate(feature_vector):
                feature_values = features[:, j]
                mean_val = np.mean(feature_values)
                std_val = np.std(feature_values)
                if std_val > 0:
                    z_score = abs((feature_value - mean_val) / std_val)
                else:
                    z_score = 0
                z_scores.append(z_score)
            
            # Anomaly if any z-score > 3 (3-sigma rule)
            max_z_score = max(z_scores)
            is_anomaly = max_z_score > 3.0
            
            anomaly_type = self._classify_anomaly_type(entry, feature_vector)
            explanation, recommendations = self._generate_anomaly_explanation(entry, anomaly_type)
            
            results.append(AnomalyDetectionResult(
                is_anomaly=is_anomaly,
                anomaly_score=float(max_z_score),
                confidence=float(min(max_z_score / 3.0, 1.0)),
                anomaly_type=anomaly_type,
                explanation=explanation,
                recommendations=recommendations
            ))
        
        return results

    def _classify_anomaly_type(self, entry: Dict[str, Any], features: np.ndarray) -> str:
        """Classify the type of anomaly"""
        readability = entry.get('readability_score', 0)
        sentiment = entry.get('sentiment_score', 0)
        word_count = entry.get('word_count', 0)
        complexity = entry.get('complexity_score', 0)
        
        if readability < 20:
            return "extremely_difficult"
        elif readability > 90:
            return "extremely_simple"
        elif abs(sentiment) > 0.8:
            return "extreme_sentiment"
        elif word_count < 20:
            return "extremely_short"
        elif word_count > 2000:
            return "extremely_long"
        elif complexity > 0.9:
            return "extremely_complex"
        elif complexity < 0.1:
            return "extremely_simple"
        else:
            return "statistical_outlier"

    def _generate_anomaly_explanation(self, entry: Dict[str, Any], anomaly_type: str) -> Tuple[str, List[str]]:
        """Generate explanation and recommendations for anomalies"""
        explanations = {
            "extremely_difficult": "Content has extremely low readability score",
            "extremely_simple": "Content has extremely high readability score",
            "extreme_sentiment": "Content has extreme positive or negative sentiment",
            "extremely_short": "Content is extremely short",
            "extremely_long": "Content is extremely long",
            "extremely_complex": "Content has extremely high complexity",
            "statistical_outlier": "Content is a statistical outlier in multiple metrics"
        }
        
        recommendations = {
            "extremely_difficult": [
                "Simplify language and sentence structure",
                "Use shorter sentences and common words",
                "Consider breaking into smaller sections"
            ],
            "extremely_simple": [
                "Add more sophisticated vocabulary",
                "Increase sentence complexity",
                "Include more detailed explanations"
            ],
            "extreme_sentiment": [
                "Balance emotional tone",
                "Consider audience appropriateness",
                "Review sentiment alignment with brand voice"
            ],
            "extremely_short": [
                "Add more detail and context",
                "Expand on key points",
                "Include examples or explanations"
            ],
            "extremely_long": [
                "Break into smaller sections",
                "Remove redundant information",
                "Focus on key messages"
            ],
            "extremely_complex": [
                "Simplify technical language",
                "Add explanations for complex terms",
                "Use more accessible vocabulary"
            ],
            "statistical_outlier": [
                "Review content against quality standards",
                "Compare with similar successful content",
                "Consider content revision"
            ]
        }
        
        return explanations.get(anomaly_type, "Content shows unusual characteristics"), \
               recommendations.get(anomaly_type, ["Review content quality"])

    def advanced_clustering(self, entries: List[Dict[str, Any]], 
                           algorithm: str = "auto", 
                           max_clusters: int = 10) -> AdvancedClusteringResult:
        """
        Perform advanced clustering with multiple algorithms and optimization
        
        Args:
            entries: List of content entries
            algorithm: Clustering algorithm ('auto', 'kmeans', 'dbscan', 'agglomerative', 'spectral')
            max_clusters: Maximum number of clusters to test
        
        Returns:
            Advanced clustering result with metrics and analysis
        """
        try:
            # Extract features
            features = self._extract_features_for_clustering(entries)
            content_texts = [entry.get('content', '') for entry in entries]
            
            if algorithm == "auto":
                return self._auto_clustering(features, content_texts, entries, max_clusters)
            else:
                return self._specific_clustering(features, content_texts, entries, algorithm, max_clusters)
                
        except Exception as e:
            logger.error(f"Error in advanced clustering: {e}")
            return self._create_empty_clustering_result()

    def _extract_features_for_clustering(self, entries: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features for clustering"""
        features = []
        
        for entry in entries:
            feature_vector = [
                entry.get('readability_score', 0),
                entry.get('sentiment_score', 0),
                entry.get('word_count', 0) / 1000.0,  # Normalize
                entry.get('sentence_count', 0),
                entry.get('avg_word_length', 0),
                entry.get('complexity_score', 0),
                entry.get('topic_diversity', 0),
                entry.get('consistency_score', 0)
            ]
            features.append(feature_vector)
        
        return np.array(features)

    def _auto_clustering(self, features: np.ndarray, content_texts: List[str], 
                        entries: List[Dict[str, Any]], max_clusters: int) -> AdvancedClusteringResult:
        """Automatically select best clustering algorithm and parameters"""
        best_result = None
        best_score = -1
        best_algorithm = "kmeans"
        
        algorithms = ["kmeans", "agglomerative", "spectral"]
        
        for algorithm in algorithms:
            try:
                result = self._specific_clustering(features, content_texts, entries, algorithm, max_clusters)
                if result.silhouette_score > best_score:
                    best_score = result.silhouette_score
                    best_result = result
                    best_algorithm = algorithm
            except Exception as e:
                logger.warning(f"Failed to run {algorithm} clustering: {e}")
                continue
        
        if best_result is None:
            return self._create_empty_clustering_result()
        
        best_result.algorithm_used = best_algorithm
        return best_result

    def _specific_clustering(self, features: np.ndarray, content_texts: List[str], 
                           entries: List[Dict[str, Any]], algorithm: str, 
                           max_clusters: int) -> AdvancedClusteringResult:
        """Perform clustering with specific algorithm"""
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Find optimal number of clusters
        optimal_clusters = self._find_optimal_clusters(features_scaled, max_clusters)
        
        # Perform clustering
        if algorithm == "kmeans":
            from sklearn.cluster import KMeans
            clusterer = KMeans(n_clusters=optimal_clusters, random_state=42)
        elif algorithm == "agglomerative":
            clusterer = AgglomerativeClustering(n_clusters=optimal_clusters)
        elif algorithm == "spectral":
            clusterer = SpectralClustering(n_clusters=optimal_clusters, random_state=42)
        elif algorithm == "dbscan":
            clusterer = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")
        
        cluster_labels = clusterer.fit_predict(features_scaled)
        
        # Calculate metrics
        if len(set(cluster_labels)) > 1:  # Need at least 2 clusters
            silhouette = silhouette_score(features_scaled, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(features_scaled, cluster_labels)
        else:
            silhouette = 0.0
            calinski_harabasz = 0.0
        
        # Organize clusters
        clusters = {}
        cluster_centers = {}
        
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
                cluster_centers[label] = {}
            clusters[label].append(entries[i].get('id', f'entry_{i}'))
        
        # Calculate cluster centers
        for cluster_id, cluster_entries in clusters.items():
            cluster_features = [features[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            if cluster_features:
                center = np.mean(cluster_features, axis=0)
                cluster_centers[cluster_id] = {
                    'readability_score': float(center[0]),
                    'sentiment_score': float(center[1]),
                    'word_count': float(center[2] * 1000),
                    'sentence_count': float(center[3]),
                    'avg_word_length': float(center[4]),
                    'complexity_score': float(center[5]),
                    'topic_diversity': float(center[6]),
                    'consistency_score': float(center[7])
                }
        
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(features_scaled, cluster_labels)
        
        return AdvancedClusteringResult(
            clusters=clusters,
            cluster_centers=cluster_centers,
            silhouette_score=float(silhouette),
            calinski_harabasz_score=float(calinski_harabasz),
            optimal_clusters=optimal_clusters,
            algorithm_used=algorithm,
            feature_importance=feature_importance
        )

    def _find_optimal_clusters(self, features: np.ndarray, max_clusters: int) -> int:
        """Find optimal number of clusters using elbow method and silhouette analysis"""
        if len(features) < 2:
            return 1
        
        max_clusters = min(max_clusters, len(features) - 1)
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=k, random_state=42)
                cluster_labels = kmeans.fit_predict(features)
                silhouette = silhouette_score(features, cluster_labels)
                silhouette_scores.append(silhouette)
            except:
                silhouette_scores.append(0)
        
        if silhouette_scores:
            optimal_k = np.argmax(silhouette_scores) + 2
            return optimal_k
        else:
            return 2

    def _calculate_feature_importance(self, features: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance for clustering"""
        feature_names = [
            'readability_score', 'sentiment_score', 'word_count', 'sentence_count',
            'avg_word_length', 'complexity_score', 'topic_diversity', 'consistency_score'
        ]
        
        # Use variance ratio to measure feature importance
        from sklearn.feature_selection import f_classif
        
        try:
            f_scores, _ = f_classif(features, cluster_labels)
            # Normalize scores
            f_scores = f_scores / np.sum(f_scores)
            
            importance = {}
            for i, name in enumerate(feature_names):
                importance[name] = float(f_scores[i])
            
            return importance
        except:
            # Fallback: equal importance
            return {name: 1.0 / len(feature_names) for name in feature_names}

    def _create_empty_clustering_result(self) -> AdvancedClusteringResult:
        """Create empty clustering result for error cases"""
        return AdvancedClusteringResult(
            clusters={},
            cluster_centers={},
            silhouette_score=0.0,
            calinski_harabasz_score=0.0,
            optimal_clusters=1,
            algorithm_used="none",
            feature_importance={}
        )

    def build_predictive_models(self, entries: List[Dict[str, Any]], 
                               target_metric: str = "readability_score",
                               model_types: List[str] = None) -> Dict[str, PredictiveModelResult]:
        """
        Build predictive models for content metrics
        
        Args:
            entries: List of content entries with metrics
            target_metric: Metric to predict
            model_types: List of model types to train
        
        Returns:
            Dictionary of trained models and their results
        """
        if model_types is None:
            model_types = ["linear_regression", "random_forest", "neural_network"]
        
        try:
            # Prepare data
            X, y = self._prepare_prediction_data(entries, target_metric)
            
            if len(X) < 10:  # Need sufficient data
                logger.warning("Insufficient data for predictive modeling")
                return {}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            results = {}
            
            for model_type in model_types:
                try:
                    result = self._train_specific_model(X_train, X_test, y_train, y_test, model_type, target_metric)
                    results[model_type] = result
                except Exception as e:
                    logger.warning(f"Failed to train {model_type}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error in predictive modeling: {e}")
            return {}

    def _prepare_prediction_data(self, entries: List[Dict[str, Any]], target_metric: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for predictive modeling"""
        features = []
        targets = []
        
        for entry in entries:
            # Extract features (exclude target metric)
            feature_vector = [
                entry.get('sentiment_score', 0),
                entry.get('word_count', 0) / 1000.0,
                entry.get('sentence_count', 0),
                entry.get('avg_word_length', 0),
                entry.get('complexity_score', 0),
                entry.get('topic_diversity', 0),
                entry.get('consistency_score', 0)
            ]
            
            # Add time-based features if available
            if 'timestamp' in entry:
                timestamp = entry['timestamp']
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                feature_vector.extend([
                    timestamp.hour / 24.0,
                    timestamp.weekday() / 7.0,
                    timestamp.month / 12.0
                ])
            else:
                feature_vector.extend([0.0, 0.0, 0.0])
            
            features.append(feature_vector)
            targets.append(entry.get(target_metric, 0))
        
        return np.array(features), np.array(targets)

    def _train_specific_model(self, X_train: np.ndarray, X_test: np.ndarray, 
                             y_train: np.ndarray, y_test: np.ndarray, 
                             model_type: str, target_metric: str) -> PredictiveModelResult:
        """Train a specific model type"""
        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if model_type == "linear_regression":
            model = LinearRegression()
        elif model_type == "ridge_regression":
            model = Ridge(alpha=1.0)
        elif model_type == "lasso_regression":
            model = Lasso(alpha=0.1)
        elif model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "neural_network":
            model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        elif model_type == "svr":
            model = SVR(kernel='rbf')
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Calculate confidence intervals (simple approximation)
        residuals = y_test - y_pred
        std_residuals = np.std(residuals)
        confidence_intervals = [
            (pred - 1.96 * std_residuals, pred + 1.96 * std_residuals)
            for pred in y_pred
        ]
        
        # Calculate feature importance
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            feature_names = [
                'sentiment_score', 'word_count', 'sentence_count', 'avg_word_length',
                'complexity_score', 'topic_diversity', 'consistency_score',
                'hour', 'weekday', 'month'
            ]
            for i, importance in enumerate(model.feature_importances_):
                if i < len(feature_names):
                    feature_importance[feature_names[i]] = float(importance)
        elif hasattr(model, 'coef_'):
            feature_names = [
                'sentiment_score', 'word_count', 'sentence_count', 'avg_word_length',
                'complexity_score', 'topic_diversity', 'consistency_score',
                'hour', 'weekday', 'month'
            ]
            for i, coef in enumerate(model.coef_):
                if i < len(feature_names):
                    feature_importance[feature_names[i]] = float(abs(coef))
        
        # Determine trend direction
        if len(y_pred) > 1:
            trend_slope = np.polyfit(range(len(y_pred)), y_pred, 1)[0]
            trend_direction = "increasing" if trend_slope > 0 else "decreasing" if trend_slope < 0 else "stable"
        else:
            trend_direction = "stable"
        
        # Next prediction (simple extrapolation)
        if len(y_pred) > 0:
            next_prediction = float(y_pred[-1] + (y_pred[-1] - y_pred[-2]) if len(y_pred) > 1 else y_pred[-1])
        else:
            next_prediction = 0.0
        
        return PredictiveModelResult(
            predictions=[float(p) for p in y_pred],
            confidence_intervals=confidence_intervals,
            model_accuracy=float(r2),
            feature_importance=feature_importance,
            trend_direction=trend_direction,
            next_prediction=next_prediction,
            model_type=model_type
        )

    def extract_advanced_features(self, content: str) -> Dict[str, Any]:
        """Extract advanced features using transformer models and advanced NLP"""
        features = {}
        
        # Basic features
        features['length'] = len(content)
        features['word_count'] = len(content.split())
        features['sentence_count'] = len(content.split('.'))
        
        # Advanced NLP features if spaCy is available
        if self.nlp:
            doc = self.nlp(content)
            
            # Named entities
            features['named_entities'] = len(doc.ents)
            features['entity_types'] = len(set([ent.label_ for ent in doc.ents]))
            
            # POS tags
            pos_counts = {}
            for token in doc:
                pos = token.pos_
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
            
            features['noun_ratio'] = pos_counts.get('NOUN', 0) / len(doc) if len(doc) > 0 else 0
            features['verb_ratio'] = pos_counts.get('VERB', 0) / len(doc) if len(doc) > 0 else 0
            features['adj_ratio'] = pos_counts.get('ADJ', 0) / len(doc) if len(doc) > 0 else 0
            
            # Dependency parsing
            features['avg_dependency_depth'] = np.mean([len(list(token.ancestors)) for token in doc])
            
            # Noun phrases
            features['noun_phrases'] = len(list(doc.noun_chunks))
        
        # Transformer-based features if available
        if self.transformer_model and self.transformer_tokenizer:
            try:
                # Get sentence embeddings
                inputs = self.transformer_tokenizer(content, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.transformer_model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                
                # Use first few dimensions as features
                for i in range(min(5, len(embeddings))):
                    features[f'embedding_dim_{i}'] = float(embeddings[i])
                    
            except Exception as e:
                logger.warning(f"Failed to extract transformer features: {e}")
        
        return features

    def save_model(self, model_name: str, model: Any, config: MLModelConfig):
        """Save a trained model to disk"""
        try:
            model_path = self.model_storage_path / f"{model_name}.pkl"
            config_path = self.model_storage_path / f"{model_name}_config.json"
            
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save config
            config_dict = {
                'model_type': config.model_type,
                'parameters': config.parameters,
                'training_data_size': config.training_data_size,
                'accuracy_score': config.accuracy_score,
                'last_trained': config.last_trained.isoformat(),
                'version': config.version
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Model {model_name} saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {e}")

    def load_model(self, model_name: str) -> Tuple[Any, MLModelConfig]:
        """Load a trained model from disk"""
        try:
            model_path = self.model_storage_path / f"{model_name}.pkl"
            config_path = self.model_storage_path / f"{model_name}_config.json"
            
            if not model_path.exists() or not config_path.exists():
                raise FileNotFoundError(f"Model {model_name} not found")
            
            # Load model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load config
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            config = MLModelConfig(
                model_type=config_dict['model_type'],
                parameters=config_dict['parameters'],
                training_data_size=config_dict['training_data_size'],
                accuracy_score=config_dict['accuracy_score'],
                last_trained=datetime.fromisoformat(config_dict['last_trained']),
                version=config_dict['version']
            )
            
            logger.info(f"Model {model_name} loaded successfully")
            return model, config
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None, None

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        model_info = {}
        
        for model_file in self.model_storage_path.glob("*.pkl"):
            model_name = model_file.stem
            config_file = self.model_storage_path / f"{model_name}_config.json"
            
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    model_info[model_name] = config
                except Exception as e:
                    logger.warning(f"Failed to load config for {model_name}: {e}")
        
        return model_info

    def cleanup_old_models(self, max_age_days: int = 30):
        """Clean up old model files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            removed_count = 0
            
            for model_file in self.model_storage_path.glob("*.pkl"):
                config_file = self.model_storage_path / f"{model_file.stem}_config.json"
                
                if config_file.exists():
                    try:
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                        
                        last_trained = datetime.fromisoformat(config['last_trained'])
                        
                        if last_trained < cutoff_date:
                            model_file.unlink()
                            config_file.unlink()
                            removed_count += 1
                            
                    except Exception as e:
                        logger.warning(f"Failed to check age of {model_file.stem}: {e}")
            
            logger.info(f"Cleaned up {removed_count} old models")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old models: {e}")


# Global ML engine instance
ml_engine = AdvancedMLEngine()

# Convenience functions
def detect_anomalies(entries: List[Dict[str, Any]], method: str = "isolation_forest") -> List[AnomalyDetectionResult]:
    """Detect anomalies in content entries"""
    return ml_engine.detect_anomalies(entries, method)

def advanced_clustering(entries: List[Dict[str, Any]], algorithm: str = "auto", max_clusters: int = 10) -> AdvancedClusteringResult:
    """Perform advanced clustering analysis"""
    return ml_engine.advanced_clustering(entries, algorithm, max_clusters)

def build_predictive_models(entries: List[Dict[str, Any]], target_metric: str = "readability_score", model_types: List[str] = None) -> Dict[str, PredictiveModelResult]:
    """Build predictive models for content metrics"""
    return ml_engine.build_predictive_models(entries, target_metric, model_types)

def extract_advanced_features(content: str) -> Dict[str, Any]:
    """Extract advanced features from content"""
    return ml_engine.extract_advanced_features(content)



























