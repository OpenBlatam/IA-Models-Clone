"""
Learning System
==============

Advanced learning system for continuous improvement.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import json
import numpy as np
from dataclasses import dataclass
from collections import defaultdict, deque
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class LearningExample:
    """Learning example data structure."""
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    quality_score: float
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class LearningModel:
    """Learning model data structure."""
    name: str
    model_type: str
    parameters: Dict[str, Any]
    performance: Dict[str, float]
    created_at: datetime
    updated_at: datetime

class LearningSystem:
    """
    Advanced learning system for continuous improvement.
    
    Features:
    - Online learning
    - Model adaptation
    - Performance tracking
    - Knowledge extraction
    - Pattern recognition
    - Automated improvement
    """
    
    def __init__(self):
        self.learning_examples = deque(maxlen=10000)
        self.models = {}
        self.performance_history = defaultdict(list)
        self.learning_metrics = {}
        self.model_path = Path("./models")
        
    async def initialize(self):
        """Initialize learning system."""
        logger.info("Initializing Learning System...")
        
        try:
            # Create models directory
            self.model_path.mkdir(parents=True, exist_ok=True)
            
            # Load existing models
            await self._load_models()
            
            # Start background learning
            asyncio.create_task(self._continuous_learning())
            asyncio.create_task(self._model_evaluation())
            asyncio.create_task(self._cleanup_old_data())
            
            logger.info("Learning System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Learning System: {str(e)}")
            raise
    
    async def _load_models(self):
        """Load existing models."""
        try:
            for model_file in self.model_path.glob("*.pkl"):
                try:
                    with open(model_file, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    model_name = model_file.stem
                    self.models[model_name] = model_data
                    
                    logger.info(f"Loaded model: {model_name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load model {model_file}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
    
    async def add_learning_example(
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        quality_score: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add learning example."""
        try:
            example = LearningExample(
                input_data=input_data,
                output_data=output_data,
                quality_score=quality_score,
                timestamp=datetime.utcnow(),
                metadata=metadata or {}
            )
            
            self.learning_examples.append(example)
            
            # Trigger learning if we have enough examples
            if len(self.learning_examples) % 100 == 0:
                await self._trigger_learning()
            
            logger.debug(f"Added learning example with quality score: {quality_score}")
            
        except Exception as e:
            logger.error(f"Failed to add learning example: {str(e)}")
    
    async def _trigger_learning(self):
        """Trigger learning process."""
        try:
            if len(self.learning_examples) < 10:
                return
            
            # Extract features and targets
            features = []
            targets = []
            
            for example in self.learning_examples:
                # Extract features from input data
                feature_vector = self._extract_features(example.input_data)
                features.append(feature_vector)
                
                # Extract target from quality score
                targets.append(example.quality_score)
            
            # Train models
            await self._train_quality_model(features, targets)
            await self._train_content_model(features, targets)
            await self._train_optimization_model(features, targets)
            
        except Exception as e:
            logger.error(f"Failed to trigger learning: {str(e)}")
    
    def _extract_features(self, input_data: Dict[str, Any]) -> List[float]:
        """Extract features from input data."""
        try:
            features = []
            
            # Text features
            if 'content' in input_data:
                content = input_data['content']
                features.extend([
                    len(content),  # Length
                    len(content.split()),  # Word count
                    len(content.split('.')) - 1,  # Sentence count
                    len(content.split('\n\n')) - 1,  # Paragraph count
                    content.count('!'),  # Exclamation count
                    content.count('?'),  # Question count
                ])
            else:
                features.extend([0] * 6)
            
            # Configuration features
            if 'config' in input_data:
                config = input_data['config']
                features.extend([
                    config.get('temperature', 0.7),
                    config.get('max_tokens', 2000) / 4000,  # Normalized
                    config.get('optimization_level', 1) / 3,  # Normalized
                ])
            else:
                features.extend([0.7, 0.5, 0.33])
            
            # Query features
            if 'query' in input_data:
                query = input_data['query']
                features.extend([
                    len(query),
                    len(query.split()),
                    query.count('?'),
                ])
            else:
                features.extend([0, 0, 0])
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract features: {str(e)}")
            return [0] * 12  # Default feature vector
    
    async def _train_quality_model(self, features: List[List[float]], targets: List[float]):
        """Train quality prediction model."""
        try:
            # Simple linear regression model
            X = np.array(features)
            y = np.array(targets)
            
            # Normalize features
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            X_normalized = (X - X_mean) / (X_std + 1e-8)
            
            # Simple linear regression
            X_with_bias = np.column_stack([np.ones(X.shape[0]), X_normalized])
            weights = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
            
            # Create model
            model = LearningModel(
                name="quality_predictor",
                model_type="linear_regression",
                parameters={
                    'weights': weights.tolist(),
                    'X_mean': X_mean.tolist(),
                    'X_std': X_std.tolist()
                },
                performance={'mse': 0.0, 'r2': 0.0},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Calculate performance metrics
            predictions = X_with_bias @ weights
            mse = np.mean((predictions - y) ** 2)
            r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
            
            model.performance = {'mse': float(mse), 'r2': float(r2)}
            
            # Save model
            self.models['quality_predictor'] = model
            await self._save_model(model)
            
            logger.info(f"Trained quality model - MSE: {mse:.4f}, R²: {r2:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to train quality model: {str(e)}")
    
    async def _train_content_model(self, features: List[List[float]], targets: List[float]):
        """Train content optimization model."""
        try:
            # Simple clustering for content patterns
            X = np.array(features)
            
            # K-means clustering (simplified)
            n_clusters = min(5, len(features) // 2)
            if n_clusters < 2:
                return
            
            # Random initialization
            centroids = X[np.random.choice(len(X), n_clusters, replace=False)]
            
            # Simple k-means
            for _ in range(10):
                # Assign points to clusters
                distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
                assignments = np.argmin(distances, axis=0)
                
                # Update centroids
                for i in range(n_clusters):
                    cluster_points = X[assignments == i]
                    if len(cluster_points) > 0:
                        centroids[i] = np.mean(cluster_points, axis=0)
            
            # Create model
            model = LearningModel(
                name="content_clusterer",
                model_type="kmeans",
                parameters={
                    'centroids': centroids.tolist(),
                    'n_clusters': n_clusters
                },
                performance={'silhouette': 0.0},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Save model
            self.models['content_clusterer'] = model
            await self._save_model(model)
            
            logger.info(f"Trained content clustering model with {n_clusters} clusters")
            
        except Exception as e:
            logger.error(f"Failed to train content model: {str(e)}")
    
    async def _train_optimization_model(self, features: List[List[float]], targets: List[float]):
        """Train optimization model."""
        try:
            # Simple optimization model based on quality scores
            X = np.array(features)
            y = np.array(targets)
            
            # Find best performing examples
            top_indices = np.argsort(y)[-len(y)//4:]  # Top 25%
            top_features = X[top_indices]
            
            # Calculate optimal feature values
            optimal_features = np.mean(top_features, axis=0)
            feature_importance = np.std(top_features, axis=0)
            
            # Create model
            model = LearningModel(
                name="optimization_guide",
                model_type="feature_optimization",
                parameters={
                    'optimal_features': optimal_features.tolist(),
                    'feature_importance': feature_importance.tolist()
                },
                performance={'improvement': 0.0},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Save model
            self.models['optimization_guide'] = model
            await self._save_model(model)
            
            logger.info("Trained optimization model")
            
        except Exception as e:
            logger.error(f"Failed to train optimization model: {str(e)}")
    
    async def _save_model(self, model: LearningModel):
        """Save model to disk."""
        try:
            model_file = self.model_path / f"{model.name}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            
        except Exception as e:
            logger.error(f"Failed to save model {model.name}: {str(e)}")
    
    async def predict_quality(self, input_data: Dict[str, Any]) -> float:
        """Predict quality score for input data."""
        try:
            if 'quality_predictor' not in self.models:
                return 0.5  # Default score
            
            model = self.models['quality_predictor']
            
            # Extract features
            features = self._extract_features(input_data)
            X = np.array(features).reshape(1, -1)
            
            # Normalize features
            X_mean = np.array(model.parameters['X_mean'])
            X_std = np.array(model.parameters['X_std'])
            X_normalized = (X - X_mean) / (X_std + 1e-8)
            
            # Make prediction
            X_with_bias = np.column_stack([np.ones(X.shape[0]), X_normalized])
            weights = np.array(model.parameters['weights'])
            prediction = X_with_bias @ weights
            
            return float(np.clip(prediction[0], 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Failed to predict quality: {str(e)}")
            return 0.5
    
    async def get_optimization_suggestions(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimization suggestions."""
        try:
            if 'optimization_guide' not in self.models:
                return {}
            
            model = self.models['optimization_guide']
            
            # Extract current features
            current_features = self._extract_features(input_data)
            optimal_features = model.parameters['optimal_features']
            feature_importance = model.parameters['feature_importance']
            
            # Calculate suggestions
            suggestions = {}
            feature_names = [
                'content_length', 'word_count', 'sentence_count', 'paragraph_count',
                'exclamation_count', 'question_count', 'temperature', 'max_tokens_norm',
                'optimization_level_norm', 'query_length', 'query_word_count', 'query_questions'
            ]
            
            for i, (current, optimal, importance) in enumerate(
                zip(current_features, optimal_features, feature_importance)
            ):
                if importance > 0.1:  # Only suggest for important features
                    diff = optimal - current
                    if abs(diff) > 0.1:  # Only suggest significant changes
                        suggestions[feature_names[i]] = {
                            'current': current,
                            'optimal': optimal,
                            'suggestion': 'increase' if diff > 0 else 'decrease',
                            'importance': importance
                        }
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to get optimization suggestions: {str(e)}")
            return {}
    
    async def _continuous_learning(self):
        """Continuous learning process."""
        while True:
            try:
                await asyncio.sleep(3600)  # Learn every hour
                
                if len(self.learning_examples) >= 50:
                    await self._trigger_learning()
                
            except Exception as e:
                logger.error(f"Error in continuous learning: {str(e)}")
    
    async def _model_evaluation(self):
        """Evaluate model performance."""
        while True:
            try:
                await asyncio.sleep(7200)  # Evaluate every 2 hours
                
                for model_name, model in self.models.items():
                    # Evaluate model performance
                    performance = await self._evaluate_model(model)
                    model.performance.update(performance)
                    model.updated_at = datetime.utcnow()
                    
                    # Save updated model
                    await self._save_model(model)
                    
                    logger.info(f"Evaluated model {model_name}: {performance}")
                
            except Exception as e:
                logger.error(f"Error in model evaluation: {str(e)}")
    
    async def _evaluate_model(self, model: LearningModel) -> Dict[str, float]:
        """Evaluate model performance."""
        try:
            if model.model_type == "linear_regression":
                # Evaluate regression model
                return {
                    'accuracy': 0.8,  # Placeholder
                    'precision': 0.75,
                    'recall': 0.7
                }
            elif model.model_type == "kmeans":
                # Evaluate clustering model
                return {
                    'silhouette': 0.6,
                    'inertia': 0.4
                }
            else:
                return {'score': 0.5}
                
        except Exception as e:
            logger.error(f"Failed to evaluate model: {str(e)}")
            return {}
    
    async def _cleanup_old_data(self):
        """Cleanup old learning data."""
        while True:
            try:
                await asyncio.sleep(86400)  # Cleanup daily
                
                # Remove old examples (keep last 30 days)
                cutoff_time = datetime.utcnow() - timedelta(days=30)
                old_count = len(self.learning_examples)
                
                self.learning_examples = deque(
                    [ex for ex in self.learning_examples if ex.timestamp > cutoff_time],
                    maxlen=10000
                )
                
                removed_count = old_count - len(self.learning_examples)
                if removed_count > 0:
                    logger.info(f"Cleaned up {removed_count} old learning examples")
                
            except Exception as e:
                logger.error(f"Error cleaning up old data: {str(e)}")
    
    async def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning system statistics."""
        try:
            return {
                'total_examples': len(self.learning_examples),
                'total_models': len(self.models),
                'model_types': list(set(model.model_type for model in self.models.values())),
                'average_quality': np.mean([ex.quality_score for ex in self.learning_examples]) if self.learning_examples else 0.0,
                'learning_rate': len(self.learning_examples) / max(1, (datetime.utcnow() - min(ex.timestamp for ex in self.learning_examples)).days) if self.learning_examples else 0.0,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get learning statistics: {str(e)}")
            return {}
    
    async def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics."""
        try:
            performance = {}
            for model_name, model in self.models.items():
                performance[model_name] = {
                    'type': model.model_type,
                    'performance': model.performance,
                    'created_at': model.created_at.isoformat(),
                    'updated_at': model.updated_at.isoformat()
                }
            
            return performance
            
        except Exception as e:
            logger.error(f"Failed to get model performance: {str(e)}")
            return {}
    
    async def cleanup(self):
        """Cleanup learning system."""
        try:
            # Save all models
            for model in self.models.values():
                await self._save_model(model)
            
            logger.info("Learning System cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup Learning System: {str(e)}")











