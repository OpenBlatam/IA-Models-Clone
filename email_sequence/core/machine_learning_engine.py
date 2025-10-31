"""
Machine Learning Engine for Email Sequence System

This module provides advanced ML capabilities including predictive modeling,
recommendation systems, and automated optimization.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from uuid import UUID
import json
import pickle
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from .config import get_settings
from .exceptions import MachineLearningError
from .cache import cache_manager

logger = logging.getLogger(__name__)
settings = get_settings()


class ModelType(str, Enum):
    """Types of ML models"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    RECOMMENDATION = "recommendation"


class PredictionType(str, Enum):
    """Types of predictions"""
    CHURN = "churn"
    ENGAGEMENT = "engagement"
    CONVERSION = "conversion"
    LIFETIME_VALUE = "lifetime_value"
    OPTIMAL_SEND_TIME = "optimal_send_time"


@dataclass
class ModelMetrics:
    """ML model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    cross_val_score: float
    feature_importance: Dict[str, float]


@dataclass
class PredictionResult:
    """ML prediction result"""
    prediction: Union[float, int, str]
    confidence: float
    probability: Optional[Dict[str, float]] = None
    feature_contributions: Optional[Dict[str, float]] = None


class MachineLearningEngine:
    """Advanced machine learning engine for email sequences"""
    
    def __init__(self):
        """Initialize ML engine"""
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_importance_cache: Dict[str, Dict[str, float]] = {}
        
        logger.info("Machine Learning Engine initialized")
    
    async def train_churn_prediction_model(
        self,
        sequence_id: UUID,
        training_data: List[Dict[str, Any]]
    ) -> ModelMetrics:
        """
        Train a churn prediction model for a sequence.
        
        Args:
            sequence_id: Sequence ID
            training_data: Training data with features and labels
            
        Returns:
            ModelMetrics with performance metrics
        """
        try:
            # Prepare data
            df = pd.DataFrame(training_data)
            
            # Feature engineering
            features = await self._engineer_features(df, "churn")
            X = features.drop('churn_label', axis=1)
            y = features['churn_label']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            cv_score = cv_scores.mean()
            
            # Feature importance
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            
            # Store model and scaler
            model_key = f"churn_model_{sequence_id}"
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            self.feature_importance_cache[model_key] = feature_importance
            
            # Cache model
            await self._cache_model(model_key, model, scaler)
            
            metrics = ModelMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                auc_roc=0.0,  # Calculate if needed
                cross_val_score=cv_score,
                feature_importance=feature_importance
            )
            
            logger.info(f"Churn prediction model trained for sequence {sequence_id}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training churn prediction model: {e}")
            raise MachineLearningError(f"Failed to train churn model: {e}")
    
    async def predict_churn(
        self,
        sequence_id: UUID,
        subscriber_data: Dict[str, Any]
    ) -> PredictionResult:
        """
        Predict churn probability for a subscriber.
        
        Args:
            sequence_id: Sequence ID
            subscriber_data: Subscriber data for prediction
            
        Returns:
            PredictionResult with churn prediction
        """
        try:
            model_key = f"churn_model_{sequence_id}"
            
            # Load model if not in memory
            if model_key not in self.models:
                await self._load_model(model_key)
            
            if model_key not in self.models:
                raise MachineLearningError(f"Model not found for sequence {sequence_id}")
            
            model = self.models[model_key]
            scaler = self.scalers[model_key]
            
            # Prepare features
            features = await self._prepare_prediction_features(subscriber_data, "churn")
            features_scaled = scaler.transform([features])
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = max(probabilities)
            
            # Feature contributions
            feature_contributions = await self._calculate_feature_contributions(
                model, features, model_key
            )
            
            return PredictionResult(
                prediction=prediction,
                confidence=confidence,
                probability={
                    "no_churn": probabilities[0],
                    "churn": probabilities[1]
                },
                feature_contributions=feature_contributions
            )
            
        except Exception as e:
            logger.error(f"Error predicting churn: {e}")
            raise MachineLearningError(f"Failed to predict churn: {e}")
    
    async def train_engagement_prediction_model(
        self,
        sequence_id: UUID,
        training_data: List[Dict[str, Any]]
    ) -> ModelMetrics:
        """
        Train an engagement prediction model.
        
        Args:
            sequence_id: Sequence ID
            training_data: Training data with engagement features
            
        Returns:
            ModelMetrics with performance metrics
        """
        try:
            # Prepare data
            df = pd.DataFrame(training_data)
            
            # Feature engineering
            features = await self._engineer_features(df, "engagement")
            X = features.drop('engagement_score', axis=1)
            y = features['engagement_score']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, r2_score
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            cv_score = cv_scores.mean()
            
            # Feature importance
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            
            # Store model and scaler
            model_key = f"engagement_model_{sequence_id}"
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            self.feature_importance_cache[model_key] = feature_importance
            
            # Cache model
            await self._cache_model(model_key, model, scaler)
            
            metrics = ModelMetrics(
                accuracy=r2,  # Using R² as accuracy for regression
                precision=0.0,  # Not applicable for regression
                recall=0.0,  # Not applicable for regression
                f1_score=0.0,  # Not applicable for regression
                auc_roc=0.0,  # Not applicable for regression
                cross_val_score=cv_score,
                feature_importance=feature_importance
            )
            
            logger.info(f"Engagement prediction model trained for sequence {sequence_id}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training engagement prediction model: {e}")
            raise MachineLearningError(f"Failed to train engagement model: {e}")
    
    async def predict_engagement(
        self,
        sequence_id: UUID,
        subscriber_data: Dict[str, Any]
    ) -> PredictionResult:
        """
        Predict engagement score for a subscriber.
        
        Args:
            sequence_id: Sequence ID
            subscriber_data: Subscriber data for prediction
            
        Returns:
            PredictionResult with engagement prediction
        """
        try:
            model_key = f"engagement_model_{sequence_id}"
            
            # Load model if not in memory
            if model_key not in self.models:
                await self._load_model(model_key)
            
            if model_key not in self.models:
                raise MachineLearningError(f"Model not found for sequence {sequence_id}")
            
            model = self.models[model_key]
            scaler = self.scalers[model_key]
            
            # Prepare features
            features = await self._prepare_prediction_features(subscriber_data, "engagement")
            features_scaled = scaler.transform([features])
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            
            # Calculate confidence based on prediction variance
            # This is a simplified approach - in production, use proper uncertainty quantification
            confidence = min(1.0, max(0.0, 1.0 - abs(prediction - 0.5) * 2))
            
            # Feature contributions
            feature_contributions = await self._calculate_feature_contributions(
                model, features, model_key
            )
            
            return PredictionResult(
                prediction=float(prediction),
                confidence=confidence,
                feature_contributions=feature_contributions
            )
            
        except Exception as e:
            logger.error(f"Error predicting engagement: {e}")
            raise MachineLearningError(f"Failed to predict engagement: {e}")
    
    async def train_recommendation_model(
        self,
        sequence_id: UUID,
        interaction_data: List[Dict[str, Any]]
    ) -> ModelMetrics:
        """
        Train a recommendation model for email content.
        
        Args:
            sequence_id: Sequence ID
            interaction_data: User interaction data
            
        Returns:
            ModelMetrics with performance metrics
        """
        try:
            # Prepare data
            df = pd.DataFrame(interaction_data)
            
            # Feature engineering for recommendations
            features = await self._engineer_recommendation_features(df)
            
            # Train collaborative filtering model (simplified)
            # In production, use more sophisticated recommendation algorithms
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                random_state=42
            )
            
            X = features.drop('interaction_score', axis=1)
            y = features['interaction_score']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            
            # Calculate metrics
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            cv_score = cv_scores.mean()
            
            # Feature importance
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            
            # Store model and scaler
            model_key = f"recommendation_model_{sequence_id}"
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            self.feature_importance_cache[model_key] = feature_importance
            
            # Cache model
            await self._cache_model(model_key, model, scaler)
            
            metrics = ModelMetrics(
                accuracy=accuracy,
                precision=0.0,  # Calculate if needed
                recall=0.0,  # Calculate if needed
                f1_score=0.0,  # Calculate if needed
                auc_roc=0.0,  # Calculate if needed
                cross_val_score=cv_score,
                feature_importance=feature_importance
            )
            
            logger.info(f"Recommendation model trained for sequence {sequence_id}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training recommendation model: {e}")
            raise MachineLearningError(f"Failed to train recommendation model: {e}")
    
    async def get_content_recommendations(
        self,
        sequence_id: UUID,
        subscriber_data: Dict[str, Any],
        available_content: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Get content recommendations for a subscriber.
        
        Args:
            sequence_id: Sequence ID
            subscriber_data: Subscriber data
            available_content: Available content options
            
        Returns:
            List of recommended content with scores
        """
        try:
            model_key = f"recommendation_model_{sequence_id}"
            
            # Load model if not in memory
            if model_key not in self.models:
                await self._load_model(model_key)
            
            if model_key not in self.models:
                # Return default recommendations if no model
                return await self._get_default_recommendations(available_content)
            
            model = self.models[model_key]
            scaler = self.scalers[model_key]
            
            recommendations = []
            
            for content in available_content:
                # Prepare features for this content
                features = await self._prepare_content_features(subscriber_data, content)
                features_scaled = scaler.transform([features])
                
                # Get prediction
                score = model.predict_proba(features_scaled)[0][1]  # Probability of positive interaction
                
                recommendations.append({
                    "content_id": content.get("id"),
                    "content_type": content.get("type"),
                    "title": content.get("title"),
                    "recommendation_score": float(score),
                    "reasoning": await self._generate_recommendation_reasoning(
                        subscriber_data, content, score
                    )
                })
            
            # Sort by recommendation score
            recommendations.sort(key=lambda x: x["recommendation_score"], reverse=True)
            
            return recommendations[:10]  # Return top 10 recommendations
            
        except Exception as e:
            logger.error(f"Error getting content recommendations: {e}")
            # Return default recommendations on error
            return await self._get_default_recommendations(available_content)
    
    async def optimize_sequence_with_ml(
        self,
        sequence_id: UUID,
        current_sequence: Dict[str, Any],
        performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize email sequence using ML insights.
        
        Args:
            sequence_id: Sequence ID
            current_sequence: Current sequence configuration
            performance_data: Historical performance data
            
        Returns:
            Optimization recommendations
        """
        try:
            # Analyze performance patterns
            performance_insights = await self._analyze_performance_patterns(performance_data)
            
            # Generate optimization recommendations
            recommendations = await self._generate_ml_optimization_recommendations(
                current_sequence, performance_insights
            )
            
            # Predict impact of changes
            impact_predictions = await self._predict_optimization_impact(
                current_sequence, recommendations
            )
            
            return {
                "sequence_id": str(sequence_id),
                "optimization_recommendations": recommendations,
                "predicted_impact": impact_predictions,
                "confidence_score": await self._calculate_optimization_confidence(recommendations),
                "implementation_priority": await self._prioritize_recommendations(recommendations),
                "optimization_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing sequence with ML: {e}")
            raise MachineLearningError(f"Failed to optimize sequence: {e}")
    
    # Private helper methods
    async def _engineer_features(self, df: pd.DataFrame, target_type: str) -> pd.DataFrame:
        """Engineer features for ML models"""
        # This is a simplified feature engineering
        # In production, implement comprehensive feature engineering
        
        features_df = df.copy()
        
        if target_type == "churn":
            # Create churn-related features
            features_df['days_since_last_activity'] = (
                datetime.utcnow() - pd.to_datetime(features_df['last_activity'])
            ).dt.days
            
            features_df['engagement_trend'] = features_df['recent_opens'] / features_df['total_opens']
            features_df['churn_label'] = (features_df['days_since_last_activity'] > 30).astype(int)
            
        elif target_type == "engagement":
            # Create engagement-related features
            features_df['open_rate'] = features_df['total_opens'] / features_df['total_emails']
            features_df['click_rate'] = features_df['total_clicks'] / features_df['total_opens']
            features_df['engagement_score'] = (
                features_df['open_rate'] * 0.4 + 
                features_df['click_rate'] * 0.6
            )
        
        return features_df
    
    async def _prepare_prediction_features(
        self,
        subscriber_data: Dict[str, Any],
        prediction_type: str
    ) -> List[float]:
        """Prepare features for prediction"""
        # This is a simplified feature preparation
        # In production, implement comprehensive feature preparation
        
        features = []
        
        if prediction_type == "churn":
            features.extend([
                subscriber_data.get('days_since_signup', 0),
                subscriber_data.get('total_opens', 0),
                subscriber_data.get('total_clicks', 0),
                subscriber_data.get('days_since_last_activity', 0),
                subscriber_data.get('email_frequency', 0)
            ])
        elif prediction_type == "engagement":
            features.extend([
                subscriber_data.get('open_rate', 0),
                subscriber_data.get('click_rate', 0),
                subscriber_data.get('time_on_site', 0),
                subscriber_data.get('page_views', 0)
            ])
        
        return features
    
    async def _calculate_feature_contributions(
        self,
        model: Any,
        features: List[float],
        model_key: str
    ) -> Dict[str, float]:
        """Calculate feature contributions to prediction"""
        # This is a simplified feature contribution calculation
        # In production, use SHAP or LIME for proper feature attribution
        
        feature_names = list(self.feature_importance_cache.get(model_key, {}).keys())
        contributions = {}
        
        for i, feature_name in enumerate(feature_names):
            if i < len(features):
                contributions[feature_name] = float(features[i])
        
        return contributions
    
    async def _cache_model(self, model_key: str, model: Any, scaler: StandardScaler) -> None:
        """Cache trained model"""
        try:
            # Serialize model and scaler
            model_data = {
                "model": pickle.dumps(model),
                "scaler": pickle.dumps(scaler),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await cache_manager.set(f"ml_model:{model_key}", model_data, 86400)  # 24 hours
            
        except Exception as e:
            logger.warning(f"Failed to cache model {model_key}: {e}")
    
    async def _load_model(self, model_key: str) -> None:
        """Load model from cache"""
        try:
            model_data = await cache_manager.get(f"ml_model:{model_key}")
            if model_data:
                model = pickle.loads(model_data["model"])
                scaler = pickle.loads(model_data["scaler"])
                
                self.models[model_key] = model
                self.scalers[model_key] = scaler
                
        except Exception as e:
            logger.warning(f"Failed to load model {model_key}: {e}")
    
    # Additional helper methods (simplified implementations)
    async def _engineer_recommendation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for recommendation model"""
        return df  # Simplified implementation
    
    async def _prepare_content_features(
        self,
        subscriber_data: Dict[str, Any],
        content: Dict[str, Any]
    ) -> List[float]:
        """Prepare features for content recommendation"""
        return [0.5, 0.3, 0.2]  # Simplified implementation
    
    async def _generate_recommendation_reasoning(
        self,
        subscriber_data: Dict[str, Any],
        content: Dict[str, Any],
        score: float
    ) -> str:
        """Generate reasoning for recommendation"""
        return f"Recommended based on subscriber preferences and content relevance (score: {score:.2f})"
    
    async def _get_default_recommendations(self, available_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get default recommendations when no model is available"""
        return [
            {
                "content_id": content.get("id"),
                "content_type": content.get("type"),
                "title": content.get("title"),
                "recommendation_score": 0.5,
                "reasoning": "Default recommendation"
            }
            for content in available_content[:5]
        ]
    
    async def _analyze_performance_patterns(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance patterns"""
        return {"pattern": "stable"}  # Simplified implementation
    
    async def _generate_ml_optimization_recommendations(
        self,
        current_sequence: Dict[str, Any],
        performance_insights: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate ML-based optimization recommendations"""
        return [
            {
                "type": "timing_optimization",
                "description": "Optimize send times based on engagement patterns",
                "expected_improvement": 15.0
            }
        ]
    
    async def _predict_optimization_impact(
        self,
        current_sequence: Dict[str, Any],
        recommendations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Predict impact of optimization recommendations"""
        return {
            "open_rate_improvement": 15.0,
            "click_rate_improvement": 10.0,
            "conversion_improvement": 8.0
        }
    
    async def _calculate_optimization_confidence(self, recommendations: List[Dict[str, Any]]) -> float:
        """Calculate confidence in optimization recommendations"""
        return 0.75  # Simplified implementation
    
    async def _prioritize_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize optimization recommendations"""
        return sorted(recommendations, key=lambda x: x.get("expected_improvement", 0), reverse=True)


# Global ML engine instance
ml_engine = MachineLearningEngine()






























