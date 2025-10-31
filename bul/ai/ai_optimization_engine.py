"""
BUL AI Optimization Engine
==========================

AI-powered optimization system for document generation, performance, and resource management.
"""

import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
import pickle
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import optuna
import mlflow
import mlflow.sklearn
import mlflow.tensorflow

from ..utils import get_logger, get_cache_manager, get_data_processor
from ..config import get_config
from ..core import BusinessArea, DocumentType

logger = get_logger(__name__)

class OptimizationType(str, Enum):
    """Optimization types"""
    PERFORMANCE = "performance"
    QUALITY = "quality"
    COST = "cost"
    RESOURCE = "resource"
    USER_EXPERIENCE = "user_experience"
    SECURITY = "security"
    SCALABILITY = "scalability"

class OptimizationTarget(str, Enum):
    """Optimization targets"""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    COST_EFFICIENCY = "cost_efficiency"
    RESOURCE_UTILIZATION = "resource_utilization"
    USER_SATISFACTION = "user_satisfaction"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"

@dataclass
class OptimizationMetrics:
    """Optimization metrics data structure"""
    timestamp: datetime
    optimization_type: OptimizationType
    target: OptimizationTarget
    current_value: float
    target_value: float
    improvement_percentage: float
    confidence: float
    parameters: Dict[str, Any]
    model_used: str

@dataclass
class OptimizationRecommendation:
    """Optimization recommendation"""
    id: str
    type: OptimizationType
    target: OptimizationTarget
    priority: int
    title: str
    description: str
    expected_improvement: float
    implementation_effort: str
    risk_level: str
    parameters: Dict[str, Any]
    created_at: datetime
    status: str

class AIOptimizationEngine:
    """AI-powered optimization engine"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.cache_manager = get_cache_manager()
        self.data_processor = get_data_processor()
        self.config = get_config()
        
        # ML Models
        self.performance_model = None
        self.quality_model = None
        self.cost_model = None
        self.resource_model = None
        
        # Optimization history
        self.optimization_history: List[OptimizationMetrics] = []
        self.recommendations: List[OptimizationRecommendation] = []
        
        # Hyperparameter optimization
        self.study = None
        self.best_params = {}
        
        # MLflow tracking
        self.mlflow_experiment = "bul-optimization"
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models for optimization"""
        try:
            # Performance optimization model
            self.performance_model = self._create_performance_model()
            
            # Quality optimization model
            self.quality_model = self._create_quality_model()
            
            # Cost optimization model
            self.cost_model = self._create_cost_model()
            
            # Resource optimization model
            self.resource_model = self._create_resource_model()
            
            # Initialize MLflow
            mlflow.set_experiment(self.mlflow_experiment)
            
            self.logger.info("AI optimization models initialized successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize AI optimization models: {e}")
    
    def _create_performance_model(self) -> keras.Model:
        """Create neural network model for performance optimization"""
        try:
            model = keras.Sequential([
                layers.Dense(128, activation='relu', input_shape=(20,)),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu'),
                layers.Dense(1, activation='linear')
            ])
            
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae', 'mse']
            )
            
            return model
        
        except Exception as e:
            self.logger.error(f"Error creating performance model: {e}")
            return None
    
    def _create_quality_model(self) -> keras.Model:
        """Create neural network model for quality optimization"""
        try:
            model = keras.Sequential([
                layers.Dense(256, activation='relu', input_shape=(25,)),
                layers.Dropout(0.4),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dense(32, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            return model
        
        except Exception as e:
            self.logger.error(f"Error creating quality model: {e}")
            return None
    
    def _create_cost_model(self) -> RandomForestRegressor:
        """Create cost optimization model"""
        try:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            return model
        
        except Exception as e:
            self.logger.error(f"Error creating cost model: {e}")
            return None
    
    def _create_resource_model(self) -> GradientBoostingRegressor:
        """Create resource optimization model"""
        try:
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            return model
        
        except Exception as e:
            self.logger.error(f"Error creating resource model: {e}")
            return None
    
    async def optimize_performance(
        self,
        current_metrics: Dict[str, float],
        target_metrics: Dict[str, float]
    ) -> OptimizationRecommendation:
        """Optimize system performance using AI"""
        try:
            # Prepare features
            features = self._prepare_performance_features(current_metrics)
            
            # Predict optimal parameters
            if self.performance_model:
                optimal_params = self._predict_optimal_parameters(
                    features, target_metrics, self.performance_model
                )
            else:
                optimal_params = self._rule_based_performance_optimization(
                    current_metrics, target_metrics
                )
            
            # Calculate expected improvement
            expected_improvement = self._calculate_performance_improvement(
                current_metrics, target_metrics, optimal_params
            )
            
            # Create recommendation
            recommendation = OptimizationRecommendation(
                id=f"perf_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type=OptimizationType.PERFORMANCE,
                target=OptimizationTarget.RESPONSE_TIME,
                priority=self._calculate_priority(expected_improvement),
                title="Performance Optimization",
                description=f"Optimize system performance to achieve {expected_improvement:.1f}% improvement",
                expected_improvement=expected_improvement,
                implementation_effort="Medium",
                risk_level="Low",
                parameters=optimal_params,
                created_at=datetime.now(),
                status="pending"
            )
            
            # Store recommendation
            self.recommendations.append(recommendation)
            
            # Record optimization metrics
            await self._record_optimization_metrics(
                OptimizationType.PERFORMANCE,
                OptimizationTarget.RESPONSE_TIME,
                current_metrics.get('response_time', 0),
                target_metrics.get('response_time', 0),
                expected_improvement,
                optimal_params
            )
            
            return recommendation
        
        except Exception as e:
            self.logger.error(f"Error optimizing performance: {e}")
            raise
    
    async def optimize_quality(
        self,
        current_quality: Dict[str, float],
        target_quality: Dict[str, float]
    ) -> OptimizationRecommendation:
        """Optimize document quality using AI"""
        try:
            # Prepare features
            features = self._prepare_quality_features(current_quality)
            
            # Predict optimal parameters
            if self.quality_model:
                optimal_params = self._predict_optimal_parameters(
                    features, target_quality, self.quality_model
                )
            else:
                optimal_params = self._rule_based_quality_optimization(
                    current_quality, target_quality
                )
            
            # Calculate expected improvement
            expected_improvement = self._calculate_quality_improvement(
                current_quality, target_quality, optimal_params
            )
            
            # Create recommendation
            recommendation = OptimizationRecommendation(
                id=f"qual_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type=OptimizationType.QUALITY,
                target=OptimizationTarget.ACCURACY,
                priority=self._calculate_priority(expected_improvement),
                title="Quality Optimization",
                description=f"Optimize document quality to achieve {expected_improvement:.1f}% improvement",
                expected_improvement=expected_improvement,
                implementation_effort="High",
                risk_level="Medium",
                parameters=optimal_params,
                created_at=datetime.now(),
                status="pending"
            )
            
            # Store recommendation
            self.recommendations.append(recommendation)
            
            # Record optimization metrics
            await self._record_optimization_metrics(
                OptimizationType.QUALITY,
                OptimizationTarget.ACCURACY,
                current_quality.get('overall_score', 0),
                target_quality.get('overall_score', 0),
                expected_improvement,
                optimal_params
            )
            
            return recommendation
        
        except Exception as e:
            self.logger.error(f"Error optimizing quality: {e}")
            raise
    
    async def optimize_cost(
        self,
        current_costs: Dict[str, float],
        target_costs: Dict[str, float]
    ) -> OptimizationRecommendation:
        """Optimize system costs using AI"""
        try:
            # Prepare features
            features = self._prepare_cost_features(current_costs)
            
            # Predict optimal parameters
            if self.cost_model:
                optimal_params = self._predict_optimal_parameters(
                    features, target_costs, self.cost_model
                )
            else:
                optimal_params = self._rule_based_cost_optimization(
                    current_costs, target_costs
                )
            
            # Calculate expected improvement
            expected_improvement = self._calculate_cost_improvement(
                current_costs, target_costs, optimal_params
            )
            
            # Create recommendation
            recommendation = OptimizationRecommendation(
                id=f"cost_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type=OptimizationType.COST,
                target=OptimizationTarget.COST_EFFICIENCY,
                priority=self._calculate_priority(expected_improvement),
                title="Cost Optimization",
                description=f"Optimize system costs to achieve {expected_improvement:.1f}% reduction",
                expected_improvement=expected_improvement,
                implementation_effort="Medium",
                risk_level="Low",
                parameters=optimal_params,
                created_at=datetime.now(),
                status="pending"
            )
            
            # Store recommendation
            self.recommendations.append(recommendation)
            
            # Record optimization metrics
            await self._record_optimization_metrics(
                OptimizationType.COST,
                OptimizationTarget.COST_EFFICIENCY,
                current_costs.get('total_cost', 0),
                target_costs.get('total_cost', 0),
                expected_improvement,
                optimal_params
            )
            
            return recommendation
        
        except Exception as e:
            self.logger.error(f"Error optimizing cost: {e}")
            raise
    
    async def optimize_resources(
        self,
        current_resources: Dict[str, float],
        target_resources: Dict[str, float]
    ) -> OptimizationRecommendation:
        """Optimize resource utilization using AI"""
        try:
            # Prepare features
            features = self._prepare_resource_features(current_resources)
            
            # Predict optimal parameters
            if self.resource_model:
                optimal_params = self._predict_optimal_parameters(
                    features, target_resources, self.resource_model
                )
            else:
                optimal_params = self._rule_based_resource_optimization(
                    current_resources, target_resources
                )
            
            # Calculate expected improvement
            expected_improvement = self._calculate_resource_improvement(
                current_resources, target_resources, optimal_params
            )
            
            # Create recommendation
            recommendation = OptimizationRecommendation(
                id=f"res_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type=OptimizationType.RESOURCE,
                target=OptimizationTarget.RESOURCE_UTILIZATION,
                priority=self._calculate_priority(expected_improvement),
                title="Resource Optimization",
                description=f"Optimize resource utilization to achieve {expected_improvement:.1f}% improvement",
                expected_improvement=expected_improvement,
                implementation_effort="Low",
                risk_level="Low",
                parameters=optimal_params,
                created_at=datetime.now(),
                status="pending"
            )
            
            # Store recommendation
            self.recommendations.append(recommendation)
            
            # Record optimization metrics
            await self._record_optimization_metrics(
                OptimizationType.RESOURCE,
                OptimizationTarget.RESOURCE_UTILIZATION,
                current_resources.get('utilization', 0),
                target_resources.get('utilization', 0),
                expected_improvement,
                optimal_params
            )
            
            return recommendation
        
        except Exception as e:
            self.logger.error(f"Error optimizing resources: {e}")
            raise
    
    def _prepare_performance_features(self, metrics: Dict[str, float]) -> np.ndarray:
        """Prepare features for performance optimization"""
        try:
            features = np.array([
                metrics.get('response_time', 0),
                metrics.get('throughput', 0),
                metrics.get('cpu_usage', 0),
                metrics.get('memory_usage', 0),
                metrics.get('disk_usage', 0),
                metrics.get('network_latency', 0),
                metrics.get('error_rate', 0),
                metrics.get('queue_length', 0),
                metrics.get('cache_hit_rate', 0),
                metrics.get('database_connections', 0),
                metrics.get('active_sessions', 0),
                metrics.get('request_size', 0),
                metrics.get('response_size', 0),
                metrics.get('concurrent_requests', 0),
                metrics.get('processing_time', 0),
                metrics.get('wait_time', 0),
                metrics.get('retry_count', 0),
                metrics.get('timeout_count', 0),
                metrics.get('success_rate', 0),
                metrics.get('availability', 0)
            ])
            
            return features.reshape(1, -1)
        
        except Exception as e:
            self.logger.error(f"Error preparing performance features: {e}")
            return np.zeros((1, 20))
    
    def _prepare_quality_features(self, metrics: Dict[str, float]) -> np.ndarray:
        """Prepare features for quality optimization"""
        try:
            features = np.array([
                metrics.get('readability', 0),
                metrics.get('coherence', 0),
                metrics.get('completeness', 0),
                metrics.get('accuracy', 0),
                metrics.get('relevance', 0),
                metrics.get('professionalism', 0),
                metrics.get('clarity', 0),
                metrics.get('structure', 0),
                metrics.get('word_count', 0),
                metrics.get('sentence_count', 0),
                metrics.get('paragraph_count', 0),
                metrics.get('avg_word_length', 0),
                metrics.get('avg_sentence_length', 0),
                metrics.get('sentiment_score', 0),
                metrics.get('topic_coherence', 0),
                metrics.get('complexity_score', 0),
                metrics.get('keyword_density', 0),
                metrics.get('section_count', 0),
                metrics.get('bullet_point_count', 0),
                metrics.get('table_count', 0),
                metrics.get('image_count', 0),
                metrics.get('link_count', 0),
                metrics.get('grammar_score', 0),
                metrics.get('spelling_score', 0),
                metrics.get('style_score', 0)
            ])
            
            return features.reshape(1, -1)
        
        except Exception as e:
            self.logger.error(f"Error preparing quality features: {e}")
            return np.zeros((1, 25))
    
    def _prepare_cost_features(self, metrics: Dict[str, float]) -> np.ndarray:
        """Prepare features for cost optimization"""
        try:
            features = np.array([
                metrics.get('compute_cost', 0),
                metrics.get('storage_cost', 0),
                metrics.get('network_cost', 0),
                metrics.get('api_cost', 0),
                metrics.get('database_cost', 0),
                metrics.get('cache_cost', 0),
                metrics.get('total_cost', 0),
                metrics.get('cost_per_request', 0),
                metrics.get('cost_per_user', 0),
                metrics.get('cost_per_document', 0)
            ])
            
            return features.reshape(1, -1)
        
        except Exception as e:
            self.logger.error(f"Error preparing cost features: {e}")
            return np.zeros((1, 10))
    
    def _prepare_resource_features(self, metrics: Dict[str, float]) -> np.ndarray:
        """Prepare features for resource optimization"""
        try:
            features = np.array([
                metrics.get('cpu_utilization', 0),
                metrics.get('memory_utilization', 0),
                metrics.get('disk_utilization', 0),
                metrics.get('network_utilization', 0),
                metrics.get('database_utilization', 0),
                metrics.get('cache_utilization', 0),
                metrics.get('queue_utilization', 0),
                metrics.get('connection_utilization', 0),
                metrics.get('storage_utilization', 0),
                metrics.get('bandwidth_utilization', 0)
            ])
            
            return features.reshape(1, -1)
        
        except Exception as e:
            self.logger.error(f"Error preparing resource features: {e}")
            return np.zeros((1, 10))
    
    def _predict_optimal_parameters(
        self,
        features: np.ndarray,
        targets: Dict[str, float],
        model: Any
    ) -> Dict[str, Any]:
        """Predict optimal parameters using AI model"""
        try:
            if model is None:
                return self._get_default_parameters()
            
            # Make prediction
            prediction = model.predict(features)
            
            # Convert prediction to parameters
            optimal_params = self._convert_prediction_to_parameters(prediction, targets)
            
            return optimal_params
        
        except Exception as e:
            self.logger.error(f"Error predicting optimal parameters: {e}")
            return self._get_default_parameters()
    
    def _convert_prediction_to_parameters(
        self,
        prediction: np.ndarray,
        targets: Dict[str, float]
    ) -> Dict[str, Any]:
        """Convert model prediction to optimization parameters"""
        try:
            # This is a simplified conversion - in practice, this would be more sophisticated
            params = {
                'cache_size': int(prediction[0] * 1000) if len(prediction) > 0 else 1000,
                'worker_count': int(prediction[0] * 10) if len(prediction) > 0 else 4,
                'timeout': int(prediction[0] * 60) if len(prediction) > 0 else 30,
                'batch_size': int(prediction[0] * 100) if len(prediction) > 0 else 50,
                'max_connections': int(prediction[0] * 100) if len(prediction) > 0 else 100
            }
            
            return params
        
        except Exception as e:
            self.logger.error(f"Error converting prediction to parameters: {e}")
            return self._get_default_parameters()
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default optimization parameters"""
        return {
            'cache_size': 1000,
            'worker_count': 4,
            'timeout': 30,
            'batch_size': 50,
            'max_connections': 100
        }
    
    def _rule_based_performance_optimization(
        self,
        current_metrics: Dict[str, float],
        target_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Rule-based performance optimization"""
        try:
            params = {}
            
            # Optimize based on response time
            if current_metrics.get('response_time', 0) > target_metrics.get('response_time', 0):
                params['cache_size'] = 2000
                params['worker_count'] = 8
                params['timeout'] = 15
            
            # Optimize based on throughput
            if current_metrics.get('throughput', 0) < target_metrics.get('throughput', 0):
                params['batch_size'] = 100
                params['max_connections'] = 200
            
            # Optimize based on resource usage
            if current_metrics.get('cpu_usage', 0) > 80:
                params['worker_count'] = 6
                params['batch_size'] = 25
            
            if current_metrics.get('memory_usage', 0) > 80:
                params['cache_size'] = 500
                params['batch_size'] = 25
            
            return params
        
        except Exception as e:
            self.logger.error(f"Error in rule-based performance optimization: {e}")
            return self._get_default_parameters()
    
    def _rule_based_quality_optimization(
        self,
        current_quality: Dict[str, float],
        target_quality: Dict[str, float]
    ) -> Dict[str, Any]:
        """Rule-based quality optimization"""
        try:
            params = {}
            
            # Optimize based on readability
            if current_quality.get('readability', 0) < target_quality.get('readability', 0):
                params['max_sentence_length'] = 20
                params['max_word_length'] = 6
                params['complexity_threshold'] = 0.5
            
            # Optimize based on structure
            if current_quality.get('structure', 0) < target_quality.get('structure', 0):
                params['min_sections'] = 3
                params['min_paragraphs'] = 5
                params['require_headings'] = True
            
            # Optimize based on completeness
            if current_quality.get('completeness', 0) < target_quality.get('completeness', 0):
                params['min_word_count'] = 500
                params['require_examples'] = True
                params['require_conclusions'] = True
            
            return params
        
        except Exception as e:
            self.logger.error(f"Error in rule-based quality optimization: {e}")
            return {}
    
    def _rule_based_cost_optimization(
        self,
        current_costs: Dict[str, float],
        target_costs: Dict[str, float]
    ) -> Dict[str, Any]:
        """Rule-based cost optimization"""
        try:
            params = {}
            
            # Optimize compute costs
            if current_costs.get('compute_cost', 0) > target_costs.get('compute_cost', 0):
                params['auto_scaling'] = True
                params['min_instances'] = 1
                params['max_instances'] = 5
            
            # Optimize storage costs
            if current_costs.get('storage_cost', 0) > target_costs.get('storage_cost', 0):
                params['data_retention_days'] = 30
                params['compression_enabled'] = True
                params['archival_enabled'] = True
            
            # Optimize API costs
            if current_costs.get('api_cost', 0) > target_costs.get('api_cost', 0):
                params['cache_ttl'] = 3600
                params['batch_processing'] = True
                params['rate_limiting'] = True
            
            return params
        
        except Exception as e:
            self.logger.error(f"Error in rule-based cost optimization: {e}")
            return {}
    
    def _rule_based_resource_optimization(
        self,
        current_resources: Dict[str, float],
        target_resources: Dict[str, float]
    ) -> Dict[str, Any]:
        """Rule-based resource optimization"""
        try:
            params = {}
            
            # Optimize CPU utilization
            if current_resources.get('cpu_utilization', 0) > 80:
                params['cpu_limit'] = 1000
                params['worker_count'] = 6
            
            # Optimize memory utilization
            if current_resources.get('memory_utilization', 0) > 80:
                params['memory_limit'] = 512
                params['cache_size'] = 500
            
            # Optimize disk utilization
            if current_resources.get('disk_utilization', 0) > 80:
                params['log_retention_days'] = 7
                params['temp_file_cleanup'] = True
            
            return params
        
        except Exception as e:
            self.logger.error(f"Error in rule-based resource optimization: {e}")
            return {}
    
    def _calculate_performance_improvement(
        self,
        current_metrics: Dict[str, float],
        target_metrics: Dict[str, float],
        optimal_params: Dict[str, Any]
    ) -> float:
        """Calculate expected performance improvement"""
        try:
            # Simplified calculation - in practice, this would use the trained model
            current_response_time = current_metrics.get('response_time', 1.0)
            target_response_time = target_metrics.get('response_time', 0.5)
            
            improvement = ((current_response_time - target_response_time) / current_response_time) * 100
            return max(0, improvement)
        
        except Exception as e:
            self.logger.error(f"Error calculating performance improvement: {e}")
            return 0.0
    
    def _calculate_quality_improvement(
        self,
        current_quality: Dict[str, float],
        target_quality: Dict[str, float],
        optimal_params: Dict[str, Any]
    ) -> float:
        """Calculate expected quality improvement"""
        try:
            current_score = current_quality.get('overall_score', 0.5)
            target_score = target_quality.get('overall_score', 0.8)
            
            improvement = ((target_score - current_score) / current_score) * 100
            return max(0, improvement)
        
        except Exception as e:
            self.logger.error(f"Error calculating quality improvement: {e}")
            return 0.0
    
    def _calculate_cost_improvement(
        self,
        current_costs: Dict[str, float],
        target_costs: Dict[str, float],
        optimal_params: Dict[str, Any]
    ) -> float:
        """Calculate expected cost improvement"""
        try:
            current_cost = current_costs.get('total_cost', 100.0)
            target_cost = target_costs.get('total_cost', 80.0)
            
            improvement = ((current_cost - target_cost) / current_cost) * 100
            return max(0, improvement)
        
        except Exception as e:
            self.logger.error(f"Error calculating cost improvement: {e}")
            return 0.0
    
    def _calculate_resource_improvement(
        self,
        current_resources: Dict[str, float],
        target_resources: Dict[str, float],
        optimal_params: Dict[str, Any]
    ) -> float:
        """Calculate expected resource improvement"""
        try:
            current_utilization = current_resources.get('utilization', 0.8)
            target_utilization = target_resources.get('utilization', 0.6)
            
            improvement = ((current_utilization - target_utilization) / current_utilization) * 100
            return max(0, improvement)
        
        except Exception as e:
            self.logger.error(f"Error calculating resource improvement: {e}")
            return 0.0
    
    def _calculate_priority(self, improvement: float) -> int:
        """Calculate recommendation priority"""
        if improvement >= 50:
            return 1  # High priority
        elif improvement >= 25:
            return 2  # Medium priority
        elif improvement >= 10:
            return 3  # Low priority
        else:
            return 4  # Very low priority
    
    async def _record_optimization_metrics(
        self,
        optimization_type: OptimizationType,
        target: OptimizationTarget,
        current_value: float,
        target_value: float,
        improvement: float,
        parameters: Dict[str, Any]
    ):
        """Record optimization metrics"""
        try:
            metrics = OptimizationMetrics(
                timestamp=datetime.now(),
                optimization_type=optimization_type,
                target=target,
                current_value=current_value,
                target_value=target_value,
                improvement_percentage=improvement,
                confidence=0.8,  # Default confidence
                parameters=parameters,
                model_used="ai_optimization_engine"
            )
            
            self.optimization_history.append(metrics)
            
            # Keep only last 1000 records
            if len(self.optimization_history) > 1000:
                self.optimization_history = self.optimization_history[-1000:]
        
        except Exception as e:
            self.logger.error(f"Error recording optimization metrics: {e}")
    
    async def get_optimization_recommendations(
        self,
        limit: int = 10
    ) -> List[OptimizationRecommendation]:
        """Get optimization recommendations"""
        try:
            # Sort by priority and return top recommendations
            sorted_recommendations = sorted(
                self.recommendations,
                key=lambda x: (x.priority, x.expected_improvement),
                reverse=True
            )
            
            return sorted_recommendations[:limit]
        
        except Exception as e:
            self.logger.error(f"Error getting optimization recommendations: {e}")
            return []
    
    async def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get optimization metrics and statistics"""
        try:
            if not self.optimization_history:
                return {"message": "No optimization data available"}
            
            # Calculate statistics
            total_optimizations = len(self.optimization_history)
            avg_improvement = np.mean([m.improvement_percentage for m in self.optimization_history])
            max_improvement = max([m.improvement_percentage for m in self.optimization_history])
            
            # Group by optimization type
            type_counts = {}
            for metrics in self.optimization_history:
                opt_type = metrics.optimization_type.value
                type_counts[opt_type] = type_counts.get(opt_type, 0) + 1
            
            return {
                "total_optimizations": total_optimizations,
                "average_improvement": round(avg_improvement, 2),
                "max_improvement": round(max_improvement, 2),
                "optimization_types": type_counts,
                "recent_optimizations": [
                    asdict(m) for m in self.optimization_history[-10:]
                ]
            }
        
        except Exception as e:
            self.logger.error(f"Error getting optimization metrics: {e}")
            return {"error": str(e)}

# Global AI optimization engine
_ai_optimization_engine: Optional[AIOptimizationEngine] = None

def get_ai_optimization_engine() -> AIOptimizationEngine:
    """Get the global AI optimization engine"""
    global _ai_optimization_engine
    if _ai_optimization_engine is None:
        _ai_optimization_engine = AIOptimizationEngine()
    return _ai_optimization_engine

# AI optimization router
ai_optimization_router = APIRouter(prefix="/ai-optimization", tags=["AI Optimization"])

@ai_optimization_router.post("/optimize-performance")
async def optimize_performance_endpoint(
    current_metrics: Dict[str, float] = Field(..., description="Current performance metrics"),
    target_metrics: Dict[str, float] = Field(..., description="Target performance metrics")
):
    """Optimize system performance using AI"""
    try:
        engine = get_ai_optimization_engine()
        recommendation = await engine.optimize_performance(current_metrics, target_metrics)
        return {"recommendation": asdict(recommendation), "success": True}
    
    except Exception as e:
        logger.error(f"Error optimizing performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to optimize performance")

@ai_optimization_router.post("/optimize-quality")
async def optimize_quality_endpoint(
    current_quality: Dict[str, float] = Field(..., description="Current quality metrics"),
    target_quality: Dict[str, float] = Field(..., description="Target quality metrics")
):
    """Optimize document quality using AI"""
    try:
        engine = get_ai_optimization_engine()
        recommendation = await engine.optimize_quality(current_quality, target_quality)
        return {"recommendation": asdict(recommendation), "success": True}
    
    except Exception as e:
        logger.error(f"Error optimizing quality: {e}")
        raise HTTPException(status_code=500, detail="Failed to optimize quality")

@ai_optimization_router.post("/optimize-cost")
async def optimize_cost_endpoint(
    current_costs: Dict[str, float] = Field(..., description="Current cost metrics"),
    target_costs: Dict[str, float] = Field(..., description="Target cost metrics")
):
    """Optimize system costs using AI"""
    try:
        engine = get_ai_optimization_engine()
        recommendation = await engine.optimize_cost(current_costs, target_costs)
        return {"recommendation": asdict(recommendation), "success": True}
    
    except Exception as e:
        logger.error(f"Error optimizing cost: {e}")
        raise HTTPException(status_code=500, detail="Failed to optimize cost")

@ai_optimization_router.post("/optimize-resources")
async def optimize_resources_endpoint(
    current_resources: Dict[str, float] = Field(..., description="Current resource metrics"),
    target_resources: Dict[str, float] = Field(..., description="Target resource metrics")
):
    """Optimize resource utilization using AI"""
    try:
        engine = get_ai_optimization_engine()
        recommendation = await engine.optimize_resources(current_resources, target_resources)
        return {"recommendation": asdict(recommendation), "success": True}
    
    except Exception as e:
        logger.error(f"Error optimizing resources: {e}")
        raise HTTPException(status_code=500, detail="Failed to optimize resources")

@ai_optimization_router.get("/recommendations")
async def get_recommendations_endpoint(limit: int = 10):
    """Get optimization recommendations"""
    try:
        engine = get_ai_optimization_engine()
        recommendations = await engine.get_optimization_recommendations(limit)
        return {"recommendations": [asdict(r) for r in recommendations]}
    
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recommendations")

@ai_optimization_router.get("/metrics")
async def get_optimization_metrics_endpoint():
    """Get optimization metrics and statistics"""
    try:
        engine = get_ai_optimization_engine()
        metrics = await engine.get_optimization_metrics()
        return {"metrics": metrics}
    
    except Exception as e:
        logger.error(f"Error getting optimization metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get optimization metrics")


