"""
Machine Learning Service
=======================

Advanced machine learning capabilities for document analysis and optimization.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from uuid import uuid4
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import pickle
import joblib
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Model type."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    NLP = "nlp"
    RECOMMENDATION = "recommendation"
    ANOMALY_DETECTION = "anomaly_detection"


class ModelStatus(str, Enum):
    """Model status."""
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    FAILED = "failed"
    RETIRED = "retired"


class TrainingStatus(str, Enum):
    """Training status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    r2_score: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingJob:
    """Training job definition."""
    job_id: str
    model_id: str
    status: TrainingStatus
    training_data: Dict[str, Any]
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metrics: Optional[ModelMetrics] = None
    error_message: Optional[str] = None
    progress: float = 0.0


@dataclass
class MLModel:
    """Machine learning model definition."""
    model_id: str
    name: str
    description: str
    model_type: ModelType
    version: str
    status: ModelStatus
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    training_data_size: int = 0
    features: List[str] = field(default_factory=list)
    target_variable: Optional[str] = None
    metrics: Optional[ModelMetrics] = None
    model_path: Optional[str] = None
    preprocessing_pipeline: Optional[Dict[str, Any]] = None
    deployment_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionRequest:
    """Prediction request."""
    request_id: str
    model_id: str
    input_data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    confidence_threshold: float = 0.5


@dataclass
class PredictionResult:
    """Prediction result."""
    request_id: str
    model_id: str
    prediction: Any
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    explanation: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)


class DocumentMLService:
    """Machine learning service for document analysis."""
    
    def __init__(self, models_dir: str = "ml_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.models: Dict[str, MLModel] = {}
        self.training_jobs: Dict[str, TrainingJob] = {}
        self.prediction_cache: Dict[str, PredictionResult] = {}
        self.feature_extractors: Dict[str, Any] = {}
        self.preprocessing_pipelines: Dict[str, Any] = {}
        
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default ML models."""
        
        # Document Classification Model
        doc_classifier = MLModel(
            model_id="doc_classifier_v1",
            name="Document Classifier",
            description="Classifies documents by type and category",
            model_type=ModelType.CLASSIFICATION,
            version="1.0",
            status=ModelStatus.TRAINED,
            features=["word_count", "sentence_count", "avg_word_length", "has_tables", "has_images"],
            target_variable="document_type",
            metrics=ModelMetrics(accuracy=0.92, precision=0.89, recall=0.91, f1_score=0.90)
        )
        
        # Content Quality Predictor
        quality_predictor = MLModel(
            model_id="quality_predictor_v1",
            name="Content Quality Predictor",
            description="Predicts content quality score",
            model_type=ModelType.REGRESSION,
            version="1.0",
            status=ModelStatus.TRAINED,
            features=["readability_score", "sentiment_score", "structure_score", "grammar_score"],
            target_variable="quality_score",
            metrics=ModelMetrics(mse=0.15, mae=0.25, r2_score=0.87)
        )
        
        # Approval Likelihood Predictor
        approval_predictor = MLModel(
            model_id="approval_predictor_v1",
            name="Approval Likelihood Predictor",
            description="Predicts document approval likelihood",
            model_type=ModelType.CLASSIFICATION,
            version="1.0",
            status=ModelStatus.TRAINED,
            features=["quality_score", "compliance_score", "reviewer_sentiment", "document_length"],
            target_variable="approval_likelihood",
            metrics=ModelMetrics(accuracy=0.88, precision=0.85, recall=0.87, f1_score=0.86)
        )
        
        # Topic Clustering Model
        topic_clusterer = MLModel(
            model_id="topic_clusterer_v1",
            name="Topic Clustering Model",
            description="Clusters documents by topics",
            model_type=ModelType.CLUSTERING,
            version="1.0",
            status=ModelStatus.TRAINED,
            features=["tfidf_features", "word_embeddings", "named_entities"],
            metrics=ModelMetrics(custom_metrics={"silhouette_score": 0.75, "inertia": 150.5})
        )
        
        # Store models
        self.models[doc_classifier.model_id] = doc_classifier
        self.models[quality_predictor.model_id] = quality_predictor
        self.models[approval_predictor.model_id] = approval_predictor
        self.models[topic_clusterer.model_id] = topic_clusterer
    
    async def create_model(
        self,
        name: str,
        description: str,
        model_type: ModelType,
        features: List[str],
        target_variable: Optional[str] = None
    ) -> MLModel:
        """Create a new ML model."""
        
        model_id = str(uuid4())
        
        model = MLModel(
            model_id=model_id,
            name=name,
            description=description,
            model_type=model_type,
            version="1.0",
            status=ModelStatus.TRAINING,
            features=features,
            target_variable=target_variable
        )
        
        self.models[model_id] = model
        
        logger.info(f"Created ML model: {name} ({model_id})")
        
        return model
    
    async def train_model(
        self,
        model_id: str,
        training_data: Dict[str, Any],
        hyperparameters: Dict[str, Any] = None
    ) -> TrainingJob:
        """Train a machine learning model."""
        
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        
        # Create training job
        job = TrainingJob(
            job_id=str(uuid4()),
            model_id=model_id,
            status=TrainingStatus.PENDING,
            training_data=training_data,
            hyperparameters=hyperparameters or {}
        )
        
        self.training_jobs[job.job_id] = job
        
        # Start training
        asyncio.create_task(self._train_model_async(job))
        
        logger.info(f"Started training for model: {model.name} ({job.job_id})")
        
        return job
    
    async def _train_model_async(self, job: TrainingJob):
        """Train model asynchronously."""
        
        try:
            job.status = TrainingStatus.IN_PROGRESS
            model = self.models[job.model_id]
            
            # Simulate training process
            for progress in range(0, 101, 10):
                job.progress = progress
                await asyncio.sleep(0.1)  # Simulate training time
            
            # Mock training completion
            if model.model_type == ModelType.CLASSIFICATION:
                metrics = ModelMetrics(
                    accuracy=0.85 + np.random.random() * 0.1,
                    precision=0.82 + np.random.random() * 0.1,
                    recall=0.84 + np.random.random() * 0.1,
                    f1_score=0.83 + np.random.random() * 0.1
                )
            elif model.model_type == ModelType.REGRESSION:
                metrics = ModelMetrics(
                    mse=0.1 + np.random.random() * 0.2,
                    mae=0.2 + np.random.random() * 0.3,
                    r2_score=0.8 + np.random.random() * 0.15
                )
            else:
                metrics = ModelMetrics(
                    custom_metrics={"silhouette_score": 0.7 + np.random.random() * 0.2}
                )
            
            job.metrics = metrics
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.now()
            
            # Update model
            model.status = ModelStatus.TRAINED
            model.metrics = metrics
            model.updated_at = datetime.now()
            model.training_data_size = len(job.training_data.get("data", []))
            
            logger.info(f"Training completed for model: {model.name}")
            
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            
            model = self.models[job.model_id]
            model.status = ModelStatus.FAILED
            
            logger.error(f"Training failed for model {model.name}: {str(e)}")
    
    async def predict(
        self,
        model_id: str,
        input_data: Dict[str, Any],
        confidence_threshold: float = 0.5
    ) -> PredictionResult:
        """Make prediction using trained model."""
        
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        
        if model.status != ModelStatus.TRAINED and model.status != ModelStatus.DEPLOYED:
            raise ValueError(f"Model {model_id} is not trained")
        
        request_id = str(uuid4())
        
        # Check cache first
        cache_key = f"{model_id}_{hash(str(sorted(input_data.items())))}"
        if cache_key in self.prediction_cache:
            cached_result = self.prediction_cache[cache_key]
            cached_result.request_id = request_id
            cached_result.timestamp = datetime.now()
            return cached_result
        
        # Extract features
        features = await self._extract_features(input_data, model.features)
        
        # Make prediction based on model type
        if model.model_type == ModelType.CLASSIFICATION:
            prediction, confidence, probabilities = await self._predict_classification(
                model, features, confidence_threshold
            )
        elif model.model_type == ModelType.REGRESSION:
            prediction, confidence = await self._predict_regression(model, features)
            probabilities = None
        elif model.model_type == ModelType.CLUSTERING:
            prediction, confidence = await self._predict_clustering(model, features)
            probabilities = None
        else:
            raise ValueError(f"Unsupported model type: {model.model_type}")
        
        # Generate explanation
        explanation = await self._generate_prediction_explanation(model, features, prediction)
        
        result = PredictionResult(
            request_id=request_id,
            model_id=model_id,
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities,
            explanation=explanation
        )
        
        # Cache result
        self.prediction_cache[cache_key] = result
        
        return result
    
    async def _extract_features(self, input_data: Dict[str, Any], feature_list: List[str]) -> Dict[str, Any]:
        """Extract features from input data."""
        
        features = {}
        
        for feature in feature_list:
            if feature == "word_count":
                content = input_data.get("content", "")
                features[feature] = len(content.split())
            elif feature == "sentence_count":
                content = input_data.get("content", "")
                features[feature] = len(re.split(r'[.!?]+', content))
            elif feature == "avg_word_length":
                content = input_data.get("content", "")
                words = content.split()
                features[feature] = sum(len(word) for word in words) / len(words) if words else 0
            elif feature == "has_tables":
                content = input_data.get("content", "")
                features[feature] = 1 if '|' in content and content.count('|') > 5 else 0
            elif feature == "has_images":
                content = input_data.get("content", "")
                features[feature] = 1 if '![' in content else 0
            elif feature == "readability_score":
                # Mock readability score
                features[feature] = 0.7 + np.random.random() * 0.3
            elif feature == "sentiment_score":
                # Mock sentiment score
                features[feature] = -1 + np.random.random() * 2
            elif feature == "structure_score":
                # Mock structure score
                features[feature] = 0.6 + np.random.random() * 0.4
            elif feature == "grammar_score":
                # Mock grammar score
                features[feature] = 0.8 + np.random.random() * 0.2
            elif feature == "quality_score":
                # Mock quality score
                features[feature] = 0.7 + np.random.random() * 0.3
            elif feature == "compliance_score":
                # Mock compliance score
                features[feature] = 0.85 + np.random.random() * 0.15
            elif feature == "reviewer_sentiment":
                # Mock reviewer sentiment
                features[feature] = -1 + np.random.random() * 2
            elif feature == "document_length":
                content = input_data.get("content", "")
                features[feature] = len(content)
            else:
                # Default feature extraction
                features[feature] = input_data.get(feature, 0)
        
        return features
    
    async def _predict_classification(
        self,
        model: MLModel,
        features: Dict[str, Any],
        confidence_threshold: float
    ) -> Tuple[Any, float, Dict[str, float]]:
        """Make classification prediction."""
        
        # Mock classification prediction
        classes = ["business", "technical", "academic", "legal", "creative"]
        probabilities = {cls: np.random.random() for cls in classes}
        
        # Normalize probabilities
        total_prob = sum(probabilities.values())
        probabilities = {cls: prob / total_prob for cls, prob in probabilities.items()}
        
        # Get prediction and confidence
        prediction = max(probabilities.items(), key=lambda x: x[1])[0]
        confidence = probabilities[prediction]
        
        return prediction, confidence, probabilities
    
    async def _predict_regression(
        self,
        model: MLModel,
        features: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Make regression prediction."""
        
        # Mock regression prediction
        prediction = 0.5 + np.random.random() * 0.5  # 0.5 to 1.0
        confidence = 0.8 + np.random.random() * 0.2  # 0.8 to 1.0
        
        return prediction, confidence
    
    async def _predict_clustering(
        self,
        model: MLModel,
        features: Dict[str, Any]
    ) -> Tuple[int, float]:
        """Make clustering prediction."""
        
        # Mock clustering prediction
        cluster = np.random.randint(0, 5)  # 5 clusters
        confidence = 0.7 + np.random.random() * 0.3  # 0.7 to 1.0
        
        return cluster, confidence
    
    async def _generate_prediction_explanation(
        self,
        model: MLModel,
        features: Dict[str, Any],
        prediction: Any
    ) -> Dict[str, Any]:
        """Generate prediction explanation."""
        
        # Mock explanation generation
        explanation = {
            "model_name": model.name,
            "prediction": prediction,
            "feature_importance": {
                feature: np.random.random()
                for feature in model.features[:5]  # Top 5 features
            },
            "reasoning": [
                f"Feature '{feature}' contributed {np.random.random():.2f} to the prediction"
                for feature in model.features[:3]
            ],
            "confidence_factors": [
                "High quality input data",
                "Model trained on similar documents",
                "Strong feature correlations"
            ]
        }
        
        return explanation
    
    async def deploy_model(self, model_id: str, deployment_config: Dict[str, Any] = None) -> bool:
        """Deploy model for production use."""
        
        if model_id not in self.models:
            return False
        
        model = self.models[model_id]
        
        if model.status != ModelStatus.TRAINED:
            return False
        
        model.status = ModelStatus.DEPLOYED
        model.deployment_config = deployment_config or {}
        model.updated_at = datetime.now()
        
        logger.info(f"Deployed model: {model.name}")
        
        return True
    
    async def retrain_model(
        self,
        model_id: str,
        new_training_data: Dict[str, Any],
        hyperparameters: Dict[str, Any] = None
    ) -> TrainingJob:
        """Retrain model with new data."""
        
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        model.status = ModelStatus.TRAINING
        model.version = f"{float(model.version) + 0.1:.1f}"
        
        # Start retraining
        return await self.train_model(model_id, new_training_data, hyperparameters)
    
    async def evaluate_model(
        self,
        model_id: str,
        test_data: Dict[str, Any]
    ) -> ModelMetrics:
        """Evaluate model performance on test data."""
        
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        
        # Mock model evaluation
        if model.model_type == ModelType.CLASSIFICATION:
            metrics = ModelMetrics(
                accuracy=0.85 + np.random.random() * 0.1,
                precision=0.82 + np.random.random() * 0.1,
                recall=0.84 + np.random.random() * 0.1,
                f1_score=0.83 + np.random.random() * 0.1
            )
        elif model.model_type == ModelType.REGRESSION:
            metrics = ModelMetrics(
                mse=0.1 + np.random.random() * 0.2,
                mae=0.2 + np.random.random() * 0.3,
                r2_score=0.8 + np.random.random() * 0.15
            )
        else:
            metrics = ModelMetrics(
                custom_metrics={"silhouette_score": 0.7 + np.random.random() * 0.2}
            )
        
        # Update model metrics
        model.metrics = metrics
        model.updated_at = datetime.now()
        
        return metrics
    
    async def get_model_performance_history(self, model_id: str) -> List[Dict[str, Any]]:
        """Get model performance history."""
        
        if model_id not in self.models:
            return []
        
        # Mock performance history
        history = []
        for i in range(10):
            history.append({
                "timestamp": (datetime.now() - timedelta(days=i)).isoformat(),
                "accuracy": 0.8 + np.random.random() * 0.15,
                "precision": 0.78 + np.random.random() * 0.15,
                "recall": 0.82 + np.random.random() * 0.15,
                "f1_score": 0.8 + np.random.random() * 0.15
            })
        
        return history
    
    async def get_model_analytics(self) -> Dict[str, Any]:
        """Get ML model analytics."""
        
        total_models = len(self.models)
        trained_models = len([m for m in self.models.values() if m.status == ModelStatus.TRAINED])
        deployed_models = len([m for m in self.models.values() if m.status == ModelStatus.DEPLOYED])
        
        # Model type distribution
        type_distribution = Counter(m.model_type.value for m in self.models.values())
        
        # Average performance metrics
        avg_accuracy = 0
        avg_precision = 0
        avg_recall = 0
        avg_f1 = 0
        
        classification_models = [m for m in self.models.values() if m.model_type == ModelType.CLASSIFICATION and m.metrics]
        if classification_models:
            avg_accuracy = sum(m.metrics.accuracy for m in classification_models) / len(classification_models)
            avg_precision = sum(m.metrics.precision for m in classification_models) / len(classification_models)
            avg_recall = sum(m.metrics.recall for m in classification_models) / len(classification_models)
            avg_f1 = sum(m.metrics.f1_score for m in classification_models) / len(classification_models)
        
        return {
            "total_models": total_models,
            "trained_models": trained_models,
            "deployed_models": deployed_models,
            "model_type_distribution": dict(type_distribution),
            "average_performance": {
                "accuracy": avg_accuracy,
                "precision": avg_precision,
                "recall": avg_recall,
                "f1_score": avg_f1
            },
            "training_jobs": {
                "total": len(self.training_jobs),
                "completed": len([j for j in self.training_jobs.values() if j.status == TrainingStatus.COMPLETED]),
                "failed": len([j for j in self.training_jobs.values() if j.status == TrainingStatus.FAILED]),
                "in_progress": len([j for j in self.training_jobs.values() if j.status == TrainingStatus.IN_PROGRESS])
            },
            "prediction_cache_size": len(self.prediction_cache)
        }
    
    async def export_model(self, model_id: str, format: str = "pickle") -> bytes:
        """Export model in specified format."""
        
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        
        # Mock model export
        model_data = {
            "model_id": model.model_id,
            "name": model.name,
            "model_type": model.model_type.value,
            "version": model.version,
            "features": model.features,
            "metrics": model.metrics.__dict__ if model.metrics else None,
            "created_at": model.created_at.isoformat(),
            "updated_at": model.updated_at.isoformat()
        }
        
        if format == "pickle":
            return pickle.dumps(model_data)
        elif format == "json":
            return json.dumps(model_data, indent=2).encode()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def import_model(self, model_data: bytes, format: str = "pickle") -> MLModel:
        """Import model from data."""
        
        if format == "pickle":
            data = pickle.loads(model_data)
        elif format == "json":
            data = json.loads(model_data.decode())
        else:
            raise ValueError(f"Unsupported import format: {format}")
        
        # Create model from imported data
        model = MLModel(
            model_id=data["model_id"],
            name=data["name"],
            model_type=ModelType(data["model_type"]),
            version=data["version"],
            status=ModelStatus.TRAINED,
            features=data["features"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )
        
        if data["metrics"]:
            model.metrics = ModelMetrics(**data["metrics"])
        
        self.models[model.model_id] = model
        
        logger.info(f"Imported model: {model.name}")
        
        return model



























