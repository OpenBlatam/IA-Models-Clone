"""
Machine Learning Service - Advanced Implementation
================================================

Advanced machine learning service with model training, prediction, and optimization.
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
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from .analytics_service import analytics_service
from .ai_service import ai_service

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Model type enumeration"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"
    NLP = "nlp"
    RECOMMENDATION = "recommendation"


class ModelStatus(str, Enum):
    """Model status enumeration"""
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    FAILED = "failed"
    RETIRED = "retired"


class MLAlgorithm(str, Enum):
    """ML algorithm enumeration"""
    RANDOM_FOREST = "random_forest"
    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_REGRESSION = "linear_regression"
    DECISION_TREE = "decision_tree"
    SVM = "svm"
    KMEANS = "kmeans"
    NEURAL_NETWORK = "neural_network"


class MachineLearningService:
    """Advanced machine learning service with model training and prediction"""
    
    def __init__(self):
        self.models = {}
        self.training_jobs = {}
        self.prediction_cache = {}
        self.ml_stats = {
            "total_models": 0,
            "trained_models": 0,
            "deployed_models": 0,
            "failed_models": 0,
            "total_predictions": 0,
            "models_by_type": {model_type.value: 0 for model_type in ModelType},
            "models_by_algorithm": {algorithm.value: 0 for algorithm in MLAlgorithm}
        }
        
        # Model storage
        self.model_storage = {}
        self.scalers = {}
        self.encoders = {}
        
        # Performance tracking
        self.performance_metrics = {}
    
    async def create_model(
        self,
        name: str,
        model_type: ModelType,
        algorithm: MLAlgorithm,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new ML model"""
        try:
            model_id = f"ml_model_{len(self.models) + 1}"
            
            model_config = {
                "id": model_id,
                "name": name,
                "type": model_type.value,
                "algorithm": algorithm.value,
                "description": description,
                "parameters": parameters or {},
                "status": ModelStatus.TRAINING.value,
                "created_at": datetime.utcnow().isoformat(),
                "trained_at": None,
                "deployed_at": None,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "mse": None,
                "r2_score": None,
                "training_data_size": 0,
                "features": [],
                "target": None,
                "model_object": None,
                "scaler": None,
                "encoder": None
            }
            
            self.models[model_id] = model_config
            self.ml_stats["total_models"] += 1
            self.ml_stats["models_by_type"][model_type.value] += 1
            self.ml_stats["models_by_algorithm"][algorithm.value] += 1
            
            logger.info(f"ML model created: {model_id} - {name}")
            return model_id
        
        except Exception as e:
            logger.error(f"Failed to create ML model: {e}")
            raise
    
    async def train_model(
        self,
        model_id: str,
        training_data: List[Dict[str, Any]],
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Train ML model with provided data"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model not found: {model_id}")
            
            model = self.models[model_id]
            model["status"] = ModelStatus.TRAINING.value
            
            # Convert data to DataFrame
            df = pd.DataFrame(training_data)
            
            # Separate features and target
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Store features and target info
            model["features"] = list(X.columns)
            model["target"] = target_column
            model["training_data_size"] = len(df)
            
            # Handle categorical variables
            categorical_columns = X.select_dtypes(include=['object']).columns
            if len(categorical_columns) > 0:
                encoder = LabelEncoder()
                for col in categorical_columns:
                    X[col] = encoder.fit_transform(X[col].astype(str))
                self.encoders[model_id] = encoder
            
            # Scale numerical features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[model_id] = scaler
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state
            )
            
            # Train model based on algorithm
            model_object = await self._train_model_by_algorithm(
                model["algorithm"], model["type"], X_train, y_train, model["parameters"]
            )
            
            # Evaluate model
            metrics = await self._evaluate_model(
                model_object, model["type"], X_test, y_test
            )
            
            # Update model
            model["model_object"] = model_object
            model["status"] = ModelStatus.TRAINED.value
            model["trained_at"] = datetime.utcnow().isoformat()
            model.update(metrics)
            
            self.ml_stats["trained_models"] += 1
            
            # Store model
            self.model_storage[model_id] = {
                "model": model_object,
                "scaler": scaler,
                "encoder": self.encoders.get(model_id),
                "features": model["features"],
                "target": model["target"]
            }
            
            # Track analytics
            await analytics_service.track_event(
                "ml_model_trained",
                {
                    "model_id": model_id,
                    "model_type": model["type"],
                    "algorithm": model["algorithm"],
                    "accuracy": metrics.get("accuracy"),
                    "training_data_size": model["training_data_size"]
                }
            )
            
            logger.info(f"ML model trained: {model_id} - {model['name']}")
            return {
                "model_id": model_id,
                "status": "trained",
                "metrics": metrics,
                "training_data_size": model["training_data_size"]
            }
        
        except Exception as e:
            logger.error(f"Failed to train ML model: {e}")
            if model_id in self.models:
                self.models[model_id]["status"] = ModelStatus.FAILED.value
                self.ml_stats["failed_models"] += 1
            raise
    
    async def _train_model_by_algorithm(
        self,
        algorithm: str,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        parameters: Dict[str, Any]
    ):
        """Train model based on algorithm"""
        try:
            if algorithm == MLAlgorithm.RANDOM_FOREST.value:
                if model_type == ModelType.CLASSIFICATION.value:
                    model = RandomForestClassifier(
                        n_estimators=parameters.get("n_estimators", 100),
                        max_depth=parameters.get("max_depth", None),
                        random_state=parameters.get("random_state", 42)
                    )
                else:  # regression
                    model = RandomForestRegressor(
                        n_estimators=parameters.get("n_estimators", 100),
                        max_depth=parameters.get("max_depth", None),
                        random_state=parameters.get("random_state", 42)
                    )
            
            elif algorithm == MLAlgorithm.LOGISTIC_REGRESSION.value:
                model = LogisticRegression(
                    random_state=parameters.get("random_state", 42),
                    max_iter=parameters.get("max_iter", 1000)
                )
            
            elif algorithm == MLAlgorithm.LINEAR_REGRESSION.value:
                model = LinearRegression()
            
            else:
                # Default to Random Forest
                if model_type == ModelType.CLASSIFICATION.value:
                    model = RandomForestClassifier(random_state=42)
                else:
                    model = RandomForestRegressor(random_state=42)
            
            model.fit(X_train, y_train)
            return model
        
        except Exception as e:
            logger.error(f"Failed to train model with algorithm {algorithm}: {e}")
            raise
    
    async def _evaluate_model(
        self,
        model,
        model_type: str,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate model performance"""
        try:
            y_pred = model.predict(X_test)
            metrics = {}
            
            if model_type == ModelType.CLASSIFICATION.value:
                metrics["accuracy"] = accuracy_score(y_test, y_pred)
                metrics["precision"] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                metrics["recall"] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                metrics["f1_score"] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            else:  # regression
                metrics["mse"] = mean_squared_error(y_test, y_pred)
                metrics["r2_score"] = r2_score(y_test, y_pred)
            
            return metrics
        
        except Exception as e:
            logger.error(f"Failed to evaluate model: {e}")
            return {}
    
    async def predict(
        self,
        model_id: str,
        input_data: List[Dict[str, Any]],
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """Make predictions using trained model"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model not found: {model_id}")
            
            model_config = self.models[model_id]
            
            if model_config["status"] != ModelStatus.TRAINED.value:
                raise ValueError(f"Model is not trained: {model_id}")
            
            if model_id not in self.model_storage:
                raise ValueError(f"Model storage not found: {model_id}")
            
            # Get model components
            model_storage = self.model_storage[model_id]
            model = model_storage["model"]
            scaler = model_storage["scaler"]
            encoder = model_storage["encoder"]
            features = model_storage["features"]
            
            # Convert input data to DataFrame
            df = pd.DataFrame(input_data)
            
            # Ensure all required features are present
            missing_features = set(features) - set(df.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Select and order features
            X = df[features]
            
            # Handle categorical variables
            if encoder:
                categorical_columns = X.select_dtypes(include=['object']).columns
                for col in categorical_columns:
                    if col in X.columns:
                        X[col] = encoder.transform(X[col].astype(str))
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Make predictions
            predictions = model.predict(X_scaled)
            
            # Get probabilities if requested and available
            probabilities = None
            if return_probabilities and hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_scaled)
            
            # Update statistics
            self.ml_stats["total_predictions"] += len(predictions)
            
            # Cache predictions
            cache_key = f"{model_id}_{hash(str(input_data))}"
            self.prediction_cache[cache_key] = {
                "predictions": predictions.tolist(),
                "probabilities": probabilities.tolist() if probabilities is not None else None,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Track analytics
            await analytics_service.track_event(
                "ml_prediction_made",
                {
                    "model_id": model_id,
                    "model_type": model_config["type"],
                    "algorithm": model_config["algorithm"],
                    "input_size": len(input_data),
                    "predictions_count": len(predictions)
                }
            )
            
            result = {
                "model_id": model_id,
                "predictions": predictions.tolist(),
                "input_size": len(input_data),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if probabilities is not None:
                result["probabilities"] = probabilities.tolist()
            
            return result
        
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise
    
    async def deploy_model(self, model_id: str) -> bool:
        """Deploy model for production use"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model not found: {model_id}")
            
            model = self.models[model_id]
            
            if model["status"] != ModelStatus.TRAINED.value:
                raise ValueError(f"Model is not trained: {model_id}")
            
            model["status"] = ModelStatus.DEPLOYED.value
            model["deployed_at"] = datetime.utcnow().isoformat()
            
            self.ml_stats["deployed_models"] += 1
            
            # Track analytics
            await analytics_service.track_event(
                "ml_model_deployed",
                {
                    "model_id": model_id,
                    "model_type": model["type"],
                    "algorithm": model["algorithm"]
                }
            )
            
            logger.info(f"ML model deployed: {model_id} - {model['name']}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            return False
    
    async def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model information"""
        try:
            return self.models.get(model_id)
        
        except Exception as e:
            logger.error(f"Failed to get model: {e}")
            return None
    
    async def list_models(
        self,
        model_type: Optional[ModelType] = None,
        status: Optional[ModelStatus] = None,
        algorithm: Optional[MLAlgorithm] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List models with filtering"""
        try:
            filtered_models = []
            
            for model in self.models.values():
                if model_type and model["type"] != model_type.value:
                    continue
                if status and model["status"] != status.value:
                    continue
                if algorithm and model["algorithm"] != algorithm.value:
                    continue
                
                filtered_models.append({
                    "id": model["id"],
                    "name": model["name"],
                    "type": model["type"],
                    "algorithm": model["algorithm"],
                    "status": model["status"],
                    "created_at": model["created_at"],
                    "trained_at": model["trained_at"],
                    "deployed_at": model["deployed_at"],
                    "accuracy": model["accuracy"],
                    "precision": model["precision"],
                    "recall": model["recall"],
                    "f1_score": model["f1_score"],
                    "mse": model["mse"],
                    "r2_score": model["r2_score"],
                    "training_data_size": model["training_data_size"]
                })
            
            # Sort by created_at (newest first)
            filtered_models.sort(key=lambda x: x["created_at"], reverse=True)
            
            return filtered_models[:limit]
        
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    async def get_model_performance(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model performance metrics"""
        try:
            if model_id not in self.models:
                return None
            
            model = self.models[model_id]
            
            return {
                "model_id": model_id,
                "name": model["name"],
                "type": model["type"],
                "algorithm": model["algorithm"],
                "status": model["status"],
                "accuracy": model["accuracy"],
                "precision": model["precision"],
                "recall": model["recall"],
                "f1_score": model["f1_score"],
                "mse": model["mse"],
                "r2_score": model["r2_score"],
                "training_data_size": model["training_data_size"],
                "features": model["features"],
                "target": model["target"]
            }
        
        except Exception as e:
            logger.error(f"Failed to get model performance: {e}")
            return None
    
    async def retrain_model(
        self,
        model_id: str,
        new_training_data: List[Dict[str, Any]],
        target_column: str,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """Retrain existing model with new data"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model not found: {model_id}")
            
            # Get original model config
            original_model = self.models[model_id]
            
            # Retrain with new data
            result = await self.train_model(
                model_id=model_id,
                training_data=new_training_data,
                target_column=target_column,
                test_size=test_size
            )
            
            # Track analytics
            await analytics_service.track_event(
                "ml_model_retrained",
                {
                    "model_id": model_id,
                    "model_type": original_model["type"],
                    "algorithm": original_model["algorithm"],
                    "new_training_data_size": len(new_training_data)
                }
            )
            
            logger.info(f"ML model retrained: {model_id} - {original_model['name']}")
            return result
        
        except Exception as e:
            logger.error(f"Failed to retrain model: {e}")
            raise
    
    async def get_ml_stats(self) -> Dict[str, Any]:
        """Get ML service statistics"""
        try:
            return {
                "total_models": self.ml_stats["total_models"],
                "trained_models": self.ml_stats["trained_models"],
                "deployed_models": self.ml_stats["deployed_models"],
                "failed_models": self.ml_stats["failed_models"],
                "total_predictions": self.ml_stats["total_predictions"],
                "models_by_type": self.ml_stats["models_by_type"],
                "models_by_algorithm": self.ml_stats["models_by_algorithm"],
                "cached_predictions": len(self.prediction_cache),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get ML stats: {e}")
            return {"error": str(e)}


# Global machine learning service instance
ml_service = MachineLearningService()

