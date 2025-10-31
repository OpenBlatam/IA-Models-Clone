"""
Advanced ML Pipeline for Microservices
Features: Real-time model training, A/B testing, model versioning, feature engineering, model serving
"""

import asyncio
import time
import json
import pickle
import hashlib
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import statistics
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

# ML imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import sklearn
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Model types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DEEP_LEARNING = "deep_learning"
    ENSEMBLE = "ensemble"

class ModelStatus(Enum):
    """Model status"""
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    RETIRED = "retired"
    FAILED = "failed"

class ExperimentType(Enum):
    """Experiment types"""
    A_B_TEST = "a_b_test"
    MULTI_ARMED_BANDIT = "multi_armed_bandit"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GRID_SEARCH = "grid_search"

@dataclass
class ModelConfig:
    """Model configuration"""
    model_id: str
    model_type: ModelType
    algorithm: str
    hyperparameters: Dict[str, Any]
    feature_columns: List[str]
    target_column: str
    validation_split: float = 0.2
    test_split: float = 0.1
    cross_validation_folds: int = 5
    early_stopping_patience: int = 10
    max_training_time: int = 3600  # seconds

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    model_id: str
    version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    training_time: float
    inference_time: float
    timestamp: float
    dataset_size: int
    feature_importance: Dict[str, float] = field(default_factory=dict)

@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    experiment_id: str
    experiment_type: ExperimentType
    models: List[str]  # Model IDs
    traffic_split: Dict[str, float]  # Model ID -> traffic percentage
    success_metric: str
    minimum_sample_size: int = 1000
    confidence_level: float = 0.95
    max_duration: int = 86400  # 24 hours

@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    feature_name: str
    feature_type: str  # numerical, categorical, text, datetime
    transformation: str  # log, sqrt, onehot, embedding, etc.
    scaling: str  # standard, minmax, robust
    missing_value_strategy: str  # mean, median, mode, drop, impute
    outlier_detection: bool = False
    outlier_threshold: float = 3.0

class FeatureEngineer:
    """
    Advanced feature engineering pipeline
    """
    
    def __init__(self):
        self.feature_configs: Dict[str, FeatureConfig] = {}
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.feature_importance: Dict[str, float] = {}
    
    def add_feature_config(self, config: FeatureConfig):
        """Add feature configuration"""
        self.feature_configs[config.feature_name] = config
        logger.info(f"Added feature config: {config.feature_name}")
    
    async def transform_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform features according to configurations"""
        try:
            transformed_data = data.copy()
            
            for feature_name, config in self.feature_configs.items():
                if feature_name not in transformed_data.columns:
                    continue
                
                # Handle missing values
                transformed_data = await self._handle_missing_values(
                    transformed_data, feature_name, config
                )
                
                # Apply transformations
                transformed_data = await self._apply_transformation(
                    transformed_data, feature_name, config
                )
                
                # Apply scaling
                transformed_data = await self._apply_scaling(
                    transformed_data, feature_name, config
                )
                
                # Detect outliers
                if config.outlier_detection:
                    transformed_data = await self._detect_outliers(
                        transformed_data, feature_name, config
                    )
            
            return transformed_data
            
        except Exception as e:
            logger.error(f"Feature transformation failed: {e}")
            return data
    
    async def _handle_missing_values(self, data: pd.DataFrame, feature_name: str, config: FeatureConfig) -> pd.DataFrame:
        """Handle missing values"""
        if data[feature_name].isnull().sum() == 0:
            return data
        
        if config.missing_value_strategy == "mean":
            data[feature_name].fillna(data[feature_name].mean(), inplace=True)
        elif config.missing_value_strategy == "median":
            data[feature_name].fillna(data[feature_name].median(), inplace=True)
        elif config.missing_value_strategy == "mode":
            data[feature_name].fillna(data[feature_name].mode()[0], inplace=True)
        elif config.missing_value_strategy == "drop":
            data.dropna(subset=[feature_name], inplace=True)
        
        return data
    
    async def _apply_transformation(self, data: pd.DataFrame, feature_name: str, config: FeatureConfig) -> pd.DataFrame:
        """Apply feature transformation"""
        if config.transformation == "log":
            data[feature_name] = np.log1p(data[feature_name])
        elif config.transformation == "sqrt":
            data[feature_name] = np.sqrt(data[feature_name])
        elif config.transformation == "square":
            data[feature_name] = data[feature_name] ** 2
        elif config.transformation == "onehot" and config.feature_type == "categorical":
            # One-hot encoding for categorical features
            dummies = pd.get_dummies(data[feature_name], prefix=feature_name)
            data = pd.concat([data, dummies], axis=1)
            data.drop(feature_name, axis=1, inplace=True)
        
        return data
    
    async def _apply_scaling(self, data: pd.DataFrame, feature_name: str, config: FeatureConfig) -> pd.DataFrame:
        """Apply feature scaling"""
        if config.scaling == "standard":
            if feature_name not in self.scalers:
                self.scalers[feature_name] = StandardScaler()
            data[feature_name] = self.scalers[feature_name].fit_transform(data[[feature_name]])
        elif config.scaling == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            if feature_name not in self.scalers:
                self.scalers[feature_name] = MinMaxScaler()
            data[feature_name] = self.scalers[feature_name].fit_transform(data[[feature_name]])
        
        return data
    
    async def _detect_outliers(self, data: pd.DataFrame, feature_name: str, config: FeatureConfig) -> pd.DataFrame:
        """Detect and handle outliers"""
        if config.feature_type != "numerical":
            return data
        
        # Z-score based outlier detection
        z_scores = np.abs((data[feature_name] - data[feature_name].mean()) / data[feature_name].std())
        outliers = z_scores > config.outlier_threshold
        
        if outliers.sum() > 0:
            logger.warning(f"Detected {outliers.sum()} outliers in feature {feature_name}")
            # Cap outliers at threshold
            data.loc[outliers, feature_name] = data[feature_name].quantile(0.95)
        
        return data
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_importance.copy()

class MLModel(ABC):
    """Abstract ML model interface"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.status = ModelStatus.TRAINING
        self.version = "1.0.0"
        self.training_history: List[Dict[str, Any]] = []
        self.feature_importance: Dict[str, float] = {}
        self.created_at = time.time()
        self.last_trained = None
    
    @abstractmethod
    async def train(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Train the model"""
        pass
    
    @abstractmethod
    async def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    async def evaluate(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Evaluate model performance"""
        pass
    
    @abstractmethod
    async def save_model(self, path: str) -> bool:
        """Save model to disk"""
        pass
    
    @abstractmethod
    async def load_model(self, path: str) -> bool:
        """Load model from disk"""
        pass

class SklearnModel(MLModel):
    """Scikit-learn model implementation"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on algorithm"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn not available")
        
        algorithm = self.config.algorithm.lower()
        
        if algorithm == "random_forest":
            self.model = RandomForestClassifier(**self.config.hyperparameters)
        elif algorithm == "gradient_boosting":
            self.model = GradientBoostingClassifier(**self.config.hyperparameters)
        elif algorithm == "logistic_regression":
            self.model = LogisticRegression(**self.config.hyperparameters)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    async def train(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Train the model"""
        try:
            start_time = time.time()
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.config.validation_split, random_state=42
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = self.model.predict(X_val)
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, average='weighted')
            recall = recall_score(y_val, y_pred, average='weighted')
            f1 = f1_score(y_val, y_pred, average='weighted')
            
            training_time = time.time() - start_time
            
            # Get feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(zip(
                    X.columns, self.model.feature_importances_
                ))
            
            metrics = ModelMetrics(
                model_id=self.config.model_id,
                version=self.version,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                auc_roc=0.0,  # Would calculate for binary classification
                training_time=training_time,
                inference_time=0.0,  # Would measure
                timestamp=time.time(),
                dataset_size=len(X)
            )
            
            self.status = ModelStatus.TRAINED
            self.last_trained = time.time()
            self.training_history.append(metrics.__dict__)
            
            logger.info(f"Model {self.config.model_id} trained successfully")
            return metrics
            
        except Exception as e:
            self.status = ModelStatus.FAILED
            logger.error(f"Model training failed: {e}")
            raise
    
    async def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        start_time = time.time()
        predictions = self.model.predict(X)
        inference_time = time.time() - start_time
        
        # Update inference time in metrics
        if self.training_history:
            self.training_history[-1]["inference_time"] = inference_time
        
        return predictions
    
    async def evaluate(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        y_pred = self.model.predict(X)
        
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')
        
        return ModelMetrics(
            model_id=self.config.model_id,
            version=self.version,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=0.0,
            training_time=0.0,
            inference_time=0.0,
            timestamp=time.time(),
            dataset_size=len(X),
            feature_importance=self.feature_importance
        )
    
    async def save_model(self, path: str) -> bool:
        """Save model to disk"""
        try:
            model_data = {
                "model": self.model,
                "config": self.config,
                "version": self.version,
                "status": self.status,
                "feature_importance": self.feature_importance,
                "training_history": self.training_history,
                "created_at": self.created_at,
                "last_trained": self.last_trained
            }
            
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    async def load_model(self, path: str) -> bool:
        """Load model from disk"""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data["model"]
            self.version = model_data["version"]
            self.status = model_data["status"]
            self.feature_importance = model_data["feature_importance"]
            self.training_history = model_data["training_history"]
            self.created_at = model_data["created_at"]
            self.last_trained = model_data["last_trained"]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

class XGBoostModel(MLModel):
    """XGBoost model implementation"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize XGBoost model"""
        if self.config.model_type == ModelType.CLASSIFICATION:
            self.model = xgb.XGBClassifier(**self.config.hyperparameters)
        else:
            self.model = xgb.XGBRegressor(**self.config.hyperparameters)
    
    async def train(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Train XGBoost model"""
        try:
            start_time = time.time()
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.config.validation_split, random_state=42
            )
            
            # Train model
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=self.config.early_stopping_patience,
                verbose=False
            )
            
            # Evaluate
            y_pred = self.model.predict(X_val)
            
            if self.config.model_type == ModelType.CLASSIFICATION:
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred, average='weighted')
                recall = recall_score(y_val, y_pred, average='weighted')
                f1 = f1_score(y_val, y_pred, average='weighted')
            else:
                from sklearn.metrics import mean_squared_error, r2_score
                accuracy = r2_score(y_val, y_pred)
                precision = 0.0
                recall = 0.0
                f1 = 0.0
            
            training_time = time.time() - start_time
            
            # Get feature importance
            self.feature_importance = dict(zip(
                X.columns, self.model.feature_importances_
            ))
            
            metrics = ModelMetrics(
                model_id=self.config.model_id,
                version=self.version,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                auc_roc=0.0,
                training_time=training_time,
                inference_time=0.0,
                timestamp=time.time(),
                dataset_size=len(X),
                feature_importance=self.feature_importance
            )
            
            self.status = ModelStatus.TRAINED
            self.last_trained = time.time()
            self.training_history.append(metrics.__dict__)
            
            logger.info(f"XGBoost model {self.config.model_id} trained successfully")
            return metrics
            
        except Exception as e:
            self.status = ModelStatus.FAILED
            logger.error(f"XGBoost model training failed: {e}")
            raise
    
    async def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        start_time = time.time()
        predictions = self.model.predict(X)
        inference_time = time.time() - start_time
        
        if self.training_history:
            self.training_history[-1]["inference_time"] = inference_time
        
        return predictions
    
    async def evaluate(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        y_pred = self.model.predict(X)
        
        if self.config.model_type == ModelType.CLASSIFICATION:
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted')
            recall = recall_score(y, y_pred, average='weighted')
            f1 = f1_score(y, y_pred, average='weighted')
        else:
            from sklearn.metrics import mean_squared_error, r2_score
            accuracy = r2_score(y, y_pred)
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        
        return ModelMetrics(
            model_id=self.config.model_id,
            version=self.version,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=0.0,
            training_time=0.0,
            inference_time=0.0,
            timestamp=time.time(),
            dataset_size=len(X),
            feature_importance=self.feature_importance
        )
    
    async def save_model(self, path: str) -> bool:
        """Save XGBoost model"""
        try:
            model_data = {
                "model": self.model,
                "config": self.config,
                "version": self.version,
                "status": self.status,
                "feature_importance": self.feature_importance,
                "training_history": self.training_history,
                "created_at": self.created_at,
                "last_trained": self.last_trained
            }
            
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save XGBoost model: {e}")
            return False
    
    async def load_model(self, path: str) -> bool:
        """Load XGBoost model"""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data["model"]
            self.version = model_data["version"]
            self.status = model_data["status"]
            self.feature_importance = model_data["feature_importance"]
            self.training_history = model_data["training_history"]
            self.created_at = model_data["created_at"]
            self.last_trained = model_data["last_trained"]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
            return False

class ExperimentManager:
    """
    A/B testing and experiment management
    """
    
    def __init__(self):
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.experiment_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.active_experiments: Dict[str, str] = {}  # user_id -> experiment_id
    
    def create_experiment(self, config: ExperimentConfig) -> bool:
        """Create a new experiment"""
        try:
            # Validate traffic split
            total_traffic = sum(config.traffic_split.values())
            if abs(total_traffic - 1.0) > 0.01:
                raise ValueError("Traffic split must sum to 1.0")
            
            self.experiments[config.experiment_id] = config
            logger.info(f"Created experiment: {config.experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            return False
    
    def assign_user_to_experiment(self, user_id: str, experiment_id: str) -> str:
        """Assign user to experiment variant"""
        if experiment_id not in self.experiments:
            return "control"
        
        experiment = self.experiments[experiment_id]
        
        # Simple hash-based assignment
        user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        assignment_value = (user_hash % 10000) / 10000.0
        
        cumulative_prob = 0.0
        for model_id, traffic_percentage in experiment.traffic_split.items():
            cumulative_prob += traffic_percentage
            if assignment_value <= cumulative_prob:
                self.active_experiments[user_id] = experiment_id
                return model_id
        
        return "control"
    
    def record_experiment_result(self, user_id: str, experiment_id: str, result: Dict[str, Any]):
        """Record experiment result"""
        self.experiment_results[experiment_id].append({
            "user_id": user_id,
            "timestamp": time.time(),
            "result": result
        })
    
    def get_experiment_stats(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment statistics"""
        if experiment_id not in self.experiments:
            return {"error": "Experiment not found"}
        
        experiment = self.experiments[experiment_id]
        results = self.experiment_results[experiment_id]
        
        if not results:
            return {"status": "no_data"}
        
        # Calculate statistics for each variant
        variant_stats = {}
        for model_id in experiment.traffic_split.keys():
            variant_results = [r for r in results if r["result"].get("model_id") == model_id]
            
            if variant_results:
                success_values = [r["result"].get(experiment.success_metric, 0) for r in variant_results]
                variant_stats[model_id] = {
                    "sample_size": len(variant_results),
                    "success_rate": statistics.mean(success_values),
                    "confidence_interval": self._calculate_confidence_interval(success_values)
                }
        
        return {
            "experiment_id": experiment_id,
            "total_participants": len(results),
            "variant_stats": variant_stats,
            "is_significant": self._is_statistically_significant(variant_stats)
        }
    
    def _calculate_confidence_interval(self, values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval"""
        if len(values) < 2:
            return (0.0, 0.0)
        
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values)
        n = len(values)
        
        # Simple confidence interval calculation
        margin_error = 1.96 * (std_dev / np.sqrt(n))  # 95% confidence
        
        return (mean - margin_error, mean + margin_error)
    
    def _is_statistically_significant(self, variant_stats: Dict[str, Any]) -> bool:
        """Check if experiment results are statistically significant"""
        if len(variant_stats) < 2:
            return False
        
        # Simple significance test (would use proper statistical test in production)
        success_rates = [stats["success_rate"] for stats in variant_stats.values()]
        return max(success_rates) - min(success_rates) > 0.05  # 5% difference threshold

class MLPipeline:
    """
    Main ML pipeline manager
    """
    
    def __init__(self):
        self.models: Dict[str, MLModel] = {}
        self.feature_engineer = FeatureEngineer()
        self.experiment_manager = ExperimentManager()
        self.training_queue: deque = deque()
        self.prediction_cache: Dict[str, Any] = {}
        self.pipeline_active = False
    
    def register_model(self, model: MLModel):
        """Register a model in the pipeline"""
        self.models[model.config.model_id] = model
        logger.info(f"Registered model: {model.config.model_id}")
    
    async def train_model(self, model_id: str, data: pd.DataFrame, target_column: str) -> ModelMetrics:
        """Train a model"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        
        # Prepare features
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Apply feature engineering
        X_transformed = await self.feature_engineer.transform_features(X)
        
        # Train model
        metrics = await model.train(X_transformed, y)
        
        # Save model
        model_path = f"models/{model_id}_{model.version}.pkl"
        await model.save_model(model_path)
        
        logger.info(f"Model {model_id} trained with accuracy: {metrics.accuracy:.4f}")
        return metrics
    
    async def predict(self, model_id: str, data: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        
        # Apply feature engineering
        X_transformed = await self.feature_engineer.transform_features(data)
        
        # Check cache
        cache_key = f"{model_id}:{hashlib.md5(X_transformed.to_string().encode()).hexdigest()}"
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        # Make predictions
        predictions = await model.predict(X_transformed)
        
        # Cache predictions
        self.prediction_cache[cache_key] = predictions
        
        return predictions
    
    async def create_experiment(self, config: ExperimentConfig) -> bool:
        """Create an experiment"""
        return self.experiment_manager.create_experiment(config)
    
    async def get_experiment_assignment(self, user_id: str, experiment_id: str) -> str:
        """Get experiment assignment for user"""
        return self.experiment_manager.assign_user_to_experiment(user_id, experiment_id)
    
    async def record_experiment_result(self, user_id: str, experiment_id: str, result: Dict[str, Any]):
        """Record experiment result"""
        self.experiment_manager.record_experiment_result(user_id, experiment_id, result)
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            "total_models": len(self.models),
            "trained_models": len([m for m in self.models.values() if m.status == ModelStatus.TRAINED]),
            "deployed_models": len([m for m in self.models.values() if m.status == ModelStatus.DEPLOYED]),
            "active_experiments": len(self.experiment_manager.experiments),
            "feature_configs": len(self.feature_engineer.feature_configs),
            "cached_predictions": len(self.prediction_cache)
        }

# Global ML pipeline
ml_pipeline = MLPipeline()

# Decorator for ML model serving
def ml_model_endpoint(model_id: str, use_cache: bool = True):
    """Decorator for ML model serving endpoints"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Get model
            if model_id not in ml_pipeline.models:
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            
            model = ml_pipeline.models[model_id]
            
            if model.status != ModelStatus.DEPLOYED:
                raise HTTPException(status_code=503, detail=f"Model {model_id} not deployed")
            
            # Extract features from request
            features = kwargs.get("features", {})
            if not features:
                raise HTTPException(status_code=400, detail="Features required")
            
            # Convert to DataFrame
            data = pd.DataFrame([features])
            
            # Make prediction
            predictions = await ml_pipeline.predict(model_id, data)
            
            return {
                "model_id": model_id,
                "version": model.version,
                "prediction": predictions[0].tolist() if hasattr(predictions[0], 'tolist') else predictions[0],
                "confidence": 0.95,  # Would calculate actual confidence
                "timestamp": time.time()
            }
        
        return async_wrapper
    return decorator






























