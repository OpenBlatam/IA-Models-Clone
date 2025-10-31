"""
Machine Learning Pipeline
=========================

Advanced machine learning pipeline for AI model analysis with automated
model training, evaluation, deployment, and monitoring capabilities.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import hashlib
import joblib
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Model types"""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"
    DEEP_LEARNING = "deep_learning"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"


class ModelStatus(str, Enum):
    """Model status"""
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATING = "evaluating"
    DEPLOYED = "deployed"
    FAILED = "failed"
    RETIRED = "retired"


class PipelineStage(str, Enum):
    """Pipeline stages"""
    DATA_PREPARATION = "data_preparation"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    MODEL_DEPLOYMENT = "model_deployment"
    MODEL_MONITORING = "model_monitoring"


@dataclass
class ModelConfiguration:
    """Model configuration"""
    config_id: str
    name: str
    description: str
    model_type: ModelType
    algorithm: str
    hyperparameters: Dict[str, Any]
    feature_columns: List[str]
    target_column: str
    preprocessing_steps: List[Dict[str, Any]]
    validation_strategy: str
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class ModelInstance:
    """Model instance"""
    model_id: str
    config_id: str
    name: str
    model_type: ModelType
    algorithm: str
    model_object: Any
    training_data_size: int
    training_time: float
    status: ModelStatus
    metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class TrainingResult:
    """Training result"""
    model_id: str
    config_id: str
    training_time: float
    validation_scores: Dict[str, float]
    test_scores: Dict[str, float]
    feature_importance: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_data_size: int
    status: str
    error_message: str = ""
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class PredictionResult:
    """Prediction result"""
    prediction_id: str
    model_id: str
    input_data: Dict[str, Any]
    predictions: List[float]
    probabilities: List[float]
    confidence: float
    processing_time: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class MachineLearningPipeline:
    """Advanced machine learning pipeline for AI model analysis"""
    
    def __init__(self, max_models: int = 1000):
        self.max_models = max_models
        
        self.model_configurations: Dict[str, ModelConfiguration] = {}
        self.model_instances: Dict[str, ModelInstance] = {}
        self.training_results: List[TrainingResult] = []
        self.prediction_results: List[PredictionResult] = []
        
        # Model registry
        self.model_registry: Dict[str, Any] = {}
        
        # Available algorithms
        self.regression_algorithms = {
            "random_forest": RandomForestRegressor,
            "linear_regression": LinearRegression,
            "svr": SVR,
            "xgboost": xgb.XGBRegressor,
            "lightgbm": lgb.LGBMRegressor
        }
        
        self.classification_algorithms = {
            "random_forest": RandomForestClassifier,
            "logistic_regression": LogisticRegression,
            "svc": SVC,
            "xgboost": xgb.XGBClassifier,
            "lightgbm": lgb.LGBMClassifier
        }
        
        # Pipeline settings
        self.default_test_size = 0.2
        self.default_random_state = 42
        self.default_cv_folds = 5
        
        # Model storage
        self.model_storage_path = "./models"
        import os
        os.makedirs(self.model_storage_path, exist_ok=True)
    
    async def create_model_configuration(self, 
                                       name: str,
                                       description: str,
                                       model_type: ModelType,
                                       algorithm: str,
                                       hyperparameters: Dict[str, Any] = None,
                                       feature_columns: List[str] = None,
                                       target_column: str = None,
                                       preprocessing_steps: List[Dict[str, Any]] = None,
                                       validation_strategy: str = "holdout") -> ModelConfiguration:
        """Create model configuration"""
        try:
            config_id = hashlib.md5(f"{name}_{model_type}_{datetime.now()}".encode()).hexdigest()
            
            if hyperparameters is None:
                hyperparameters = {}
            if feature_columns is None:
                feature_columns = []
            if preprocessing_steps is None:
                preprocessing_steps = []
            
            config = ModelConfiguration(
                config_id=config_id,
                name=name,
                description=description,
                model_type=model_type,
                algorithm=algorithm,
                hyperparameters=hyperparameters,
                feature_columns=feature_columns,
                target_column=target_column,
                preprocessing_steps=preprocessing_steps,
                validation_strategy=validation_strategy
            )
            
            self.model_configurations[config_id] = config
            
            logger.info(f"Created model configuration: {name}")
            
            return config
            
        except Exception as e:
            logger.error(f"Error creating model configuration: {str(e)}")
            raise e
    
    async def train_model(self, 
                        config_id: str,
                        training_data: pd.DataFrame,
                        target_data: pd.Series = None) -> TrainingResult:
        """Train model with given configuration"""
        try:
            if config_id not in self.model_configurations:
                raise ValueError(f"Model configuration {config_id} not found")
            
            config = self.model_configurations[config_id]
            model_id = hashlib.md5(f"{config_id}_{datetime.now()}".encode()).hexdigest()
            
            start_time = time.time()
            
            # Prepare data
            X, y = await self._prepare_training_data(config, training_data, target_data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.default_test_size, random_state=self.default_random_state
            )
            
            # Create model
            model = await self._create_model(config)
            
            # Train model
            model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            
            # Evaluate model
            validation_scores, test_scores = await self._evaluate_model(
                model, X_train, y_train, X_test, y_test, config.model_type
            )
            
            # Get feature importance
            feature_importance = await self._get_feature_importance(model, config.feature_columns)
            
            # Create training result
            result = TrainingResult(
                model_id=model_id,
                config_id=config_id,
                training_time=training_time,
                validation_scores=validation_scores,
                test_scores=test_scores,
                feature_importance=feature_importance,
                hyperparameters=config.hyperparameters,
                training_data_size=len(X_train),
                status="completed"
            )
            
            # Create model instance
            model_instance = ModelInstance(
                model_id=model_id,
                config_id=config_id,
                name=config.name,
                model_type=config.model_type,
                algorithm=config.algorithm,
                model_object=model,
                training_data_size=len(X_train),
                training_time=training_time,
                status=ModelStatus.TRAINED,
                metrics=test_scores,
                feature_importance=feature_importance
            )
            
            # Store results
            self.training_results.append(result)
            self.model_instances[model_id] = model_instance
            
            # Save model
            await self._save_model(model_instance)
            
            logger.info(f"Trained model: {config.name} ({model_id})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            
            # Create failed result
            failed_result = TrainingResult(
                model_id=model_id if 'model_id' in locals() else "",
                config_id=config_id,
                training_time=0.0,
                validation_scores={},
                test_scores={},
                feature_importance={},
                hyperparameters=config.hyperparameters if 'config' in locals() else {},
                training_data_size=0,
                status="failed",
                error_message=str(e)
            )
            
            self.training_results.append(failed_result)
            
            return failed_result
    
    async def predict(self, 
                     model_id: str,
                     input_data: Dict[str, Any]) -> PredictionResult:
        """Make prediction with trained model"""
        try:
            if model_id not in self.model_instances:
                raise ValueError(f"Model {model_id} not found")
            
            model_instance = self.model_instances[model_id]
            
            if model_instance.status != ModelStatus.TRAINED:
                raise ValueError(f"Model {model_id} is not trained")
            
            start_time = time.time()
            
            # Prepare input data
            X = await self._prepare_prediction_data(model_instance, input_data)
            
            # Make prediction
            predictions = model_instance.model_object.predict(X)
            
            # Get probabilities if available
            probabilities = []
            if hasattr(model_instance.model_object, 'predict_proba'):
                probabilities = model_instance.model_object.predict_proba(X).tolist()
            
            # Calculate confidence
            confidence = await self._calculate_prediction_confidence(
                model_instance, predictions, probabilities
            )
            
            processing_time = time.time() - start_time
            
            # Create prediction result
            result = PredictionResult(
                prediction_id=hashlib.md5(f"{model_id}_{datetime.now()}".encode()).hexdigest(),
                model_id=model_id,
                input_data=input_data,
                predictions=predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
                probabilities=probabilities,
                confidence=confidence,
                processing_time=processing_time
            )
            
            self.prediction_results.append(result)
            
            logger.info(f"Made prediction with model {model_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise e
    
    async def evaluate_model_performance(self, 
                                       model_id: str,
                                       test_data: pd.DataFrame,
                                       target_data: pd.Series = None) -> Dict[str, Any]:
        """Evaluate model performance on test data"""
        try:
            if model_id not in self.model_instances:
                raise ValueError(f"Model {model_id} not found")
            
            model_instance = self.model_instances[model_id]
            config = self.model_configurations[model_instance.config_id]
            
            # Prepare test data
            X_test, y_test = await self._prepare_training_data(config, test_data, target_data)
            
            # Make predictions
            y_pred = model_instance.model_object.predict(X_test)
            
            # Calculate metrics
            metrics = await self._calculate_metrics(
                y_test, y_pred, model_instance.model_type
            )
            
            # Calculate additional performance metrics
            performance_metrics = {
                "model_id": model_id,
                "model_name": model_instance.name,
                "evaluation_timestamp": datetime.now().isoformat(),
                "test_data_size": len(X_test),
                "metrics": metrics,
                "feature_importance": model_instance.feature_importance,
                "model_metadata": {
                    "algorithm": model_instance.algorithm,
                    "training_time": model_instance.training_time,
                    "training_data_size": model_instance.training_data_size
                }
            }
            
            logger.info(f"Evaluated model performance: {model_id}")
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model performance: {str(e)}")
            return {"error": str(e)}
    
    async def hyperparameter_optimization(self, 
                                        config_id: str,
                                        training_data: pd.DataFrame,
                                        target_data: pd.Series = None,
                                        param_grid: Dict[str, List] = None) -> Dict[str, Any]:
        """Perform hyperparameter optimization"""
        try:
            if config_id not in self.model_configurations:
                raise ValueError(f"Model configuration {config_id} not found")
            
            config = self.model_configurations[config_id]
            
            # Prepare data
            X, y = await self._prepare_training_data(config, training_data, target_data)
            
            # Create base model
            base_model = await self._create_model(config)
            
            # Default parameter grid if not provided
            if param_grid is None:
                param_grid = await self._get_default_param_grid(config.algorithm)
            
            # Perform grid search
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=self.default_cv_folds,
                scoring='accuracy' if config.model_type == ModelType.CLASSIFICATION else 'r2',
                n_jobs=-1
            )
            
            start_time = time.time()
            grid_search.fit(X, y)
            optimization_time = time.time() - start_time
            
            # Get results
            optimization_results = {
                "config_id": config_id,
                "best_params": grid_search.best_params_,
                "best_score": grid_search.best_score_,
                "optimization_time": optimization_time,
                "cv_results": {
                    "mean_test_score": grid_search.cv_results_['mean_test_score'].tolist(),
                    "std_test_score": grid_search.cv_results_['std_test_score'].tolist(),
                    "params": grid_search.cv_results_['params']
                },
                "param_grid": param_grid
            }
            
            logger.info(f"Completed hyperparameter optimization: {config_id}")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {str(e)}")
            return {"error": str(e)}
    
    async def create_ensemble_model(self, 
                                  model_ids: List[str],
                                  ensemble_method: str = "voting",
                                  weights: List[float] = None) -> ModelInstance:
        """Create ensemble model from multiple models"""
        try:
            if not model_ids:
                raise ValueError("No model IDs provided")
            
            # Validate all models exist and are trained
            models = []
            for model_id in model_ids:
                if model_id not in self.model_instances:
                    raise ValueError(f"Model {model_id} not found")
                
                model_instance = self.model_instances[model_id]
                if model_instance.status != ModelStatus.TRAINED:
                    raise ValueError(f"Model {model_id} is not trained")
                
                models.append(model_instance)
            
            # Create ensemble model
            ensemble_id = hashlib.md5(f"ensemble_{'_'.join(model_ids)}_{datetime.now()}".encode()).hexdigest()
            
            if ensemble_method == "voting":
                from sklearn.ensemble import VotingRegressor, VotingClassifier
                
                # Determine if regression or classification
                model_type = models[0].model_type
                
                if model_type == ModelType.REGRESSION:
                    ensemble_model = VotingRegressor(
                        [(f"model_{i}", model.model_object) for i, model in enumerate(models)],
                        weights=weights
                    )
                else:
                    ensemble_model = VotingClassifier(
                        [(f"model_{i}", model.model_object) for i, model in enumerate(models)],
                        weights=weights,
                        voting='soft' if all(hasattr(m.model_object, 'predict_proba') for m in models) else 'hard'
                    )
            
            # Create ensemble model instance
            ensemble_instance = ModelInstance(
                model_id=ensemble_id,
                config_id="ensemble",
                name=f"Ensemble Model ({len(models)} models)",
                model_type=models[0].model_type,
                algorithm="ensemble",
                model_object=ensemble_model,
                training_data_size=sum(m.training_data_size for m in models),
                training_time=sum(m.training_time for m in models),
                status=ModelStatus.TRAINED,
                metrics={},
                feature_importance={}
            )
            
            self.model_instances[ensemble_id] = ensemble_instance
            
            logger.info(f"Created ensemble model: {ensemble_id}")
            
            return ensemble_instance
            
        except Exception as e:
            logger.error(f"Error creating ensemble model: {str(e)}")
            raise e
    
    async def get_model_analytics(self, 
                                time_range_days: int = 30) -> Dict[str, Any]:
        """Get model analytics"""
        try:
            cutoff_date = datetime.now() - timedelta(days=time_range_days)
            
            # Filter recent data
            recent_training_results = [r for r in self.training_results if r.created_at >= cutoff_date]
            recent_predictions = [p for p in self.prediction_results if p.timestamp >= cutoff_date]
            
            analytics = {
                "total_models": len(self.model_instances),
                "trained_models": len([m for m in self.model_instances.values() if m.status == ModelStatus.TRAINED]),
                "total_training_runs": len(recent_training_results),
                "successful_training_runs": len([r for r in recent_training_results if r.status == "completed"]),
                "total_predictions": len(recent_predictions),
                "average_training_time": 0.0,
                "average_prediction_time": 0.0,
                "model_types": {},
                "algorithms": {},
                "performance_distribution": {},
                "top_performing_models": []
            }
            
            if recent_training_results:
                # Calculate average training time
                training_times = [r.training_time for r in recent_training_results if r.training_time > 0]
                if training_times:
                    analytics["average_training_time"] = sum(training_times) / len(training_times)
            
            if recent_predictions:
                # Calculate average prediction time
                prediction_times = [p.processing_time for p in recent_predictions if p.processing_time > 0]
                if prediction_times:
                    analytics["average_prediction_time"] = sum(prediction_times) / len(prediction_times)
            
            # Analyze model types
            for model in self.model_instances.values():
                model_type = model.model_type.value
                if model_type not in analytics["model_types"]:
                    analytics["model_types"][model_type] = 0
                analytics["model_types"][model_type] += 1
            
            # Analyze algorithms
            for model in self.model_instances.values():
                algorithm = model.algorithm
                if algorithm not in analytics["algorithms"]:
                    analytics["algorithms"][algorithm] = 0
                analytics["algorithms"][algorithm] += 1
            
            # Find top performing models
            trained_models = [m for m in self.model_instances.values() if m.status == ModelStatus.TRAINED and m.metrics]
            if trained_models:
                # Sort by primary metric
                primary_metric = "accuracy" if trained_models[0].model_type == ModelType.CLASSIFICATION else "r2_score"
                
                top_models = sorted(
                    trained_models,
                    key=lambda m: m.metrics.get(primary_metric, 0),
                    reverse=True
                )[:10]
                
                analytics["top_performing_models"] = [
                    {
                        "model_id": m.model_id,
                        "name": m.name,
                        "algorithm": m.algorithm,
                        "primary_metric": m.metrics.get(primary_metric, 0),
                        "all_metrics": m.metrics
                    }
                    for m in top_models
                ]
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting model analytics: {str(e)}")
            return {"error": str(e)}
    
    # Private helper methods
    async def _prepare_training_data(self, 
                                   config: ModelConfiguration, 
                                   training_data: pd.DataFrame, 
                                   target_data: pd.Series = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data"""
        try:
            # Select features
            if config.feature_columns:
                X = training_data[config.feature_columns]
            else:
                X = training_data.select_dtypes(include=[np.number])
            
            # Get target
            if target_data is not None:
                y = target_data
            elif config.target_column:
                y = training_data[config.target_column]
            else:
                raise ValueError("Target data not provided")
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise e
    
    async def _create_model(self, config: ModelConfiguration) -> Any:
        """Create model instance"""
        try:
            algorithm = config.algorithm.lower()
            
            if config.model_type == ModelType.REGRESSION:
                if algorithm not in self.regression_algorithms:
                    raise ValueError(f"Unsupported regression algorithm: {algorithm}")
                model_class = self.regression_algorithms[algorithm]
            elif config.model_type == ModelType.CLASSIFICATION:
                if algorithm not in self.classification_algorithms:
                    raise ValueError(f"Unsupported classification algorithm: {algorithm}")
                model_class = self.classification_algorithms[algorithm]
            else:
                raise ValueError(f"Unsupported model type: {config.model_type}")
            
            # Create model with hyperparameters
            model = model_class(**config.hyperparameters)
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            raise e
    
    async def _evaluate_model(self, 
                            model: Any, 
                            X_train: pd.DataFrame, 
                            y_train: pd.Series, 
                            X_test: pd.DataFrame, 
                            y_test: pd.Series, 
                            model_type: ModelType) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Evaluate model performance"""
        try:
            # Cross-validation on training data
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=self.default_cv_folds,
                scoring='accuracy' if model_type == ModelType.CLASSIFICATION else 'r2'
            )
            
            validation_scores = {
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std()
            }
            
            # Test set evaluation
            y_pred = model.predict(X_test)
            test_scores = await self._calculate_metrics(y_test, y_pred, model_type)
            
            return validation_scores, test_scores
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {}, {}
    
    async def _calculate_metrics(self, 
                               y_true: pd.Series, 
                               y_pred: np.ndarray, 
                               model_type: ModelType) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        try:
            metrics = {}
            
            if model_type == ModelType.CLASSIFICATION:
                metrics["accuracy"] = accuracy_score(y_true, y_pred)
                metrics["precision"] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics["recall"] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics["f1_score"] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            else:
                metrics["r2_score"] = r2_score(y_true, y_pred)
                metrics["mse"] = mean_squared_error(y_true, y_pred)
                metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
                metrics["mae"] = np.mean(np.abs(y_true - y_pred))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    async def _get_feature_importance(self, model: Any, feature_columns: List[str]) -> Dict[str, float]:
        """Get feature importance"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(feature_columns, model.feature_importances_))
                return importance_dict
            elif hasattr(model, 'coef_'):
                # For linear models
                if len(model.coef_.shape) == 1:
                    importance_dict = dict(zip(feature_columns, model.coef_))
                else:
                    # Multi-class case
                    importance_dict = dict(zip(feature_columns, np.mean(np.abs(model.coef_), axis=0)))
                return importance_dict
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    async def _prepare_prediction_data(self, 
                                     model_instance: ModelInstance, 
                                     input_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare data for prediction"""
        try:
            config = self.model_configurations[model_instance.config_id]
            
            # Convert input data to DataFrame
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            else:
                df = pd.DataFrame(input_data)
            
            # Select features
            if config.feature_columns:
                X = df[config.feature_columns]
            else:
                X = df.select_dtypes(include=[np.number])
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            return X
            
        except Exception as e:
            logger.error(f"Error preparing prediction data: {str(e)}")
            raise e
    
    async def _calculate_prediction_confidence(self, 
                                             model_instance: ModelInstance, 
                                             predictions: np.ndarray, 
                                             probabilities: List[float]) -> float:
        """Calculate prediction confidence"""
        try:
            if probabilities:
                # Use maximum probability as confidence
                return max(probabilities)
            else:
                # Simple confidence based on model performance
                primary_metric = "accuracy" if model_instance.model_type == ModelType.CLASSIFICATION else "r2_score"
                return model_instance.metrics.get(primary_metric, 0.5)
                
        except Exception as e:
            logger.error(f"Error calculating prediction confidence: {str(e)}")
            return 0.5
    
    async def _get_default_param_grid(self, algorithm: str) -> Dict[str, List]:
        """Get default parameter grid for hyperparameter optimization"""
        try:
            param_grids = {
                "random_forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10]
                },
                "linear_regression": {
                    "fit_intercept": [True, False]
                },
                "logistic_regression": {
                    "C": [0.1, 1, 10],
                    "penalty": ["l1", "l2"]
                },
                "svr": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"],
                    "gamma": ["scale", "auto"]
                },
                "svc": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"],
                    "gamma": ["scale", "auto"]
                },
                "xgboost": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 6, 9],
                    "learning_rate": [0.01, 0.1, 0.2]
                },
                "lightgbm": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 6, 9],
                    "learning_rate": [0.01, 0.1, 0.2]
                }
            }
            
            return param_grids.get(algorithm.lower(), {})
            
        except Exception as e:
            logger.error(f"Error getting default param grid: {str(e)}")
            return {}
    
    async def _save_model(self, model_instance: ModelInstance) -> None:
        """Save model to disk"""
        try:
            model_path = f"{self.model_storage_path}/{model_instance.model_id}.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_instance, f)
            
            logger.info(f"Saved model: {model_instance.model_id}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")


# Global ML pipeline instance
_ml_pipeline: Optional[MachineLearningPipeline] = None


def get_machine_learning_pipeline(max_models: int = 1000) -> MachineLearningPipeline:
    """Get or create global machine learning pipeline instance"""
    global _ml_pipeline
    if _ml_pipeline is None:
        _ml_pipeline = MachineLearningPipeline(max_models)
    return _ml_pipeline


# Example usage
async def main():
    """Example usage of the machine learning pipeline"""
    pipeline = get_machine_learning_pipeline()
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)  # Regression target
    
    # Convert to DataFrame
    feature_columns = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_columns)
    df["target"] = y
    
    # Create model configuration
    config = await pipeline.create_model_configuration(
        name="Performance Predictor",
        description="Predicts AI model performance",
        model_type=ModelType.REGRESSION,
        algorithm="random_forest",
        hyperparameters={"n_estimators": 100, "random_state": 42},
        feature_columns=feature_columns,
        target_column="target"
    )
    print(f"Created model configuration: {config.config_id}")
    
    # Train model
    training_result = await pipeline.train_model(
        config_id=config.config_id,
        training_data=df
    )
    print(f"Trained model: {training_result.model_id}")
    print(f"Training time: {training_result.training_time:.2f}s")
    print(f"Test R²: {training_result.test_scores.get('r2_score', 0):.3f}")
    
    # Make prediction
    prediction = await pipeline.predict(
        model_id=training_result.model_id,
        input_data={f"feature_{i}": np.random.randn() for i in range(n_features)}
    )
    print(f"Prediction: {prediction.predictions[0]:.3f}")
    print(f"Confidence: {prediction.confidence:.3f}")
    
    # Hyperparameter optimization
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20]
    }
    
    optimization_result = await pipeline.hyperparameter_optimization(
        config_id=config.config_id,
        training_data=df,
        param_grid=param_grid
    )
    print(f"Best parameters: {optimization_result.get('best_params', {})}")
    print(f"Best score: {optimization_result.get('best_score', 0):.3f}")
    
    # Get analytics
    analytics = await pipeline.get_model_analytics()
    print(f"Model analytics: {analytics.get('total_models', 0)} models")


if __name__ == "__main__":
    import time
    asyncio.run(main())

























