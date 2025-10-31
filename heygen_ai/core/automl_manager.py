#!/usr/bin/env python3
"""
AutoML Manager for Enhanced HeyGen AI
Automatically selects, trains, and optimizes the best models for different tasks.
"""

import asyncio
import time
import json
import structlog
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import optuna
import joblib
import pickle
from pathlib import Path
from sklearn.metrics import mean_absolute_error

logger = structlog.get_logger()

class TaskType(Enum):
    """Types of machine learning tasks."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"
    COMPUTER_VISION = "computer_vision"
    NLP = "nlp"

class ModelFamily(Enum):
    """Model families for AutoML."""
    LINEAR = "linear"
    TREE = "tree"
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"

class OptimizationMetric(Enum):
    """Optimization metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    MSE = "mse"
    RMSE = "rmse"
    MAE = "mae"
    R2_SCORE = "r2_score"

@dataclass
class AutoMLConfig:
    """AutoML configuration."""
    task_type: TaskType
    optimization_metric: OptimizationMetric
    max_trials: int = 100
    timeout_minutes: int = 60
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    enable_feature_engineering: bool = True
    enable_hyperparameter_tuning: bool = True
    enable_ensemble_methods: bool = True
    max_models_per_family: int = 3

@dataclass
class ModelCandidate:
    """Model candidate for AutoML."""
    name: str
    model_family: ModelFamily
    model: Any
    hyperparameters: Dict[str, Any]
    score: float
    training_time: float
    prediction_time: float
    memory_usage: float
    interpretability_score: float

@dataclass
class AutoMLResult:
    """AutoML optimization result."""
    best_model: ModelCandidate
    all_candidates: List[ModelCandidate]
    optimization_history: List[Dict[str, Any]]
    feature_importance: Optional[Dict[str, float]] = None
    model_explanations: Optional[Dict[str, Any]] = None
    training_curves: Optional[Dict[str, List[float]]] = None

class AutoMLManager:
    """Comprehensive AutoML management for HeyGen AI."""
    
    def __init__(
        self,
        models_dir: str = "./automl_models",
        enable_advanced_features: bool = True,
        max_workers: int = 4
    ):
        self.models_dir = Path(models_dir)
        self.enable_advanced_features = enable_advanced_features
        self.max_workers = max_workers
        
        # Create models directory
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread pool for parallel model training
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Model registry
        self.model_registry: Dict[str, ModelCandidate] = {}
        
        # Initialize model families
        self._initialize_model_families()
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        
        logger.info("AutoML Manager initialized successfully")
    
    def _initialize_model_families(self):
        """Initialize available model families."""
        self.model_families = {
            ModelFamily.LINEAR: {
                TaskType.CLASSIFICATION: [LogisticRegression],
                TaskType.REGRESSION: [LinearRegression]
            },
            ModelFamily.TREE: {
                TaskType.CLASSIFICATION: [RandomForestClassifier],
                TaskType.REGRESSION: [RandomForestRegressor]
            },
            ModelFamily.SVM: {
                TaskType.CLASSIFICATION: [SVC],
                TaskType.REGRESSION: [SVR]
            },
            ModelFamily.NEURAL_NETWORK: {
                TaskType.CLASSIFICATION: [MLPClassifier],
                TaskType.REGRESSION: [MLPRegressor]
            }
        }
    
    async def run_automl(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: AutoMLConfig
    ) -> AutoMLResult:
        """Run AutoML optimization."""
        try:
            logger.info(f"Starting AutoML optimization for {config.task_type.value} task")
            
            # Data preprocessing
            X_processed, y_processed = await self._preprocess_data(X, y, config)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed,
                test_size=config.test_size,
                random_state=config.random_state
            )
            
            # Generate model candidates
            candidates = await self._generate_model_candidates(config)
            
            # Evaluate candidates
            evaluated_candidates = await self._evaluate_candidates(
                candidates, X_train, X_test, y_train, y_test, config
            )
            
            # Select best model
            best_model = max(evaluated_candidates, key=lambda x: x.score)
            
            # Create result
            result = AutoMLResult(
                best_model=best_model,
                all_candidates=evaluated_candidates,
                optimization_history=self.performance_history
            )
            
            # Save best model
            await self._save_model(best_model, config)
            
            # Update registry
            self.model_registry[best_model.name] = best_model
            
            logger.info(f"AutoML optimization completed. Best model: {best_model.name}")
            return result
            
        except Exception as e:
            logger.error(f"AutoML optimization failed: {e}")
            raise
    
    async def _preprocess_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: AutoMLConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for AutoML."""
        try:
            X_processed = X.copy()
            y_processed = y.copy()
            
            # Handle missing values
            if np.isnan(X_processed).any():
                X_processed = np.nan_to_num(X_processed, nan=0.0)
            
            # Feature scaling
            if config.enable_feature_engineering:
                scaler = StandardScaler()
                X_processed = scaler.fit_transform(X_processed)
                
                # Save scaler for later use
                scaler_path = self.models_dir / "scaler.pkl"
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
            
            # Label encoding for classification
            if config.task_type == TaskType.CLASSIFICATION and y_processed.dtype == 'object':
                label_encoder = LabelEncoder()
                y_processed = label_encoder.fit_transform(y_processed)
                
                # Save encoder for later use
                encoder_path = self.models_dir / "label_encoder.pkl"
                with open(encoder_path, 'wb') as f:
                    pickle.dump(label_encoder, f)
            
            logger.info(f"Data preprocessing completed. Shape: {X_processed.shape}")
            return X_processed, y_processed
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            raise
    
    async def _generate_model_candidates(self, config: AutoMLConfig) -> List[ModelCandidate]:
        """Generate model candidates for the given task."""
        try:
            candidates = []
            
            # Generate candidates for each model family
            for family, task_models in self.model_families.items():
                if config.task_type in task_models:
                    models = task_models[config.task_type]
                    
                    for i, model_class in enumerate(models[:config.max_models_per_family]):
                        # Generate hyperparameter combinations
                        hyperparams_list = self._generate_hyperparameters(
                            model_class, family, config
                        )
                        
                        for j, hyperparams in enumerate(hyperparams_list):
                            candidate = ModelCandidate(
                                name=f"{family.value}_{model_class.__name__}_{i}_{j}",
                                model_family=family,
                                model=model_class,
                                hyperparameters=hyperparams,
                                score=0.0,
                                training_time=0.0,
                                prediction_time=0.0,
                                memory_usage=0.0,
                                interpretability_score=self._calculate_interpretability_score(family)
                            )
                            candidates.append(candidate)
            
            # Add custom models if enabled
            if config.enable_advanced_features:
                custom_candidates = await self._generate_custom_models(config)
                candidates.extend(custom_candidates)
            
            logger.info(f"Generated {len(candidates)} model candidates")
            return candidates
            
        except Exception as e:
            logger.error(f"Model candidate generation failed: {e}")
            raise
    
    def _generate_hyperparameters(
        self,
        model_class: Any,
        family: ModelFamily,
        config: AutoMLConfig
    ) -> List[Dict[str, Any]]:
        """Generate hyperparameter combinations for a model."""
        try:
            hyperparams_list = []
            
            if family == ModelFamily.LINEAR:
                if model_class == LogisticRegression:
                    hyperparams_list = [
                        {'C': 0.1, 'penalty': 'l2', 'solver': 'lbfgs'},
                        {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'},
                        {'C': 10.0, 'penalty': 'l2', 'solver': 'lbfgs'},
                        {'C': 0.1, 'penalty': 'l1', 'solver': 'liblinear'},
                        {'C': 1.0, 'penalty': 'l1', 'solver': 'liblinear'}
                    ]
                elif model_class == LinearRegression:
                    hyperparams_list = [{}]  # No hyperparameters for LinearRegression
            
            elif family == ModelFamily.TREE:
                if model_class in [RandomForestClassifier, RandomForestRegressor]:
                    hyperparams_list = [
                        {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2},
                        {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5},
                        {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 10},
                        {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2},
                        {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 5}
                    ]
            
            elif family == ModelFamily.SVM:
                if model_class == SVC:
                    hyperparams_list = [
                        {'C': 0.1, 'kernel': 'rbf', 'gamma': 'scale'},
                        {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'},
                        {'C': 10.0, 'kernel': 'rbf', 'gamma': 'scale'},
                        {'C': 1.0, 'kernel': 'linear'},
                        {'C': 1.0, 'kernel': 'poly', 'degree': 2}
                    ]
                elif model_class == SVR:
                    hyperparams_list = [
                        {'C': 0.1, 'kernel': 'rbf', 'gamma': 'scale'},
                        {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'},
                        {'C': 10.0, 'kernel': 'rbf', 'gamma': 'scale'}
                    ]
            
            elif family == ModelFamily.NEURAL_NETWORK:
                if model_class in [MLPClassifier, MLPRegressor]:
                    hyperparams_list = [
                        {'hidden_layer_sizes': (100,), 'alpha': 0.0001, 'max_iter': 1000},
                        {'hidden_layer_sizes': (100, 50), 'alpha': 0.0001, 'max_iter': 1000},
                        {'hidden_layer_sizes': (200, 100), 'alpha': 0.001, 'max_iter': 1000},
                        {'hidden_layer_sizes': (100, 50, 25), 'alpha': 0.0001, 'max_iter': 1000}
                    ]
            
            # If no specific hyperparameters defined, use default
            if not hyperparams_list:
                hyperparams_list = [{}]
            
            return hyperparams_list
            
        except Exception as e:
            logger.error(f"Hyperparameter generation failed: {e}")
            return [{}]
    
    async def _generate_custom_models(self, config: AutoMLConfig) -> List[ModelCandidate]:
        """Generate custom model candidates."""
        try:
            custom_candidates = []
            
            # Add ensemble methods if enabled
            if config.enable_ensemble_methods:
                # Voting classifier for classification
                if config.task_type == TaskType.CLASSIFICATION:
                    from sklearn.ensemble import VotingClassifier
                    
                    # Create base estimators
                    base_estimators = [
                        ('rf', RandomForestClassifier(n_estimators=100)),
                        ('svm', SVC(probability=True)),
                        ('mlp', MLPClassifier(hidden_layer_sizes=(100,)))
                    ]
                    
                    custom_candidates.append(ModelCandidate(
                        name="ensemble_voting_classifier",
                        model_family=ModelFamily.ENSEMBLE,
                        model=VotingClassifier,
                        hyperparameters={'estimators': base_estimators, 'voting': 'soft'},
                        score=0.0,
                        training_time=0.0,
                        prediction_time=0.0,
                        memory_usage=0.0,
                        interpretability_score=0.3
                    ))
                
                # Stacking regressor for regression
                elif config.task_type == TaskType.REGRESSION:
                    from sklearn.ensemble import StackingRegressor
                    
                    base_estimators = [
                        ('rf', RandomForestRegressor(n_estimators=100)),
                        ('svr', SVR()),
                        ('mlp', MLPRegressor(hidden_layer_sizes=(100,)))
                    ]
                    
                    custom_candidates.append(ModelCandidate(
                        name="ensemble_stacking_regressor",
                        model_family=ModelFamily.ENSEMBLE,
                        model=StackingRegressor,
                        hyperparameters={'estimators': base_estimators, 'final_estimator': LinearRegression()},
                        score=0.0,
                        training_time=0.0,
                        prediction_time=0.0,
                        memory_usage=0.0,
                        interpretability_score=0.4
                    ))
            
            return custom_candidates
            
        except Exception as e:
            logger.error(f"Custom model generation failed: {e}")
            return []
    
    def _calculate_interpretability_score(self, family: ModelFamily) -> float:
        """Calculate interpretability score for a model family."""
        interpretability_scores = {
            ModelFamily.LINEAR: 0.9,
            ModelFamily.TREE: 0.8,
            ModelFamily.SVM: 0.6,
            ModelFamily.NEURAL_NETWORK: 0.3,
            ModelFamily.ENSEMBLE: 0.5,
            ModelFamily.CUSTOM: 0.4
        }
        return interpretability_scores.get(family, 0.5)
    
    async def _evaluate_candidates(
        self,
        candidates: List[ModelCandidate],
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        config: AutoMLConfig
    ) -> List[ModelCandidate]:
        """Evaluate all model candidates."""
        try:
            evaluated_candidates = []
            
            # Evaluate candidates in parallel
            loop = asyncio.get_event_loop()
            tasks = []
            
            for candidate in candidates:
                task = loop.run_in_executor(
                    self.thread_pool,
                    self._evaluate_single_candidate,
                    candidate, X_train, X_test, y_train, y_test, config
                )
                tasks.append(task)
            
            # Wait for all evaluations to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Evaluation failed for {candidates[i].name}: {result}")
                    continue
                
                evaluated_candidates.append(result)
                self.performance_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'model_name': result.name,
                    'score': result.score,
                    'training_time': result.training_time,
                    'prediction_time': result.prediction_time,
                    'memory_usage': result.memory_usage
                })
            
            # Sort by score
            evaluated_candidates.sort(key=lambda x: x.score, reverse=True)
            
            logger.info(f"Evaluated {len(evaluated_candidates)} candidates")
            return evaluated_candidates
            
        except Exception as e:
            logger.error(f"Candidate evaluation failed: {e}")
            raise
    
    def _evaluate_single_candidate(
        self,
        candidate: ModelCandidate,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        config: AutoMLConfig
    ) -> ModelCandidate:
        """Evaluate a single model candidate."""
        try:
            start_time = time.time()
            
            # Instantiate model with hyperparameters
            if candidate.model == VotingClassifier:
                model = candidate.model(**candidate.hyperparameters)
            elif candidate.model == StackingRegressor:
                model = candidate.model(**candidate.hyperparameters)
            else:
                model = candidate.model(**candidate.hyperparameters)
            
            # Train model
            train_start = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - train_start
            
            # Make predictions
            pred_start = time.time()
            y_pred = model.predict(X_test)
            prediction_time = time.time() - pred_start
            
            # Calculate score
            score = self._calculate_score(y_test, y_pred, config)
            
            # Estimate memory usage
            memory_usage = self._estimate_memory_usage(model)
            
            # Update candidate
            candidate.score = score
            candidate.training_time = training_time
            candidate.prediction_time = prediction_time
            candidate.memory_usage = memory_usage
            
            # Store trained model
            candidate.model = model
            
            total_time = time.time() - start_time
            logger.debug(f"Evaluated {candidate.name}: score={score:.4f}, time={total_time:.2f}s")
            
            return candidate
            
        except Exception as e:
            logger.error(f"Evaluation failed for {candidate.name}: {e}")
            candidate.score = -float('inf')  # Mark as failed
            return candidate
    
    def _calculate_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        config: AutoMLConfig
    ) -> float:
        """Calculate score based on optimization metric."""
        try:
            if config.task_type == TaskType.CLASSIFICATION:
                if config.optimization_metric == OptimizationMetric.ACCURACY:
                    return accuracy_score(y_true, y_pred)
                elif config.optimization_metric == OptimizationMetric.PRECISION:
                    return precision_score(y_true, y_pred, average='weighted')
                elif config.optimization_metric == OptimizationMetric.RECALL:
                    return recall_score(y_true, y_pred, average='weighted')
                elif config.optimization_metric == OptimizationMetric.F1_SCORE:
                    return f1_score(y_true, y_pred, average='weighted')
                else:
                    return accuracy_score(y_true, y_pred)
            
            elif config.task_type == TaskType.REGRESSION:
                if config.optimization_metric == OptimizationMetric.MSE:
                    return -mean_squared_error(y_true, y_pred)  # Negative because we maximize
                elif config.optimization_metric == OptimizationMetric.RMSE:
                    return -np.sqrt(mean_squared_error(y_true, y_pred))
                elif config.optimization_metric == OptimizationMetric.MAE:
                    return -mean_absolute_error(y_true, y_pred)
                elif config.optimization_metric == OptimizationMetric.R2_SCORE:
                    return r2_score(y_true, y_pred)
                else:
                    return r2_score(y_true, y_pred)
            
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Score calculation failed: {e}")
            return -float('inf')
    
    def _estimate_memory_usage(self, model: Any) -> float:
        """Estimate memory usage of a trained model."""
        try:
            # Use joblib to estimate size
            model_bytes = len(pickle.dumps(model))
            return model_bytes / (1024 * 1024)  # Convert to MB
            
        except Exception:
            return 0.0
    
    async def _save_model(self, model_candidate: ModelCandidate, config: AutoMLConfig):
        """Save the best model."""
        try:
            model_path = self.models_dir / f"{model_candidate.name}.pkl"
            
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(model_candidate.model, f)
            
            # Save metadata
            metadata = {
                'name': model_candidate.name,
                'model_family': model_candidate.model_family.value,
                'hyperparameters': model_candidate.hyperparameters,
                'score': model_candidate.score,
                'training_time': model_candidate.training_time,
                'prediction_time': model_candidate.prediction_time,
                'memory_usage': model_candidate.memory_usage,
                'interpretability_score': model_candidate.interpretability_score,
                'task_type': config.task_type.value,
                'optimization_metric': config.optimization_metric.value,
                'created_at': datetime.now().isoformat()
            }
            
            metadata_path = self.models_dir / f"{model_candidate.name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved: {model_path}")
            
        except Exception as e:
            logger.error(f"Model saving failed: {e}")
            raise
    
    async def load_model(self, model_name: str) -> Optional[ModelCandidate]:
        """Load a saved model."""
        try:
            model_path = self.models_dir / f"{model_name}.pkl"
            metadata_path = self.models_dir / f"{model_name}_metadata.json"
            
            if not model_path.exists() or not metadata_path.exists():
                logger.warning(f"Model {model_name} not found")
                return None
            
            # Load model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Create model candidate
            candidate = ModelCandidate(
                name=metadata['name'],
                model_family=ModelFamily(metadata['model_family']),
                model=model,
                hyperparameters=metadata['hyperparameters'],
                score=metadata['score'],
                training_time=metadata['training_time'],
                prediction_time=metadata['prediction_time'],
                memory_usage=metadata['memory_usage'],
                interpretability_score=metadata['interpretability_score']
            )
            
            logger.info(f"Model loaded: {model_name}")
            return candidate
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return None
    
    async def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of model performance."""
        try:
            if not self.performance_history:
                return {"message": "No performance data available"}
            
            # Calculate statistics
            scores = [entry['score'] for entry in self.performance_history]
            training_times = [entry['training_time'] for entry in self.performance_history]
            prediction_times = [entry['prediction_time'] for entry in self.performance_history]
            
            summary = {
                'total_models_evaluated': len(self.performance_history),
                'best_score': max(scores),
                'average_score': np.mean(scores),
                'score_std': np.std(scores),
                'average_training_time': np.mean(training_times),
                'average_prediction_time': np.mean(prediction_times),
                'top_models': sorted(
                    self.performance_history,
                    key=lambda x: x['score'],
                    reverse=True
                )[:5]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Performance summary generation failed: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_models(self, days_to_keep: int = 30):
        """Clean up old model files."""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(days=days_to_keep)
            
            cleaned_count = 0
            for file_path in self.models_dir.glob("*.pkl"):
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    file_path.unlink()
                    
                    # Also remove metadata file
                    metadata_path = file_path.with_suffix('_metadata.json')
                    if metadata_path.exists():
                        metadata_path.unlink()
                    
                    cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} old model files")
            
        except Exception as e:
            logger.error(f"Model cleanup failed: {e}")
    
    async def shutdown(self):
        """Shutdown the AutoML manager."""
        try:
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            logger.info("AutoML manager shutdown complete")
            
        except Exception as e:
            logger.error(f"AutoML manager shutdown error: {e}")

# Global AutoML manager instance
automl_manager: Optional[AutoMLManager] = None

def get_automl_manager() -> AutoMLManager:
    """Get global AutoML manager instance."""
    global automl_manager
    if automl_manager is None:
        automl_manager = AutoMLManager()
    return automl_manager

async def shutdown_automl_manager():
    """Shutdown global AutoML manager."""
    global automl_manager
    if automl_manager:
        await automl_manager.shutdown()
        automl_manager = None

