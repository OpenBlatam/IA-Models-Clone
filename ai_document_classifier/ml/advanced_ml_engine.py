"""
Advanced Machine Learning Engine
===============================

Advanced ML engine with deep learning models, ensemble methods,
auto-tuning, and continuous learning capabilities.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import pickle
import joblib
from pathlib import Path
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.naive_bayes import MultinomialNB, GaussianNB
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.pipeline import Pipeline
    from sklearn.base import BaseEstimator, TransformerMixin
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available")

try:
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False
    logging.warning("Advanced ML libraries (XGBoost, LightGBM, CatBoost) not available")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    logging.warning("PyTorch and Transformers not available")

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Model types"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"
    NAIVE_BAYES = "naive_bayes"
    NEURAL_NETWORK = "neural_network"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"

class TrainingStatus(Enum):
    """Training status"""
    PENDING = "pending"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]
    training_time: float
    prediction_time: float
    cross_val_scores: List[float] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)

@dataclass
class ModelConfig:
    """Model configuration"""
    model_type: ModelType
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    feature_extraction: Dict[str, Any] = field(default_factory=dict)
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    auto_tune: bool = False
    ensemble_models: List[str] = field(default_factory=list)

@dataclass
class TrainingJob:
    """Training job definition"""
    id: str
    model_config: ModelConfig
    training_data: str  # Path to training data
    validation_data: Optional[str] = None
    status: TrainingStatus = TrainingStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics: Optional[ModelMetrics] = None
    error_message: Optional[str] = None
    progress: float = 0.0

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Custom text feature extractor"""
    
    def __init__(self, use_tfidf=True, use_count=True, use_ngrams=True, max_features=10000):
        self.use_tfidf = use_tfidf
        self.use_count = use_count
        self.use_ngrams = use_ngrams
        self.max_features = max_features
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        
    def fit(self, X, y=None):
        if self.use_tfidf:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2) if self.use_ngrams else (1, 1),
                stop_words='english'
            )
            self.tfidf_vectorizer.fit(X)
        
        if self.use_count:
            self.count_vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2) if self.use_ngrams else (1, 1),
                stop_words='english'
            )
            self.count_vectorizer.fit(X)
        
        return self
    
    def transform(self, X):
        features = []
        
        if self.use_tfidf and self.tfidf_vectorizer:
            tfidf_features = self.tfidf_vectorizer.transform(X).toarray()
            features.append(tfidf_features)
        
        if self.use_count and self.count_vectorizer:
            count_features = self.count_vectorizer.transform(X).toarray()
            features.append(count_features)
        
        # Add text statistics
        text_stats = np.array([
            [len(text), len(text.split()), text.count('.'), text.count('!'), text.count('?')]
            for text in X
        ])
        features.append(text_stats)
        
        return np.hstack(features) if features else np.array([])

class AdvancedMLEngine:
    """
    Advanced machine learning engine with multiple algorithms and auto-tuning
    """
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize ML engine
        
        Args:
            models_dir: Directory for storing models
        """
        self.models_dir = Path(models_dir) if models_dir else Path(__file__).parent / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Model registry
        self.trained_models: Dict[str, Any] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.training_jobs: Dict[str, TrainingJob] = {}
        
        # Feature extractors
        self.feature_extractors: Dict[str, Any] = {}
        
        # Training executor
        self.training_executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize default configurations
        self._initialize_default_configs()
        
        # Load existing models
        self._load_existing_models()
    
    def _initialize_default_configs(self):
        """Initialize default model configurations"""
        if not SKLEARN_AVAILABLE:
            return
        
        # Random Forest
        self.model_configs["random_forest"] = ModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            name="Random Forest Classifier",
            parameters={
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42
            },
            feature_extraction={
                "use_tfidf": True,
                "use_count": True,
                "use_ngrams": True,
                "max_features": 10000
            },
            auto_tune=True
        )
        
        # Gradient Boosting
        self.model_configs["gradient_boosting"] = ModelConfig(
            model_type=ModelType.GRADIENT_BOOSTING,
            name="Gradient Boosting Classifier",
            parameters={
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "random_state": 42
            },
            feature_extraction={
                "use_tfidf": True,
                "use_count": False,
                "use_ngrams": True,
                "max_features": 15000
            },
            auto_tune=True
        )
        
        # XGBoost
        if ADVANCED_ML_AVAILABLE:
            self.model_configs["xgboost"] = ModelConfig(
                model_type=ModelType.XGBOOST,
                name="XGBoost Classifier",
                parameters={
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "random_state": 42
                },
                feature_extraction={
                    "use_tfidf": True,
                    "use_count": False,
                    "use_ngrams": True,
                    "max_features": 20000
                },
                auto_tune=True
            )
        
        # LightGBM
        if ADVANCED_ML_AVAILABLE:
            self.model_configs["lightgbm"] = ModelConfig(
                model_type=ModelType.LIGHTGBM,
                name="LightGBM Classifier",
                parameters={
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "random_state": 42
                },
                feature_extraction={
                    "use_tfidf": True,
                    "use_count": False,
                    "use_ngrams": True,
                    "max_features": 20000
                },
                auto_tune=True
            )
        
        # Neural Network
        self.model_configs["neural_network"] = ModelConfig(
            model_type=ModelType.NEURAL_NETWORK,
            name="Neural Network Classifier",
            parameters={
                "hidden_layer_sizes": (100, 50),
                "activation": "relu",
                "solver": "adam",
                "alpha": 0.001,
                "max_iter": 1000,
                "random_state": 42
            },
            feature_extraction={
                "use_tfidf": True,
                "use_count": False,
                "use_ngrams": True,
                "max_features": 10000
            },
            preprocessing={
                "scaler": "standard"
            }
        )
        
        # Ensemble
        self.model_configs["ensemble"] = ModelConfig(
            model_type=ModelType.ENSEMBLE,
            name="Ensemble Classifier",
            parameters={
                "voting": "soft"
            },
            ensemble_models=["random_forest", "gradient_boosting", "neural_network"],
            feature_extraction={
                "use_tfidf": True,
                "use_count": True,
                "use_ngrams": True,
                "max_features": 15000
            }
        )
    
    def _load_existing_models(self):
        """Load existing trained models"""
        for model_file in self.models_dir.glob("*.pkl"):
            try:
                model_name = model_file.stem
                model = joblib.load(model_file)
                self.trained_models[model_name] = model
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Error loading model {model_file}: {e}")
    
    def create_training_job(
        self,
        model_config_name: str,
        training_data_path: str,
        validation_data_path: Optional[str] = None
    ) -> str:
        """
        Create a new training job
        
        Args:
            model_config_name: Name of model configuration
            training_data_path: Path to training data
            validation_data_path: Path to validation data
            
        Returns:
            Training job ID
        """
        if model_config_name not in self.model_configs:
            raise ValueError(f"Model configuration {model_config_name} not found")
        
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_config_name}"
        
        training_job = TrainingJob(
            id=job_id,
            model_config=self.model_configs[model_config_name],
            training_data=training_data_path,
            validation_data=validation_data_path
        )
        
        self.training_jobs[job_id] = training_job
        
        # Start training asynchronously
        self.training_executor.submit(self._train_model, job_id)
        
        return job_id
    
    def _train_model(self, job_id: str):
        """Train model for a specific job"""
        job = self.training_jobs[job_id]
        job.status = TrainingStatus.TRAINING
        job.started_at = datetime.now()
        
        try:
            # Load training data
            training_data = self._load_training_data(job.training_data)
            X_train, y_train = training_data['text'], training_data['label']
            
            # Load validation data if provided
            validation_data = None
            if job.validation_data:
                validation_data = self._load_training_data(job.validation_data)
            
            # Create feature extractor
            feature_extractor = TextFeatureExtractor(**job.model_config.feature_extraction)
            
            # Extract features
            job.progress = 0.1
            X_train_features = feature_extractor.fit_transform(X_train)
            
            # Preprocessing
            if job.model_config.preprocessing.get("scaler"):
                scaler = StandardScaler()
                X_train_features = scaler.fit_transform(X_train_features)
            
            job.progress = 0.3
            
            # Create and train model
            model = self._create_model(job.model_config)
            
            # Auto-tuning if enabled
            if job.model_config.auto_tune:
                model = self._auto_tune_model(model, X_train_features, y_train)
            
            job.progress = 0.5
            
            # Train model
            start_time = datetime.now()
            model.fit(X_train_features, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            job.progress = 0.7
            
            # Evaluate model
            metrics = self._evaluate_model(model, X_train_features, y_train, validation_data, feature_extractor)
            metrics.training_time = training_time
            
            job.progress = 0.9
            
            # Save model
            model_path = self.models_dir / f"{job.model_config.name.lower().replace(' ', '_')}.pkl"
            joblib.dump({
                'model': model,
                'feature_extractor': feature_extractor,
                'scaler': scaler if job.model_config.preprocessing.get("scaler") else None,
                'config': job.model_config,
                'metrics': metrics
            }, model_path)
            
            self.trained_models[job.model_config.name] = {
                'model': model,
                'feature_extractor': feature_extractor,
                'scaler': scaler if job.model_config.preprocessing.get("scaler") else None,
                'config': job.model_config,
                'metrics': metrics
            }
            
            job.metrics = metrics
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.now()
            job.progress = 1.0
            
            logger.info(f"Training completed for job {job_id}")
            
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            logger.error(f"Training failed for job {job_id}: {e}")
    
    def _load_training_data(self, data_path: str) -> Dict[str, List]:
        """Load training data from file"""
        data_path = Path(data_path)
        
        if data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
            return {
                'text': df['text'].tolist(),
                'label': df['label'].tolist()
            }
        elif data_path.suffix == '.json':
            with open(data_path, 'r') as f:
                data = json.load(f)
            return data
        else:
            raise ValueError(f"Unsupported data format: {data_path.suffix}")
    
    def _create_model(self, config: ModelConfig) -> Any:
        """Create model based on configuration"""
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("Scikit-learn not available")
        
        if config.model_type == ModelType.RANDOM_FOREST:
            return RandomForestClassifier(**config.parameters)
        elif config.model_type == ModelType.GRADIENT_BOOSTING:
            return GradientBoostingClassifier(**config.parameters)
        elif config.model_type == ModelType.XGBOOST and ADVANCED_ML_AVAILABLE:
            return xgb.XGBClassifier(**config.parameters)
        elif config.model_type == ModelType.LIGHTGBM and ADVANCED_ML_AVAILABLE:
            return lgb.LGBMClassifier(**config.parameters)
        elif config.model_type == ModelType.CATBOOST and ADVANCED_ML_AVAILABLE:
            return cb.CatBoostClassifier(**config.parameters)
        elif config.model_type == ModelType.LOGISTIC_REGRESSION:
            return LogisticRegression(**config.parameters)
        elif config.model_type == ModelType.SVM:
            return SVC(**config.parameters)
        elif config.model_type == ModelType.NAIVE_BAYES:
            return MultinomialNB(**config.parameters)
        elif config.model_type == ModelType.NEURAL_NETWORK:
            return MLPClassifier(**config.parameters)
        elif config.model_type == ModelType.ENSEMBLE:
            return self._create_ensemble_model(config)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
    
    def _create_ensemble_model(self, config: ModelConfig) -> VotingClassifier:
        """Create ensemble model"""
        estimators = []
        
        for model_name in config.ensemble_models:
            if model_name in self.model_configs:
                model_config = self.model_configs[model_name]
                model = self._create_model(model_config)
                estimators.append((model_name, model))
        
        return VotingClassifier(estimators=estimators, **config.parameters)
    
    def _auto_tune_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> Any:
        """Auto-tune model hyperparameters"""
        if not SKLEARN_AVAILABLE:
            return model
        
        # Define parameter grids for different models
        param_grids = {
            'RandomForestClassifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'GradientBoostingClassifier': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'MLPClassifier': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
        
        model_name = model.__class__.__name__
        if model_name in param_grids:
            grid_search = GridSearchCV(
                model, 
                param_grids[model_name], 
                cv=3, 
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X, y)
            return grid_search.best_estimator_
        
        return model
    
    def _evaluate_model(
        self, 
        model: Any, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        validation_data: Optional[Dict[str, List]],
        feature_extractor: TextFeatureExtractor
    ) -> ModelMetrics:
        """Evaluate model performance"""
        if not SKLEARN_AVAILABLE:
            return ModelMetrics(0, 0, 0, 0, [], {}, 0, 0)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        # Predictions
        start_time = datetime.now()
        y_pred = model.predict(X_train)
        prediction_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate metrics
        accuracy = accuracy_score(y_train, y_pred)
        
        # Classification report
        report = classification_report(y_train, y_pred, output_dict=True)
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1 = report['weighted avg']['f1-score']
        
        # Confusion matrix
        cm = confusion_matrix(y_train, y_pred)
        
        # Feature importance (if available)
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            # This is a simplified version - in practice, you'd map back to feature names
            feature_importance = {
                f"feature_{i}": importance 
                for i, importance in enumerate(model.feature_importances_)
            }
        
        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=cm.tolist(),
            classification_report=report,
            training_time=0,  # Will be set by caller
            prediction_time=prediction_time,
            cross_val_scores=cv_scores.tolist(),
            feature_importance=feature_importance
        )
    
    def predict(self, model_name: str, text: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Make predictions using trained model
        
        Args:
            model_name: Name of trained model
            text: Text or list of texts to classify
            
        Returns:
            Prediction results
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found")
        
        model_data = self.trained_models[model_name]
        model = model_data['model']
        feature_extractor = model_data['feature_extractor']
        scaler = model_data.get('scaler')
        
        # Prepare input
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        # Extract features
        X_features = feature_extractor.transform(texts)
        
        # Apply scaling if available
        if scaler:
            X_features = scaler.transform(X_features)
        
        # Make predictions
        start_time = datetime.now()
        predictions = model.predict(X_features)
        prediction_time = (datetime.now() - start_time).total_seconds()
        
        # Get prediction probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_features)
        
        results = {
            "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
            "prediction_time": prediction_time,
            "model_name": model_name,
            "input_count": len(texts)
        }
        
        if probabilities is not None:
            results["probabilities"] = probabilities.tolist()
        
        return results
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a trained model"""
        if model_name not in self.trained_models:
            return {"error": f"Model {model_name} not found"}
        
        model_data = self.trained_models[model_name]
        config = model_data['config']
        metrics = model_data['metrics']
        
        return {
            "name": model_name,
            "model_type": config.model_type.value,
            "parameters": config.parameters,
            "metrics": {
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "training_time": metrics.training_time,
                "prediction_time": metrics.prediction_time,
                "cross_val_scores": metrics.cross_val_scores
            },
            "feature_extraction": config.feature_extraction,
            "preprocessing": config.preprocessing
        }
    
    def get_training_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get training job status"""
        if job_id not in self.training_jobs:
            return {"error": f"Training job {job_id} not found"}
        
        job = self.training_jobs[job_id]
        
        return {
            "id": job.id,
            "model_config": job.model_config.name,
            "status": job.status.value,
            "progress": job.progress,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "error_message": job.error_message,
            "metrics": {
                "accuracy": job.metrics.accuracy if job.metrics else None,
                "f1_score": job.metrics.f1_score if job.metrics else None,
                "training_time": job.metrics.training_time if job.metrics else None
            } if job.metrics else None
        }
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all trained models"""
        return [
            {
                "name": name,
                "model_type": data['config'].model_type.value,
                "accuracy": data['metrics'].accuracy,
                "training_time": data['metrics'].training_time
            }
            for name, data in self.trained_models.items()
        ]
    
    def list_training_jobs(self) -> List[Dict[str, Any]]:
        """List all training jobs"""
        return [
            {
                "id": job.id,
                "model_config": job.model_config.name,
                "status": job.status.value,
                "progress": job.progress,
                "started_at": job.started_at.isoformat() if job.started_at else None
            }
            for job in self.training_jobs.values()
        ]
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a trained model"""
        if model_name not in self.trained_models:
            return False
        
        # Remove from memory
        del self.trained_models[model_name]
        
        # Remove file
        model_path = self.models_dir / f"{model_name.lower().replace(' ', '_')}.pkl"
        if model_path.exists():
            model_path.unlink()
        
        return True
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """Get ML engine statistics"""
        return {
            "total_models": len(self.trained_models),
            "total_training_jobs": len(self.training_jobs),
            "completed_jobs": len([j for j in self.training_jobs.values() if j.status == TrainingStatus.COMPLETED]),
            "failed_jobs": len([j for j in self.training_jobs.values() if j.status == TrainingStatus.FAILED]),
            "running_jobs": len([j for j in self.training_jobs.values() if j.status == TrainingStatus.TRAINING]),
            "available_configs": len(self.model_configs),
            "sklearn_available": SKLEARN_AVAILABLE,
            "advanced_ml_available": ADVANCED_ML_AVAILABLE,
            "deep_learning_available": DEEP_LEARNING_AVAILABLE
        }

# Example usage
if __name__ == "__main__":
    # Initialize ML engine
    ml_engine = AdvancedMLEngine()
    
    # Get engine statistics
    stats = ml_engine.get_engine_statistics()
    print("ML Engine Statistics:")
    print(f"Total models: {stats['total_models']}")
    print(f"Available configs: {stats['available_configs']}")
    print(f"Scikit-learn available: {stats['sklearn_available']}")
    print(f"Advanced ML available: {stats['advanced_ml_available']}")
    
    # List available model configurations
    print("\nAvailable Model Configurations:")
    for name, config in ml_engine.model_configs.items():
        print(f"  {name}: {config.name} ({config.model_type.value})")
    
    print("\nAdvanced ML engine initialized successfully")


























