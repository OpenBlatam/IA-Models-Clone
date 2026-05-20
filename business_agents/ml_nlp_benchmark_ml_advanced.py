"""
ML NLP Benchmark Advanced Machine Learning System
Real, working advanced ML for ML NLP Benchmark system
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import pickle
import json
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
import threading

logger = logging.getLogger(__name__)

@dataclass
class MLModel:
    """ML Model structure"""
    model_id: str
    name: str
    type: str
    algorithm: str
    parameters: Dict[str, Any]
    training_data_size: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    created_at: datetime
    last_updated: datetime
    is_trained: bool
    model_data: Optional[bytes]

@dataclass
class TrainingResult:
    """Training result structure"""
    model_id: str
    training_time: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    cross_val_scores: List[float]
    best_parameters: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]]
    confusion_matrix: Optional[List[List[int]]]
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class PredictionResult:
    """Prediction result structure"""
    model_id: str
    input_data: Any
    prediction: Any
    probability: Optional[float]
    confidence: float
    processing_time: float
    timestamp: datetime
    metadata: Dict[str, Any]

class MLNLPBenchmarkMLAdvanced:
    """Advanced Machine Learning system for ML NLP Benchmark"""
    
    def __init__(self):
        self.models = {}
        self.training_results = []
        self.prediction_results = []
        self.lock = threading.RLock()
        
        # Available algorithms
        self.algorithms = {
            "classification": {
                "random_forest": RandomForestClassifier,
                "gradient_boosting": GradientBoostingClassifier,
                "logistic_regression": LogisticRegression,
                "svm": SVC,
                "naive_bayes": MultinomialNB
            },
            "regression": {
                "linear_regression": LinearRegression,
                "random_forest": RandomForestClassifier,
                "gradient_boosting": GradientBoostingClassifier
            },
            "clustering": {
                "kmeans": KMeans,
                "dbscan": DBSCAN
            },
            "dimensionality_reduction": {
                "lda": LatentDirichletAllocation,
                "nmf": NMF,
                "svd": TruncatedSVD
            }
        }
        
        # Feature extraction methods
        self.feature_extractors = {
            "tfidf": TfidfVectorizer,
            "count": CountVectorizer
        }
        
        # Default parameters for algorithms
        self.default_parameters = {
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            "gradient_boosting": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42
            },
            "logistic_regression": {
                "random_state": 42,
                "max_iter": 1000
            },
            "svm": {
                "random_state": 42,
                "probability": True
            },
            "naive_bayes": {
                "alpha": 1.0
            },
            "linear_regression": {},
            "kmeans": {
                "n_clusters": 5,
                "random_state": 42
            },
            "dbscan": {
                "eps": 0.5,
                "min_samples": 5
            },
            "lda": {
                "n_components": 10,
                "random_state": 42
            },
            "nmf": {
                "n_components": 10,
                "random_state": 42
            },
            "svd": {
                "n_components": 10,
                "random_state": 42
            }
        }
        
        # Hyperparameter search spaces
        self.param_grids = {
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10]
            },
            "gradient_boosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7]
            },
            "logistic_regression": {
                "C": [0.1, 1, 10, 100],
                "penalty": ["l1", "l2"]
            },
            "svm": {
                "C": [0.1, 1, 10, 100],
                "kernel": ["linear", "rbf", "poly"],
                "gamma": ["scale", "auto"]
            },
            "naive_bayes": {
                "alpha": [0.1, 0.5, 1.0, 2.0]
            }
        }
    
    def create_model(self, name: str, algorithm: str, model_type: str, 
                    parameters: Optional[Dict[str, Any]] = None) -> str:
        """Create a new ML model"""
        model_id = f"{name}_{int(time.time())}"
        
        if algorithm not in self.algorithms.get(model_type, {}):
            raise ValueError(f"Unknown algorithm {algorithm} for type {model_type}")
        
        # Get default parameters
        default_params = self.default_parameters.get(algorithm, {})
        if parameters:
            default_params.update(parameters)
        
        model = MLModel(
            model_id=model_id,
            name=name,
            type=model_type,
            algorithm=algorithm,
            parameters=default_params,
            training_data_size=0,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            training_time=0.0,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            is_trained=False,
            model_data=None
        )
        
        with self.lock:
            self.models[model_id] = model
        
        logger.info(f"Created model {model_id}: {name} ({algorithm})")
        return model_id
    
    def train_model(self, model_id: str, X: Union[List[str], np.ndarray], 
                   y: Union[List[str], np.ndarray], 
                   feature_extractor: str = "tfidf",
                   test_size: float = 0.2,
                   hyperparameter_tuning: bool = False) -> TrainingResult:
        """Train an ML model"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        start_time = time.time()
        
        try:
            # Prepare data
            if model.type in ["classification", "regression"]:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
            else:
                X_train, X_test = X, X
                y_train, y_test = y, y
            
            # Feature extraction for text data
            if isinstance(X_train[0], str) and feature_extractor in self.feature_extractors:
                vectorizer = self.feature_extractors[feature_extractor]()
                X_train_vectorized = vectorizer.fit_transform(X_train)
                X_test_vectorized = vectorizer.transform(X_test)
            else:
                X_train_vectorized = X_train
                X_test_vectorized = X_test
                vectorizer = None
            
            # Get algorithm class
            algorithm_class = self.algorithms[model.type][model.algorithm]
            
            # Create model instance
            ml_model = algorithm_class(**model.parameters)
            
            # Hyperparameter tuning
            if hyperparameter_tuning and model.algorithm in self.param_grids:
                param_grid = self.param_grids[model.algorithm]
                grid_search = GridSearchCV(
                    ml_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
                )
                grid_search.fit(X_train_vectorized, y_train)
                ml_model = grid_search.best_estimator_
                best_parameters = grid_search.best_params_
            else:
                best_parameters = model.parameters
            
            # Train model
            ml_model.fit(X_train_vectorized, y_train)
            
            # Make predictions
            y_pred = ml_model.predict(X_test_vectorized)
            
            # Calculate metrics
            if model.type == "classification":
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Cross-validation scores
                cv_scores = cross_val_score(ml_model, X_train_vectorized, y_train, cv=5)
                
                # Feature importance (if available)
                feature_importance = None
                if hasattr(ml_model, 'feature_importances_') and vectorizer:
                    feature_names = vectorizer.get_feature_names_out()
                    feature_importance = dict(zip(feature_names, ml_model.feature_importances_))
                
                # Confusion matrix
                from sklearn.metrics import confusion_matrix
                confusion_matrix_result = confusion_matrix(y_test, y_pred).tolist()
                
            elif model.type == "regression":
                from sklearn.metrics import mean_squared_error, r2_score
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                accuracy = r2  # Use R² as accuracy for regression
                precision = recall = f1 = 0.0
                cv_scores = cross_val_score(ml_model, X_train_vectorized, y_train, cv=5, scoring='r2')
                feature_importance = None
                confusion_matrix_result = None
                
            else:  # clustering or dimensionality reduction
                accuracy = precision = recall = f1 = 0.0
                cv_scores = []
                feature_importance = None
                confusion_matrix_result = None
            
            training_time = time.time() - start_time
            
            # Create training result
            result = TrainingResult(
                model_id=model_id,
                training_time=training_time,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                cross_val_scores=cv_scores.tolist() if hasattr(cv_scores, 'tolist') else list(cv_scores),
                best_parameters=best_parameters,
                feature_importance=feature_importance,
                confusion_matrix=confusion_matrix_result,
                timestamp=datetime.now(),
                metadata={
                    "training_data_size": len(X_train),
                    "test_data_size": len(X_test),
                    "feature_extractor": feature_extractor,
                    "hyperparameter_tuning": hyperparameter_tuning,
                    "vectorizer": vectorizer is not None
                }
            )
            
            # Update model
            with self.lock:
                model.accuracy = accuracy
                model.precision = precision
                model.recall = recall
                model.f1_score = f1
                model.training_time = training_time
                model.training_data_size = len(X_train)
                model.is_trained = True
                model.last_updated = datetime.now()
                model.model_data = pickle.dumps(ml_model)
                
                # Store vectorizer if used
                if vectorizer:
                    model.parameters["vectorizer"] = pickle.dumps(vectorizer)
            
            # Store training result
            with self.lock:
                self.training_results.append(result)
            
            logger.info(f"Trained model {model_id} in {training_time:.2f}s with accuracy {accuracy:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error training model {model_id}: {e}")
            raise
    
    def predict(self, model_id: str, X: Union[List[str], np.ndarray]) -> PredictionResult:
        """Make predictions with a trained model"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        
        if not model.is_trained:
            raise ValueError(f"Model {model_id} is not trained")
        
        start_time = time.time()
        
        try:
            # Load model
            ml_model = pickle.loads(model.model_data)
            
            # Prepare input data
            if isinstance(X, list) and isinstance(X[0], str):
                # Text data - need vectorizer
                if "vectorizer" in model.parameters:
                    vectorizer = pickle.loads(model.parameters["vectorizer"])
                    X_vectorized = vectorizer.transform(X)
                else:
                    raise ValueError("No vectorizer found for text data")
            else:
                X_vectorized = X
            
            # Make prediction
            prediction = ml_model.predict(X_vectorized)
            
            # Get probability if available
            probability = None
            if hasattr(ml_model, 'predict_proba'):
                proba = ml_model.predict_proba(X_vectorized)
                probability = proba.max() if len(proba.shape) == 1 else proba[0].max()
            
            # Calculate confidence
            confidence = probability if probability is not None else 0.8
            
            processing_time = time.time() - start_time
            
            # Create prediction result
            result = PredictionResult(
                model_id=model_id,
                input_data=X[0] if len(X) == 1 else X,
                prediction=prediction[0] if len(prediction) == 1 else prediction,
                probability=probability,
                confidence=confidence,
                processing_time=processing_time,
                timestamp=datetime.now(),
                metadata={
                    "input_size": len(X),
                    "model_type": model.type,
                    "algorithm": model.algorithm
                }
            )
            
            # Store prediction result
            with self.lock:
                self.prediction_results.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction with model {model_id}: {e}")
            raise
    
    def batch_predict(self, model_id: str, X: Union[List[str], np.ndarray]) -> List[PredictionResult]:
        """Make batch predictions"""
        results = []
        
        # Process in chunks to avoid memory issues
        chunk_size = 1000
        for i in range(0, len(X), chunk_size):
            chunk = X[i:i + chunk_size]
            result = self.predict(model_id, chunk)
            results.append(result)
        
        return results
    
    def evaluate_model(self, model_id: str, X: Union[List[str], np.ndarray], 
                      y: Union[List[str], np.ndarray]) -> Dict[str, Any]:
        """Evaluate model performance"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        
        if not model.is_trained:
            raise ValueError(f"Model {model_id} is not trained")
        
        try:
            # Load model
            ml_model = pickle.loads(model.model_data)
            
            # Prepare data
            if isinstance(X[0], str) and "vectorizer" in model.parameters:
                vectorizer = pickle.loads(model.parameters["vectorizer"])
                X_vectorized = vectorizer.transform(X)
            else:
                X_vectorized = X
            
            # Make predictions
            y_pred = ml_model.predict(X_vectorized)
            
            # Calculate metrics
            if model.type == "classification":
                accuracy = accuracy_score(y, y_pred)
                precision = precision_score(y, y_pred, average='weighted')
                recall = recall_score(y, y_pred, average='weighted')
                f1 = f1_score(y, y_pred, average='weighted')
                
                # Classification report
                report = classification_report(y, y_pred, output_dict=True)
                
                return {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "classification_report": report,
                    "model_type": model.type
                }
                
            elif model.type == "regression":
                from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                
                return {
                    "mse": mse,
                    "r2_score": r2,
                    "mae": mae,
                    "model_type": model.type
                }
            
            else:
                return {
                    "model_type": model.type,
                    "note": "Evaluation not implemented for this model type"
                }
                
        except Exception as e:
            logger.error(f"Error evaluating model {model_id}: {e}")
            raise
    
    def get_model(self, model_id: str) -> Optional[MLModel]:
        """Get model information"""
        return self.models.get(model_id)
    
    def list_models(self, model_type: Optional[str] = None, trained_only: bool = False) -> List[MLModel]:
        """List available models"""
        models = list(self.models.values())
        
        if model_type:
            models = [m for m in models if m.type == model_type]
        
        if trained_only:
            models = [m for m in models if m.is_trained]
        
        return models
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model"""
        if model_id in self.models:
            with self.lock:
                del self.models[model_id]
            logger.info(f"Deleted model {model_id}")
            return True
        return False
    
    def export_model(self, model_id: str) -> Optional[bytes]:
        """Export model to bytes"""
        if model_id not in self.models:
            return None
        
        model = self.models[model_id]
        
        export_data = {
            "model_info": {
                "model_id": model.model_id,
                "name": model.name,
                "type": model.type,
                "algorithm": model.algorithm,
                "parameters": model.parameters,
                "accuracy": model.accuracy,
                "precision": model.precision,
                "recall": model.recall,
                "f1_score": model.f1_score,
                "training_time": model.training_time,
                "created_at": model.created_at.isoformat(),
                "last_updated": model.last_updated.isoformat(),
                "is_trained": model.is_trained
            },
            "model_data": model.model_data,
            "export_timestamp": datetime.now().isoformat()
        }
        
        return pickle.dumps(export_data)
    
    def import_model(self, model_data: bytes) -> str:
        """Import model from bytes"""
        try:
            export_data = pickle.loads(model_data)
            model_info = export_data["model_info"]
            
            # Create model
            model = MLModel(
                model_id=model_info["model_id"],
                name=model_info["name"],
                type=model_info["type"],
                algorithm=model_info["algorithm"],
                parameters=model_info["parameters"],
                training_data_size=0,
                accuracy=model_info["accuracy"],
                precision=model_info["precision"],
                recall=model_info["recall"],
                f1_score=model_info["f1_score"],
                training_time=model_info["training_time"],
                created_at=datetime.fromisoformat(model_info["created_at"]),
                last_updated=datetime.fromisoformat(model_info["last_updated"]),
                is_trained=model_info["is_trained"],
                model_data=export_data["model_data"]
            )
            
            with self.lock:
                self.models[model.model_id] = model
            
            logger.info(f"Imported model {model.model_id}")
            return model.model_id
            
        except Exception as e:
            logger.error(f"Error importing model: {e}")
            raise
    
    def get_training_results(self, model_id: Optional[str] = None) -> List[TrainingResult]:
        """Get training results"""
        results = self.training_results
        
        if model_id:
            results = [r for r in results if r.model_id == model_id]
        
        return results
    
    def get_prediction_results(self, model_id: Optional[str] = None) -> List[PredictionResult]:
        """Get prediction results"""
        results = self.prediction_results
        
        if model_id:
            results = [r for r in results if r.model_id == model_id]
        
        return results
    
    def get_ml_summary(self) -> Dict[str, Any]:
        """Get ML system summary"""
        with self.lock:
            total_models = len(self.models)
            trained_models = len([m for m in self.models.values() if m.is_trained])
            total_training_results = len(self.training_results)
            total_predictions = len(self.prediction_results)
            
            # Model type distribution
            model_types = {}
            for model in self.models.values():
                model_types[model.type] = model_types.get(model.type, 0) + 1
            
            # Algorithm distribution
            algorithms = {}
            for model in self.models.values():
                algorithms[model.algorithm] = algorithms.get(model.algorithm, 0) + 1
            
            # Average performance
            trained_models_list = [m for m in self.models.values() if m.is_trained]
            avg_accuracy = np.mean([m.accuracy for m in trained_models_list]) if trained_models_list else 0
            avg_training_time = np.mean([m.training_time for m in trained_models_list]) if trained_models_list else 0
            
            return {
                "total_models": total_models,
                "trained_models": trained_models,
                "untrained_models": total_models - trained_models,
                "total_training_results": total_training_results,
                "total_predictions": total_predictions,
                "model_types": model_types,
                "algorithms": algorithms,
                "average_accuracy": avg_accuracy,
                "average_training_time": avg_training_time,
                "available_algorithms": self.algorithms,
                "available_feature_extractors": list(self.feature_extractors.keys())
            }
    
    def clear_ml_data(self):
        """Clear all ML data"""
        with self.lock:
            self.models.clear()
            self.training_results.clear()
            self.prediction_results.clear()
        logger.info("ML data cleared")

# Global ML instance
ml_nlp_benchmark_ml_advanced = MLNLPBenchmarkMLAdvanced()

def get_ml_advanced() -> MLNLPBenchmarkMLAdvanced:
    """Get the global ML advanced instance"""
    return ml_nlp_benchmark_ml_advanced

def create_model(name: str, algorithm: str, model_type: str, 
                parameters: Optional[Dict[str, Any]] = None) -> str:
    """Create a new ML model"""
    return ml_nlp_benchmark_ml_advanced.create_model(name, algorithm, model_type, parameters)

def train_model(model_id: str, X: Union[List[str], np.ndarray], 
               y: Union[List[str], np.ndarray], 
               feature_extractor: str = "tfidf",
               test_size: float = 0.2,
               hyperparameter_tuning: bool = False) -> TrainingResult:
    """Train an ML model"""
    return ml_nlp_benchmark_ml_advanced.train_model(model_id, X, y, feature_extractor, test_size, hyperparameter_tuning)

def predict(model_id: str, X: Union[List[str], np.ndarray]) -> PredictionResult:
    """Make predictions with a trained model"""
    return ml_nlp_benchmark_ml_advanced.predict(model_id, X)

def batch_predict(model_id: str, X: Union[List[str], np.ndarray]) -> List[PredictionResult]:
    """Make batch predictions"""
    return ml_nlp_benchmark_ml_advanced.batch_predict(model_id, X)

def evaluate_model(model_id: str, X: Union[List[str], np.ndarray], 
                  y: Union[List[str], np.ndarray]) -> Dict[str, Any]:
    """Evaluate model performance"""
    return ml_nlp_benchmark_ml_advanced.evaluate_model(model_id, X, y)

def get_model(model_id: str) -> Optional[MLModel]:
    """Get model information"""
    return ml_nlp_benchmark_ml_advanced.get_model(model_id)

def list_models(model_type: Optional[str] = None, trained_only: bool = False) -> List[MLModel]:
    """List available models"""
    return ml_nlp_benchmark_ml_advanced.list_models(model_type, trained_only)

def delete_model(model_id: str) -> bool:
    """Delete a model"""
    return ml_nlp_benchmark_ml_advanced.delete_model(model_id)

def export_model(model_id: str) -> Optional[bytes]:
    """Export model to bytes"""
    return ml_nlp_benchmark_ml_advanced.export_model(model_id)

def import_model(model_data: bytes) -> str:
    """Import model from bytes"""
    return ml_nlp_benchmark_ml_advanced.import_model(model_data)

def get_ml_summary() -> Dict[str, Any]:
    """Get ML system summary"""
    return ml_nlp_benchmark_ml_advanced.get_ml_summary()

def clear_ml_data():
    """Clear all ML data"""
    ml_nlp_benchmark_ml_advanced.clear_ml_data()











