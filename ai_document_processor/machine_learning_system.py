"""
Machine Learning System for AI Document Processor
Real, working ML features for document processing
"""

import asyncio
import logging
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

logger = logging.getLogger(__name__)

class MLSystem:
    """Real working machine learning system for AI document processing"""
    
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.scalers = {}
        self.encoders = {}
        self.training_data = {}
        self.model_metrics = {}
        self.auto_ml_config = {
            "max_iterations": 100,
            "cv_folds": 5,
            "test_size": 0.2,
            "random_state": 42
        }
        
        # ML stats
        self.stats = {
            "total_models": 0,
            "trained_models": 0,
            "failed_models": 0,
            "predictions_made": 0,
            "start_time": time.time()
        }
        
        # Initialize default models
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default ML models"""
        self.models = {
            "text_classifier": {
                "type": "classification",
                "model": None,
                "vectorizer": None,
                "trained": False,
                "accuracy": 0.0,
                "created_at": None
            },
            "sentiment_classifier": {
                "type": "classification",
                "model": None,
                "vectorizer": None,
                "trained": False,
                "accuracy": 0.0,
                "created_at": None
            },
            "topic_model": {
                "type": "clustering",
                "model": None,
                "vectorizer": None,
                "trained": False,
                "silhouette_score": 0.0,
                "created_at": None
            },
            "document_clusterer": {
                "type": "clustering",
                "model": None,
                "vectorizer": None,
                "trained": False,
                "silhouette_score": 0.0,
                "created_at": None
            },
            "complexity_predictor": {
                "type": "regression",
                "model": None,
                "vectorizer": None,
                "trained": False,
                "r2_score": 0.0,
                "created_at": None
            }
        }
    
    async def train_model(self, model_name: str, training_data: List[Dict[str, Any]], 
                         model_type: str = "classification", algorithm: str = "random_forest") -> Dict[str, Any]:
        """Train a machine learning model"""
        try:
            if model_name not in self.models:
                self.models[model_name] = {
                    "type": model_type,
                    "model": None,
                    "vectorizer": None,
                    "trained": False,
                    "accuracy": 0.0,
                    "created_at": None
                }
            
            # Prepare training data
            X, y = self._prepare_training_data(training_data, model_type)
            
            if len(X) == 0:
                return {"error": "No training data provided"}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.auto_ml_config["test_size"], 
                random_state=self.auto_ml_config["random_state"]
            )
            
            # Create vectorizer
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            X_train_vectorized = vectorizer.fit_transform(X_train)
            X_test_vectorized = vectorizer.transform(X_test)
            
            # Create and train model
            model = self._create_model(algorithm, model_type)
            model.fit(X_train_vectorized, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_vectorized)
            metrics = self._calculate_metrics(y_test, y_pred, model_type)
            
            # Store model
            self.models[model_name]["model"] = model
            self.models[model_name]["vectorizer"] = vectorizer
            self.models[model_name]["trained"] = True
            self.models[model_name]["accuracy"] = metrics.get("accuracy", 0.0)
            self.models[model_name]["created_at"] = datetime.now().isoformat()
            
            # Store metrics
            self.model_metrics[model_name] = metrics
            
            # Update stats
            self.stats["total_models"] += 1
            self.stats["trained_models"] += 1
            
            return {
                "status": "trained",
                "model_name": model_name,
                "model_type": model_type,
                "algorithm": algorithm,
                "metrics": metrics,
                "training_samples": len(X_train),
                "test_samples": len(X_test)
            }
            
        except Exception as e:
            self.stats["failed_models"] += 1
            logger.error(f"Error training model {model_name}: {e}")
            return {"error": str(e)}
    
    def _prepare_training_data(self, training_data: List[Dict[str, Any]], model_type: str) -> tuple:
        """Prepare training data for ML model"""
        try:
            X = []
            y = []
            
            for item in training_data:
                if "text" in item:
                    X.append(item["text"])
                    
                    if model_type == "classification":
                        y.append(item.get("label", "unknown"))
                    elif model_type == "regression":
                        y.append(item.get("target", 0.0))
                    elif model_type == "clustering":
                        y.append(item.get("cluster", 0))
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return [], []
    
    def _create_model(self, algorithm: str, model_type: str):
        """Create ML model based on algorithm and type"""
        if model_type == "classification":
            if algorithm == "random_forest":
                return RandomForestClassifier(n_estimators=100, random_state=42)
            elif algorithm == "logistic_regression":
                return LogisticRegression(random_state=42)
            elif algorithm == "svm":
                return SVC(random_state=42)
            elif algorithm == "naive_bayes":
                return MultinomialNB()
            elif algorithm == "gradient_boosting":
                return GradientBoostingClassifier(random_state=42)
            else:
                return RandomForestClassifier(n_estimators=100, random_state=42)
        
        elif model_type == "regression":
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor
            if algorithm == "random_forest":
                return RandomForestRegressor(n_estimators=100, random_state=42)
            elif algorithm == "linear_regression":
                return LinearRegression()
            else:
                return RandomForestRegressor(n_estimators=100, random_state=42)
        
        elif model_type == "clustering":
            if algorithm == "kmeans":
                return KMeans(n_clusters=5, random_state=42)
            elif algorithm == "dbscan":
                return DBSCAN(eps=0.5, min_samples=5)
            else:
                return KMeans(n_clusters=5, random_state=42)
        
        else:
            return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def _calculate_metrics(self, y_true, y_pred, model_type: str) -> Dict[str, float]:
        """Calculate model metrics"""
        try:
            metrics = {}
            
            if model_type == "classification":
                metrics["accuracy"] = accuracy_score(y_true, y_pred)
                metrics["precision"] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics["recall"] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics["f1_score"] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            elif model_type == "regression":
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                metrics["r2_score"] = r2_score(y_true, y_pred)
                metrics["mse"] = mean_squared_error(y_true, y_pred)
                metrics["mae"] = mean_absolute_error(y_true, y_pred)
            
            elif model_type == "clustering":
                from sklearn.metrics import silhouette_score
                if len(set(y_pred)) > 1:  # Only if more than one cluster
                    metrics["silhouette_score"] = silhouette_score(y_true, y_pred)
                else:
                    metrics["silhouette_score"] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}
    
    async def predict(self, model_name: str, text: str) -> Dict[str, Any]:
        """Make prediction using trained model"""
        try:
            if model_name not in self.models:
                return {"error": f"Model '{model_name}' not found"}
            
            model_info = self.models[model_name]
            
            if not model_info["trained"]:
                return {"error": f"Model '{model_name}' is not trained"}
            
            model = model_info["model"]
            vectorizer = model_info["vectorizer"]
            
            # Vectorize text
            text_vectorized = vectorizer.transform([text])
            
            # Make prediction
            prediction = model.predict(text_vectorized)[0]
            prediction_proba = None
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(text_vectorized)[0].tolist()
            
            self.stats["predictions_made"] += 1
            
            return {
                "model_name": model_name,
                "prediction": prediction,
                "prediction_proba": prediction_proba,
                "confidence": max(prediction_proba) if prediction_proba else 1.0,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {"error": str(e)}
    
    async def batch_predict(self, model_name: str, texts: List[str]) -> Dict[str, Any]:
        """Make batch predictions"""
        try:
            if model_name not in self.models:
                return {"error": f"Model '{model_name}' not found"}
            
            model_info = self.models[model_name]
            
            if not model_info["trained"]:
                return {"error": f"Model '{model_name}' is not trained"}
            
            model = model_info["model"]
            vectorizer = model_info["vectorizer"]
            
            # Vectorize texts
            texts_vectorized = vectorizer.transform(texts)
            
            # Make predictions
            predictions = model.predict(texts_vectorized)
            predictions_proba = None
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                predictions_proba = model.predict_proba(texts_vectorized).tolist()
            
            self.stats["predictions_made"] += len(texts)
            
            return {
                "model_name": model_name,
                "predictions": predictions.tolist(),
                "predictions_proba": predictions_proba,
                "total_predictions": len(texts),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error making batch predictions: {e}")
            return {"error": str(e)}
    
    async def auto_ml(self, training_data: List[Dict[str, Any]], model_type: str = "classification") -> Dict[str, Any]:
        """Automated machine learning"""
        try:
            algorithms = []
            
            if model_type == "classification":
                algorithms = ["random_forest", "logistic_regression", "svm", "naive_bayes", "gradient_boosting"]
            elif model_type == "regression":
                algorithms = ["random_forest", "linear_regression"]
            elif model_type == "clustering":
                algorithms = ["kmeans", "dbscan"]
            
            best_model = None
            best_score = -1
            best_algorithm = None
            
            results = []
            
            for algorithm in algorithms:
                try:
                    # Train model
                    model_name = f"auto_ml_{algorithm}_{int(time.time())}"
                    result = await self.train_model(model_name, training_data, model_type, algorithm)
                    
                    if "error" not in result:
                        score = result["metrics"].get("accuracy", result["metrics"].get("r2_score", result["metrics"].get("silhouette_score", 0)))
                        
                        if score > best_score:
                            best_score = score
                            best_model = model_name
                            best_algorithm = algorithm
                        
                        results.append({
                            "algorithm": algorithm,
                            "model_name": model_name,
                            "score": score,
                            "metrics": result["metrics"]
                        })
                
                except Exception as e:
                    logger.error(f"Error in auto ML for {algorithm}: {e}")
                    continue
            
            return {
                "status": "completed",
                "best_model": best_model,
                "best_algorithm": best_algorithm,
                "best_score": best_score,
                "results": results,
                "total_algorithms_tested": len(algorithms)
            }
            
        except Exception as e:
            logger.error(f"Error in auto ML: {e}")
            return {"error": str(e)}
    
    async def save_model(self, model_name: str, file_path: str) -> Dict[str, Any]:
        """Save trained model to file"""
        try:
            if model_name not in self.models:
                return {"error": f"Model '{model_name}' not found"}
            
            model_info = self.models[model_name]
            
            if not model_info["trained"]:
                return {"error": f"Model '{model_name}' is not trained"}
            
            # Save model and vectorizer
            model_data = {
                "model": model_info["model"],
                "vectorizer": model_info["vectorizer"],
                "model_type": model_info["type"],
                "accuracy": model_info["accuracy"],
                "created_at": model_info["created_at"]
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            return {
                "status": "saved",
                "model_name": model_name,
                "file_path": file_path
            }
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return {"error": str(e)}
    
    async def load_model(self, model_name: str, file_path: str) -> Dict[str, Any]:
        """Load model from file"""
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models[model_name] = {
                "type": model_data["model_type"],
                "model": model_data["model"],
                "vectorizer": model_data["vectorizer"],
                "trained": True,
                "accuracy": model_data["accuracy"],
                "created_at": model_data["created_at"]
            }
            
            return {
                "status": "loaded",
                "model_name": model_name,
                "model_type": model_data["model_type"],
                "accuracy": model_data["accuracy"]
            }
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return {"error": str(e)}
    
    def get_models(self) -> Dict[str, Any]:
        """Get all models"""
        return {
            "models": self.models,
            "model_count": len(self.models)
        }
    
    def get_model_metrics(self, model_name: str = None) -> Dict[str, Any]:
        """Get model metrics"""
        if model_name:
            if model_name in self.model_metrics:
                return self.model_metrics[model_name]
            else:
                return {"error": f"Metrics for model '{model_name}' not found"}
        else:
            return self.model_metrics
    
    def get_ml_stats(self) -> Dict[str, Any]:
        """Get ML statistics"""
        uptime = time.time() - self.stats["start_time"]
        return {
            "stats": self.stats.copy(),
            "uptime_seconds": round(uptime, 2),
            "uptime_hours": round(uptime / 3600, 2),
            "models_count": len(self.models),
            "trained_models_count": len([m for m in self.models.values() if m["trained"]]),
            "model_metrics_count": len(self.model_metrics)
        }

# Global instance
ml_system = MLSystem()













