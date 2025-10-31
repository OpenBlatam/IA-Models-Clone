"""
Gamma App - Advanced Machine Learning Engine
Advanced ML capabilities with predictive analytics, recommendation systems, and automated optimization
"""

import asyncio
import json
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import pickle
from pathlib import Path
import sqlite3
import redis
from collections import defaultdict, Counter
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import pipeline, AutoTokenizer, AutoModel
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import networkx as nx

logger = logging.getLogger(__name__)

class MLTaskType(Enum):
    """Types of ML tasks"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    RECOMMENDATION = "recommendation"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"

class ModelType(Enum):
    """Types of ML models"""
    SKLEARN = "sklearn"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    TRANSFORMERS = "transformers"
    CUSTOM = "custom"

class OptimizationAlgorithm(Enum):
    """Optimization algorithms"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    NEURAL_ARCHITECTURE = "neural_architecture"

@dataclass
class MLDataset:
    """ML Dataset definition"""
    name: str
    data: pd.DataFrame
    target_column: Optional[str] = None
    feature_columns: List[str] = None
    metadata: Dict[str, Any] = None
    created_at: datetime = None

@dataclass
class MLModel:
    """ML Model definition"""
    model_id: str
    name: str
    task_type: MLTaskType
    model_type: ModelType
    model_object: Any = None
    accuracy: float = 0.0
    training_data: str = None
    hyperparameters: Dict[str, Any] = None
    created_at: datetime = None
    last_trained: datetime = None

@dataclass
class PredictionResult:
    """Prediction result"""
    prediction: Any
    confidence: float
    probabilities: Dict[str, float] = None
    metadata: Dict[str, Any] = None

class AdvancedMLEngine:
    """Advanced Machine Learning Engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config.get("database_path", "ml_engine.db")
        self.redis_client = None
        self.models = {}
        self.datasets = {}
        self.pipelines = {}
        self.recommendation_models = {}
        self.anomaly_detectors = {}
        self.time_series_models = {}
        
        # Initialize components
        self._init_database()
        self._init_redis()
        self._init_tensorflow()
        self._init_pytorch()
    
    def _init_database(self):
        """Initialize ML database"""
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ml_models (
                    model_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    model_data BLOB,
                    accuracy REAL DEFAULT 0.0,
                    training_data TEXT,
                    hyperparameters TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_trained DATETIME
                )
            """)
            
            # Create datasets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ml_datasets (
                    dataset_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    data_path TEXT NOT NULL,
                    target_column TEXT,
                    feature_columns TEXT,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    prediction_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    input_data TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL,
                    probabilities TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES ml_models (model_id)
                )
            """)
            
            conn.commit()
        
        logger.info("ML Engine database initialized")
    
    def _init_redis(self):
        """Initialize Redis for caching"""
        try:
            redis_url = self.config.get("redis_url", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            logger.info("Redis client initialized for ML Engine")
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
    
    def _init_tensorflow(self):
        """Initialize TensorFlow"""
        try:
            # Configure TensorFlow for optimal performance
            tf.config.optimizer.set_jit(True)
            tf.config.threading.set_inter_op_parallelism_threads(0)
            tf.config.threading.set_intra_op_parallelism_threads(0)
            
            # Enable mixed precision
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            logger.info("TensorFlow initialized with optimizations")
        except Exception as e:
            logger.warning(f"TensorFlow initialization failed: {e}")
    
    def _init_pytorch(self):
        """Initialize PyTorch"""
        try:
            # Set random seeds for reproducibility
            torch.manual_seed(42)
            np.random.seed(42)
            
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            logger.info("PyTorch initialized with optimizations")
        except Exception as e:
            logger.warning(f"PyTorch initialization failed: {e}")
    
    async def create_dataset(
        self,
        name: str,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new ML dataset"""
        
        dataset_id = f"dataset_{int(time.time())}"
        
        # Prepare dataset
        dataset = MLDataset(
            name=name,
            data=data.copy(),
            target_column=target_column,
            feature_columns=feature_columns or list(data.columns),
            metadata=metadata or {},
            created_at=datetime.now()
        )
        
        # Store dataset
        dataset_path = Path(f"data/datasets/{dataset_id}.pkl")
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        # Store metadata in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ml_datasets
                (dataset_id, name, data_path, target_column, feature_columns, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                dataset_id,
                name,
                str(dataset_path),
                target_column,
                json.dumps(feature_columns or []),
                json.dumps(metadata or {}),
                dataset.created_at.isoformat()
            ))
            conn.commit()
        
        # Cache dataset
        self.datasets[dataset_id] = dataset
        
        logger.info(f"Dataset created: {dataset_id}")
        return dataset_id
    
    async def train_model(
        self,
        dataset_id: str,
        task_type: MLTaskType,
        model_type: ModelType = ModelType.SKLEARN,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Train a new ML model"""
        
        # Load dataset
        dataset = self.datasets.get(dataset_id)
        if not dataset:
            dataset = await self._load_dataset(dataset_id)
        
        if not dataset:
            raise ValueError(f"Dataset not found: {dataset_id}")
        
        model_id = f"model_{int(time.time())}"
        
        # Prepare data
        X = dataset.data[dataset.feature_columns]
        y = dataset.data[dataset.target_column] if dataset.target_column else None
        
        # Train model based on type
        if model_type == ModelType.SKLEARN:
            model = await self._train_sklearn_model(X, y, task_type, hyperparameters)
        elif model_type == ModelType.TENSORFLOW:
            model = await self._train_tensorflow_model(X, y, task_type, hyperparameters)
        elif model_type == ModelType.PYTORCH:
            model = await self._train_pytorch_model(X, y, task_type, hyperparameters)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Evaluate model
        accuracy = await self._evaluate_model(model, X, y, task_type)
        
        # Create model object
        ml_model = MLModel(
            model_id=model_id,
            name=f"{task_type.value}_{model_type.value}",
            task_type=task_type,
            model_type=model_type,
            model_object=model,
            accuracy=accuracy,
            training_data=dataset_id,
            hyperparameters=hyperparameters or {},
            created_at=datetime.now(),
            last_trained=datetime.now()
        )
        
        # Store model
        await self._save_model(ml_model)
        
        # Cache model
        self.models[model_id] = ml_model
        
        logger.info(f"Model trained: {model_id} with accuracy: {accuracy:.4f}")
        return model_id
    
    async def _train_sklearn_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: MLTaskType,
        hyperparameters: Optional[Dict[str, Any]]
    ) -> Any:
        """Train sklearn model"""
        
        if task_type == MLTaskType.CLASSIFICATION:
            model = RandomForestClassifier(
                n_estimators=hyperparameters.get('n_estimators', 100),
                max_depth=hyperparameters.get('max_depth', 10),
                random_state=42
            )
        elif task_type == MLTaskType.REGRESSION:
            model = GradientBoostingRegressor(
                n_estimators=hyperparameters.get('n_estimators', 100),
                max_depth=hyperparameters.get('max_depth', 6),
                random_state=42
            )
        elif task_type == MLTaskType.CLUSTERING:
            model = KMeans(
                n_clusters=hyperparameters.get('n_clusters', 3),
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported sklearn task type: {task_type}")
        
        # Train model
        if task_type == MLTaskType.CLUSTERING:
            model.fit(X)
        else:
            model.fit(X, y)
        
        return model
    
    async def _train_tensorflow_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: MLTaskType,
        hyperparameters: Optional[Dict[str, Any]]
    ) -> Any:
        """Train TensorFlow model"""
        
        # Prepare data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create model
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(
            hyperparameters.get('hidden_units', 128),
            activation='relu',
            input_shape=(X_scaled.shape[1],)
        ))
        
        # Hidden layers
        for _ in range(hyperparameters.get('hidden_layers', 2)):
            model.add(layers.Dropout(hyperparameters.get('dropout_rate', 0.2)))
            model.add(layers.Dense(
                hyperparameters.get('hidden_units', 128),
                activation='relu'
            ))
        
        # Output layer
        if task_type == MLTaskType.CLASSIFICATION:
            n_classes = len(y.unique())
            if n_classes == 2:
                model.add(layers.Dense(1, activation='sigmoid'))
                loss = 'binary_crossentropy'
            else:
                model.add(layers.Dense(n_classes, activation='softmax'))
                loss = 'sparse_categorical_crossentropy'
        else:  # Regression
            model.add(layers.Dense(1, activation='linear'))
            loss = 'mse'
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=hyperparameters.get('learning_rate', 0.001)
            ),
            loss=loss,
            metrics=['accuracy'] if task_type == MLTaskType.CLASSIFICATION else ['mae']
        )
        
        # Train model
        model.fit(
            X_scaled, y,
            epochs=hyperparameters.get('epochs', 100),
            batch_size=hyperparameters.get('batch_size', 32),
            validation_split=0.2,
            verbose=0
        )
        
        return model
    
    async def _train_pytorch_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: MLTaskType,
        hyperparameters: Optional[Dict[str, Any]]
    ) -> Any:
        """Train PyTorch model"""
        
        # Prepare data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y.values) if task_type == MLTaskType.REGRESSION else torch.LongTensor(y.values)
        
        # Create model
        class PyTorchModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, task_type):
                super(PyTorchModel, self).__init__()
                self.task_type = task_type
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, output_size)
                self.dropout = nn.Dropout(0.2)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
        
        # Initialize model
        input_size = X_scaled.shape[1]
        hidden_size = hyperparameters.get('hidden_size', 128)
        output_size = 1 if task_type == MLTaskType.REGRESSION else len(y.unique())
        
        model = PyTorchModel(input_size, hidden_size, output_size, task_type)
        
        # Define loss and optimizer
        if task_type == MLTaskType.CLASSIFICATION:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        optimizer = optim.Adam(model.parameters(), lr=hyperparameters.get('learning_rate', 0.001))
        
        # Train model
        epochs = hyperparameters.get('epochs', 100)
        batch_size = hyperparameters.get('batch_size', 32)
        
        for epoch in range(epochs):
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        return model
    
    async def _evaluate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: MLTaskType
    ) -> float:
        """Evaluate model performance"""
        
        try:
            if task_type == MLTaskType.CLASSIFICATION:
                predictions = model.predict(X)
                if hasattr(predictions, 'argmax'):
                    predictions = predictions.argmax(axis=1)
                accuracy = accuracy_score(y, predictions)
            elif task_type == MLTaskType.REGRESSION:
                predictions = model.predict(X)
                # Calculate R² score
                ss_res = np.sum((y - predictions) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                accuracy = 1 - (ss_res / ss_tot)
            else:
                accuracy = 0.0
            
            return max(0.0, min(1.0, accuracy))  # Clamp between 0 and 1
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return 0.0
    
    async def predict(
        self,
        model_id: str,
        input_data: Union[pd.DataFrame, Dict[str, Any], List[Any]]
    ) -> PredictionResult:
        """Make prediction using trained model"""
        
        # Load model
        model = self.models.get(model_id)
        if not model:
            model = await self._load_model(model_id)
        
        if not model:
            raise ValueError(f"Model not found: {model_id}")
        
        # Prepare input data
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            input_df = pd.DataFrame(input_data)
        else:
            input_df = input_data
        
        # Make prediction
        try:
            if model.model_type == ModelType.SKLEARN:
                prediction = model.model_object.predict(input_df)
                probabilities = None
                
                if hasattr(model.model_object, 'predict_proba'):
                    probabilities = model.model_object.predict_proba(input_df)[0]
                    probabilities = {
                        f"class_{i}": prob for i, prob in enumerate(probabilities)
                    }
                
            elif model.model_type == ModelType.TENSORFLOW:
                # Preprocess data (simplified)
                scaler = StandardScaler()
                input_scaled = scaler.fit_transform(input_df)
                prediction = model.model_object.predict(input_scaled)
                
                if model.task_type == MLTaskType.CLASSIFICATION:
                    probabilities = {
                        f"class_{i}": prob for i, prob in enumerate(prediction[0])
                    }
                    prediction = prediction.argmax(axis=1)[0]
                else:
                    prediction = prediction[0][0]
                    probabilities = None
            
            elif model.model_type == ModelType.PYTORCH:
                model.model_object.eval()
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(input_df.values)
                    prediction = model.model_object(input_tensor)
                    
                    if model.task_type == MLTaskType.CLASSIFICATION:
                        probabilities = torch.softmax(prediction, dim=1).numpy()[0]
                        probabilities = {
                            f"class_{i}": prob for i, prob in enumerate(probabilities)
                        }
                        prediction = prediction.argmax(dim=1).item()
                    else:
                        prediction = prediction.item()
                        probabilities = None
            
            # Calculate confidence
            confidence = 0.8  # Simplified confidence calculation
            
            result = PredictionResult(
                prediction=prediction,
                confidence=confidence,
                probabilities=probabilities,
                metadata={
                    "model_id": model_id,
                    "model_type": model.model_type.value,
                    "task_type": model.task_type.value,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Store prediction
            await self._store_prediction(model_id, input_data, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise e
    
    async def create_recommendation_system(
        self,
        user_data: pd.DataFrame,
        item_data: pd.DataFrame,
        interaction_data: pd.DataFrame,
        algorithm: str = "collaborative_filtering"
    ) -> str:
        """Create recommendation system"""
        
        system_id = f"rec_sys_{int(time.time())}"
        
        if algorithm == "collaborative_filtering":
            model = await self._create_collaborative_filtering_model(
                user_data, item_data, interaction_data
            )
        elif algorithm == "content_based":
            model = await self._create_content_based_model(
                user_data, item_data, interaction_data
            )
        elif algorithm == "hybrid":
            model = await self._create_hybrid_model(
                user_data, item_data, interaction_data
            )
        else:
            raise ValueError(f"Unsupported recommendation algorithm: {algorithm}")
        
        self.recommendation_models[system_id] = {
            "model": model,
            "algorithm": algorithm,
            "user_data": user_data,
            "item_data": item_data,
            "interaction_data": interaction_data,
            "created_at": datetime.now()
        }
        
        logger.info(f"Recommendation system created: {system_id}")
        return system_id
    
    async def _create_collaborative_filtering_model(
        self,
        user_data: pd.DataFrame,
        item_data: pd.DataFrame,
        interaction_data: pd.DataFrame
    ) -> Any:
        """Create collaborative filtering model"""
        
        # Create user-item matrix
        user_item_matrix = interaction_data.pivot_table(
            index='user_id',
            columns='item_id',
            values='rating',
            fill_value=0
        )
        
        # Apply SVD for dimensionality reduction
        from sklearn.decomposition import TruncatedSVD
        
        svd = TruncatedSVD(n_components=50, random_state=42)
        user_factors = svd.fit_transform(user_item_matrix)
        item_factors = svd.components_.T
        
        return {
            "type": "collaborative_filtering",
            "user_factors": user_factors,
            "item_factors": item_factors,
            "user_item_matrix": user_item_matrix,
            "svd": svd
        }
    
    async def _create_content_based_model(
        self,
        user_data: pd.DataFrame,
        item_data: pd.DataFrame,
        interaction_data: pd.DataFrame
    ) -> Any:
        """Create content-based model"""
        
        # Create item feature matrix
        item_features = item_data.select_dtypes(include=[np.number])
        
        # Create user profile from interaction history
        user_profiles = {}
        for user_id in user_data['user_id']:
            user_interactions = interaction_data[interaction_data['user_id'] == user_id]
            if len(user_interactions) > 0:
                user_items = user_interactions['item_id'].values
                user_profile = item_features[item_features.index.isin(user_items)].mean()
                user_profiles[user_id] = user_profile
        
        return {
            "type": "content_based",
            "item_features": item_features,
            "user_profiles": user_profiles
        }
    
    async def _create_hybrid_model(
        self,
        user_data: pd.DataFrame,
        item_data: pd.DataFrame,
        interaction_data: pd.DataFrame
    ) -> Any:
        """Create hybrid recommendation model"""
        
        # Create both collaborative and content-based models
        cf_model = await self._create_collaborative_filtering_model(
            user_data, item_data, interaction_data
        )
        cb_model = await self._create_content_based_model(
            user_data, item_data, interaction_data
        )
        
        return {
            "type": "hybrid",
            "collaborative_model": cf_model,
            "content_model": cb_model,
            "weights": {"collaborative": 0.6, "content": 0.4}
        }
    
    async def get_recommendations(
        self,
        system_id: str,
        user_id: str,
        n_recommendations: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recommendations for a user"""
        
        system = self.recommendation_models.get(system_id)
        if not system:
            raise ValueError(f"Recommendation system not found: {system_id}")
        
        model = system["model"]
        algorithm = system["algorithm"]
        
        if algorithm == "collaborative_filtering":
            recommendations = await self._get_cf_recommendations(
                model, user_id, n_recommendations
            )
        elif algorithm == "content_based":
            recommendations = await self._get_cb_recommendations(
                model, user_id, n_recommendations
            )
        elif algorithm == "hybrid":
            recommendations = await self._get_hybrid_recommendations(
                model, user_id, n_recommendations
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        return recommendations
    
    async def _get_cf_recommendations(
        self,
        model: Dict[str, Any],
        user_id: str,
        n_recommendations: int
    ) -> List[Dict[str, Any]]:
        """Get collaborative filtering recommendations"""
        
        user_item_matrix = model["user_item_matrix"]
        user_factors = model["user_factors"]
        item_factors = model["item_factors"]
        
        # Find user index
        if user_id not in user_item_matrix.index:
            return []
        
        user_idx = user_item_matrix.index.get_loc(user_id)
        user_vector = user_factors[user_idx]
        
        # Calculate scores for all items
        scores = np.dot(item_factors, user_vector)
        
        # Get top recommendations
        top_items = np.argsort(scores)[::-1][:n_recommendations]
        
        recommendations = []
        for item_idx in top_items:
            item_id = user_item_matrix.columns[item_idx]
            score = scores[item_idx]
            recommendations.append({
                "item_id": item_id,
                "score": float(score),
                "type": "collaborative_filtering"
            })
        
        return recommendations
    
    async def _get_cb_recommendations(
        self,
        model: Dict[str, Any],
        user_id: str,
        n_recommendations: int
    ) -> List[Dict[str, Any]]:
        """Get content-based recommendations"""
        
        user_profiles = model["user_profiles"]
        item_features = model["item_features"]
        
        if user_id not in user_profiles:
            return []
        
        user_profile = user_profiles[user_id]
        
        # Calculate similarity scores
        scores = []
        for item_id, item_features_row in item_features.iterrows():
            similarity = np.dot(user_profile, item_features_row) / (
                np.linalg.norm(user_profile) * np.linalg.norm(item_features_row)
            )
            scores.append((item_id, similarity))
        
        # Sort by similarity
        scores.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for item_id, score in scores[:n_recommendations]:
            recommendations.append({
                "item_id": item_id,
                "score": float(score),
                "type": "content_based"
            })
        
        return recommendations
    
    async def _get_hybrid_recommendations(
        self,
        model: Dict[str, Any],
        user_id: str,
        n_recommendations: int
    ) -> List[Dict[str, Any]]:
        """Get hybrid recommendations"""
        
        # Get recommendations from both models
        cf_recs = await self._get_cf_recommendations(
            model["collaborative_model"], user_id, n_recommendations * 2
        )
        cb_recs = await self._get_cb_recommendations(
            model["content_model"], user_id, n_recommendations * 2
        )
        
        # Combine recommendations
        combined_scores = {}
        weights = model["weights"]
        
        for rec in cf_recs:
            item_id = rec["item_id"]
            combined_scores[item_id] = combined_scores.get(item_id, 0) + (
                rec["score"] * weights["collaborative"]
            )
        
        for rec in cb_recs:
            item_id = rec["item_id"]
            combined_scores[item_id] = combined_scores.get(item_id, 0) + (
                rec["score"] * weights["content"]
            )
        
        # Sort by combined score
        sorted_items = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        recommendations = []
        for item_id, score in sorted_items[:n_recommendations]:
            recommendations.append({
                "item_id": item_id,
                "score": float(score),
                "type": "hybrid"
            })
        
        return recommendations
    
    async def detect_anomalies(
        self,
        data: pd.DataFrame,
        algorithm: str = "isolation_forest",
        contamination: float = 0.1
    ) -> Dict[str, Any]:
        """Detect anomalies in data"""
        
        from sklearn.ensemble import IsolationForest
        from sklearn.cluster import DBSCAN
        
        if algorithm == "isolation_forest":
            detector = IsolationForest(
                contamination=contamination,
                random_state=42
            )
            anomalies = detector.fit_predict(data)
            scores = detector.decision_function(data)
            
        elif algorithm == "dbscan":
            detector = DBSCAN(eps=0.5, min_samples=5)
            clusters = detector.fit_predict(data)
            anomalies = (clusters == -1).astype(int)
            scores = -detector.core_sample_indices_  # Simplified scoring
            
        else:
            raise ValueError(f"Unsupported anomaly detection algorithm: {algorithm}")
        
        # Create results
        anomaly_indices = np.where(anomalies == -1)[0]
        normal_indices = np.where(anomalies == 1)[0]
        
        result = {
            "algorithm": algorithm,
            "total_samples": len(data),
            "anomalies_detected": len(anomaly_indices),
            "anomaly_rate": len(anomaly_indices) / len(data),
            "anomaly_indices": anomaly_indices.tolist(),
            "normal_indices": normal_indices.tolist(),
            "anomaly_scores": scores.tolist(),
            "detected_at": datetime.now().isoformat()
        }
        
        return result
    
    async def analyze_time_series(
        self,
        data: pd.DataFrame,
        target_column: str,
        time_column: str,
        analysis_type: str = "trend"
    ) -> Dict[str, Any]:
        """Analyze time series data"""
        
        # Sort by time
        data_sorted = data.sort_values(time_column)
        time_series = data_sorted[target_column].values
        
        if analysis_type == "trend":
            # Linear trend analysis
            x = np.arange(len(time_series))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, time_series)
            
            result = {
                "type": "trend",
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_value ** 2,
                "p_value": p_value,
                "trend_direction": "increasing" if slope > 0 else "decreasing",
                "trend_strength": abs(r_value)
            }
            
        elif analysis_type == "seasonality":
            # Seasonal decomposition (simplified)
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Ensure we have enough data points
            if len(time_series) < 24:
                result = {"type": "seasonality", "error": "Insufficient data for seasonal analysis"}
            else:
                decomposition = seasonal_decompose(time_series, model='additive', period=12)
                
                result = {
                    "type": "seasonality",
                    "trend": decomposition.trend.tolist(),
                    "seasonal": decomposition.seasonal.tolist(),
                    "residual": decomposition.resid.tolist(),
                    "seasonal_strength": np.var(decomposition.seasonal) / np.var(time_series)
                }
        
        elif analysis_type == "forecast":
            # Simple forecasting using linear regression
            x = np.arange(len(time_series))
            slope, intercept, _, _, _ = stats.linregress(x, time_series)
            
            # Forecast next 10 periods
            future_x = np.arange(len(time_series), len(time_series) + 10)
            forecast = slope * future_x + intercept
            
            result = {
                "type": "forecast",
                "forecast_values": forecast.tolist(),
                "forecast_periods": 10,
                "confidence_interval": {
                    "lower": (forecast - 2 * np.std(time_series)).tolist(),
                    "upper": (forecast + 2 * np.std(time_series)).tolist()
                }
            }
        
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
        
        result["analyzed_at"] = datetime.now().isoformat()
        return result
    
    async def optimize_hyperparameters(
        self,
        model_id: str,
        optimization_algorithm: OptimizationAlgorithm = OptimizationAlgorithm.GRID_SEARCH,
        param_grid: Optional[Dict[str, List]] = None
    ) -> Dict[str, Any]:
        """Optimize model hyperparameters"""
        
        model = self.models.get(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")
        
        # Load training data
        dataset = await self._load_dataset(model.training_data)
        if not dataset:
            raise ValueError(f"Training dataset not found: {model.training_data}")
        
        X = dataset.data[dataset.feature_columns]
        y = dataset.data[dataset.target_column] if dataset.target_column else None
        
        if optimization_algorithm == OptimizationAlgorithm.GRID_SEARCH:
            result = await self._grid_search_optimization(
                model, X, y, param_grid
            )
        elif optimization_algorithm == OptimizationAlgorithm.RANDOM_SEARCH:
            result = await self._random_search_optimization(
                model, X, y, param_grid
            )
        else:
            raise ValueError(f"Unsupported optimization algorithm: {optimization_algorithm}")
        
        return result
    
    async def _grid_search_optimization(
        self,
        model: MLModel,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Optional[Dict[str, List]]
    ) -> Dict[str, Any]:
        """Grid search hyperparameter optimization"""
        
        from sklearn.model_selection import GridSearchCV
        
        if not param_grid:
            # Default parameter grids
            if model.task_type == MLTaskType.CLASSIFICATION:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10]
                }
            else:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
        
        # Create base model
        if model.task_type == MLTaskType.CLASSIFICATION:
            base_model = RandomForestClassifier(random_state=42)
        else:
            base_model = GradientBoostingRegressor(random_state=42)
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy' if model.task_type == MLTaskType.CLASSIFICATION else 'r2',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        return {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "optimization_method": "grid_search",
            "cv_scores": grid_search.cv_results_['mean_test_score'].tolist(),
            "optimized_at": datetime.now().isoformat()
        }
    
    async def _random_search_optimization(
        self,
        model: MLModel,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Optional[Dict[str, List]]
    ) -> Dict[str, Any]:
        """Random search hyperparameter optimization"""
        
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import randint, uniform
        
        if not param_grid:
            # Default parameter distributions
            if model.task_type == MLTaskType.CLASSIFICATION:
                param_dist = {
                    'n_estimators': randint(50, 300),
                    'max_depth': randint(3, 20),
                    'min_samples_split': randint(2, 20)
                }
            else:
                param_dist = {
                    'n_estimators': randint(50, 300),
                    'max_depth': randint(3, 15),
                    'learning_rate': uniform(0.01, 0.3)
                }
        else:
            param_dist = param_grid
        
        # Create base model
        if model.task_type == MLTaskType.CLASSIFICATION:
            base_model = RandomForestClassifier(random_state=42)
        else:
            base_model = GradientBoostingRegressor(random_state=42)
        
        # Perform random search
        random_search = RandomizedSearchCV(
            base_model,
            param_dist,
            n_iter=50,
            cv=5,
            scoring='accuracy' if model.task_type == MLTaskType.CLASSIFICATION else 'r2',
            random_state=42,
            n_jobs=-1
        )
        
        random_search.fit(X, y)
        
        return {
            "best_params": random_search.best_params_,
            "best_score": random_search.best_score_,
            "optimization_method": "random_search",
            "cv_scores": random_search.cv_results_['mean_test_score'].tolist(),
            "optimized_at": datetime.now().isoformat()
        }
    
    async def _load_dataset(self, dataset_id: str) -> Optional[MLDataset]:
        """Load dataset from storage"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT data_path FROM ml_datasets WHERE dataset_id = ?
            """, (dataset_id,))
            row = cursor.fetchone()
            
            if row:
                dataset_path = Path(row[0])
                if dataset_path.exists():
                    with open(dataset_path, 'rb') as f:
                        dataset = pickle.load(f)
                    self.datasets[dataset_id] = dataset
                    return dataset
        
        return None
    
    async def _load_model(self, model_id: str) -> Optional[MLModel]:
        """Load model from storage"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM ml_models WHERE model_id = ?
            """, (model_id,))
            row = cursor.fetchone()
            
            if row:
                # Load model data
                model_data = pickle.loads(row[3]) if row[3] else None
                
                model = MLModel(
                    model_id=row[0],
                    name=row[1],
                    task_type=MLTaskType(row[2]),
                    model_type=ModelType(row[3]),
                    model_object=model_data,
                    accuracy=row[4],
                    training_data=row[5],
                    hyperparameters=json.loads(row[6]) if row[6] else {},
                    created_at=datetime.fromisoformat(row[7]),
                    last_trained=datetime.fromisoformat(row[8]) if row[8] else None
                )
                
                self.models[model_id] = model
                return model
        
        return None
    
    async def _save_model(self, model: MLModel):
        """Save model to storage"""
        
        # Serialize model object
        model_data = pickle.dumps(model.model_object)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO ml_models
                (model_id, name, task_type, model_type, model_data, accuracy, 
                 training_data, hyperparameters, created_at, last_trained)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model.model_id,
                model.name,
                model.task_type.value,
                model.model_type.value,
                model_data,
                model.accuracy,
                model.training_data,
                json.dumps(model.hyperparameters),
                model.created_at.isoformat(),
                model.last_trained.isoformat() if model.last_trained else None
            ))
            conn.commit()
    
    async def _store_prediction(
        self,
        model_id: str,
        input_data: Any,
        result: PredictionResult
    ):
        """Store prediction result"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ml_predictions
                (prediction_id, model_id, input_data, prediction, confidence, 
                 probabilities, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                f"pred_{int(time.time())}",
                model_id,
                json.dumps(input_data),
                json.dumps(result.prediction),
                result.confidence,
                json.dumps(result.probabilities) if result.probabilities else None,
                datetime.now().isoformat()
            ))
            conn.commit()
    
    async def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Get model performance metrics"""
        
        model = self.models.get(model_id)
        if not model:
            model = await self._load_model(model_id)
        
        if not model:
            raise ValueError(f"Model not found: {model_id}")
        
        # Get recent predictions
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT confidence, created_at FROM ml_predictions 
                WHERE model_id = ? 
                ORDER BY created_at DESC 
                LIMIT 100
            """, (model_id,))
            recent_predictions = cursor.fetchall()
        
        # Calculate performance metrics
        avg_confidence = np.mean([p[0] for p in recent_predictions]) if recent_predictions else 0
        prediction_count = len(recent_predictions)
        
        return {
            "model_id": model_id,
            "model_name": model.name,
            "task_type": model.task_type.value,
            "model_type": model.model_type.value,
            "accuracy": model.accuracy,
            "average_confidence": avg_confidence,
            "total_predictions": prediction_count,
            "created_at": model.created_at.isoformat(),
            "last_trained": model.last_trained.isoformat() if model.last_trained else None
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        
        if self.redis_client:
            self.redis_client.close()
        
        # Clear GPU memory if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("ML Engine cleanup completed")

# Global instance
ml_engine = None

async def get_ml_engine() -> AdvancedMLEngine:
    """Get global ML engine instance"""
    global ml_engine
    if not ml_engine:
        config = {
            "database_path": "data/ml_engine.db",
            "redis_url": "redis://localhost:6379"
        }
        ml_engine = AdvancedMLEngine(config)
    return ml_engine



