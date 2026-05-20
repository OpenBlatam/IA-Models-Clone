"""
Content ML Engine - Advanced machine learning for content analysis and prediction
Following FastAPI best practices: functional programming, RORO pattern, async operations
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, AsyncGenerator, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sentence_transformers import SentenceTransformer
import joblib

logger = logging.getLogger(__name__)


@dataclass
class MLModel:
    """Machine learning model definition"""
    model_id: str
    model_type: str
    model_name: str
    version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_data_size: int
    features_count: int
    created_at: datetime
    last_trained: datetime
    is_active: bool = True


@dataclass
class MLPrediction:
    """ML prediction result"""
    model_id: str
    prediction: Any
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    feature_importance: Optional[Dict[str, float]] = None
    prediction_timestamp: datetime = None


@dataclass
class MLTrainingResult:
    """ML training result"""
    model_id: str
    training_accuracy: float
    validation_accuracy: float
    training_loss: float
    validation_loss: float
    epochs_trained: int
    training_time: float
    best_epoch: int
    training_timestamp: datetime


class ContentDataset(Dataset):
    """Custom dataset for content analysis"""
    
    def __init__(self, texts: List[str], labels: List[Any] = None, embeddings: np.ndarray = None):
        self.texts = texts
        self.labels = labels
        self.embeddings = embeddings
        self.length = len(texts)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        item = {"text": self.texts[idx]}
        if self.labels is not None:
            item["label"] = self.labels[idx]
        if self.embeddings is not None:
            item["embedding"] = self.embeddings[idx]
        return item


class ContentClassifier(nn.Module):
    """Neural network classifier for content analysis"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, dropout: float = 0.3):
        super(ContentClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        logits = self.network(x)
        probabilities = self.softmax(logits)
        return logits, probabilities


class ContentMLEngine:
    """Advanced machine learning engine for content analysis"""
    
    def __init__(self):
        self.models: Dict[str, MLModel] = {}
        self.trained_models: Dict[str, Any] = {}
        self.vectorizers: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.label_encoders: Dict[str, Any] = {}
        self.sentence_transformer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_loaded = False
        self.model_storage_path = Path("models")
        self.model_storage_path.mkdir(exist_ok=True)
        
    async def initialize(self) -> None:
        """Initialize the ML engine"""
        try:
            logger.info("Initializing Content ML Engine...")
            
            # Load sentence transformer
            await self._load_sentence_transformer()
            
            # Load existing models
            await self._load_existing_models()
            
            self.models_loaded = True
            logger.info("Content ML Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Content ML Engine: {e}")
            raise
    
    async def _load_sentence_transformer(self) -> None:
        """Load sentence transformer for embeddings"""
        try:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")
            self.sentence_transformer = None
    
    async def _load_existing_models(self) -> None:
        """Load existing trained models"""
        try:
            model_files = list(self.model_storage_path.glob("*.pkl"))
            for model_file in model_files:
                model_id = model_file.stem
                try:
                    with open(model_file, 'rb') as f:
                        model_data = pickle.load(f)
                        self.trained_models[model_id] = model_data
                    logger.info(f"Loaded model: {model_id}")
                except Exception as e:
                    logger.warning(f"Failed to load model {model_id}: {e}")
        except Exception as e:
            logger.warning(f"Failed to load existing models: {e}")
    
    async def train_content_classifier(
        self,
        texts: List[str],
        labels: List[str],
        model_name: str = "content_classifier",
        model_type: str = "neural_network"
    ) -> str:
        """Train a content classification model"""
        
        try:
            model_id = f"{model_name}_{int(datetime.now().timestamp())}"
            
            if model_type == "neural_network":
                return await self._train_neural_network_classifier(texts, labels, model_id)
            elif model_type == "random_forest":
                return await self._train_random_forest_classifier(texts, labels, model_id)
            elif model_type == "gradient_boosting":
                return await self._train_gradient_boosting_classifier(texts, labels, model_id)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Error training content classifier: {e}")
            raise
    
    async def _train_neural_network_classifier(
        self,
        texts: List[str],
        labels: List[str],
        model_id: str
    ) -> str:
        """Train neural network classifier"""
        
        try:
            # Generate embeddings
            embeddings = await self._generate_embeddings(texts)
            
            # Encode labels
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                embeddings, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
            )
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_train_tensor = torch.LongTensor(y_train).to(self.device)
            y_val_tensor = torch.LongTensor(y_val).to(self.device)
            
            # Create model
            input_size = embeddings.shape[1]
            num_classes = len(np.unique(encoded_labels))
            model = ContentClassifier(
                input_size=input_size,
                hidden_sizes=[512, 256, 128],
                num_classes=num_classes
            ).to(self.device)
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
            
            # Training loop
            best_val_accuracy = 0
            best_epoch = 0
            training_losses = []
            validation_losses = []
            
            for epoch in range(50):
                # Training
                model.train()
                optimizer.zero_grad()
                logits, _ = model(X_train_tensor)
                loss = criterion(logits, y_train_tensor)
                loss.backward()
                optimizer.step()
                training_losses.append(loss.item())
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_logits, _ = model(X_val_tensor)
                    val_loss = criterion(val_logits, y_val_tensor)
                    validation_losses.append(val_loss.item())
                    
                    # Calculate accuracy
                    predictions = torch.argmax(val_logits, dim=1)
                    val_accuracy = (predictions == y_val_tensor).float().mean().item()
                    
                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        best_epoch = epoch
                
                scheduler.step(val_loss)
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Final evaluation
            model.eval()
            with torch.no_grad():
                train_logits, _ = model(X_train_tensor)
                train_predictions = torch.argmax(train_logits, dim=1)
                train_accuracy = (train_predictions == y_train_tensor).float().mean().item()
                
                val_logits, _ = model(X_val_tensor)
                val_predictions = torch.argmax(val_logits, dim=1)
                val_accuracy = (val_predictions == y_val_tensor).float().mean().item()
            
            # Calculate metrics
            precision = precision_score(y_val, val_predictions.cpu().numpy(), average='weighted')
            recall = recall_score(y_val, val_predictions.cpu().numpy(), average='weighted')
            f1 = f1_score(y_val, val_predictions.cpu().numpy(), average='weighted')
            
            # Create model record
            ml_model = MLModel(
                model_id=model_id,
                model_type="neural_network",
                model_name="content_classifier",
                version="1.0.0",
                accuracy=val_accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                training_data_size=len(texts),
                features_count=input_size,
                created_at=datetime.now(),
                last_trained=datetime.now()
            )
            
            # Store model
            self.models[model_id] = ml_model
            self.trained_models[model_id] = {
                "model": model,
                "label_encoder": label_encoder,
                "scaler": None
            }
            
            # Save model
            await self._save_model(model_id)
            
            logger.info(f"Neural network classifier trained successfully: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Error training neural network classifier: {e}")
            raise
    
    async def _train_random_forest_classifier(
        self,
        texts: List[str],
        labels: List[str],
        model_id: str
    ) -> str:
        """Train random forest classifier"""
        
        try:
            # Generate embeddings
            embeddings = await self._generate_embeddings(texts)
            
            # Encode labels
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                embeddings, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
            )
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Evaluate model
            train_predictions = model.predict(X_train)
            val_predictions = model.predict(X_val)
            
            train_accuracy = accuracy_score(y_train, train_predictions)
            val_accuracy = accuracy_score(y_val, val_predictions)
            precision = precision_score(y_val, val_predictions, average='weighted')
            recall = recall_score(y_val, val_predictions, average='weighted')
            f1 = f1_score(y_val, val_predictions, average='weighted')
            
            # Create model record
            ml_model = MLModel(
                model_id=model_id,
                model_type="random_forest",
                model_name="content_classifier",
                version="1.0.0",
                accuracy=val_accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                training_data_size=len(texts),
                features_count=embeddings.shape[1],
                created_at=datetime.now(),
                last_trained=datetime.now()
            )
            
            # Store model
            self.models[model_id] = ml_model
            self.trained_models[model_id] = {
                "model": model,
                "label_encoder": label_encoder,
                "scaler": None
            }
            
            # Save model
            await self._save_model(model_id)
            
            logger.info(f"Random forest classifier trained successfully: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Error training random forest classifier: {e}")
            raise
    
    async def _train_gradient_boosting_classifier(
        self,
        texts: List[str],
        labels: List[str],
        model_id: str
    ) -> str:
        """Train gradient boosting classifier"""
        
        try:
            # Generate embeddings
            embeddings = await self._generate_embeddings(texts)
            
            # Encode labels
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                embeddings, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
            )
            
            # Train model
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Evaluate model
            train_predictions = model.predict(X_train)
            val_predictions = model.predict(X_val)
            
            # Convert to classification predictions
            train_predictions_class = np.round(train_predictions).astype(int)
            val_predictions_class = np.round(val_predictions).astype(int)
            
            train_accuracy = accuracy_score(y_train, train_predictions_class)
            val_accuracy = accuracy_score(y_val, val_predictions_class)
            precision = precision_score(y_val, val_predictions_class, average='weighted')
            recall = recall_score(y_val, val_predictions_class, average='weighted')
            f1 = f1_score(y_val, val_predictions_class, average='weighted')
            
            # Create model record
            ml_model = MLModel(
                model_id=model_id,
                model_type="gradient_boosting",
                model_name="content_classifier",
                version="1.0.0",
                accuracy=val_accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                training_data_size=len(texts),
                features_count=embeddings.shape[1],
                created_at=datetime.now(),
                last_trained=datetime.now()
            )
            
            # Store model
            self.models[model_id] = ml_model
            self.trained_models[model_id] = {
                "model": model,
                "label_encoder": label_encoder,
                "scaler": None
            }
            
            # Save model
            await self._save_model(model_id)
            
            logger.info(f"Gradient boosting classifier trained successfully: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Error training gradient boosting classifier: {e}")
            raise
    
    async def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        try:
            if self.sentence_transformer:
                embeddings = self.sentence_transformer.encode(texts)
                return embeddings
            else:
                # Fallback to TF-IDF
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                embeddings = vectorizer.fit_transform(texts).toarray()
                return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    async def predict_content_class(
        self,
        text: str,
        model_id: str
    ) -> MLPrediction:
        """Predict content class using trained model"""
        
        try:
            if model_id not in self.trained_models:
                raise ValueError(f"Model {model_id} not found")
            
            model_data = self.trained_models[model_id]
            model = model_data["model"]
            label_encoder = model_data["label_encoder"]
            
            # Generate embedding
            embedding = await self._generate_embeddings([text])
            
            if self.models[model_id].model_type == "neural_network":
                # Neural network prediction
                model.eval()
                with torch.no_grad():
                    embedding_tensor = torch.FloatTensor(embedding).to(self.device)
                    logits, probabilities = model(embedding_tensor)
                    prediction_idx = torch.argmax(logits, dim=1).item()
                    confidence = probabilities[0][prediction_idx].item()
                    
                    # Get class probabilities
                    class_probs = {}
                    for i, class_name in enumerate(label_encoder.classes_):
                        class_probs[class_name] = probabilities[0][i].item()
                    
                    predicted_class = label_encoder.inverse_transform([prediction_idx])[0]
            else:
                # Traditional ML model prediction
                prediction_idx = model.predict(embedding)[0]
                predicted_class = label_encoder.inverse_transform([prediction_idx])[0]
                
                # Get prediction probabilities if available
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(embedding)[0]
                    confidence = max(probabilities)
                    class_probs = {}
                    for i, class_name in enumerate(label_encoder.classes_):
                        class_probs[class_name] = probabilities[i]
                else:
                    confidence = 1.0
                    class_probs = {predicted_class: 1.0}
            
            return MLPrediction(
                model_id=model_id,
                prediction=predicted_class,
                confidence=confidence,
                probabilities=class_probs,
                prediction_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error predicting content class: {e}")
            raise
    
    async def train_content_clustering(
        self,
        texts: List[str],
        n_clusters: int = 5,
        model_name: str = "content_clustering"
    ) -> str:
        """Train content clustering model"""
        
        try:
            model_id = f"{model_name}_{int(datetime.now().timestamp())}"
            
            # Generate embeddings
            embeddings = await self._generate_embeddings(texts)
            
            # Train clustering model
            clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clustering_model.fit_predict(embeddings)
            
            # Calculate silhouette score
            from sklearn.metrics import silhouette_score
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
            
            # Create model record
            ml_model = MLModel(
                model_id=model_id,
                model_type="clustering",
                model_name="content_clustering",
                version="1.0.0",
                accuracy=silhouette_avg,
                precision=0.0,  # Not applicable for clustering
                recall=0.0,     # Not applicable for clustering
                f1_score=0.0,   # Not applicable for clustering
                training_data_size=len(texts),
                features_count=embeddings.shape[1],
                created_at=datetime.now(),
                last_trained=datetime.now()
            )
            
            # Store model
            self.models[model_id] = ml_model
            self.trained_models[model_id] = {
                "model": clustering_model,
                "label_encoder": None,
                "scaler": None
            }
            
            # Save model
            await self._save_model(model_id)
            
            logger.info(f"Content clustering model trained successfully: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Error training content clustering: {e}")
            raise
    
    async def predict_content_cluster(
        self,
        text: str,
        model_id: str
    ) -> MLPrediction:
        """Predict content cluster using trained model"""
        
        try:
            if model_id not in self.trained_models:
                raise ValueError(f"Model {model_id} not found")
            
            model_data = self.trained_models[model_id]
            clustering_model = model_data["model"]
            
            # Generate embedding
            embedding = await self._generate_embeddings([text])
            
            # Predict cluster
            cluster_id = clustering_model.predict(embedding)[0]
            
            # Calculate distance to cluster center
            cluster_center = clustering_model.cluster_centers_[cluster_id]
            distance = np.linalg.norm(embedding[0] - cluster_center)
            
            # Convert distance to confidence (closer = higher confidence)
            max_distance = np.max([np.linalg.norm(embedding[0] - center) for center in clustering_model.cluster_centers_])
            confidence = 1.0 - (distance / max_distance)
            
            return MLPrediction(
                model_id=model_id,
                prediction=cluster_id,
                confidence=confidence,
                prediction_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error predicting content cluster: {e}")
            raise
    
    async def train_topic_modeling(
        self,
        texts: List[str],
        n_topics: int = 10,
        model_name: str = "topic_modeling"
    ) -> str:
        """Train topic modeling using LDA"""
        
        try:
            model_id = f"{model_name}_{int(datetime.now().timestamp())}"
            
            # Create vectorizer
            vectorizer = CountVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Fit and transform texts
            doc_term_matrix = vectorizer.fit_transform(texts)
            
            # Train LDA model
            lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=100
            )
            lda_model.fit(doc_term_matrix)
            
            # Create model record
            ml_model = MLModel(
                model_id=model_id,
                model_type="topic_modeling",
                model_name="topic_modeling",
                version="1.0.0",
                accuracy=0.0,  # Not applicable for topic modeling
                precision=0.0,  # Not applicable for topic modeling
                recall=0.0,     # Not applicable for topic modeling
                f1_score=0.0,   # Not applicable for topic modeling
                training_data_size=len(texts),
                features_count=doc_term_matrix.shape[1],
                created_at=datetime.now(),
                last_trained=datetime.now()
            )
            
            # Store model
            self.models[model_id] = ml_model
            self.trained_models[model_id] = {
                "model": lda_model,
                "vectorizer": vectorizer,
                "scaler": None
            }
            
            # Save model
            await self._save_model(model_id)
            
            logger.info(f"Topic modeling trained successfully: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Error training topic modeling: {e}")
            raise
    
    async def predict_topics(
        self,
        text: str,
        model_id: str,
        top_n: int = 3
    ) -> MLPrediction:
        """Predict topics for text using trained LDA model"""
        
        try:
            if model_id not in self.trained_models:
                raise ValueError(f"Model {model_id} not found")
            
            model_data = self.trained_models[model_id]
            lda_model = model_data["model"]
            vectorizer = model_data["vectorizer"]
            
            # Transform text
            doc_term_matrix = vectorizer.transform([text])
            
            # Predict topics
            topic_probs = lda_model.transform(doc_term_matrix)[0]
            
            # Get top topics
            top_topic_indices = np.argsort(topic_probs)[-top_n:][::-1]
            top_topics = [(idx, topic_probs[idx]) for idx in top_topic_indices]
            
            # Get topic words
            feature_names = vectorizer.get_feature_names_out()
            topic_words = {}
            for topic_idx, _ in top_topics:
                top_words_idx = np.argsort(lda_model.components_[topic_idx])[-10:][::-1]
                topic_words[topic_idx] = [feature_names[i] for i in top_words_idx]
            
            return MLPrediction(
                model_id=model_id,
                prediction=top_topics,
                confidence=max(topic_probs),
                probabilities={f"topic_{idx}": prob for idx, prob in top_topics},
                prediction_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error predicting topics: {e}")
            raise
    
    async def _save_model(self, model_id: str) -> None:
        """Save trained model to disk"""
        try:
            model_file = self.model_storage_path / f"{model_id}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(self.trained_models[model_id], f)
            logger.info(f"Model saved: {model_id}")
        except Exception as e:
            logger.error(f"Error saving model {model_id}: {e}")
    
    async def get_model_info(self, model_id: str) -> Optional[MLModel]:
        """Get information about a trained model"""
        return self.models.get(model_id)
    
    async def list_models(self) -> List[MLModel]:
        """List all trained models"""
        return list(self.models.values())
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete a trained model"""
        try:
            if model_id in self.models:
                del self.models[model_id]
            if model_id in self.trained_models:
                del self.trained_models[model_id]
            
            # Delete model file
            model_file = self.model_storage_path / f"{model_id}.pkl"
            if model_file.exists():
                model_file.unlink()
            
            logger.info(f"Model deleted: {model_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting model {model_id}: {e}")
            return False
    
    async def get_model_metrics(self) -> Dict[str, Any]:
        """Get ML engine metrics"""
        return {
            "total_models": len(self.models),
            "active_models": len([m for m in self.models.values() if m.is_active]),
            "model_types": list(set(m.model_type for m in self.models.values())),
            "average_accuracy": np.mean([m.accuracy for m in self.models.values()]) if self.models else 0,
            "total_training_data": sum(m.training_data_size for m in self.models.values()),
            "models_loaded": self.models_loaded,
            "sentence_transformer_loaded": self.sentence_transformer is not None,
            "device": str(self.device),
            "timestamp": datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of ML engine"""
        return {
            "status": "healthy" if self.models_loaded else "unhealthy",
            "models_loaded": self.models_loaded,
            "total_models": len(self.models),
            "sentence_transformer_loaded": self.sentence_transformer is not None,
            "device": str(self.device),
            "model_storage_path": str(self.model_storage_path),
            "timestamp": datetime.now().isoformat()
        }


# Global ML engine instance
content_ml_engine = ContentMLEngine()


async def initialize_content_ml_engine() -> None:
    """Initialize the global ML engine"""
    await content_ml_engine.initialize()


async def train_content_classifier(
    texts: List[str],
    labels: List[str],
    model_name: str = "content_classifier",
    model_type: str = "neural_network"
) -> str:
    """Train a content classification model"""
    return await content_ml_engine.train_content_classifier(texts, labels, model_name, model_type)


async def predict_content_class(text: str, model_id: str) -> MLPrediction:
    """Predict content class using trained model"""
    return await content_ml_engine.predict_content_class(text, model_id)


async def train_content_clustering(
    texts: List[str],
    n_clusters: int = 5,
    model_name: str = "content_clustering"
) -> str:
    """Train content clustering model"""
    return await content_ml_engine.train_content_clustering(texts, n_clusters, model_name)


async def predict_content_cluster(text: str, model_id: str) -> MLPrediction:
    """Predict content cluster using trained model"""
    return await content_ml_engine.predict_content_cluster(text, model_id)


async def train_topic_modeling(
    texts: List[str],
    n_topics: int = 10,
    model_name: str = "topic_modeling"
) -> str:
    """Train topic modeling using LDA"""
    return await content_ml_engine.train_topic_modeling(texts, n_topics, model_name)


async def predict_topics(text: str, model_id: str, top_n: int = 3) -> MLPrediction:
    """Predict topics for text using trained LDA model"""
    return await content_ml_engine.predict_topics(text, model_id, top_n)


async def get_ml_model_info(model_id: str) -> Optional[MLModel]:
    """Get information about a trained model"""
    return await content_ml_engine.get_model_info(model_id)


async def list_ml_models() -> List[MLModel]:
    """List all trained models"""
    return await content_ml_engine.list_models()


async def get_ml_engine_metrics() -> Dict[str, Any]:
    """Get ML engine metrics"""
    return await content_ml_engine.get_model_metrics()


async def get_ml_engine_health() -> Dict[str, Any]:
    """Get ML engine health status"""
    return await content_ml_engine.health_check()




