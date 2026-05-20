"""
Servicio de Deep Learning para análisis musical avanzado
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr, zscore
import json
import os
from datetime import datetime
from itertools import product
import base64
from io import BytesIO

# Optional experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from ..models.music_transformer import MusicClassifier, MusicModelTrainer, MusicDataset
from .spotify_service import SpotifyService

logger = logging.getLogger(__name__)


class DeepLearningService:
    """Servicio de Deep Learning para análisis musical"""
    
    def __init__(self):
        self.spotify = SpotifyService()
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Inicializar modelo (sin entrenar por defecto)
        self.model = None
        self.model_loaded = False
        self.sentiment_pipeline = None
        
        # Géneros y emociones
        self.genres = [
            "Pop", "Rock", "Electronic", "Hip-Hop", "R&B", "Jazz",
            "Classical", "Country", "Reggae", "Blues", "Folk", "Metal"
        ]
        
        self.emotions = [
            "happy", "sad", "energetic", "calm", "angry",
            "romantic", "nostalgic", "mysterious"
        ]
        
        # Historial de entrenamiento
        self.training_history = []
        self.best_model_state = None
        self.best_metrics = None
        
        # Experiment tracking
        self.wandb_run = None
        self.tensorboard_writer = None
        
        # Production monitoring
        self.prediction_history = []
        self.performance_metrics = {
            "total_predictions": 0,
            "avg_latency": 0.0,
            "error_rate": 0.0,
            "last_updated": None
        }
        
        # Model cache
        self.model_cache = {}
        self.embedding_cache = {}
        
        # Ensemble models
        self.ensemble_models = []
        
        # Calibration
        self.calibration_model = None
        
        # Active learning
        self.active_learning_pool = []
        self.labeled_samples = []
        
        # Transfer learning
        self.source_model_state = None
        
        # Meta-learning
        self.meta_learning_history = []
        
        # Few-shot learning
        self.few_shot_examples = {}
        
        # Concept analysis
        self.concept_embeddings = {}
    
    def initialize_model(
        self,
        feature_dim: int = 13,
        d_model: int = 256,
        num_layers: int = 4
    ) -> Dict[str, Any]:
        """Inicializa el modelo"""
        try:
            self.model = MusicClassifier(
                feature_dim=feature_dim,
                d_model=d_model,
                num_layers=num_layers,
                num_genres=len(self.genres),
                num_emotions=len(self.emotions)
            ).to(self.device)
            
            self.model_loaded = True
            
            return {
                "success": True,
                "model_initialized": True,
                "device": self.device,
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            return {"error": str(e)}
    
    def predict_with_model(self, track_id: str, monitor: bool = True) -> Dict[str, Any]:
        """Predice usando el modelo de deep learning"""
        import time
        start_time = time.time()
        
        try:
            if not self.model_loaded or self.model is None:
                # Inicializar modelo si no está cargado
                self.initialize_model()
            
            # Obtener características de audio
            audio_features = self.spotify.get_track_audio_features(track_id)
            if not audio_features:
                if monitor:
                    self.monitor_prediction(track_id, time.time() - start_time, success=False, error="No audio features")
                return {"error": "No hay características de audio disponibles"}
            
            # Extraer características
            features = self._extract_features(audio_features)
            
            # Convertir a tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)  # [1, 1, feature_dim]
            
            # Predicción
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(features_tensor)
            
            # Procesar resultados
            genre_probs = torch.softmax(outputs["genre_logits"], dim=1).cpu().numpy()[0]
            emotion_probs = torch.softmax(outputs["emotion_logits"], dim=1).cpu().numpy()[0]
            popularity_pred = outputs["popularity_pred"].cpu().numpy()[0]
            
            prediction_time = time.time() - start_time
            
            # Top predictions
            top_genres = [
                {
                    "genre": self.genres[i],
                    "probability": float(genre_probs[i])
                }
                for i in np.argsort(genre_probs)[::-1][:3]
            ]
            
            top_emotions = [
                {
                    "emotion": self.emotions[i],
                    "probability": float(emotion_probs[i])
                }
                for i in np.argsort(emotion_probs)[::-1][:3]
            ]
            
            result = {
                "track_id": track_id,
                "predictions": {
                    "genre": {
                        "primary": top_genres[0]["genre"],
                        "confidence": top_genres[0]["probability"],
                        "top_3": top_genres
                    },
                    "emotion": {
                        "primary": top_emotions[0]["emotion"],
                        "confidence": top_emotions[0]["probability"],
                        "top_3": top_emotions
                    },
                    "popularity": {
                        "predicted": float(popularity_pred * 100),
                        "confidence": "Medium"  # Placeholder
                    }
                },
                "model_info": {
                    "device": self.device,
                    "model_type": "Transformer"
                }
            }
            
            # Monitorear predicción
            if monitor:
                self.monitor_prediction(track_id, prediction_time, success=True)
            
            return result
        except Exception as e:
            self.logger.error(f"Error predicting with model: {e}")
            if monitor:
                self.monitor_prediction(track_id, time.time() - start_time, success=False, error=str(e))
            return {"error": str(e)}
    
    def _extract_features(self, audio_features: Dict) -> np.ndarray:
        """Extrae características de audio features"""
        feature_keys = [
            "acousticness", "danceability", "energy", "instrumentalness",
            "liveness", "loudness", "speechiness", "tempo", "valence",
            "key", "mode", "time_signature", "duration_ms"
        ]
        
        features = []
        for key in feature_keys:
            value = audio_features.get(key, 0)
            # Normalizar valores
            if key == "loudness":
                value = (value + 60) / 60  # Normalizar a [0, 1]
            elif key == "tempo":
                value = value / 200  # Normalizar a [0, 1]
            elif key == "duration_ms":
                value = value / 600000  # Normalizar a [0, 1]
            elif key in ["key", "time_signature"]:
                value = value / 12  # Normalizar
            
            features.append(float(value))
        
        return np.array(features)
    
    def batch_predict(self, track_ids: List[str]) -> Dict[str, Any]:
        """Predicción en batch"""
        try:
            if not self.model_loaded or self.model is None:
                self.initialize_model()
            
            # Obtener características de todos los tracks
            features_list = []
            valid_track_ids = []
            
            for track_id in track_ids[:50]:  # Limitar a 50
                audio_features = self.spotify.get_track_audio_features(track_id)
                if audio_features:
                    features = self._extract_features(audio_features)
                    features_list.append(features)
                    valid_track_ids.append(track_id)
            
            if not features_list:
                return {"error": "No se pudieron obtener características"}
            
            # Convertir a tensor
            features_tensor = torch.FloatTensor(features_list).unsqueeze(1)  # [batch_size, 1, feature_dim]
            
            # Predicción batch
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(features_tensor)
            
            # Procesar resultados
            genre_probs = torch.softmax(outputs["genre_logits"], dim=1).cpu().numpy()
            emotion_probs = torch.softmax(outputs["emotion_logits"], dim=1).cpu().numpy()
            popularity_preds = outputs["popularity_pred"].cpu().numpy()
            
            predictions = []
            for i, track_id in enumerate(valid_track_ids):
                top_genre_idx = np.argmax(genre_probs[i])
                top_emotion_idx = np.argmax(emotion_probs[i])
                
                predictions.append({
                    "track_id": track_id,
                    "genre": {
                        "primary": self.genres[top_genre_idx],
                        "confidence": float(genre_probs[i][top_genre_idx])
                    },
                    "emotion": {
                        "primary": self.emotions[top_emotion_idx],
                        "confidence": float(emotion_probs[i][top_emotion_idx])
                    },
                    "popularity_predicted": float(popularity_preds[i] * 100)
                })
            
            return {
                "predictions": predictions,
                "total_predicted": len(predictions),
                "batch_size": len(track_ids)
            }
        except Exception as e:
            self.logger.error(f"Error in batch prediction: {e}")
            return {"error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtiene información del modelo"""
        if not self.model_loaded or self.model is None:
            return {
                "model_loaded": False,
                "message": "Modelo no inicializado"
            }
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_loaded": True,
            "model_type": "MusicClassifier (Transformer)",
            "device": self.device,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "num_genres": len(self.genres),
            "num_emotions": len(self.emotions),
            "genres": self.genres,
            "emotions": self.emotions
        }
    
    def analyze_lyrics_with_transformer(self, lyrics: str) -> Dict[str, Any]:
        """Analiza letras usando un modelo Transformer pre-entrenado"""
        try:
            # Cargar modelo de sentimiento si no está cargado
            if not hasattr(self, 'sentiment_pipeline') or self.sentiment_pipeline is None:
                try:
                    self.sentiment_pipeline = pipeline(
                        "sentiment-analysis",
                        model="distilbert-base-uncased-finetuned-sst-2-english",
                        device=0 if self.device == "cuda" else -1
                    )
                except Exception as e:
                    self.logger.warning(f"No se pudo cargar modelo de sentimiento: {e}")
                    return {"error": "Modelo de sentimiento no disponible"}
            
            # Analizar sentimiento
            result = self.sentiment_pipeline(lyrics[:512])  # Limitar a 512 tokens
            
            # Análisis adicional
            words = lyrics.lower().split()
            word_count = len(words)
            unique_words = len(set(words))
            vocabulary_richness = unique_words / word_count if word_count > 0 else 0
            
            return {
                "sentiment": {
                    "label": result[0]["label"],
                    "score": float(result[0]["score"])
                },
                "statistics": {
                    "word_count": word_count,
                    "unique_words": unique_words,
                    "vocabulary_richness": round(vocabulary_richness, 4)
                },
                "model": "distilbert-base-uncased-finetuned-sst-2-english"
            }
        except Exception as e:
            self.logger.error(f"Error analyzing lyrics with transformer: {e}")
            return {"error": str(e)}
    
    def train_model(
        self,
        track_ids: List[str],
        genres: Optional[List[int]] = None,
        emotions: Optional[List[int]] = None,
        popularities: Optional[List[float]] = None,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-4
    ) -> Dict[str, Any]:
        """Entrena el modelo con datos de tracks"""
        try:
            if not self.model_loaded or self.model is None:
                self.initialize_model()
            
            # Obtener características de audio
            features_list = []
            valid_genres = []
            valid_emotions = []
            valid_popularities = []
            
            for i, track_id in enumerate(track_ids):
                audio_features = self.spotify.get_track_audio_features(track_id)
                if audio_features:
                    features = self._extract_features(audio_features)
                    features_list.append(features)
                    
                    if genres and i < len(genres):
                        valid_genres.append(genres[i])
                    if emotions and i < len(emotions):
                        valid_emotions.append(emotions[i])
                    if popularities and i < len(popularities):
                        valid_popularities.append(popularities[i] / 100.0)
            
            if len(features_list) < batch_size:
                return {"error": f"Se necesitan al menos {batch_size} tracks para entrenar"}
            
            # Crear dataset
            dataset = MusicDataset(
                features=features_list,
                genres=valid_genres if valid_genres else None,
                emotions=valid_emotions if valid_emotions else None,
                popularities=valid_popularities if valid_popularities else None
            )
            
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Crear trainer
            trainer = MusicModelTrainer(
                model=self.model,
                device=self.device,
                learning_rate=learning_rate
            )
            
            # Entrenar
            training_history = []
            for epoch in range(epochs):
                metrics = trainer.train_epoch(dataloader)
                training_history.append({
                    "epoch": epoch + 1,
                    **metrics
                })
                self.logger.info(f"Epoch {epoch + 1}/{epochs}: {metrics}")
            
            # Evaluar
            eval_metrics = trainer.evaluate(dataloader)
            
            return {
                "success": True,
                "epochs_trained": epochs,
                "training_samples": len(features_list),
                "final_metrics": eval_metrics,
                "training_history": training_history
            }
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            return {"error": str(e)}
    
    def save_model(self, path: str) -> Dict[str, Any]:
        """Guarda el modelo entrenado"""
        try:
            if not self.model_loaded or self.model is None:
                return {"error": "No hay modelo para guardar"}
            
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "genres": self.genres,
                "emotions": self.emotions,
                "device": self.device
            }, path)
            
            return {
                "success": True,
                "path": path,
                "message": "Modelo guardado exitosamente"
            }
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return {"error": str(e)}
    
    def load_model(self, path: str) -> Dict[str, Any]:
        """Carga un modelo pre-entrenado"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Inicializar modelo si no está inicializado
            if not self.model_loaded or self.model is None:
                self.initialize_model()
            
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            
            if "genres" in checkpoint:
                self.genres = checkpoint["genres"]
            if "emotions" in checkpoint:
                self.emotions = checkpoint["emotions"]
            
            return {
                "success": True,
                "path": path,
                "message": "Modelo cargado exitosamente"
            }
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return {"error": str(e)}
    
    def evaluate_model_advanced(
        self,
        track_ids: List[str],
        genres: Optional[List[int]] = None,
        emotions: Optional[List[int]] = None,
        popularities: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Evaluación avanzada con métricas detalladas"""
        try:
            if not self.model_loaded or self.model is None:
                self.initialize_model()
            
            # Obtener características
            features_list = []
            valid_genres = []
            valid_emotions = []
            valid_popularities = []
            
            for i, track_id in enumerate(track_ids):
                audio_features = self.spotify.get_track_audio_features(track_id)
                if audio_features:
                    features = self._extract_features(audio_features)
                    features_list.append(features)
                    
                    if genres and i < len(genres):
                        valid_genres.append(genres[i])
                    if emotions and i < len(emotions):
                        valid_emotions.append(emotions[i])
                    if popularities and i < len(popularities):
                        valid_popularities.append(popularities[i] / 100.0)
            
            if not features_list:
                return {"error": "No se pudieron obtener características"}
            
            # Crear dataset y dataloader
            dataset = MusicDataset(
                features=features_list,
                genres=valid_genres if valid_genres else None,
                emotions=valid_emotions if valid_emotions else None,
                popularities=valid_popularities if valid_popularities else None
            )
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            # Evaluar
            self.model.eval()
            all_genre_preds = []
            all_genre_labels = []
            all_emotion_preds = []
            all_emotion_labels = []
            all_popularity_preds = []
            all_popularity_labels = []
            
            with torch.no_grad():
                for batch in dataloader:
                    features = batch["features"].to(self.device)
                    outputs = self.model(features)
                    
                    if "genre" in batch:
                        genre_preds = outputs["genre_logits"].argmax(dim=1).cpu().numpy()
                        genre_labels = batch["genre"].squeeze().cpu().numpy()
                        all_genre_preds.extend(genre_preds)
                        all_genre_labels.extend(genre_labels)
                    
                    if "emotion" in batch:
                        emotion_preds = outputs["emotion_logits"].argmax(dim=1).cpu().numpy()
                        emotion_labels = batch["emotion"].squeeze().cpu().numpy()
                        all_emotion_preds.extend(emotion_preds)
                        all_emotion_labels.extend(emotion_labels)
                    
                    if "popularity" in batch:
                        popularity_preds = outputs["popularity_pred"].cpu().numpy()
                        popularity_labels = batch["popularity"].squeeze().cpu().numpy()
                        all_popularity_preds.extend(popularity_preds)
                        all_popularity_labels.extend(popularity_labels)
            
            # Calcular métricas
            metrics = {}
            
            if all_genre_labels:
                genre_accuracy = accuracy_score(all_genre_labels, all_genre_preds)
                genre_precision, genre_recall, genre_f1, _ = precision_recall_fscore_support(
                    all_genre_labels, all_genre_preds, average='weighted', zero_division=0
                )
                genre_cm = confusion_matrix(all_genre_labels, all_genre_preds).tolist()
                
                metrics["genre"] = {
                    "accuracy": float(genre_accuracy),
                    "precision": float(genre_precision),
                    "recall": float(genre_recall),
                    "f1_score": float(genre_f1),
                    "confusion_matrix": genre_cm,
                    "classification_report": classification_report(
                        all_genre_labels, all_genre_preds, 
                        target_names=self.genres, 
                        output_dict=True, 
                        zero_division=0
                    )
                }
            
            if all_emotion_labels:
                emotion_accuracy = accuracy_score(all_emotion_labels, all_emotion_preds)
                emotion_precision, emotion_recall, emotion_f1, _ = precision_recall_fscore_support(
                    all_emotion_labels, all_emotion_preds, average='weighted', zero_division=0
                )
                emotion_cm = confusion_matrix(all_emotion_labels, all_emotion_preds).tolist()
                
                metrics["emotion"] = {
                    "accuracy": float(emotion_accuracy),
                    "precision": float(emotion_precision),
                    "recall": float(emotion_recall),
                    "f1_score": float(emotion_f1),
                    "confusion_matrix": emotion_cm,
                    "classification_report": classification_report(
                        all_emotion_labels, all_emotion_preds,
                        target_names=self.emotions,
                        output_dict=True,
                        zero_division=0
                    )
                }
            
            if all_popularity_labels:
                popularity_mse = np.mean((np.array(all_popularity_labels) - np.array(all_popularity_preds)) ** 2)
                popularity_mae = np.mean(np.abs(np.array(all_popularity_labels) - np.array(all_popularity_preds)))
                popularity_rmse = np.sqrt(popularity_mse)
                
                metrics["popularity"] = {
                    "mse": float(popularity_mse),
                    "mae": float(popularity_mae),
                    "rmse": float(popularity_rmse),
                    "r2_score": float(1 - (popularity_mse / np.var(all_popularity_labels))) if np.var(all_popularity_labels) > 0 else 0.0
                }
            
            return {
                "success": True,
                "metrics": metrics,
                "samples_evaluated": len(features_list)
            }
        except Exception as e:
            self.logger.error(f"Error in advanced evaluation: {e}")
            return {"error": str(e)}
    
    def train_with_validation(
        self,
        train_track_ids: List[str],
        val_track_ids: List[str],
        train_genres: Optional[List[int]] = None,
        train_emotions: Optional[List[int]] = None,
        train_popularities: Optional[List[float]] = None,
        val_genres: Optional[List[int]] = None,
        val_emotions: Optional[List[int]] = None,
        val_popularities: Optional[List[float]] = None,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        early_stopping_patience: int = 5,
        min_delta: float = 0.001
    ) -> Dict[str, Any]:
        """Entrenamiento con validación y early stopping"""
        try:
            if not self.model_loaded or self.model is None:
                self.initialize_model()
            
            # Preparar datos de entrenamiento
            train_features, train_labels = self._prepare_training_data(
                train_track_ids, train_genres, train_emotions, train_popularities
            )
            
            # Preparar datos de validación
            val_features, val_labels = self._prepare_training_data(
                val_track_ids, val_genres, val_emotions, val_popularities
            )
            
            if len(train_features) < batch_size:
                return {"error": f"Se necesitan al menos {batch_size} tracks para entrenar"}
            
            # Crear datasets
            train_dataset = MusicDataset(
                features=train_features,
                genres=train_labels.get("genres"),
                emotions=train_labels.get("emotions"),
                popularities=train_labels.get("popularities")
            )
            val_dataset = MusicDataset(
                features=val_features,
                genres=val_labels.get("genres"),
                emotions=val_labels.get("emotions"),
                popularities=val_labels.get("popularities")
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Crear trainer
            trainer = MusicModelTrainer(
                model=self.model,
                device=self.device,
                learning_rate=learning_rate
            )
            
            # Entrenar con early stopping
            best_val_loss = float('inf')
            patience_counter = 0
            training_history = []
            
            for epoch in range(epochs):
                # Entrenar
                train_metrics = trainer.train_epoch(train_loader)
                
                # Validar
                val_metrics = trainer.evaluate(val_loader)
                
                training_history.append({
                    "epoch": epoch + 1,
                    "train": train_metrics,
                    "val": val_metrics
                })
                
                # Early stopping
                val_loss = val_metrics.get("loss", float('inf'))
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Guardar mejor modelo
                    self.best_model_state = self.model.state_dict().copy()
                    self.best_metrics = val_metrics
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch + 1}")
                        # Restaurar mejor modelo
                        if self.best_model_state:
                            self.model.load_state_dict(self.best_model_state)
                        break
                
                self.logger.info(f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_metrics.get('total_loss', 0):.4f}, Val Loss: {val_loss:.4f}")
            
            self.training_history = training_history
            
            return {
                "success": True,
                "epochs_trained": epoch + 1,
                "training_samples": len(train_features),
                "validation_samples": len(val_features),
                "best_val_loss": float(best_val_loss),
                "training_history": training_history,
                "early_stopped": patience_counter >= early_stopping_patience
            }
        except Exception as e:
            self.logger.error(f"Error training with validation: {e}")
            return {"error": str(e)}
    
    def _prepare_training_data(
        self,
        track_ids: List[str],
        genres: Optional[List[int]] = None,
        emotions: Optional[List[int]] = None,
        popularities: Optional[List[float]] = None
    ) -> Tuple[List[np.ndarray], Dict[str, List]]:
        """Prepara datos para entrenamiento"""
        features_list = []
        labels = {"genres": [], "emotions": [], "popularities": []}
        
        for i, track_id in enumerate(track_ids):
            audio_features = self.spotify.get_track_audio_features(track_id)
            if audio_features:
                features = self._extract_features(audio_features)
                features_list.append(features)
                
                if genres and i < len(genres):
                    labels["genres"].append(genres[i])
                if emotions and i < len(emotions):
                    labels["emotions"].append(emotions[i])
                if popularities and i < len(popularities):
                    labels["popularities"].append(popularities[i] / 100.0)
        
        return features_list, labels
    
    def extract_embeddings(self, track_ids: List[str]) -> Dict[str, Any]:
        """Extrae embeddings musicales del modelo"""
        try:
            if not self.model_loaded or self.model is None:
                self.initialize_model()
            
            features_list = []
            valid_track_ids = []
            
            for track_id in track_ids:
                audio_features = self.spotify.get_track_audio_features(track_id)
                if audio_features:
                    features = self._extract_features(audio_features)
                    features_list.append(features)
                    valid_track_ids.append(track_id)
            
            if not features_list:
                return {"error": "No se pudieron obtener características"}
            
            # Convertir a tensor
            features_tensor = torch.FloatTensor(features_list).unsqueeze(1)
            
            # Extraer embeddings
            self.model.eval()
            with torch.no_grad():
                # Obtener embeddings del encoder
                encoded = self.model.encoder(features_tensor.to(self.device))
                # Pooling global
                embeddings = encoded.mean(dim=1).cpu().numpy()
            
            return {
                "success": True,
                "embeddings": embeddings.tolist(),
                "track_ids": valid_track_ids,
                "embedding_dim": embeddings.shape[1],
                "num_tracks": len(valid_track_ids)
            }
        except Exception as e:
            self.logger.error(f"Error extracting embeddings: {e}")
            return {"error": str(e)}
    
    def save_training_history(self, path: str) -> Dict[str, Any]:
        """Guarda el historial de entrenamiento"""
        try:
            with open(path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            
            return {
                "success": True,
                "path": path,
                "message": "Historial de entrenamiento guardado"
            }
        except Exception as e:
            self.logger.error(f"Error saving training history: {e}")
            return {"error": str(e)}
    
    def initialize_experiment_tracking(
        self,
        experiment_name: str,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        project_name: str = "music-analyzer-ai"
    ) -> Dict[str, Any]:
        """Inicializa tracking de experimentos"""
        try:
            if use_wandb and WANDB_AVAILABLE:
                wandb.init(
                    project=project_name,
                    name=experiment_name,
                    config={}
                )
                self.wandb_run = wandb.run
                self.logger.info(f"Wandb initialized for experiment: {experiment_name}")
            
            if use_tensorboard and TENSORBOARD_AVAILABLE:
                log_dir = f"./runs/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.makedirs(log_dir, exist_ok=True)
                self.tensorboard_writer = SummaryWriter(log_dir)
                self.logger.info(f"TensorBoard initialized at: {log_dir}")
            
            return {
                "success": True,
                "experiment_name": experiment_name,
                "wandb_enabled": use_wandb and WANDB_AVAILABLE,
                "tensorboard_enabled": use_tensorboard and TENSORBOARD_AVAILABLE
            }
        except Exception as e:
            self.logger.error(f"Error initializing experiment tracking: {e}")
            return {"error": str(e)}
    
    def log_metrics(self, metrics: Dict[str, float], step: int = 0):
        """Registra métricas en wandb/tensorboard"""
        try:
            if self.wandb_run:
                wandb.log(metrics, step=step)
            
            if self.tensorboard_writer:
                for key, value in metrics.items():
                    self.tensorboard_writer.add_scalar(key, value, step)
        except Exception as e:
            self.logger.warning(f"Error logging metrics: {e}")
    
    def find_similar_tracks(
        self,
        reference_track_id: str,
        candidate_track_ids: List[str],
        top_k: int = 10,
        metric: str = "cosine"
    ) -> Dict[str, Any]:
        """Encuentra tracks similares usando embeddings"""
        try:
            # Extraer embeddings
            all_track_ids = [reference_track_id] + candidate_track_ids
            result = self.extract_embeddings(all_track_ids)
            
            if "error" in result:
                return result
            
            embeddings = np.array(result["embeddings"])
            track_ids = result["track_ids"]
            
            # Encontrar índice del track de referencia
            ref_idx = track_ids.index(reference_track_id)
            ref_embedding = embeddings[ref_idx:ref_idx+1]
            
            # Calcular similitudes
            similarities = []
            for i, track_id in enumerate(track_ids):
                if track_id == reference_track_id:
                    continue
                
                candidate_embedding = embeddings[i:i+1]
                
                if metric == "cosine":
                    similarity = cosine_similarity(ref_embedding, candidate_embedding)[0][0]
                elif metric == "euclidean":
                    similarity = 1 / (1 + euclidean(ref_embedding[0], candidate_embedding[0]))
                else:
                    similarity = cosine_similarity(ref_embedding, candidate_embedding)[0][0]
                
                similarities.append({
                    "track_id": track_id,
                    "similarity": float(similarity),
                    "rank": 0  # Se actualizará después
                })
            
            # Ordenar por similitud
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Asignar rankings
            for i, sim in enumerate(similarities[:top_k]):
                sim["rank"] = i + 1
            
            return {
                "success": True,
                "reference_track_id": reference_track_id,
                "similar_tracks": similarities[:top_k],
                "metric": metric,
                "total_candidates": len(candidate_track_ids)
            }
        except Exception as e:
            self.logger.error(f"Error finding similar tracks: {e}")
            return {"error": str(e)}
    
    def recommend_based_on_embeddings(
        self,
        seed_track_ids: List[str],
        candidate_track_ids: List[str],
        top_k: int = 20,
        diversity_weight: float = 0.3
    ) -> Dict[str, Any]:
        """Recomendaciones basadas en embeddings con diversidad"""
        try:
            # Extraer embeddings
            all_track_ids = seed_track_ids + candidate_track_ids
            result = self.extract_embeddings(all_track_ids)
            
            if "error" in result:
                return result
            
            embeddings = np.array(result["embeddings"])
            track_ids = result["track_ids"]
            
            # Calcular embedding promedio de seeds
            seed_indices = [track_ids.index(tid) for tid in seed_track_ids if tid in track_ids]
            if not seed_indices:
                return {"error": "No se encontraron seed tracks"}
            
            seed_embeddings = embeddings[seed_indices]
            avg_seed_embedding = seed_embeddings.mean(axis=0).reshape(1, -1)
            
            # Calcular similitudes con candidatos
            candidate_indices = [track_ids.index(tid) for tid in candidate_track_ids if tid in track_ids]
            recommendations = []
            
            for idx in candidate_indices:
                candidate_embedding = embeddings[idx:idx+1]
                similarity = cosine_similarity(avg_seed_embedding, candidate_embedding)[0][0]
                
                # Calcular diversidad (distancia promedio a otros candidatos ya seleccionados)
                diversity = 1.0
                if recommendations:
                    selected_embeddings = np.array([embeddings[track_ids.index(r["track_id"])] for r in recommendations])
                    avg_diversity = cosine_similarity(candidate_embedding, selected_embeddings).mean()
                    diversity = 1 - avg_diversity
                
                # Score combinado
                score = (1 - diversity_weight) * similarity + diversity_weight * diversity
                
                recommendations.append({
                    "track_id": track_ids[idx],
                    "similarity": float(similarity),
                    "diversity": float(diversity),
                    "score": float(score)
                })
            
            # Ordenar por score
            recommendations.sort(key=lambda x: x["score"], reverse=True)
            
            return {
                "success": True,
                "seed_tracks": seed_track_ids,
                "recommendations": recommendations[:top_k],
                "diversity_weight": diversity_weight,
                "total_candidates": len(candidate_track_ids)
            }
        except Exception as e:
            self.logger.error(f"Error in embedding-based recommendations: {e}")
            return {"error": str(e)}
    
    def optimize_hyperparameters(
        self,
        train_track_ids: List[str],
        val_track_ids: List[str],
        train_genres: Optional[List[int]] = None,
        train_emotions: Optional[List[int]] = None,
        train_popularities: Optional[List[float]] = None,
        val_genres: Optional[List[int]] = None,
        val_emotions: Optional[List[int]] = None,
        val_popularities: Optional[List[float]] = None,
        param_grid: Optional[Dict[str, List]] = None,
        max_trials: int = 10
    ) -> Dict[str, Any]:
        """Optimización básica de hiperparámetros"""
        try:
            if param_grid is None:
                param_grid = {
                    "learning_rate": [1e-5, 1e-4, 5e-4],
                    "batch_size": [16, 32, 64],
                    "d_model": [128, 256, 512],
                    "num_layers": [2, 4, 6]
                }
            
            # Generar combinaciones
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            combinations = list(product(*param_values))
            
            # Limitar número de trials
            if len(combinations) > max_trials:
                # Seleccionar aleatoriamente
                import random
                combinations = random.sample(combinations, max_trials)
            
            best_score = float('inf')
            best_params = None
            best_model_state = None
            results = []
            
            for i, combo in enumerate(combinations):
                params = dict(zip(param_names, combo))
                self.logger.info(f"Trial {i+1}/{len(combinations)}: {params}")
                
                # Inicializar modelo con nuevos parámetros
                d_model = params.get("d_model", 256)
                num_layers = params.get("num_layers", 4)
                self.initialize_model(d_model=d_model, num_layers=num_layers)
                
                # Entrenar
                learning_rate = params.get("learning_rate", 1e-4)
                batch_size = params.get("batch_size", 32)
                
                train_result = self.train_with_validation(
                    train_track_ids=train_track_ids,
                    val_track_ids=val_track_ids,
                    train_genres=train_genres,
                    train_emotions=train_emotions,
                    train_popularities=train_popularities,
                    val_genres=val_genres,
                    val_emotions=val_emotions,
                    val_popularities=val_popularities,
                    epochs=5,  # Menos épocas para búsqueda rápida
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    early_stopping_patience=3
                )
                
                if "error" in train_result:
                    continue
                
                val_loss = train_result.get("best_val_loss", float('inf'))
                results.append({
                    "trial": i + 1,
                    "params": params,
                    "val_loss": float(val_loss)
                })
                
                if val_loss < best_score:
                    best_score = val_loss
                    best_params = params
                    best_model_state = self.model.state_dict().copy()
            
            # Restaurar mejor modelo
            if best_model_state:
                self.initialize_model(
                    d_model=best_params.get("d_model", 256),
                    num_layers=best_params.get("num_layers", 4)
                )
                self.model.load_state_dict(best_model_state)
            
            return {
                "success": True,
                "best_params": best_params,
                "best_score": float(best_score),
                "total_trials": len(results),
                "all_results": results
            }
        except Exception as e:
            self.logger.error(f"Error optimizing hyperparameters: {e}")
            return {"error": str(e)}
    
    def close_experiment_tracking(self):
        """Cierra tracking de experimentos"""
        try:
            if self.wandb_run:
                wandb.finish()
                self.wandb_run = None
            
            if self.tensorboard_writer:
                self.tensorboard_writer.close()
                self.tensorboard_writer = None
        except Exception as e:
            self.logger.warning(f"Error closing experiment tracking: {e}")
    
    def cluster_tracks(
        self,
        track_ids: List[str],
        n_clusters: int = 5,
        method: str = "kmeans",
        use_pca: bool = False,
        pca_components: int = 2
    ) -> Dict[str, Any]:
        """Agrupa tracks usando embeddings con clustering"""
        try:
            # Extraer embeddings
            result = self.extract_embeddings(track_ids)
            
            if "error" in result:
                return result
            
            embeddings = np.array(result["embeddings"])
            track_ids = result["track_ids"]
            
            # Reducción de dimensionalidad opcional
            if use_pca and embeddings.shape[1] > pca_components:
                pca = PCA(n_components=pca_components)
                embeddings_reduced = pca.fit_transform(embeddings)
                explained_variance = pca.explained_variance_ratio_.tolist()
            else:
                embeddings_reduced = embeddings
                explained_variance = None
            
            # Clustering
            if method == "kmeans":
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = clusterer.fit_predict(embeddings_reduced)
                cluster_centers = clusterer.cluster_centers_.tolist()
            elif method == "dbscan":
                clusterer = DBSCAN(eps=0.5, min_samples=3)
                cluster_labels = clusterer.fit_predict(embeddings_reduced)
                cluster_centers = None
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            else:
                return {"error": f"Método de clustering '{method}' no soportado"}
            
            # Organizar resultados por cluster
            clusters = {}
            for i, (track_id, label) in enumerate(zip(track_ids, cluster_labels)):
                if label not in clusters:
                    clusters[label] = {
                        "cluster_id": int(label),
                        "tracks": [],
                        "size": 0
                    }
                clusters[label]["tracks"].append({
                    "track_id": track_id,
                    "embedding_index": i
                })
                clusters[label]["size"] += 1
            
            # Calcular estadísticas por cluster
            for cluster_id, cluster_data in clusters.items():
                if cluster_id == -1:  # Noise points en DBSCAN
                    continue
                cluster_indices = [t["embedding_index"] for t in cluster_data["tracks"]]
                cluster_embeddings = embeddings[cluster_indices]
                cluster_data["avg_embedding"] = cluster_embeddings.mean(axis=0).tolist()
                cluster_data["std_embedding"] = cluster_embeddings.std(axis=0).tolist()
            
            return {
                "success": True,
                "method": method,
                "n_clusters": n_clusters,
                "clusters": list(clusters.values()),
                "cluster_labels": cluster_labels.tolist(),
                "cluster_centers": cluster_centers,
                "explained_variance": explained_variance,
                "total_tracks": len(track_ids),
                "noise_points": int(np.sum(cluster_labels == -1)) if method == "dbscan" else 0
            }
        except Exception as e:
            self.logger.error(f"Error clustering tracks: {e}")
            return {"error": str(e)}
    
    def analyze_feature_importance(
        self,
        track_ids: List[str],
        target_values: List[float],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analiza importancia de características usando correlación"""
        try:
            if len(track_ids) != len(target_values):
                return {"error": "track_ids y target_values deben tener la misma longitud"}
            
            # Extraer embeddings y características
            features_list = []
            valid_track_ids = []
            valid_targets = []
            
            for i, track_id in enumerate(track_ids):
                audio_features = self.spotify.get_track_audio_features(track_id)
                if audio_features:
                    features = self._extract_features(audio_features)
                    features_list.append(features)
                    valid_track_ids.append(track_id)
                    valid_targets.append(target_values[i])
            
            if not features_list:
                return {"error": "No se pudieron obtener características"}
            
            features_array = np.array(features_list)
            
            # Nombres de características por defecto
            if feature_names is None:
                feature_names = [
                    "acousticness", "danceability", "energy", "instrumentalness",
                    "liveness", "loudness", "speechiness", "tempo", "valence",
                    "key", "mode", "time_signature", "duration_ms"
                ]
            
            # Calcular correlaciones
            correlations = []
            for i, feature_name in enumerate(feature_names):
                if i < features_array.shape[1]:
                    feature_values = features_array[:, i]
                    correlation, p_value = pearsonr(feature_values, valid_targets)
                    correlations.append({
                        "feature": feature_name,
                        "correlation": float(correlation),
                        "p_value": float(p_value),
                        "abs_correlation": float(abs(correlation))
                    })
            
            # Ordenar por importancia (correlación absoluta)
            correlations.sort(key=lambda x: x["abs_correlation"], reverse=True)
            
            return {
                "success": True,
                "total_features": len(correlations),
                "correlations": correlations,
                "top_features": [c["feature"] for c in correlations[:5]],
                "samples_analyzed": len(valid_track_ids)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing feature importance: {e}")
            return {"error": str(e)}
    
    def compare_models(
        self,
        model_paths: List[str],
        test_track_ids: List[str],
        test_genres: Optional[List[int]] = None,
        test_emotions: Optional[List[int]] = None,
        test_popularities: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Compara múltiples modelos en el mismo conjunto de prueba"""
        try:
            if len(model_paths) < 2:
                return {"error": "Se necesitan al menos 2 modelos para comparar"}
            
            results = []
            original_model_state = self.model.state_dict() if self.model else None
            
            for i, model_path in enumerate(model_paths):
                # Cargar modelo
                load_result = self.load_model(model_path)
                if "error" in load_result:
                    results.append({
                        "model_path": model_path,
                        "error": load_result["error"]
                    })
                    continue
                
                # Evaluar modelo
                eval_result = self.evaluate_model_advanced(
                    track_ids=test_track_ids,
                    genres=test_genres,
                    emotions=test_emotions,
                    popularities=test_popularities
                )
                
                if "error" in eval_result:
                    results.append({
                        "model_path": model_path,
                        "error": eval_result["error"]
                    })
                    continue
                
                # Extraer métricas clave
                metrics = eval_result.get("metrics", {})
                model_metrics = {
                    "model_path": model_path,
                    "model_index": i + 1
                }
                
                if "genre" in metrics:
                    model_metrics["genre_accuracy"] = metrics["genre"].get("accuracy", 0)
                    model_metrics["genre_f1"] = metrics["genre"].get("f1_score", 0)
                
                if "emotion" in metrics:
                    model_metrics["emotion_accuracy"] = metrics["emotion"].get("accuracy", 0)
                    model_metrics["emotion_f1"] = metrics["emotion"].get("f1_score", 0)
                
                if "popularity" in metrics:
                    model_metrics["popularity_rmse"] = metrics["popularity"].get("rmse", 0)
                    model_metrics["popularity_r2"] = metrics["popularity"].get("r2_score", 0)
                
                results.append(model_metrics)
            
            # Restaurar modelo original si existe
            if original_model_state and self.model:
                self.model.load_state_dict(original_model_state)
            
            # Determinar mejor modelo
            best_model = None
            if results and "error" not in results[0]:
                # Calcular score promedio
                for result in results:
                    if "error" in result:
                        continue
                    scores = []
                    if "genre_accuracy" in result:
                        scores.append(result["genre_accuracy"])
                    if "emotion_accuracy" in result:
                        scores.append(result["emotion_accuracy"])
                    if "popularity_r2" in result:
                        scores.append(result["popularity_r2"])
                    result["average_score"] = float(np.mean(scores)) if scores else 0
                
                best_model = max(
                    [r for r in results if "error" not in r],
                    key=lambda x: x.get("average_score", 0)
                )
            
            return {
                "success": True,
                "models_compared": len(model_paths),
                "results": results,
                "best_model": best_model,
                "test_samples": len(test_track_ids)
            }
        except Exception as e:
            self.logger.error(f"Error comparing models: {e}")
            return {"error": str(e)}
    
    def export_training_results(
        self,
        output_path: str,
        include_embeddings: bool = False,
        track_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Exporta resultados de entrenamiento en formato JSON"""
        try:
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "model_info": self.get_model_info(),
                "training_history": self.training_history,
                "best_metrics": self.best_metrics
            }
            
            if include_embeddings and track_ids:
                embeddings_result = self.extract_embeddings(track_ids)
                if "error" not in embeddings_result:
                    export_data["embeddings"] = {
                        "track_ids": embeddings_result["track_ids"],
                        "embeddings": embeddings_result["embeddings"],
                        "embedding_dim": embeddings_result["embedding_dim"]
                    }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return {
                "success": True,
                "output_path": output_path,
                "file_size": os.path.getsize(output_path),
                "includes_embeddings": include_embeddings
            }
        except Exception as e:
            self.logger.error(f"Error exporting training results: {e}")
            return {"error": str(e)}
    
    def analyze_embedding_trends(
        self,
        track_ids_by_time: Dict[str, List[str]],
        time_labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analiza tendencias temporales en embeddings"""
        try:
            all_track_ids = []
            for tracks in track_ids_by_time.values():
                all_track_ids.extend(tracks)
            
            # Extraer embeddings
            result = self.extract_embeddings(all_track_ids)
            if "error" in result:
                return result
            
            embeddings = np.array(result["embeddings"])
            track_ids = result["track_ids"]
            
            # Organizar por período de tiempo
            trends = {}
            for period, period_tracks in track_ids_by_time.items():
                period_indices = [track_ids.index(tid) for tid in period_tracks if tid in track_ids]
                if period_indices:
                    period_embeddings = embeddings[period_indices]
                    trends[period] = {
                        "avg_embedding": period_embeddings.mean(axis=0).tolist(),
                        "std_embedding": period_embeddings.std(axis=0).tolist(),
                        "num_tracks": len(period_indices),
                        "embedding_norm": float(np.linalg.norm(period_embeddings.mean(axis=0)))
                    }
            
            # Calcular cambios entre períodos
            periods = list(trends.keys())
            changes = []
            for i in range(len(periods) - 1):
                period1 = periods[i]
                period2 = periods[i + 1]
                
                emb1 = np.array(trends[period1]["avg_embedding"])
                emb2 = np.array(trends[period2]["avg_embedding"])
                
                cosine_change = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
                euclidean_change = euclidean(emb1, emb2)
                
                changes.append({
                    "from_period": period1,
                    "to_period": period2,
                    "cosine_similarity": float(cosine_change),
                    "euclidean_distance": float(euclidean_change),
                    "change_magnitude": float(euclidean_change)
                })
            
            return {
                "success": True,
                "trends": trends,
                "changes": changes,
                "total_periods": len(periods),
                "total_tracks": len(all_track_ids)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing embedding trends: {e}")
            return {"error": str(e)}
    
    def analyze_bias_fairness(
        self,
        track_ids: List[str],
        genres: List[int],
        predictions: Optional[Dict[str, List]] = None
    ) -> Dict[str, Any]:
        """Analiza bias y fairness en predicciones del modelo"""
        try:
            if not predictions:
                # Obtener predicciones del modelo
                if not self.model_loaded or self.model is None:
                    self.initialize_model()
                
                features_list = []
                valid_track_ids = []
                valid_genres = []
                
                for i, track_id in enumerate(track_ids):
                    audio_features = self.spotify.get_track_audio_features(track_id)
                    if audio_features and i < len(genres):
                        features = self._extract_features(audio_features)
                        features_list.append(features)
                        valid_track_ids.append(track_id)
                        valid_genres.append(genres[i])
                
                if not features_list:
                    return {"error": "No se pudieron obtener características"}
                
                # Predecir
                features_tensor = torch.FloatTensor(features_list).unsqueeze(1)
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(features_tensor.to(self.device))
                
                genre_probs = torch.softmax(outputs["genre_logits"], dim=1).cpu().numpy()
                genre_preds = genre_probs.argmax(axis=1)
            else:
                genre_preds = np.array(predictions.get("genres", []))
                valid_genres = genres
                valid_track_ids = track_ids
            
            # Calcular métricas por género
            genre_metrics = {}
            for genre_idx in range(len(self.genres)):
                genre_mask = np.array(valid_genres) == genre_idx
                pred_mask = genre_preds == genre_idx
                
                if np.sum(genre_mask) > 0:
                    precision = np.sum(genre_mask & pred_mask) / (np.sum(pred_mask) + 1e-8)
                    recall = np.sum(genre_mask & pred_mask) / (np.sum(genre_mask) + 1e-8)
                    
                    genre_metrics[self.genres[genre_idx]] = {
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1_score": float(2 * precision * recall / (precision + recall + 1e-8)),
                        "support": int(np.sum(genre_mask)),
                        "predicted_count": int(np.sum(pred_mask))
                    }
            
            # Calcular fairness (igualdad de precisión entre géneros)
            precisions = [m["precision"] for m in genre_metrics.values()]
            recalls = [m["recall"] for m in genre_metrics.values()]
            
            precision_std = float(np.std(precisions)) if precisions else 0.0
            recall_std = float(np.std(recalls)) if recalls else 0.0
            
            return {
                "success": True,
                "genre_metrics": genre_metrics,
                "fairness": {
                    "precision_std": precision_std,
                    "recall_std": recall_std,
                    "precision_range": float(max(precisions) - min(precisions)) if precisions else 0.0,
                    "recall_range": float(max(recalls) - min(recalls)) if recalls else 0.0
                },
                "total_samples": len(valid_track_ids)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing bias and fairness: {e}")
            return {"error": str(e)}
    
    def generate_training_report(
        self,
        include_visualizations: bool = False
    ) -> Dict[str, Any]:
        """Genera un reporte completo del entrenamiento"""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "model_info": self.get_model_info(),
                "training_summary": {
                    "total_epochs": len(self.training_history),
                    "best_metrics": self.best_metrics
                }
            }
            
            if self.training_history:
                # Estadísticas de entrenamiento
                train_losses = [h["train"].get("total_loss", 0) for h in self.training_history if "train" in h]
                val_losses = [h["val"].get("loss", 0) for h in self.training_history if "val" in h]
                
                report["training_statistics"] = {
                    "final_train_loss": float(train_losses[-1]) if train_losses else None,
                    "final_val_loss": float(val_losses[-1]) if val_losses else None,
                    "best_train_loss": float(min(train_losses)) if train_losses else None,
                    "best_val_loss": float(min(val_losses)) if val_losses else None,
                    "loss_improvement": float(train_losses[0] - train_losses[-1]) if len(train_losses) > 1 else None
                }
                
                # Métricas por época
                report["epoch_metrics"] = self.training_history
            
            # Recomendaciones
            recommendations = []
            if self.training_history:
                final_val_loss = val_losses[-1] if val_losses else None
                if final_val_loss and final_val_loss > 0.5:
                    recommendations.append("Considerar más épocas de entrenamiento")
                if train_losses and val_losses and abs(train_losses[-1] - val_losses[-1]) > 0.3:
                    recommendations.append("Posible overfitting detectado, considerar regularización")
            
            report["recommendations"] = recommendations
            
            return {
                "success": True,
                "report": report
            }
        except Exception as e:
            self.logger.error(f"Error generating training report: {e}")
            return {"error": str(e)}
    
    def fine_tune_model(
        self,
        base_model_path: str,
        fine_tune_track_ids: List[str],
        fine_tune_genres: Optional[List[int]] = None,
        fine_tune_emotions: Optional[List[int]] = None,
        fine_tune_popularities: Optional[List[float]] = None,
        epochs: int = 5,
        learning_rate: float = 1e-5,
        freeze_encoder: bool = False
    ) -> Dict[str, Any]:
        """Fine-tuning de un modelo pre-entrenado"""
        try:
            # Cargar modelo base
            load_result = self.load_model(base_model_path)
            if "error" in load_result:
                return load_result
            
            # Congelar encoder si se solicita
            if freeze_encoder and self.model:
                for param in self.model.encoder.parameters():
                    param.requires_grad = False
            
            # Preparar datos
            features_list = []
            labels = {"genres": [], "emotions": [], "popularities": []}
            
            for i, track_id in enumerate(fine_tune_track_ids):
                audio_features = self.spotify.get_track_audio_features(track_id)
                if audio_features:
                    features = self._extract_features(audio_features)
                    features_list.append(features)
                    
                    if fine_tune_genres and i < len(fine_tune_genres):
                        labels["genres"].append(fine_tune_genres[i])
                    if fine_tune_emotions and i < len(fine_tune_emotions):
                        labels["emotions"].append(fine_tune_emotions[i])
                    if fine_tune_popularities and i < len(fine_tune_popularities):
                        labels["popularities"].append(fine_tune_popularities[i] / 100.0)
            
            if not features_list:
                return {"error": "No se pudieron obtener características"}
            
            # Crear dataset
            dataset = MusicDataset(
                features=features_list,
                genres=labels["genres"] if labels["genres"] else None,
                emotions=labels["emotions"] if labels["emotions"] else None,
                popularities=labels["popularities"] if labels["popularities"] else None
            )
            
            dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
            
            # Crear trainer con learning rate más bajo
            trainer = MusicModelTrainer(
                model=self.model,
                device=self.device,
                learning_rate=learning_rate
            )
            
            # Fine-tuning
            fine_tune_history = []
            for epoch in range(epochs):
                metrics = trainer.train_epoch(dataloader)
                fine_tune_history.append({
                    "epoch": epoch + 1,
                    **metrics
                })
            
            return {
                "success": True,
                "base_model": base_model_path,
                "epochs_fine_tuned": epochs,
                "learning_rate": learning_rate,
                "freeze_encoder": freeze_encoder,
                "fine_tune_history": fine_tune_history,
                "samples_used": len(features_list)
            }
        except Exception as e:
            self.logger.error(f"Error fine-tuning model: {e}")
            return {"error": str(e)}
    
    def explain_prediction(
        self,
        track_id: str,
        method: str = "gradient"
    ) -> Dict[str, Any]:
        """Explica una predicción usando interpretabilidad"""
        try:
            if not self.model_loaded or self.model is None:
                self.initialize_model()
            
            # Obtener características
            audio_features = self.spotify.get_track_audio_features(track_id)
            if not audio_features:
                return {"error": "No se pudieron obtener características"}
            
            features = self._extract_features(audio_features)
            feature_names = [
                "acousticness", "danceability", "energy", "instrumentalness",
                "liveness", "loudness", "speechiness", "tempo", "valence",
                "key", "mode", "time_signature", "duration_ms"
            ]
            
            features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(self.device)
            features_tensor.requires_grad = True
            
            # Forward pass
            self.model.eval()
            outputs = self.model(features_tensor)
            
            # Calcular gradientes para interpretabilidad
            if method == "gradient":
                # Gradient-based importance
                genre_logits = outputs["genre_logits"]
                emotion_logits = outputs["emotion_logits"]
                popularity_pred = outputs["popularity_pred"]
                
                # Calcular importancia para género
                genre_probs = torch.softmax(genre_logits, dim=1)
                top_genre_idx = genre_probs.argmax(dim=1)
                
                genre_logits[0, top_genre_idx].backward(retain_graph=True)
                genre_gradients = features_tensor.grad.abs().squeeze().cpu().numpy()
                
                # Reset gradients
                features_tensor.grad.zero_()
                
                # Calcular importancia para emoción
                emotion_probs = torch.softmax(emotion_logits, dim=1)
                top_emotion_idx = emotion_probs.argmax(dim=1)
                
                emotion_logits[0, top_emotion_idx].backward(retain_graph=True)
                emotion_gradients = features_tensor.grad.abs().squeeze().cpu().numpy()
                
                # Reset gradients
                features_tensor.grad.zero_()
                
                # Calcular importancia para popularidad
                popularity_pred.backward()
                popularity_gradients = features_tensor.grad.abs().squeeze().cpu().numpy()
                
                # Normalizar
                genre_importance = (genre_gradients / (genre_gradients.sum() + 1e-8)).tolist()
                emotion_importance = (emotion_gradients / (emotion_gradients.sum() + 1e-8)).tolist()
                popularity_importance = (popularity_gradients / (popularity_gradients.sum() + 1e-8)).tolist()
                
                return {
                    "success": True,
                    "track_id": track_id,
                    "method": method,
                    "feature_importance": {
                        "genre": [
                            {"feature": name, "importance": float(imp)}
                            for name, imp in zip(feature_names, genre_importance)
                        ],
                        "emotion": [
                            {"feature": name, "importance": float(imp)}
                            for name, imp in zip(feature_names, emotion_importance)
                        ],
                        "popularity": [
                            {"feature": name, "importance": float(imp)}
                            for name, imp in zip(feature_names, popularity_importance)
                        ]
                    },
                    "top_features": {
                        "genre": sorted(
                            [{"feature": name, "importance": float(imp)} 
                             for name, imp in zip(feature_names, genre_importance)],
                            key=lambda x: x["importance"],
                            reverse=True
                        )[:5],
                        "emotion": sorted(
                            [{"feature": name, "importance": float(imp)} 
                             for name, imp in zip(feature_names, emotion_importance)],
                            key=lambda x: x["importance"],
                            reverse=True
                        )[:5],
                        "popularity": sorted(
                            [{"feature": name, "importance": float(imp)} 
                             for name, imp in zip(feature_names, popularity_importance)],
                            key=lambda x: x["importance"],
                            reverse=True
                        )[:5]
                    }
                }
            else:
                return {"error": f"Método '{method}' no soportado"}
        except Exception as e:
            self.logger.error(f"Error explaining prediction: {e}")
            return {"error": str(e)}
    
    def ab_test_models(
        self,
        model_a_path: str,
        model_b_path: str,
        test_track_ids: List[str],
        test_genres: Optional[List[int]] = None,
        test_emotions: Optional[List[int]] = None,
        test_popularities: Optional[List[float]] = None,
        metric: str = "accuracy"
    ) -> Dict[str, Any]:
        """A/B testing entre dos modelos"""
        try:
            original_model_state = self.model.state_dict() if self.model else None
            
            # Evaluar modelo A
            load_result_a = self.load_model(model_a_path)
            if "error" in load_result_a:
                return load_result_a
            
            eval_result_a = self.evaluate_model_advanced(
                track_ids=test_track_ids,
                genres=test_genres,
                emotions=test_emotions,
                popularities=test_popularities
            )
            
            if "error" in eval_result_a:
                return eval_result_a
            
            metrics_a = eval_result_a.get("metrics", {})
            
            # Evaluar modelo B
            load_result_b = self.load_model(model_b_path)
            if "error" in load_result_b:
                return load_result_b
            
            eval_result_b = self.evaluate_model_advanced(
                track_ids=test_track_ids,
                genres=test_genres,
                emotions=test_emotions,
                popularities=test_popularities
            )
            
            if "error" in eval_result_b:
                return eval_result_b
            
            metrics_b = eval_result_b.get("metrics", {})
            
            # Restaurar modelo original
            if original_model_state and self.model:
                self.model.load_state_dict(original_model_state)
            
            # Comparar métricas
            comparison = {}
            winner = None
            
            if "genre" in metrics_a and "genre" in metrics_b:
                genre_acc_a = metrics_a["genre"].get("accuracy", 0)
                genre_acc_b = metrics_b["genre"].get("accuracy", 0)
                comparison["genre_accuracy"] = {
                    "model_a": float(genre_acc_a),
                    "model_b": float(genre_acc_b),
                    "difference": float(genre_acc_a - genre_acc_b),
                    "winner": "A" if genre_acc_a > genre_acc_b else "B"
                }
            
            if "emotion" in metrics_a and "emotion" in metrics_b:
                emotion_acc_a = metrics_a["emotion"].get("accuracy", 0)
                emotion_acc_b = metrics_b["emotion"].get("accuracy", 0)
                comparison["emotion_accuracy"] = {
                    "model_a": float(emotion_acc_a),
                    "model_b": float(emotion_acc_b),
                    "difference": float(emotion_acc_a - emotion_acc_b),
                    "winner": "A" if emotion_acc_a > emotion_acc_b else "B"
                }
            
            if "popularity" in metrics_a and "popularity" in metrics_b:
                pop_rmse_a = metrics_a["popularity"].get("rmse", float('inf'))
                pop_rmse_b = metrics_b["popularity"].get("rmse", float('inf'))
                comparison["popularity_rmse"] = {
                    "model_a": float(pop_rmse_a),
                    "model_b": float(pop_rmse_b),
                    "difference": float(pop_rmse_a - pop_rmse_b),
                    "winner": "A" if pop_rmse_a < pop_rmse_b else "B"
                }
            
            # Determinar ganador general
            wins_a = sum(1 for comp in comparison.values() if comp["winner"] == "A")
            wins_b = sum(1 for comp in comparison.values() if comp["winner"] == "B")
            winner = "A" if wins_a > wins_b else "B" if wins_b > wins_a else "Tie"
            
            return {
                "success": True,
                "model_a": model_a_path,
                "model_b": model_b_path,
                "comparison": comparison,
                "winner": winner,
                "test_samples": len(test_track_ids)
            }
        except Exception as e:
            self.logger.error(f"Error in A/B testing: {e}")
            return {"error": str(e)}
    
    def analyze_robustness(
        self,
        track_ids: List[str],
        noise_level: float = 0.1,
        num_perturbations: int = 10
    ) -> Dict[str, Any]:
        """Analiza robustez del modelo ante perturbaciones"""
        try:
            if not self.model_loaded or self.model is None:
                self.initialize_model()
            
            original_predictions = []
            perturbed_predictions = []
            
            for track_id in track_ids:
                # Obtener características originales
                audio_features = self.spotify.get_track_audio_features(track_id)
                if not audio_features:
                    continue
                
                features = self._extract_features(audio_features)
                features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(self.device)
                
                # Predicción original
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(features_tensor)
                    genre_probs = torch.softmax(outputs["genre_logits"], dim=1).cpu().numpy()[0]
                    emotion_probs = torch.softmax(outputs["emotion_logits"], dim=1).cpu().numpy()[0]
                    popularity_pred = outputs["popularity_pred"].cpu().numpy()[0]
                
                original_predictions.append({
                    "track_id": track_id,
                    "genre": int(np.argmax(genre_probs)),
                    "emotion": int(np.argmax(emotion_probs)),
                    "popularity": float(popularity_pred)
                })
                
                # Predicciones con perturbaciones
                track_perturbed = []
                for _ in range(num_perturbations):
                    # Agregar ruido
                    noise = np.random.normal(0, noise_level, features.shape)
                    perturbed_features = features + noise
                    perturbed_features = np.clip(perturbed_features, 0, 1)  # Mantener en rango válido
                    
                    perturbed_tensor = torch.FloatTensor(perturbed_features).unsqueeze(0).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(perturbed_tensor)
                        genre_probs = torch.softmax(outputs["genre_logits"], dim=1).cpu().numpy()[0]
                        emotion_probs = torch.softmax(outputs["emotion_logits"], dim=1).cpu().numpy()[0]
                        popularity_pred = outputs["popularity_pred"].cpu().numpy()[0]
                    
                    track_perturbed.append({
                        "genre": int(np.argmax(genre_probs)),
                        "emotion": int(np.argmax(emotion_probs)),
                        "popularity": float(popularity_pred)
                    })
                
                perturbed_predictions.append({
                    "track_id": track_id,
                    "perturbations": track_perturbed
                })
            
            # Calcular estabilidad
            stability_scores = []
            for orig, pert in zip(original_predictions, perturbed_predictions):
                genre_matches = sum(1 for p in pert["perturbations"] if p["genre"] == orig["genre"])
                emotion_matches = sum(1 for p in pert["perturbations"] if p["emotion"] == orig["emotion"])
                
                genre_stability = genre_matches / num_perturbations
                emotion_stability = emotion_matches / num_perturbations
                
                # Estabilidad de popularidad (diferencia promedio)
                pop_diffs = [abs(p["popularity"] - orig["popularity"]) for p in pert["perturbations"]]
                pop_stability = 1 - (np.mean(pop_diffs) if pop_diffs else 0)
                
                stability_scores.append({
                    "track_id": orig["track_id"],
                    "genre_stability": float(genre_stability),
                    "emotion_stability": float(emotion_stability),
                    "popularity_stability": float(pop_stability),
                    "overall_stability": float((genre_stability + emotion_stability + pop_stability) / 3)
                })
            
            avg_stability = np.mean([s["overall_stability"] for s in stability_scores])
            
            return {
                "success": True,
                "noise_level": noise_level,
                "num_perturbations": num_perturbations,
                "stability_scores": stability_scores,
                "average_stability": float(avg_stability),
                "total_tracks": len(original_predictions)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing robustness: {e}")
            return {"error": str(e)}
    
    def version_model(
        self,
        version_name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Versiona el modelo actual"""
        try:
            if not self.model_loaded or self.model is None:
                return {"error": "No hay modelo para versionar"}
            
            # Crear directorio de versiones
            versions_dir = "./model_versions"
            os.makedirs(versions_dir, exist_ok=True)
            
            # Guardar modelo versionado
            version_path = os.path.join(versions_dir, f"model_{version_name}.pt")
            
            version_data = {
                "version": version_name,
                "timestamp": datetime.now().isoformat(),
                "description": description,
                "metadata": metadata or {},
                "model_info": self.get_model_info(),
                "best_metrics": self.best_metrics
            }
            
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "version_data": version_data,
                "genres": self.genres,
                "emotions": self.emotions
            }, version_path)
            
            # Guardar metadata en JSON
            metadata_path = os.path.join(versions_dir, f"metadata_{version_name}.json")
            with open(metadata_path, 'w') as f:
                json.dump(version_data, f, indent=2)
            
            return {
                "success": True,
                "version": version_name,
                "model_path": version_path,
                "metadata_path": metadata_path,
                "version_data": version_data
            }
        except Exception as e:
            self.logger.error(f"Error versioning model: {e}")
            return {"error": str(e)}
    
    def list_model_versions(self) -> Dict[str, Any]:
        """Lista todas las versiones del modelo"""
        try:
            versions_dir = "./model_versions"
            if not os.path.exists(versions_dir):
                return {
                    "success": True,
                    "versions": [],
                    "total": 0
                }
            
            versions = []
            for filename in os.listdir(versions_dir):
                if filename.startswith("metadata_") and filename.endswith(".json"):
                    version_name = filename.replace("metadata_", "").replace(".json", "")
                    metadata_path = os.path.join(versions_dir, filename)
                    
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        versions.append({
                            "version": version_name,
                            "timestamp": metadata.get("timestamp"),
                            "description": metadata.get("description"),
                            "metadata": metadata.get("metadata", {})
                        })
                    except Exception as e:
                        self.logger.warning(f"Error reading metadata for {version_name}: {e}")
            
            # Ordenar por timestamp
            versions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            return {
                "success": True,
                "versions": versions,
                "total": len(versions)
            }
        except Exception as e:
            self.logger.error(f"Error listing model versions: {e}")
            return {"error": str(e)}
    
    def monitor_prediction(
        self,
        track_id: str,
        prediction_time: float,
        success: bool = True,
        error: Optional[str] = None
    ):
        """Registra una predicción para monitoreo"""
        try:
            self.prediction_history.append({
                "track_id": track_id,
                "timestamp": datetime.now().isoformat(),
                "prediction_time": prediction_time,
                "success": success,
                "error": error
            })
            
            # Mantener solo últimas 1000 predicciones
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
            
            # Actualizar métricas
            self.performance_metrics["total_predictions"] += 1
            recent_predictions = self.prediction_history[-100:] if len(self.prediction_history) >= 100 else self.prediction_history
            
            if recent_predictions:
                avg_latency = np.mean([p["prediction_time"] for p in recent_predictions])
                error_count = sum(1 for p in recent_predictions if not p["success"])
                error_rate = error_count / len(recent_predictions)
                
                self.performance_metrics["avg_latency"] = float(avg_latency)
                self.performance_metrics["error_rate"] = float(error_rate)
                self.performance_metrics["last_updated"] = datetime.now().isoformat()
        except Exception as e:
            self.logger.warning(f"Error monitoring prediction: {e}")
    
    def get_production_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de producción"""
        try:
            recent_predictions = self.prediction_history[-100:] if len(self.prediction_history) >= 100 else self.prediction_history
            
            if not recent_predictions:
                return {
                    "success": True,
                    "metrics": self.performance_metrics,
                    "message": "No hay predicciones recientes"
                }
            
            latencies = [p["prediction_time"] for p in recent_predictions]
            errors = [p for p in recent_predictions if not p["success"]]
            
            return {
                "success": True,
                "metrics": {
                    **self.performance_metrics,
                    "recent_predictions": len(recent_predictions),
                    "min_latency": float(np.min(latencies)) if latencies else 0.0,
                    "max_latency": float(np.max(latencies)) if latencies else 0.0,
                    "p95_latency": float(np.percentile(latencies, 95)) if latencies else 0.0,
                    "p99_latency": float(np.percentile(latencies, 99)) if latencies else 0.0,
                    "error_count": len(errors),
                    "throughput": len(recent_predictions) / 60.0 if recent_predictions else 0.0  # predictions per minute
                },
                "recent_errors": [
                    {
                        "track_id": e["track_id"],
                        "timestamp": e["timestamp"],
                        "error": e.get("error", "Unknown")
                    }
                    for e in errors[-10:]  # Últimos 10 errores
                ]
            }
        except Exception as e:
            self.logger.error(f"Error getting production metrics: {e}")
            return {"error": str(e)}
    
    def detect_data_drift(
        self,
        reference_track_ids: List[str],
        current_track_ids: List[str],
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """Detecta drift en los datos"""
        try:
            # Extraer embeddings de referencia
            ref_result = self.extract_embeddings(reference_track_ids)
            if "error" in ref_result:
                return ref_result
            
            ref_embeddings = np.array(ref_result["embeddings"])
            
            # Extraer embeddings actuales
            curr_result = self.extract_embeddings(current_track_ids)
            if "error" in curr_result:
                return curr_result
            
            curr_embeddings = np.array(curr_result["embeddings"])
            
            # Calcular estadísticas
            ref_mean = ref_embeddings.mean(axis=0)
            ref_std = ref_embeddings.std(axis=0)
            curr_mean = curr_embeddings.mean(axis=0)
            curr_std = curr_embeddings.std(axis=0)
            
            # Calcular drift (distancia entre medias normalizada)
            mean_diff = np.abs(curr_mean - ref_mean)
            normalized_diff = mean_diff / (ref_std + 1e-8)
            drift_score = float(normalized_diff.mean())
            
            # Calcular cambio en distribución (KL divergence aproximado)
            std_ratio = curr_std / (ref_std + 1e-8)
            distribution_change = float(np.abs(np.log(std_ratio + 1e-8)).mean())
            
            drift_detected = drift_score > threshold or distribution_change > threshold
            
            return {
                "success": True,
                "drift_detected": drift_detected,
                "drift_score": drift_score,
                "distribution_change": distribution_change,
                "threshold": threshold,
                "reference_samples": len(reference_track_ids),
                "current_samples": len(current_track_ids),
                "recommendation": "Retrain model" if drift_detected else "No action needed"
            }
        except Exception as e:
            self.logger.error(f"Error detecting data drift: {e}")
            return {"error": str(e)}
    
    def check_model_degradation(
        self,
        baseline_track_ids: List[str],
        baseline_genres: Optional[List[int]] = None,
        baseline_emotions: Optional[List[int]] = None,
        baseline_popularities: Optional[List[float]] = None,
        current_track_ids: Optional[List[str]] = None,
        degradation_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """Verifica degradación del modelo comparando con baseline"""
        try:
            # Evaluar baseline
            baseline_eval = self.evaluate_model_advanced(
                track_ids=baseline_track_ids,
                genres=baseline_genres,
                emotions=baseline_emotions,
                popularities=baseline_popularities
            )
            
            if "error" in baseline_eval:
                return baseline_eval
            
            baseline_metrics = baseline_eval.get("metrics", {})
            
            # Evaluar actual (si se proporciona)
            if current_track_ids:
                current_eval = self.evaluate_model_advanced(
                    track_ids=current_track_ids,
                    genres=baseline_genres[:len(current_track_ids)] if baseline_genres else None,
                    emotions=baseline_emotions[:len(current_track_ids)] if baseline_emotions else None,
                    popularities=baseline_popularities[:len(current_track_ids)] if baseline_popularities else None
                )
                
                if "error" not in current_eval:
                    current_metrics = current_eval.get("metrics", {})
                    
                    # Comparar métricas
                    degradation = {}
                    degraded = False
                    
                    if "genre" in baseline_metrics and "genre" in current_metrics:
                        baseline_acc = baseline_metrics["genre"].get("accuracy", 0)
                        current_acc = current_metrics["genre"].get("accuracy", 0)
                        diff = baseline_acc - current_acc
                        
                        degradation["genre_accuracy"] = {
                            "baseline": float(baseline_acc),
                            "current": float(current_acc),
                            "degradation": float(diff),
                            "degraded": diff > degradation_threshold
                        }
                        if diff > degradation_threshold:
                            degraded = True
                    
                    if "emotion" in baseline_metrics and "emotion" in current_metrics:
                        baseline_acc = baseline_metrics["emotion"].get("accuracy", 0)
                        current_acc = current_metrics["emotion"].get("accuracy", 0)
                        diff = baseline_acc - current_acc
                        
                        degradation["emotion_accuracy"] = {
                            "baseline": float(baseline_acc),
                            "current": float(current_acc),
                            "degradation": float(diff),
                            "degraded": diff > degradation_threshold
                        }
                        if diff > degradation_threshold:
                            degraded = True
                    
                    return {
                        "success": True,
                        "degradation_detected": degraded,
                        "degradation_metrics": degradation,
                        "threshold": degradation_threshold,
                        "recommendation": "Retrain model" if degraded else "Model performing well"
                    }
            
            # Si no hay datos actuales, usar métricas de baseline como referencia
            return {
                "success": True,
                "baseline_metrics": baseline_metrics,
                "message": "Baseline established. Monitor current performance to detect degradation."
            }
        except Exception as e:
            self.logger.error(f"Error checking model degradation: {e}")
            return {"error": str(e)}
    
    def auto_retrain(
        self,
        trigger: str = "degradation",
        train_track_ids: List[str] = None,
        val_track_ids: List[str] = None,
        train_genres: Optional[List[int]] = None,
        train_emotions: Optional[List[int]] = None,
        train_popularities: Optional[List[float]] = None,
        val_genres: Optional[List[int]] = None,
        val_emotions: Optional[List[int]] = None,
        val_popularities: Optional[List[float]] = None,
        epochs: int = 5,
        improvement_threshold: float = 0.01
    ) -> Dict[str, Any]:
        """Auto-retraining del modelo basado en triggers"""
        try:
            if not train_track_ids or not val_track_ids:
                return {"error": "Se necesitan datos de entrenamiento y validación"}
            
            # Guardar modelo actual
            backup_path = f"./model_backups/model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            os.makedirs("./model_backups", exist_ok=True)
            
            if self.model:
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "genres": self.genres,
                    "emotions": self.emotions
                }, backup_path)
            
            # Evaluar modelo actual
            current_eval = self.evaluate_model_advanced(
                track_ids=val_track_ids,
                genres=val_genres,
                emotions=val_emotions,
                popularities=val_popularities
            )
            
            current_metrics = current_eval.get("metrics", {}) if "error" not in current_eval else {}
            current_val_loss = current_metrics.get("genre", {}).get("accuracy", 0) if "genre" in current_metrics else 0
            
            # Entrenar nuevo modelo
            train_result = self.train_with_validation(
                train_track_ids=train_track_ids,
                val_track_ids=val_track_ids,
                train_genres=train_genres,
                train_emotions=train_emotions,
                train_popularities=train_popularities,
                val_genres=val_genres,
                val_emotions=val_emotions,
                val_popularities=val_popularities,
                epochs=epochs,
                early_stopping_patience=3
            )
            
            if "error" in train_result:
                # Restaurar modelo anterior
                if os.path.exists(backup_path):
                    self.load_model(backup_path)
                return train_result
            
            # Evaluar nuevo modelo
            new_eval = self.evaluate_model_advanced(
                track_ids=val_track_ids,
                genres=val_genres,
                emotions=val_emotions,
                popularities=val_popularities
            )
            
            new_metrics = new_eval.get("metrics", {}) if "error" not in new_eval else {}
            new_val_loss = new_metrics.get("genre", {}).get("accuracy", 0) if "genre" in new_metrics else 0
            
            # Verificar mejora
            improvement = new_val_loss - current_val_loss
            improved = improvement > improvement_threshold
            
            if not improved:
                # Restaurar modelo anterior si no hay mejora
                if os.path.exists(backup_path):
                    self.load_model(backup_path)
                    return {
                        "success": False,
                        "message": "New model did not improve. Previous model restored.",
                        "improvement": float(improvement),
                        "threshold": improvement_threshold
                    }
            
            return {
                "success": True,
                "trigger": trigger,
                "improvement": float(improvement),
                "improved": improved,
                "backup_path": backup_path,
                "training_result": train_result,
                "old_metrics": current_metrics,
                "new_metrics": new_metrics
            }
        except Exception as e:
            self.logger.error(f"Error in auto-retraining: {e}")
            return {"error": str(e)}
    
    def analyze_prediction_confidence(
        self,
        track_ids: List[str],
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Analiza la confianza de las predicciones"""
        try:
            if not self.model_loaded or self.model is None:
                self.initialize_model()
            
            predictions = []
            low_confidence_count = 0
            
            for track_id in track_ids:
                audio_features = self.spotify.get_track_audio_features(track_id)
                if not audio_features:
                    continue
                
                features = self._extract_features(audio_features)
                features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(self.device)
                
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(features_tensor)
                
                genre_probs = torch.softmax(outputs["genre_logits"], dim=1).cpu().numpy()[0]
                emotion_probs = torch.softmax(outputs["emotion_logits"], dim=1).cpu().numpy()[0]
                
                genre_confidence = float(np.max(genre_probs))
                emotion_confidence = float(np.max(emotion_probs))
                avg_confidence = (genre_confidence + emotion_confidence) / 2
                
                is_low_confidence = avg_confidence < confidence_threshold
                if is_low_confidence:
                    low_confidence_count += 1
                
                predictions.append({
                    "track_id": track_id,
                    "genre_confidence": genre_confidence,
                    "emotion_confidence": emotion_confidence,
                    "average_confidence": avg_confidence,
                    "low_confidence": is_low_confidence
                })
            
            return {
                "success": True,
                "total_predictions": len(predictions),
                "low_confidence_count": low_confidence_count,
                "low_confidence_rate": float(low_confidence_count / len(predictions)) if predictions else 0.0,
                "average_confidence": float(np.mean([p["average_confidence"] for p in predictions])) if predictions else 0.0,
                "confidence_threshold": confidence_threshold,
                "predictions": predictions
            }
        except Exception as e:
            self.logger.error(f"Error analyzing prediction confidence: {e}")
            return {"error": str(e)}
    
    def detect_outliers(
        self,
        track_ids: List[str],
        method: str = "zscore",
        threshold: float = 3.0
    ) -> Dict[str, Any]:
        """Detecta outliers en los embeddings"""
        try:
            # Extraer embeddings
            result = self.extract_embeddings(track_ids)
            if "error" in result:
                return result
            
            embeddings = np.array(result["embeddings"])
            track_ids = result["track_ids"]
            
            outliers = []
            
            if method == "zscore":
                # Detectar outliers usando Z-score
                z_scores = np.abs(zscore(embeddings, axis=0))
                outlier_scores = z_scores.max(axis=1)
                
                for i, (track_id, score) in enumerate(zip(track_ids, outlier_scores)):
                    if score > threshold:
                        outliers.append({
                            "track_id": track_id,
                            "outlier_score": float(score),
                            "method": method
                        })
            
            elif method == "isolation":
                # Usar distancia promedio como proxy de isolation
                from sklearn.neighbors import NearestNeighbors
                nbrs = NearestNeighbors(n_neighbors=min(5, len(embeddings) - 1))
                nbrs.fit(embeddings)
                distances, _ = nbrs.kneighbors(embeddings)
                avg_distances = distances.mean(axis=1)
                
                threshold_dist = np.percentile(avg_distances, 95)
                
                for i, (track_id, dist) in enumerate(zip(track_ids, avg_distances)):
                    if dist > threshold_dist:
                        outliers.append({
                            "track_id": track_id,
                            "outlier_score": float(dist),
                            "method": method
                        })
            
            return {
                "success": True,
                "method": method,
                "threshold": threshold,
                "total_tracks": len(track_ids),
                "outliers": outliers,
                "outlier_count": len(outliers),
                "outlier_rate": float(len(outliers) / len(track_ids)) if track_ids else 0.0
            }
        except Exception as e:
            self.logger.error(f"Error detecting outliers: {e}")
            return {"error": str(e)}
    
    def create_ensemble(
        self,
        model_paths: List[str],
        weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Crea un ensemble de modelos"""
        try:
            if len(model_paths) < 2:
                return {"error": "Se necesitan al menos 2 modelos para ensemble"}
            
            if len(model_paths) > 5:
                return {"error": "Máximo 5 modelos en ensemble"}
            
            ensemble_models = []
            original_model_state = self.model.state_dict() if self.model else None
            
            for model_path in model_paths:
                load_result = self.load_model(model_path)
                if "error" not in load_result:
                    ensemble_models.append({
                        "path": model_path,
                        "model_state": self.model.state_dict().copy()
                    })
            
            if not ensemble_models:
                return {"error": "No se pudieron cargar modelos para ensemble"}
            
            # Normalizar pesos
            if weights:
                if len(weights) != len(ensemble_models):
                    weights = None
                else:
                    weights = np.array(weights)
                    weights = weights / weights.sum()
            else:
                weights = np.ones(len(ensemble_models)) / len(ensemble_models)
            
            self.ensemble_models = ensemble_models
            
            # Restaurar modelo original
            if original_model_state and self.model:
                self.model.load_state_dict(original_model_state)
            
            return {
                "success": True,
                "ensemble_size": len(ensemble_models),
                "model_paths": [m["path"] for m in ensemble_models],
                "weights": weights.tolist()
            }
        except Exception as e:
            self.logger.error(f"Error creating ensemble: {e}")
            return {"error": str(e)}
    
    def predict_with_ensemble(
        self,
        track_id: str
    ) -> Dict[str, Any]:
        """Predice usando ensemble de modelos"""
        try:
            if not self.ensemble_models:
                return {"error": "No hay ensemble creado"}
            
            audio_features = self.spotify.get_track_audio_features(track_id)
            if not audio_features:
                return {"error": "No hay características de audio disponibles"}
            
            features = self._extract_features(audio_features)
            features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
            
            # Predicciones de cada modelo
            genre_predictions = []
            emotion_predictions = []
            popularity_predictions = []
            
            original_model_state = self.model.state_dict() if self.model else None
            
            for i, ensemble_model in enumerate(self.ensemble_models):
                self.model.load_state_dict(ensemble_model["model_state"])
                self.model.eval()
                
                with torch.no_grad():
                    outputs = self.model(features_tensor.to(self.device))
                
                genre_probs = torch.softmax(outputs["genre_logits"], dim=1).cpu().numpy()[0]
                emotion_probs = torch.softmax(outputs["emotion_logits"], dim=1).cpu().numpy()[0]
                popularity_pred = outputs["popularity_pred"].cpu().numpy()[0]
                
                genre_predictions.append(genre_probs)
                emotion_predictions.append(emotion_probs)
                popularity_predictions.append(popularity_pred)
            
            # Restaurar modelo original
            if original_model_state and self.model:
                self.model.load_state_dict(original_model_state)
            
            # Promedio ponderado
            weights = np.ones(len(self.ensemble_models)) / len(self.ensemble_models)
            
            ensemble_genre = np.average(genre_predictions, axis=0, weights=weights)
            ensemble_emotion = np.average(emotion_predictions, axis=0, weights=weights)
            ensemble_popularity = np.average(popularity_predictions, axis=0, weights=weights)
            
            # Top predictions
            top_genre_idx = np.argmax(ensemble_genre)
            top_emotion_idx = np.argmax(ensemble_emotion)
            
            return {
                "success": True,
                "track_id": track_id,
                "predictions": {
                    "genre": {
                        "primary": self.genres[top_genre_idx],
                        "confidence": float(ensemble_genre[top_genre_idx]),
                        "all_probs": ensemble_genre.tolist()
                    },
                    "emotion": {
                        "primary": self.emotions[top_emotion_idx],
                        "confidence": float(ensemble_emotion[top_emotion_idx]),
                        "all_probs": ensemble_emotion.tolist()
                    },
                    "popularity": {
                        "predicted": float(ensemble_popularity * 100)
                    }
                },
                "ensemble_size": len(self.ensemble_models),
                "method": "weighted_average"
            }
        except Exception as e:
            self.logger.error(f"Error predicting with ensemble: {e}")
            return {"error": str(e)}
    
    def batch_process_advanced(
        self,
        track_ids: List[str],
        batch_size: int = 32,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Procesamiento batch avanzado con caching"""
        try:
            if len(track_ids) > 500:
                return {"error": "Máximo 500 tracks por batch"}
            
            results = []
            cached_count = 0
            
            # Procesar en batches
            for i in range(0, len(track_ids), batch_size):
                batch = track_ids[i:i+batch_size]
                batch_results = []
                
                for track_id in batch:
                    # Verificar cache
                    if use_cache and track_id in self.embedding_cache:
                        cached_count += 1
                        results.append(self.embedding_cache[track_id])
                        continue
                    
                    # Procesar
                    prediction = self.predict_with_model(track_id, monitor=False)
                    if "error" not in prediction:
                        batch_results.append(prediction)
                        
                        # Guardar en cache
                        if use_cache:
                            self.embedding_cache[track_id] = prediction
                
                results.extend(batch_results)
            
            # Limpiar cache si es muy grande
            if len(self.embedding_cache) > 10000:
                # Mantener solo los más recientes
                self.embedding_cache = dict(list(self.embedding_cache.items())[-5000:])
            
            return {
                "success": True,
                "total_processed": len(results),
                "cached_count": cached_count,
                "new_predictions": len(results) - cached_count,
                "results": results,
                "batch_size": batch_size
            }
        except Exception as e:
            self.logger.error(f"Error in advanced batch processing: {e}")
            return {"error": str(e)}
    
    def clear_cache(self) -> Dict[str, Any]:
        """Limpia el cache de modelos y embeddings"""
        try:
            cache_size = len(self.embedding_cache)
            self.embedding_cache.clear()
            self.model_cache.clear()
            
            return {
                "success": True,
                "message": "Cache cleared",
                "cleared_items": cache_size
            }
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return {"error": str(e)}
    
    def calibrate_model(
        self,
        track_ids: List[str],
        true_genres: Optional[List[int]] = None,
        true_emotions: Optional[List[int]] = None,
        method: str = "isotonic"
    ) -> Dict[str, Any]:
        """Calibra las probabilidades del modelo"""
        try:
            if not self.model_loaded or self.model is None:
                self.initialize_model()
            
            if not track_ids:
                return {"error": "Se necesitan track_ids para calibración"}
            
            # Extraer predicciones
            predictions = []
            true_labels_genre = []
            true_labels_emotion = []
            
            for i, track_id in enumerate(track_ids):
                audio_features = self.spotify.get_track_audio_features(track_id)
                if not audio_features:
                    continue
                
                features = self._extract_features(audio_features)
                features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(self.device)
                
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(features_tensor)
                
                genre_probs = torch.softmax(outputs["genre_logits"], dim=1).cpu().numpy()[0]
                emotion_probs = torch.softmax(outputs["emotion_logits"], dim=1).cpu().numpy()[0]
                
                predictions.append({
                    "genre_probs": genre_probs,
                    "emotion_probs": emotion_probs
                })
                
                if true_genres and i < len(true_genres):
                    true_labels_genre.append(true_genres[i])
                if true_emotions and i < len(true_emotions):
                    true_labels_emotion.append(true_emotions[i])
            
            # Calibrar usando isotonic regression
            if method == "isotonic" and true_labels_genre:
                genre_probs_array = np.array([p["genre_probs"] for p in predictions])
                calibrated_genre = np.zeros_like(genre_probs_array)
                
                for class_idx in range(len(self.genres)):
                    class_probs = genre_probs_array[:, class_idx]
                    class_labels = (np.array(true_labels_genre) == class_idx).astype(int)
                    
                    if len(np.unique(class_labels)) > 1:
                        iso_reg = IsotonicRegression(out_of_bounds='clip')
                        iso_reg.fit(class_probs, class_labels)
                        calibrated_genre[:, class_idx] = iso_reg.transform(class_probs)
                    else:
                        calibrated_genre[:, class_idx] = class_probs
                
                # Normalizar
                calibrated_genre = calibrated_genre / calibrated_genre.sum(axis=1, keepdims=True)
                
                self.calibration_model = {
                    "method": method,
                    "calibrated": True,
                    "genre_calibration": True
                }
                
                return {
                    "success": True,
                    "method": method,
                    "calibrated": True,
                    "total_samples": len(predictions),
                    "message": "Modelo calibrado exitosamente"
                }
            
            return {
                "success": False,
                "message": "Se necesitan labels verdaderos para calibración"
            }
        except Exception as e:
            self.logger.error(f"Error calibrating model: {e}")
            return {"error": str(e)}
    
    def analyze_uncertainty(
        self,
        track_ids: List[str],
        num_samples: int = 10
    ) -> Dict[str, Any]:
        """Analiza la incertidumbre usando Monte Carlo Dropout"""
        try:
            if not self.model_loaded or self.model is None:
                self.initialize_model()
            
            uncertainties = []
            
            for track_id in track_ids:
                audio_features = self.spotify.get_track_audio_features(track_id)
                if not audio_features:
                    continue
                
                features = self._extract_features(audio_features)
                features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(self.device)
                
                # Múltiples forward passes con dropout activo
                self.model.train()  # Activar dropout
                predictions_genre = []
                predictions_emotion = []
                predictions_popularity = []
                
                with torch.no_grad():
                    for _ in range(num_samples):
                        outputs = self.model(features_tensor)
                        genre_probs = torch.softmax(outputs["genre_logits"], dim=1).cpu().numpy()[0]
                        emotion_probs = torch.softmax(outputs["emotion_logits"], dim=1).cpu().numpy()[0]
                        popularity = outputs["popularity_pred"].cpu().numpy()[0]
                        
                        predictions_genre.append(genre_probs)
                        predictions_emotion.append(emotion_probs)
                        predictions_popularity.append(popularity)
                
                # Calcular incertidumbre (varianza)
                genre_std = np.std(predictions_genre, axis=0)
                emotion_std = np.std(predictions_emotion, axis=0)
                popularity_std = np.std(predictions_popularity, axis=0)
                
                # Promedio de predicciones
                genre_mean = np.mean(predictions_genre, axis=0)
                emotion_mean = np.mean(predictions_emotion, axis=0)
                popularity_mean = np.mean(predictions_popularity, axis=0)
                
                # Incertidumbre total (entropía promedio)
                genre_entropy = -np.sum(genre_mean * np.log(genre_mean + 1e-10))
                emotion_entropy = -np.sum(emotion_mean * np.log(emotion_mean + 1e-10))
                
                uncertainties.append({
                    "track_id": track_id,
                    "genre_uncertainty": float(np.mean(genre_std)),
                    "emotion_uncertainty": float(np.mean(emotion_std)),
                    "popularity_uncertainty": float(popularity_std),
                    "genre_entropy": float(genre_entropy),
                    "emotion_entropy": float(emotion_entropy),
                    "total_uncertainty": float((np.mean(genre_std) + np.mean(emotion_std) + popularity_std) / 3)
                })
            
            return {
                "success": True,
                "num_samples": num_samples,
                "uncertainties": uncertainties,
                "average_uncertainty": float(np.mean([u["total_uncertainty"] for u in uncertainties])) if uncertainties else 0.0
            }
        except Exception as e:
            self.logger.error(f"Error analyzing uncertainty: {e}")
            return {"error": str(e)}
    
    def active_learning_query(
        self,
        track_ids: List[str],
        query_strategy: str = "uncertainty",
        num_samples: int = 10
    ) -> Dict[str, Any]:
        """Selecciona muestras para etiquetar usando active learning"""
        try:
            if query_strategy == "uncertainty":
                # Usar incertidumbre para seleccionar muestras
                uncertainty_result = self.analyze_uncertainty(track_ids, num_samples=5)
                if "error" in uncertainty_result:
                    return uncertainty_result
                
                # Ordenar por incertidumbre
                uncertainties = uncertainty_result["uncertainties"]
                uncertainties.sort(key=lambda x: x["total_uncertainty"], reverse=True)
                
                # Seleccionar top N más inciertas
                selected = uncertainties[:num_samples]
                
                return {
                    "success": True,
                    "strategy": query_strategy,
                    "selected_tracks": [s["track_id"] for s in selected],
                    "uncertainties": selected,
                    "num_selected": len(selected)
                }
            
            elif query_strategy == "diversity":
                # Seleccionar muestras diversas usando clustering
                result = self.extract_embeddings(track_ids)
                if "error" in result:
                    return result
                
                embeddings = np.array(result["embeddings"])
                
                # K-Means para diversidad
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=min(num_samples, len(embeddings)))
                kmeans.fit(embeddings)
                
                # Seleccionar el track más cercano al centroide de cada cluster
                selected_indices = []
                for i in range(kmeans.n_clusters):
                    cluster_center = kmeans.cluster_centers_[i]
                    distances = np.linalg.norm(embeddings - cluster_center, axis=1)
                    closest_idx = np.argmin(distances)
                    selected_indices.append(closest_idx)
                
                selected_tracks = [result["track_ids"][idx] for idx in selected_indices]
                
                return {
                    "success": True,
                    "strategy": query_strategy,
                    "selected_tracks": selected_tracks,
                    "num_selected": len(selected_tracks)
                }
            
            return {"error": f"Estrategia desconocida: {query_strategy}"}
        except Exception as e:
            self.logger.error(f"Error in active learning query: {e}")
            return {"error": str(e)}
    
    def transfer_learning_analysis(
        self,
        source_track_ids: List[str],
        target_track_ids: List[str],
        source_labels: Optional[Dict[str, List[int]]] = None
    ) -> Dict[str, Any]:
        """Analiza transfer learning entre dominios"""
        try:
            # Extraer embeddings de ambos dominios
            source_result = self.extract_embeddings(source_track_ids)
            target_result = self.extract_embeddings(target_track_ids)
            
            if "error" in source_result or "error" in target_result:
                return {"error": "Error extrayendo embeddings"}
            
            source_embeddings = np.array(source_result["embeddings"])
            target_embeddings = np.array(target_result["embeddings"])
            
            # Calcular distancia entre distribuciones
            source_mean = np.mean(source_embeddings, axis=0)
            target_mean = np.mean(target_embeddings, axis=0)
            
            source_std = np.std(source_embeddings, axis=0)
            target_std = np.std(target_embeddings, axis=0)
            
            # Distancia entre medias
            mean_distance = np.linalg.norm(source_mean - target_mean)
            
            # Divergencia KL aproximada
            kl_div = np.sum(
                source_std * np.log((source_std + 1e-10) / (target_std + 1e-10)) +
                (target_mean - source_mean) ** 2 / (2 * target_std ** 2 + 1e-10)
            )
            
            # Similitud coseno entre distribuciones
            cosine_sim = np.dot(source_mean, target_mean) / (
                np.linalg.norm(source_mean) * np.linalg.norm(target_mean) + 1e-10
            )
            
            transferability_score = float(cosine_sim)  # Mayor similitud = mejor transferencia
            
            return {
                "success": True,
                "source_domain": {
                    "num_samples": len(source_track_ids),
                    "mean_norm": float(np.linalg.norm(source_mean)),
                    "std_mean": float(np.mean(source_std))
                },
                "target_domain": {
                    "num_samples": len(target_track_ids),
                    "mean_norm": float(np.linalg.norm(target_mean)),
                    "std_mean": float(np.mean(target_std))
                },
                "domain_distance": {
                    "mean_distance": float(mean_distance),
                    "kl_divergence": float(kl_div),
                    "cosine_similarity": float(cosine_sim)
                },
                "transferability_score": transferability_score,
                "recommendation": "Good transferability" if transferability_score > 0.7 else "Limited transferability"
            }
        except Exception as e:
            self.logger.error(f"Error in transfer learning analysis: {e}")
            return {"error": str(e)}
    
    def detect_adversarial_examples(
        self,
        track_id: str,
        epsilon: float = 0.01,
        num_perturbations: int = 10
    ) -> Dict[str, Any]:
        """Detecta si un track es vulnerable a adversarial examples"""
        try:
            if not self.model_loaded or self.model is None:
                self.initialize_model()
            
            audio_features = self.spotify.get_track_audio_features(track_id)
            if not audio_features:
                return {"error": "No hay características de audio disponibles"}
            
            features = self._extract_features(audio_features)
            features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(self.device)
            features_tensor.requires_grad = True
            
            # Predicción original
            self.model.eval()
            with torch.no_grad():
                original_outputs = self.model(features_tensor)
                original_genre = torch.softmax(original_outputs["genre_logits"], dim=1)
                original_emotion = torch.softmax(original_outputs["emotion_logits"], dim=1)
            
            # Calcular gradientes
            self.model.train()
            outputs = self.model(features_tensor)
            loss = outputs["genre_logits"].sum()  # Simple loss para gradientes
            loss.backward()
            
            gradients = features_tensor.grad.data
            
            # Crear perturbaciones adversarias
            perturbations = []
            prediction_changes = []
            
            for _ in range(num_perturbations):
                # Perturbación basada en gradientes
                perturbation = epsilon * torch.sign(gradients)
                perturbed_features = features_tensor.data + perturbation
                
                # Predicción con perturbación
                with torch.no_grad():
                    perturbed_outputs = self.model(perturbed_features)
                    perturbed_genre = torch.softmax(perturbed_outputs["genre_logits"], dim=1)
                    perturbed_emotion = torch.softmax(perturbed_outputs["emotion_logits"], dim=1)
                
                # Cambio en predicciones
                genre_change = torch.abs(perturbed_genre - original_genre).sum().item()
                emotion_change = torch.abs(perturbed_emotion - original_emotion).sum().item()
                
                perturbations.append({
                    "epsilon": epsilon,
                    "genre_change": float(genre_change),
                    "emotion_change": float(emotion_change),
                    "total_change": float(genre_change + emotion_change)
                })
                
                prediction_changes.append(float(genre_change + emotion_change))
            
            # Calcular robustez
            avg_change = np.mean(prediction_changes)
            max_change = np.max(prediction_changes)
            
            is_vulnerable = avg_change > 0.1 or max_change > 0.3
            
            return {
                "success": True,
                "track_id": track_id,
                "epsilon": epsilon,
                "num_perturbations": num_perturbations,
                "robustness": {
                    "average_change": float(avg_change),
                    "max_change": float(max_change),
                    "is_vulnerable": is_vulnerable
                },
                "perturbations": perturbations,
                "recommendation": "Model is vulnerable to adversarial examples" if is_vulnerable else "Model is relatively robust"
            }
        except Exception as e:
            self.logger.error(f"Error detecting adversarial examples: {e}")
            return {"error": str(e)}
    
    def meta_learning_adapt(
        self,
        support_track_ids: List[str],
        support_labels: Dict[str, List[int]],
        query_track_ids: List[str],
        adaptation_steps: int = 5
    ) -> Dict[str, Any]:
        """Adapta el modelo usando meta-learning (MAML-like)"""
        try:
            if not self.model_loaded or self.model is None:
                self.initialize_model()
            
            if len(support_track_ids) < 5:
                return {"error": "Se necesitan al menos 5 tracks de soporte"}
            
            # Guardar estado original
            original_state = self.model.state_dict().copy()
            
            # Preparar datos de soporte
            support_features = []
            support_genres = []
            support_emotions = []
            
            for i, track_id in enumerate(support_track_ids):
                audio_features = self.spotify.get_track_audio_features(track_id)
                if not audio_features:
                    continue
                
                features = self._extract_features(audio_features)
                support_features.append(features)
                
                if "genres" in support_labels and i < len(support_labels["genres"]):
                    support_genres.append(support_labels["genres"][i])
                if "emotions" in support_labels and i < len(support_labels["emotions"]):
                    support_emotions.append(support_labels["emotions"][i])
            
            if not support_features:
                return {"error": "No se pudieron extraer características"}
            
            # Adaptación rápida (few gradient steps)
            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            for step in range(adaptation_steps):
                optimizer.zero_grad()
                total_loss = 0
                
                for i, features in enumerate(support_features):
                    features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(self.device)
                    outputs = self.model(features_tensor)
                    
                    # Loss para género
                    if i < len(support_genres):
                        genre_loss = torch.nn.functional.cross_entropy(
                            outputs["genre_logits"],
                            torch.LongTensor([support_genres[i]]).to(self.device)
                        )
                        total_loss += genre_loss
                    
                    # Loss para emoción
                    if i < len(support_emotions):
                        emotion_loss = torch.nn.functional.cross_entropy(
                            outputs["emotion_logits"],
                            torch.LongTensor([support_emotions[i]]).to(self.device)
                        )
                        total_loss += emotion_loss
                
                if total_loss > 0:
                    total_loss = total_loss / len(support_features)
                    total_loss.backward()
                    optimizer.step()
            
            # Evaluar en queries
            query_predictions = []
            for track_id in query_track_ids:
                audio_features = self.spotify.get_track_audio_features(track_id)
                if not audio_features:
                    continue
                
                features = self._extract_features(audio_features)
                features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(self.device)
                
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(features_tensor)
                
                genre_probs = torch.softmax(outputs["genre_logits"], dim=1).cpu().numpy()[0]
                emotion_probs = torch.softmax(outputs["emotion_logits"], dim=1).cpu().numpy()[0]
                
                query_predictions.append({
                    "track_id": track_id,
                    "genre": self.genres[np.argmax(genre_probs)],
                    "emotion": self.emotions[np.argmax(emotion_probs)],
                    "genre_confidence": float(np.max(genre_probs)),
                    "emotion_confidence": float(np.max(emotion_probs))
                })
            
            # Restaurar estado original
            self.model.load_state_dict(original_state)
            
            return {
                "success": True,
                "adaptation_steps": adaptation_steps,
                "support_samples": len(support_features),
                "query_predictions": query_predictions,
                "num_queries": len(query_predictions)
            }
        except Exception as e:
            self.logger.error(f"Error in meta-learning adaptation: {e}")
            return {"error": str(e)}
    
    def few_shot_learning(
        self,
        task_name: str,
        example_track_ids: List[str],
        example_labels: Dict[str, List[int]],
        query_track_ids: List[str]
    ) -> Dict[str, Any]:
        """Aprende de pocos ejemplos (few-shot learning)"""
        try:
            if len(example_track_ids) < 2:
                return {"error": "Se necesitan al menos 2 ejemplos"}
            
            # Guardar ejemplos
            self.few_shot_examples[task_name] = {
                "track_ids": example_track_ids,
                "labels": example_labels
            }
            
            # Extraer embeddings de ejemplos
            example_result = self.extract_embeddings(example_track_ids)
            if "error" in example_result:
                return example_result
            
            example_embeddings = np.array(example_result["embeddings"])
            
            # Extraer embeddings de queries
            query_result = self.extract_embeddings(query_track_ids)
            if "error" in query_result:
                return query_result
            
            query_embeddings = np.array(query_result["embeddings"])
            
            # Encontrar el ejemplo más cercano para cada query
            predictions = []
            for i, query_emb in enumerate(query_embeddings):
                # Calcular distancias a todos los ejemplos
                distances = np.linalg.norm(example_embeddings - query_emb, axis=1)
                closest_idx = np.argmin(distances)
                
                # Predecir basado en el ejemplo más cercano
                if "genres" in example_labels and closest_idx < len(example_labels["genres"]):
                    predicted_genre_idx = example_labels["genres"][closest_idx]
                    predicted_genre = self.genres[predicted_genre_idx] if predicted_genre_idx < len(self.genres) else "unknown"
                else:
                    predicted_genre = "unknown"
                
                if "emotions" in example_labels and closest_idx < len(example_labels["emotions"]):
                    predicted_emotion_idx = example_labels["emotions"][closest_idx]
                    predicted_emotion = self.emotions[predicted_emotion_idx] if predicted_emotion_idx < len(self.emotions) else "unknown"
                else:
                    predicted_emotion = "unknown"
                
                predictions.append({
                    "track_id": query_track_ids[i],
                    "predicted_genre": predicted_genre,
                    "predicted_emotion": predicted_emotion,
                    "closest_example": example_track_ids[closest_idx],
                    "distance": float(distances[closest_idx])
                })
            
            return {
                "success": True,
                "task_name": task_name,
                "num_examples": len(example_track_ids),
                "num_queries": len(query_track_ids),
                "predictions": predictions
            }
        except Exception as e:
            self.logger.error(f"Error in few-shot learning: {e}")
            return {"error": str(e)}
    
    def analyze_causality(
        self,
        track_ids: List[str],
        target_variable: str = "popularity",
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analiza relaciones causales entre características"""
        try:
            if not track_ids:
                return {"error": "Se necesitan track_ids"}
            
            # Extraer características
            features_list = []
            targets = []
            
            for track_id in track_ids:
                audio_features = self.spotify.get_track_audio_features(track_id)
                if not audio_features:
                    continue
                
                features = self._extract_features(audio_features)
                features_list.append(features)
                
                if target_variable == "popularity":
                    track_info = self.spotify.get_track_info(track_id)
                    if track_info and "popularity" in track_info:
                        targets.append(track_info["popularity"] / 100.0)
                    else:
                        targets.append(0.5)  # Default
            
            if not features_list:
                return {"error": "No se pudieron extraer características"}
            
            features_array = np.array(features_list)
            targets_array = np.array(targets)
            
            # Calcular correlaciones (proxy de causalidad)
            correlations = []
            default_feature_names = [
                "danceability", "energy", "key", "loudness", "mode",
                "speechiness", "acousticness", "instrumentalness",
                "liveness", "valence", "tempo", "duration_ms", "time_signature"
            ]
            
            feature_names = feature_names or default_feature_names[:len(features_array[0])]
            
            for i, feat_name in enumerate(feature_names):
                if i < features_array.shape[1]:
                    feat_values = features_array[:, i]
                    corr, p_value = pearsonr(feat_values, targets_array)
                    
                    correlations.append({
                        "feature": feat_name,
                        "correlation": float(corr),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05
                    })
            
            # Ordenar por correlación absoluta
            correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            
            return {
                "success": True,
                "target_variable": target_variable,
                "num_samples": len(features_list),
                "correlations": correlations,
                "top_causal_features": [c["feature"] for c in correlations[:5]]
            }
        except Exception as e:
            self.logger.error(f"Error analyzing causality: {e}")
            return {"error": str(e)}
    
    def explain_prediction_advanced(
        self,
        track_id: str,
        method: str = "gradient",
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Explicación avanzada de predicciones"""
        try:
            if not self.model_loaded or self.model is None:
                self.initialize_model()
            
            audio_features = self.spotify.get_track_audio_features(track_id)
            if not audio_features:
                return {"error": "No hay características de audio disponibles"}
            
            features = self._extract_features(audio_features)
            features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(self.device)
            features_tensor.requires_grad = True
            
            feature_names = [
                "danceability", "energy", "key", "loudness", "mode",
                "speechiness", "acousticness", "instrumentalness",
                "liveness", "valence", "tempo", "duration_ms", "time_signature"
            ]
            
            if method == "gradient":
                # Método basado en gradientes
                self.model.eval()
                outputs = self.model(features_tensor)
                
                genre_probs = torch.softmax(outputs["genre_logits"], dim=1)
                emotion_probs = torch.softmax(outputs["emotion_logits"], dim=1)
                
                # Gradientes para género
                genre_probs.sum().backward(retain_graph=True)
                genre_gradients = features_tensor.grad.data.abs().cpu().numpy()[0, 0]
                
                # Gradientes para emoción
                features_tensor.grad.zero_()
                emotion_probs.sum().backward()
                emotion_gradients = features_tensor.grad.data.abs().cpu().numpy()[0, 0]
                
                # Top features para género
                top_genre_indices = np.argsort(genre_gradients)[-top_k:][::-1]
                top_genre_features = [
                    {
                        "feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
                        "importance": float(genre_gradients[i]),
                        "value": float(features[i])
                    }
                    for i in top_genre_indices
                ]
                
                # Top features para emoción
                top_emotion_indices = np.argsort(emotion_gradients)[-top_k:][::-1]
                top_emotion_features = [
                    {
                        "feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
                        "importance": float(emotion_gradients[i]),
                        "value": float(features[i])
                    }
                    for i in top_emotion_indices
                ]
                
                return {
                    "success": True,
                    "track_id": track_id,
                    "method": method,
                    "explanations": {
                        "genre": {
                            "top_features": top_genre_features,
                            "predicted_genre": self.genres[torch.argmax(genre_probs, dim=1).item()],
                            "confidence": float(torch.max(genre_probs).item())
                        },
                        "emotion": {
                            "top_features": top_emotion_features,
                            "predicted_emotion": self.emotions[torch.argmax(emotion_probs, dim=1).item()],
                            "confidence": float(torch.max(emotion_probs).item())
                        }
                    }
                }
            
            return {"error": f"Método desconocido: {method}"}
        except Exception as e:
            self.logger.error(f"Error explaining prediction: {e}")
            return {"error": str(e)}
    
    def analyze_concepts(
        self,
        concept_track_ids: Dict[str, List[str]],
        query_track_ids: List[str]
    ) -> Dict[str, Any]:
        """Analiza conceptos musicales y su presencia en tracks"""
        try:
            # Extraer embeddings de conceptos
            concept_embeddings_dict = {}
            
            for concept_name, track_ids in concept_track_ids.items():
                result = self.extract_embeddings(track_ids)
                if "error" not in result:
                    concept_embeddings = np.array(result["embeddings"])
                    concept_embeddings_dict[concept_name] = {
                        "mean_embedding": np.mean(concept_embeddings, axis=0),
                        "num_tracks": len(track_ids)
                    }
            
            if not concept_embeddings_dict:
                return {"error": "No se pudieron extraer embeddings de conceptos"}
            
            # Extraer embeddings de queries
            query_result = self.extract_embeddings(query_track_ids)
            if "error" in query_result:
                return query_result
            
            query_embeddings = np.array(query_result["embeddings"])
            
            # Analizar presencia de conceptos
            concept_analysis = []
            
            for i, query_emb in enumerate(query_embeddings):
                concept_scores = {}
                
                for concept_name, concept_data in concept_embeddings_dict.items():
                    # Similitud coseno
                    cosine_sim = np.dot(query_emb, concept_data["mean_embedding"]) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(concept_data["mean_embedding"]) + 1e-10
                    )
                    concept_scores[concept_name] = float(cosine_sim)
                
                # Ordenar conceptos por similitud
                sorted_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
                
                concept_analysis.append({
                    "track_id": query_track_ids[i],
                    "concept_scores": concept_scores,
                    "dominant_concept": sorted_concepts[0][0] if sorted_concepts else None,
                    "dominant_score": sorted_concepts[0][1] if sorted_concepts else 0.0
                })
            
            return {
                "success": True,
                "concepts": list(concept_track_ids.keys()),
                "num_queries": len(query_track_ids),
                "analysis": concept_analysis
            }
        except Exception as e:
            self.logger.error(f"Error analyzing concepts: {e}")
            return {"error": str(e)}

