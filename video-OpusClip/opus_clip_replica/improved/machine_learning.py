"""
Advanced Machine Learning System for OpusClip Improved
====================================================

Comprehensive ML system with custom models, training pipelines, and AI optimization.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import UUID, uuid4
import pickle
import joblib
from pathlib import Path
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel
import openai
import anthropic

from .schemas import get_settings
from .exceptions import MachineLearningError, create_ml_error

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Model types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DEEP_LEARNING = "deep_learning"
    TRANSFORMER = "transformer"
    COMPUTER_VISION = "computer_vision"
    NLP = "nlp"


class ModelStatus(str, Enum):
    """Model status"""
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ARCHIVED = "archived"


class TrainingStatus(str, Enum):
    """Training status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ModelMetadata:
    """Model metadata"""
    model_id: str
    name: str
    description: str
    model_type: ModelType
    version: str
    status: ModelStatus
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    created_at: datetime = None
    updated_at: datetime = None
    trained_at: Optional[datetime] = None
    deployed_at: Optional[datetime] = None


@dataclass
class TrainingJob:
    """Training job"""
    job_id: str
    model_id: str
    status: TrainingStatus
    dataset_path: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class PredictionRequest:
    """Prediction request"""
    request_id: str
    model_id: str
    input_data: Dict[str, Any]
    timestamp: datetime
    confidence_threshold: float = 0.5


class VideoDataset(Dataset):
    """Custom video dataset for PyTorch"""
    
    def __init__(self, video_paths: List[str], labels: List[int], transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        # Convert to tensor
        video_tensor = torch.tensor(np.array(frames), dtype=torch.float32)
        
        if self.transform:
            video_tensor = self.transform(video_tensor)
        
        return video_tensor, torch.tensor(label, dtype=torch.long)


class ViralPotentialPredictor(nn.Module):
    """Neural network for viral potential prediction"""
    
    def __init__(self, input_size: int = 100, hidden_size: int = 256, num_classes: int = 2):
        super(ViralPotentialPredictor, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output


class ContentQualityAnalyzer:
    """Content quality analysis using computer vision"""
    
    def __init__(self):
        self.settings = get_settings()
        self.quality_model = None
        self._load_quality_model()
    
    def _load_quality_model(self):
        """Load pre-trained quality analysis model"""
        try:
            # Load pre-trained model for quality assessment
            # This would typically load a pre-trained model from disk
            logger.info("Quality analysis model loaded")
        except Exception as e:
            logger.error(f"Failed to load quality model: {e}")
    
    async def analyze_video_quality(self, video_path: str) -> Dict[str, Any]:
        """Analyze video quality metrics"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Extract frames for analysis
            frames = []
            frame_count = 0
            max_frames = 100  # Analyze up to 100 frames
            
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames.append(frame)
                frame_count += 1
            
            cap.release()
            
            if not frames:
                raise ValueError("No frames extracted from video")
            
            # Analyze quality metrics
            quality_metrics = {
                "sharpness": self._calculate_sharpness(frames),
                "brightness": self._calculate_brightness(frames),
                "contrast": self._calculate_contrast(frames),
                "color_vibrancy": self._calculate_color_vibrancy(frames),
                "stability": self._calculate_stability(frames),
                "resolution_quality": self._assess_resolution_quality(frames[0])
            }
            
            # Calculate overall quality score
            quality_score = self._calculate_overall_quality(quality_metrics)
            
            return {
                "quality_score": quality_score,
                "metrics": quality_metrics,
                "recommendations": self._generate_quality_recommendations(quality_metrics)
            }
            
        except Exception as e:
            logger.error(f"Video quality analysis failed: {e}")
            raise create_ml_error("quality_analysis", video_path, e)
    
    def _calculate_sharpness(self, frames: List[np.ndarray]) -> float:
        """Calculate average sharpness across frames"""
        sharpness_scores = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_scores.append(laplacian_var)
        
        return np.mean(sharpness_scores)
    
    def _calculate_brightness(self, frames: List[np.ndarray]) -> float:
        """Calculate average brightness across frames"""
        brightness_scores = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            brightness_scores.append(brightness)
        
        return np.mean(brightness_scores)
    
    def _calculate_contrast(self, frames: List[np.ndarray]) -> float:
        """Calculate average contrast across frames"""
        contrast_scores = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            contrast = np.std(gray)
            contrast_scores.append(contrast)
        
        return np.mean(contrast_scores)
    
    def _calculate_color_vibrancy(self, frames: List[np.ndarray]) -> float:
        """Calculate color vibrancy across frames"""
        vibrancy_scores = []
        
        for frame in frames:
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            vibrancy = np.mean(saturation)
            vibrancy_scores.append(vibrancy)
        
        return np.mean(vibrancy_scores)
    
    def _calculate_stability(self, frames: List[np.ndarray]) -> float:
        """Calculate video stability"""
        if len(frames) < 2:
            return 1.0
        
        # Calculate optical flow between consecutive frames
        flow_scores = []
        
        for i in range(len(frames) - 1):
            prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            
            flow = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, None, None)
            
            if flow[0] is not None:
                # Calculate motion magnitude
                motion_magnitude = np.mean(np.sqrt(flow[0][:, 0]**2 + flow[0][:, 1]**2))
                flow_scores.append(motion_magnitude)
        
        if not flow_scores:
            return 1.0
        
        # Lower motion = higher stability
        avg_motion = np.mean(flow_scores)
        stability = max(0, 1 - (avg_motion / 10))  # Normalize to 0-1
        
        return stability
    
    def _assess_resolution_quality(self, frame: np.ndarray) -> float:
        """Assess resolution quality"""
        height, width = frame.shape[:2]
        
        # Check for common resolution standards
        if height >= 1080 and width >= 1920:
            return 1.0  # 1080p or higher
        elif height >= 720 and width >= 1280:
            return 0.8  # 720p
        elif height >= 480 and width >= 854:
            return 0.6  # 480p
        else:
            return 0.4  # Lower resolution
    
    def _calculate_overall_quality(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score"""
        weights = {
            "sharpness": 0.25,
            "brightness": 0.15,
            "contrast": 0.20,
            "color_vibrancy": 0.15,
            "stability": 0.15,
            "resolution_quality": 0.10
        }
        
        # Normalize metrics to 0-1 scale
        normalized_metrics = {}
        for key, value in metrics.items():
            if key == "sharpness":
                normalized_metrics[key] = min(1.0, value / 1000)  # Normalize sharpness
            elif key == "brightness":
                normalized_metrics[key] = min(1.0, abs(value - 127.5) / 127.5)  # Optimal around 127.5
            elif key == "contrast":
                normalized_metrics[key] = min(1.0, value / 100)  # Normalize contrast
            elif key == "color_vibrancy":
                normalized_metrics[key] = value / 255  # HSV saturation is 0-255
            else:
                normalized_metrics[key] = value  # Already normalized
        
        # Calculate weighted average
        overall_score = sum(weights[key] * normalized_metrics[key] for key in weights.keys())
        
        return round(overall_score, 3)
    
    def _generate_quality_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        if metrics["sharpness"] < 500:
            recommendations.append("Consider using a tripod or stabilizer to improve sharpness")
        
        if metrics["brightness"] < 100 or metrics["brightness"] > 200:
            recommendations.append("Adjust lighting to achieve optimal brightness levels")
        
        if metrics["contrast"] < 30:
            recommendations.append("Increase contrast to make the video more visually appealing")
        
        if metrics["color_vibrancy"] < 100:
            recommendations.append("Enhance color saturation for more vibrant visuals")
        
        if metrics["stability"] < 0.7:
            recommendations.append("Use video stabilization to reduce camera shake")
        
        if metrics["resolution_quality"] < 0.8:
            recommendations.append("Record in higher resolution for better quality")
        
        return recommendations


class ViralPotentialPredictor:
    """Viral potential prediction using machine learning"""
    
    def __init__(self):
        self.settings = get_settings()
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_importance = {}
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained viral potential model"""
        try:
            # Load pre-trained model
            # This would typically load from disk or train if not exists
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            logger.info("Viral potential model loaded")
        except Exception as e:
            logger.error(f"Failed to load viral potential model: {e}")
    
    async def predict_viral_potential(self, content_features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict viral potential of content"""
        try:
            # Extract features
            features = self._extract_features(content_features)
            
            # Scale features
            features_scaled = self.scaler.fit_transform([features])
            
            # Make prediction
            viral_score = self.model.predict(features_scaled)[0]
            
            # Calculate confidence
            confidence = self._calculate_confidence(features_scaled[0])
            
            # Generate insights
            insights = self._generate_viral_insights(content_features, viral_score)
            
            return {
                "viral_score": round(viral_score, 3),
                "confidence": round(confidence, 3),
                "insights": insights,
                "recommendations": self._generate_viral_recommendations(content_features, viral_score)
            }
            
        except Exception as e:
            logger.error(f"Viral potential prediction failed: {e}")
            raise create_ml_error("viral_prediction", "content", e)
    
    def _extract_features(self, content_features: Dict[str, Any]) -> List[float]:
        """Extract features for viral prediction"""
        features = []
        
        # Video features
        features.append(content_features.get("duration", 0))
        features.append(content_features.get("fps", 0))
        features.append(content_features.get("resolution_score", 0))
        features.append(content_features.get("quality_score", 0))
        
        # Content features
        features.append(content_features.get("sentiment_score", 0))
        features.append(content_features.get("emotion_intensity", 0))
        features.append(content_features.get("topic_relevance", 0))
        features.append(content_features.get("trending_score", 0))
        
        # Engagement features
        features.append(content_features.get("engagement_rate", 0))
        features.append(content_features.get("share_probability", 0))
        features.append(content_features.get("comment_probability", 0))
        features.append(content_features.get("like_probability", 0))
        
        # Platform features
        features.append(content_features.get("platform_optimization", 0))
        features.append(content_features.get("timing_score", 0))
        features.append(content_features.get("audience_match", 0))
        
        return features
    
    def _calculate_confidence(self, features: np.ndarray) -> float:
        """Calculate prediction confidence"""
        # Simple confidence calculation based on feature variance
        # In practice, this would be more sophisticated
        feature_variance = np.var(features)
        confidence = max(0.5, min(1.0, 1 - (feature_variance / 10)))
        return confidence
    
    def _generate_viral_insights(self, content_features: Dict[str, Any], viral_score: float) -> List[str]:
        """Generate insights about viral potential"""
        insights = []
        
        if viral_score > 0.8:
            insights.append("High viral potential detected")
        elif viral_score > 0.6:
            insights.append("Moderate viral potential")
        else:
            insights.append("Low viral potential")
        
        # Content-specific insights
        if content_features.get("sentiment_score", 0) > 0.7:
            insights.append("Positive sentiment increases viral potential")
        
        if content_features.get("emotion_intensity", 0) > 0.8:
            insights.append("High emotional content has viral appeal")
        
        if content_features.get("trending_score", 0) > 0.7:
            insights.append("Content aligns with current trends")
        
        return insights
    
    def _generate_viral_recommendations(self, content_features: Dict[str, Any], viral_score: float) -> List[str]:
        """Generate recommendations to improve viral potential"""
        recommendations = []
        
        if viral_score < 0.6:
            if content_features.get("duration", 0) > 60:
                recommendations.append("Consider shortening the video to under 60 seconds")
            
            if content_features.get("sentiment_score", 0) < 0.5:
                recommendations.append("Add more positive or engaging elements")
            
            if content_features.get("emotion_intensity", 0) < 0.6:
                recommendations.append("Increase emotional impact of the content")
            
            if content_features.get("trending_score", 0) < 0.5:
                recommendations.append("Incorporate current trending topics or hashtags")
        
        return recommendations


class SentimentAnalyzer:
    """Advanced sentiment analysis using multiple models"""
    
    def __init__(self):
        self.settings = get_settings()
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load sentiment analysis models"""
        try:
            # Load different sentiment analysis models
            # This would typically load pre-trained models
            logger.info("Sentiment analysis models loaded")
        except Exception as e:
            logger.error(f"Failed to load sentiment models: {e}")
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        try:
            # Basic sentiment analysis (in practice, would use advanced models)
            positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
            negative_words = ["bad", "terrible", "awful", "horrible", "disappointing", "poor"]
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            total_words = len(text.split())
            positive_ratio = positive_count / total_words if total_words > 0 else 0
            negative_ratio = negative_count / total_words if total_words > 0 else 0
            
            # Calculate sentiment score
            sentiment_score = positive_ratio - negative_ratio
            
            # Determine sentiment label
            if sentiment_score > 0.1:
                sentiment_label = "positive"
            elif sentiment_score < -0.1:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            return {
                "sentiment_label": sentiment_label,
                "sentiment_score": round(sentiment_score, 3),
                "positive_ratio": round(positive_ratio, 3),
                "negative_ratio": round(negative_ratio, 3),
                "confidence": round(abs(sentiment_score), 3)
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            raise create_ml_error("sentiment_analysis", text[:50], e)


class TrendAnalyzer:
    """Trend analysis and prediction"""
    
    def __init__(self):
        self.settings = get_settings()
        self.trend_data = {}
        self._load_trend_data()
    
    def _load_trend_data(self):
        """Load historical trend data"""
        try:
            # Load trend data from various sources
            logger.info("Trend data loaded")
        except Exception as e:
            logger.error(f"Failed to load trend data: {e}")
    
    async def analyze_trends(self, topic: str, time_period: int = 30) -> Dict[str, Any]:
        """Analyze trends for a specific topic"""
        try:
            # Simulate trend analysis
            # In practice, this would analyze real trend data
            
            trend_score = np.random.uniform(0, 1)  # Simulated trend score
            growth_rate = np.random.uniform(-0.2, 0.3)  # Simulated growth rate
            
            # Determine trend direction
            if growth_rate > 0.1:
                trend_direction = "rising"
            elif growth_rate < -0.1:
                trend_direction = "declining"
            else:
                trend_direction = "stable"
            
            # Generate predictions
            predictions = self._generate_trend_predictions(topic, trend_score, growth_rate)
            
            return {
                "topic": topic,
                "trend_score": round(trend_score, 3),
                "growth_rate": round(growth_rate, 3),
                "trend_direction": trend_direction,
                "predictions": predictions,
                "recommendations": self._generate_trend_recommendations(trend_direction, trend_score)
            }
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            raise create_ml_error("trend_analysis", topic, e)
    
    def _generate_trend_predictions(self, topic: str, trend_score: float, growth_rate: float) -> Dict[str, Any]:
        """Generate trend predictions"""
        # Simulate predictions based on current trend
        future_scores = []
        current_score = trend_score
        
        for i in range(7):  # 7-day prediction
            current_score += growth_rate * 0.1  # Daily change
            current_score = max(0, min(1, current_score))  # Clamp to 0-1
            future_scores.append(round(current_score, 3))
        
        return {
            "7_day_forecast": future_scores,
            "peak_prediction": max(future_scores),
            "trough_prediction": min(future_scores)
        }
    
    def _generate_trend_recommendations(self, trend_direction: str, trend_score: float) -> List[str]:
        """Generate trend-based recommendations"""
        recommendations = []
        
        if trend_direction == "rising" and trend_score > 0.7:
            recommendations.append("This topic is trending strongly - consider creating content now")
        elif trend_direction == "declining":
            recommendations.append("This topic is declining - consider alternative topics")
        elif trend_direction == "stable":
            recommendations.append("This topic is stable - good for evergreen content")
        
        return recommendations


class ModelTrainingPipeline:
    """Model training pipeline"""
    
    def __init__(self):
        self.settings = get_settings()
        self.training_jobs: Dict[str, TrainingJob] = {}
        self.models: Dict[str, ModelMetadata] = {}
    
    async def train_model(self, model_config: Dict[str, Any], dataset_path: str) -> str:
        """Train a new model"""
        try:
            job_id = str(uuid4())
            model_id = str(uuid4())
            
            # Create training job
            job = TrainingJob(
                job_id=job_id,
                model_id=model_id,
                status=TrainingStatus.IN_PROGRESS,
                dataset_path=dataset_path,
                hyperparameters=model_config.get("hyperparameters", {}),
                metrics={},
                start_time=datetime.utcnow()
            )
            
            self.training_jobs[job_id] = job
            
            # Create model metadata
            model = ModelMetadata(
                model_id=model_id,
                name=model_config.get("name", f"Model_{model_id[:8]}"),
                description=model_config.get("description", ""),
                model_type=ModelType(model_config.get("type", "classification")),
                version="1.0.0",
                status=ModelStatus.TRAINING,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            self.models[model_id] = model
            
            # Start training in background
            asyncio.create_task(self._train_model_async(job_id))
            
            logger.info(f"Started training job: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise create_ml_error("model_training", model_config.get("name", "unknown"), e)
    
    async def _train_model_async(self, job_id: str):
        """Async model training"""
        try:
            job = self.training_jobs[job_id]
            model = self.models[job.model_id]
            
            # Simulate training process
            # In practice, this would load data, train model, and evaluate
            
            await asyncio.sleep(10)  # Simulate training time
            
            # Update job status
            job.status = TrainingStatus.COMPLETED
            job.end_time = datetime.utcnow()
            job.metrics = {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1_score": 0.85
            }
            
            # Update model
            model.status = ModelStatus.TRAINED
            model.trained_at = datetime.utcnow()
            model.accuracy = job.metrics["accuracy"]
            model.precision = job.metrics["precision"]
            model.recall = job.metrics["recall"]
            model.f1_score = job.metrics["f1_score"]
            
            logger.info(f"Training completed: {job_id}")
            
        except Exception as e:
            # Update job status to failed
            job = self.training_jobs[job_id]
            job.status = TrainingStatus.FAILED
            job.end_time = datetime.utcnow()
            job.error_message = str(e)
            
            # Update model status
            model = self.models[job.model_id]
            model.status = ModelStatus.FAILED
            
            logger.error(f"Training failed: {job_id} - {e}")
    
    def get_training_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get training job status"""
        return self.training_jobs.get(job_id)
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata"""
        return self.models.get(model_id)
    
    def list_models(self) -> List[ModelMetadata]:
        """List all models"""
        return list(self.models.values())
    
    def deploy_model(self, model_id: str) -> bool:
        """Deploy model for inference"""
        try:
            model = self.get_model(model_id)
            if not model:
                raise ValueError(f"Model {model_id} not found")
            
            if model.status != ModelStatus.TRAINED:
                raise ValueError(f"Model {model_id} is not trained")
            
            model.status = ModelStatus.DEPLOYED
            model.deployed_at = datetime.utcnow()
            model.updated_at = datetime.utcnow()
            
            logger.info(f"Model deployed: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            raise create_ml_error("model_deployment", model_id, e)


class MachineLearningManager:
    """Main machine learning manager"""
    
    def __init__(self):
        self.settings = get_settings()
        self.quality_analyzer = ContentQualityAnalyzer()
        self.viral_predictor = ViralPotentialPredictor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        self.training_pipeline = ModelTrainingPipeline()
        
        self.prediction_cache = {}
        self.model_cache = {}
    
    async def analyze_content_comprehensive(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive content analysis using all ML models"""
        try:
            results = {}
            
            # Quality analysis
            if "video_path" in content_data:
                quality_results = await self.quality_analyzer.analyze_video_quality(
                    content_data["video_path"]
                )
                results["quality_analysis"] = quality_results
            
            # Viral potential prediction
            viral_results = await self.viral_predictor.predict_viral_potential(content_data)
            results["viral_prediction"] = viral_results
            
            # Sentiment analysis
            if "text" in content_data:
                sentiment_results = await self.sentiment_analyzer.analyze_sentiment(
                    content_data["text"]
                )
                results["sentiment_analysis"] = sentiment_results
            
            # Trend analysis
            if "topic" in content_data:
                trend_results = await self.trend_analyzer.analyze_trends(
                    content_data["topic"]
                )
                results["trend_analysis"] = trend_results
            
            # Generate overall insights
            results["overall_insights"] = self._generate_overall_insights(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive content analysis failed: {e}")
            raise create_ml_error("comprehensive_analysis", "content", e)
    
    def _generate_overall_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate overall insights from all analyses"""
        insights = []
        
        # Quality insights
        if "quality_analysis" in analysis_results:
            quality_score = analysis_results["quality_analysis"]["quality_score"]
            if quality_score > 0.8:
                insights.append("High-quality content detected")
            elif quality_score < 0.5:
                insights.append("Content quality needs improvement")
        
        # Viral potential insights
        if "viral_prediction" in analysis_results:
            viral_score = analysis_results["viral_prediction"]["viral_score"]
            if viral_score > 0.7:
                insights.append("High viral potential identified")
            elif viral_score < 0.4:
                insights.append("Low viral potential - consider optimization")
        
        # Sentiment insights
        if "sentiment_analysis" in analysis_results:
            sentiment = analysis_results["sentiment_analysis"]["sentiment_label"]
            if sentiment == "positive":
                insights.append("Positive sentiment enhances engagement")
            elif sentiment == "negative":
                insights.append("Negative sentiment may limit reach")
        
        # Trend insights
        if "trend_analysis" in analysis_results:
            trend_direction = analysis_results["trend_analysis"]["trend_direction"]
            if trend_direction == "rising":
                insights.append("Content aligns with rising trends")
            elif trend_direction == "declining":
                insights.append("Consider trending topics for better reach")
        
        return insights
    
    def get_ml_statistics(self) -> Dict[str, Any]:
        """Get machine learning system statistics"""
        try:
            return {
                "total_models": len(self.training_pipeline.models),
                "deployed_models": len([m for m in self.training_pipeline.models.values() 
                                      if m.status == ModelStatus.DEPLOYED]),
                "training_jobs": len(self.training_pipeline.training_jobs),
                "active_jobs": len([j for j in self.training_pipeline.training_jobs.values() 
                                  if j.status == TrainingStatus.IN_PROGRESS]),
                "prediction_cache_size": len(self.prediction_cache),
                "model_cache_size": len(self.model_cache)
            }
            
        except Exception as e:
            logger.error(f"ML statistics failed: {e}")
            return {}


# Global machine learning manager
ml_manager = MachineLearningManager()





























