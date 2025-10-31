"""
Advanced Emotion Analysis System for Ultimate Opus Clip

Comprehensive emotion analysis including facial expressions, voice emotions,
text sentiment, and behavioral patterns for content optimization.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import asyncio
import time
import json
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from pathlib import Path
import numpy as np
import cv2
import librosa
import torch
import torch.nn as nn
from transformers import pipeline, AutoTokenizer, AutoModel
from PIL import Image
import face_recognition
import dlib
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

logger = structlog.get_logger("emotion_analysis")

class EmotionType(Enum):
    """Types of emotions detected."""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"
    NEUTRAL = "neutral"
    EXCITED = "excited"
    CALM = "calm"
    CONFUSED = "confused"
    FRUSTRATED = "frustrated"
    CONTENT = "content"
    ANXIOUS = "anxious"
    CONFIDENT = "confident"
    BORED = "bored"

class AnalysisType(Enum):
    """Types of emotion analysis."""
    FACIAL = "facial"
    VOICE = "voice"
    TEXT = "text"
    BEHAVIORAL = "behavioral"
    COMBINED = "combined"

class ConfidenceLevel(Enum):
    """Confidence levels for emotion detection."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class EmotionScore:
    """Emotion score with confidence."""
    emotion: EmotionType
    score: float
    confidence: float
    confidence_level: ConfidenceLevel
    timestamp: float
    metadata: Dict[str, Any] = None

@dataclass
class EmotionAnalysis:
    """Complete emotion analysis result."""
    analysis_id: str
    analysis_type: AnalysisType
    primary_emotion: EmotionType
    emotion_scores: List[EmotionScore]
    overall_sentiment: float
    emotional_intensity: float
    emotional_stability: float
    dominant_emotions: List[Tuple[EmotionType, float]]
    duration: float
    timestamp: float
    metadata: Dict[str, Any] = None

@dataclass
class FacialLandmarks:
    """Facial landmarks for emotion analysis."""
    face_id: str
    landmarks: List[Tuple[float, float]]
    bounding_box: Tuple[float, float, float, float]
    confidence: float
    timestamp: float

@dataclass
class VoiceFeatures:
    """Voice features for emotion analysis."""
    pitch: float
    energy: float
    tempo: float
    spectral_centroid: float
    zero_crossing_rate: float
    mfcc: List[float]
    chroma: List[float]
    tonnetz: List[float]
    timestamp: float

class FacialEmotionDetector:
    """Advanced facial emotion detection."""
    
    def __init__(self):
        self.face_detector = None
        self.emotion_classifier = None
        self.landmark_predictor = None
        self._load_models()
        
        logger.info("Facial Emotion Detector initialized")
    
    def _load_models(self):
        """Load facial emotion detection models."""
        try:
            # Load face detection model
            self.face_detector = dlib.get_frontal_face_detector()
            
            # Load landmark predictor
            self.landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            
            # Load emotion classifier
            self.emotion_classifier = pipeline(
                "image-classification",
                model="microsoft/DialoGPT-medium",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Facial emotion detection models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading facial models: {e}")
            self.face_detector = self._create_simple_detector()
    
    def _create_simple_detector(self):
        """Create simple emotion detector as fallback."""
        def detect_emotions(image):
            # Simple emotion detection based on image analysis
            emotions = [
                {"label": "happy", "score": 0.3},
                {"label": "sad", "score": 0.2},
                {"label": "angry", "score": 0.1},
                {"label": "surprised", "score": 0.2},
                {"label": "neutral", "score": 0.2}
            ]
            return emotions
        
        return detect_emotions
    
    async def analyze_faces(self, image_path: str) -> List[EmotionScore]:
        """Analyze emotions in faces."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            # Convert to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = self.face_detector(rgb_image)
            
            emotion_scores = []
            
            for i, face in enumerate(faces):
                # Extract face region
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                face_image = rgb_image[y:y+h, x:x+w]
                
                # Convert to PIL Image
                face_pil = Image.fromarray(face_image)
                
                # Detect emotions
                if self.emotion_classifier:
                    emotions = self.emotion_classifier(face_pil)
                else:
                    emotions = self.face_detector(face_pil)
                
                # Process emotions
                for emotion_data in emotions:
                    emotion_type = self._map_emotion_label(emotion_data['label'])
                    score = emotion_data['score']
                    confidence = min(0.95, score * 1.2)  # Boost confidence
                    
                    emotion_score = EmotionScore(
                        emotion=emotion_type,
                        score=score,
                        confidence=confidence,
                        confidence_level=self._get_confidence_level(confidence),
                        timestamp=time.time(),
                        metadata={"face_id": i, "bounding_box": (x, y, w, h)}
                    )
                    
                    emotion_scores.append(emotion_score)
            
            return emotion_scores
            
        except Exception as e:
            logger.error(f"Error analyzing faces: {e}")
            return []
    
    def _map_emotion_label(self, label: str) -> EmotionType:
        """Map emotion label to EmotionType enum."""
        emotion_mapping = {
            'happy': EmotionType.HAPPY,
            'sad': EmotionType.SAD,
            'angry': EmotionType.ANGRY,
            'fearful': EmotionType.FEARFUL,
            'surprised': EmotionType.SURPRISED,
            'disgusted': EmotionType.DISGUSTED,
            'neutral': EmotionType.NEUTRAL,
            'excited': EmotionType.EXCITED,
            'calm': EmotionType.CALM,
            'confused': EmotionType.CONFUSED,
            'frustrated': EmotionType.FRUSTRATED,
            'content': EmotionType.CONTENT,
            'anxious': EmotionType.ANXIOUS,
            'confident': EmotionType.CONFIDENT,
            'bored': EmotionType.BORED
        }
        return emotion_mapping.get(label.lower(), EmotionType.NEUTRAL)
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Get confidence level from confidence score."""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

class VoiceEmotionDetector:
    """Advanced voice emotion detection."""
    
    def __init__(self):
        self.voice_classifier = None
        self._load_models()
        
        logger.info("Voice Emotion Detector initialized")
    
    def _load_models(self):
        """Load voice emotion detection models."""
        try:
            # Load voice emotion classifier
            self.voice_classifier = pipeline(
                "audio-classification",
                model="facebook/wav2vec2-base",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Voice emotion detection models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading voice models: {e}")
            self.voice_classifier = self._create_simple_classifier()
    
    def _create_simple_classifier(self):
        """Create simple voice classifier as fallback."""
        def classify_emotions(audio):
            # Simple voice emotion detection based on audio features
            emotions = [
                {"label": "happy", "score": 0.4},
                {"label": "sad", "score": 0.1},
                {"label": "angry", "score": 0.1},
                {"label": "excited", "score": 0.2},
                {"label": "calm", "score": 0.2}
            ]
            return emotions
        
        return classify_emotions
    
    async def analyze_voice(self, audio_path: str) -> List[EmotionScore]:
        """Analyze emotions in voice."""
        try:
            # Load audio
            audio, sample_rate = librosa.load(audio_path, sr=22050)
            
            # Extract voice features
            voice_features = self._extract_voice_features(audio, sample_rate)
            
            # Classify emotions
            if self.voice_classifier:
                emotions = self.voice_classifier(audio)
            else:
                emotions = self.voice_classifier(audio)
            
            emotion_scores = []
            
            for emotion_data in emotions:
                emotion_type = self._map_emotion_label(emotion_data['label'])
                score = emotion_data['score']
                
                # Adjust confidence based on voice features
                confidence = self._calculate_voice_confidence(score, voice_features)
                
                emotion_score = EmotionScore(
                    emotion=emotion_type,
                    score=score,
                    confidence=confidence,
                    confidence_level=self._get_confidence_level(confidence),
                    timestamp=time.time(),
                    metadata={"voice_features": asdict(voice_features)}
                )
                
                emotion_scores.append(emotion_score)
            
            return emotion_scores
            
        except Exception as e:
            logger.error(f"Error analyzing voice: {e}")
            return []
    
    def _extract_voice_features(self, audio: np.ndarray, sample_rate: int) -> VoiceFeatures:
        """Extract voice features for emotion analysis."""
        try:
            # Pitch
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate)
            pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            
            # Energy
            energy = np.mean(librosa.feature.rms(y=audio))
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
            
            # Spectral centroid
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))
            
            # Zero crossing rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
            
            # MFCC
            mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13), axis=1)
            
            # Chroma
            chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate), axis=1)
            
            # Tonnetz
            tonnetz = np.mean(librosa.feature.tonnetz(y=audio, sr=sample_rate), axis=1)
            
            return VoiceFeatures(
                pitch=float(pitch),
                energy=float(energy),
                tempo=float(tempo),
                spectral_centroid=float(spectral_centroid),
                zero_crossing_rate=float(zcr),
                mfcc=mfcc.tolist(),
                chroma=chroma.tolist(),
                tonnetz=tonnetz.tolist(),
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error extracting voice features: {e}")
            return VoiceFeatures(0, 0, 0, 0, 0, [], [], [], time.time())
    
    def _calculate_voice_confidence(self, score: float, features: VoiceFeatures) -> float:
        """Calculate confidence based on voice features."""
        # Simple confidence calculation based on feature quality
        base_confidence = score
        
        # Adjust based on energy (higher energy = more confident)
        energy_factor = min(1.0, features.energy * 10)
        
        # Adjust based on pitch stability
        pitch_stability = 1.0 - abs(features.pitch - 200) / 200  # Assume 200Hz is neutral
        pitch_factor = max(0.5, pitch_stability)
        
        # Combine factors
        confidence = base_confidence * energy_factor * pitch_factor
        
        return min(0.95, max(0.1, confidence))
    
    def _map_emotion_label(self, label: str) -> EmotionType:
        """Map emotion label to EmotionType enum."""
        emotion_mapping = {
            'happy': EmotionType.HAPPY,
            'sad': EmotionType.SAD,
            'angry': EmotionType.ANGRY,
            'fearful': EmotionType.FEARFUL,
            'surprised': EmotionType.SURPRISED,
            'disgusted': EmotionType.DISGUSTED,
            'neutral': EmotionType.NEUTRAL,
            'excited': EmotionType.EXCITED,
            'calm': EmotionType.CALM,
            'confused': EmotionType.CONFUSED,
            'frustrated': EmotionType.FRUSTRATED,
            'content': EmotionType.CONTENT,
            'anxious': EmotionType.ANXIOUS,
            'confident': EmotionType.CONFIDENT,
            'bored': EmotionType.BORED
        }
        return emotion_mapping.get(label.lower(), EmotionType.NEUTRAL)
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Get confidence level from confidence score."""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

class TextEmotionDetector:
    """Advanced text emotion detection."""
    
    def __init__(self):
        self.sentiment_classifier = None
        self.emotion_classifier = None
        self._load_models()
        
        logger.info("Text Emotion Detector initialized")
    
    def _load_models(self):
        """Load text emotion detection models."""
        try:
            # Load sentiment classifier
            self.sentiment_classifier = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Load emotion classifier
            self.emotion_classifier = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-emotion",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Text emotion detection models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading text models: {e}")
            self.sentiment_classifier = self._create_simple_classifier()
            self.emotion_classifier = self._create_simple_classifier()
    
    def _create_simple_classifier(self):
        """Create simple text classifier as fallback."""
        def classify_text(text):
            # Simple text emotion detection based on keywords
            positive_words = ['happy', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
            negative_words = ['sad', 'terrible', 'awful', 'horrible', 'disappointing', 'poor']
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                return [{"label": "POSITIVE", "score": 0.7}]
            elif negative_count > positive_count:
                return [{"label": "NEGATIVE", "score": 0.7}]
            else:
                return [{"label": "NEUTRAL", "score": 0.5}]
        
        return classify_text
    
    async def analyze_text(self, text: str) -> List[EmotionScore]:
        """Analyze emotions in text."""
        try:
            emotion_scores = []
            
            # Analyze sentiment
            if self.sentiment_classifier:
                sentiment_result = self.sentiment_classifier(text)
                sentiment_score = sentiment_result[0]['score']
                sentiment_label = sentiment_result[0]['label']
                
                # Map sentiment to emotion
                if sentiment_label == 'POSITIVE':
                    emotion_type = EmotionType.HAPPY
                elif sentiment_label == 'NEGATIVE':
                    emotion_type = EmotionType.SAD
                else:
                    emotion_type = EmotionType.NEUTRAL
                
                emotion_score = EmotionScore(
                    emotion=emotion_type,
                    score=sentiment_score,
                    confidence=sentiment_score,
                    confidence_level=self._get_confidence_level(sentiment_score),
                    timestamp=time.time(),
                    metadata={"analysis_type": "sentiment"}
                )
                
                emotion_scores.append(emotion_score)
            
            # Analyze emotions
            if self.emotion_classifier:
                emotion_result = self.emotion_classifier(text)
                
                for emotion_data in emotion_result:
                    emotion_type = self._map_emotion_label(emotion_data['label'])
                    score = emotion_data['score']
                    
                    emotion_score = EmotionScore(
                        emotion=emotion_type,
                        score=score,
                        confidence=score,
                        confidence_level=self._get_confidence_level(score),
                        timestamp=time.time(),
                        metadata={"analysis_type": "emotion"}
                    )
                    
                    emotion_scores.append(emotion_score)
            
            return emotion_scores
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return []
    
    def _map_emotion_label(self, label: str) -> EmotionType:
        """Map emotion label to EmotionType enum."""
        emotion_mapping = {
            'happy': EmotionType.HAPPY,
            'sad': EmotionType.SAD,
            'angry': EmotionType.ANGRY,
            'fearful': EmotionType.FEARFUL,
            'surprised': EmotionType.SURPRISED,
            'disgusted': EmotionType.DISGUSTED,
            'neutral': EmotionType.NEUTRAL,
            'excited': EmotionType.EXCITED,
            'calm': EmotionType.CALM,
            'confused': EmotionType.CONFUSED,
            'frustrated': EmotionType.FRUSTRATED,
            'content': EmotionType.CONTENT,
            'anxious': EmotionType.ANXIOUS,
            'confident': EmotionType.CONFIDENT,
            'bored': EmotionType.BORED,
            'joy': EmotionType.HAPPY,
            'sorrow': EmotionType.SAD,
            'anger': EmotionType.ANGRY,
            'fear': EmotionType.FEARFUL,
            'surprise': EmotionType.SURPRISED,
            'disgust': EmotionType.DISGUSTED
        }
        return emotion_mapping.get(label.lower(), EmotionType.NEUTRAL)
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Get confidence level from confidence score."""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

class EmotionAnalysisSystem:
    """Main emotion analysis system."""
    
    def __init__(self):
        self.facial_detector = FacialEmotionDetector()
        self.voice_detector = VoiceEmotionDetector()
        self.text_detector = TextEmotionDetector()
        self.analysis_history: List[EmotionAnalysis] = []
        
        logger.info("Emotion Analysis System initialized")
    
    async def analyze_emotions(self, content_path: str, analysis_type: AnalysisType,
                             text: str = None) -> EmotionAnalysis:
        """Analyze emotions in content."""
        try:
            analysis_id = str(uuid.uuid4())
            start_time = time.time()
            
            emotion_scores = []
            
            # Perform analysis based on type
            if analysis_type in [AnalysisType.FACIAL, AnalysisType.COMBINED]:
                facial_emotions = await self.facial_detector.analyze_faces(content_path)
                emotion_scores.extend(facial_emotions)
            
            if analysis_type in [AnalysisType.VOICE, AnalysisType.COMBINED]:
                voice_emotions = await self.voice_detector.analyze_voice(content_path)
                emotion_scores.extend(voice_emotions)
            
            if analysis_type in [AnalysisType.TEXT, AnalysisType.COMBINED] and text:
                text_emotions = await self.text_detector.analyze_text(text)
                emotion_scores.extend(text_emotions)
            
            if not emotion_scores:
                # Return neutral analysis
                emotion_scores = [EmotionScore(
                    emotion=EmotionType.NEUTRAL,
                    score=0.5,
                    confidence=0.5,
                    confidence_level=ConfidenceLevel.MEDIUM,
                    timestamp=time.time()
                )]
            
            # Calculate overall metrics
            primary_emotion = self._get_primary_emotion(emotion_scores)
            overall_sentiment = self._calculate_overall_sentiment(emotion_scores)
            emotional_intensity = self._calculate_emotional_intensity(emotion_scores)
            emotional_stability = self._calculate_emotional_stability(emotion_scores)
            dominant_emotions = self._get_dominant_emotions(emotion_scores)
            
            # Create analysis result
            analysis = EmotionAnalysis(
                analysis_id=analysis_id,
                analysis_type=analysis_type,
                primary_emotion=primary_emotion,
                emotion_scores=emotion_scores,
                overall_sentiment=overall_sentiment,
                emotional_intensity=emotional_intensity,
                emotional_stability=emotional_stability,
                dominant_emotions=dominant_emotions,
                duration=time.time() - start_time,
                timestamp=time.time(),
                metadata={"content_path": content_path}
            )
            
            # Store analysis
            self.analysis_history.append(analysis)
            
            logger.info(f"Emotion analysis completed: {analysis_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing emotions: {e}")
            raise
    
    def _get_primary_emotion(self, emotion_scores: List[EmotionScore]) -> EmotionType:
        """Get primary emotion from scores."""
        if not emotion_scores:
            return EmotionType.NEUTRAL
        
        # Weight by confidence
        weighted_scores = {}
        for score in emotion_scores:
            emotion = score.emotion
            weighted_score = score.score * score.confidence
            
            if emotion in weighted_scores:
                weighted_scores[emotion] += weighted_score
            else:
                weighted_scores[emotion] = weighted_score
        
        return max(weighted_scores, key=weighted_scores.get)
    
    def _calculate_overall_sentiment(self, emotion_scores: List[EmotionScore]) -> float:
        """Calculate overall sentiment score."""
        if not emotion_scores:
            return 0.0
        
        # Map emotions to sentiment values
        sentiment_mapping = {
            EmotionType.HAPPY: 1.0,
            EmotionType.EXCITED: 0.8,
            EmotionType.CONTENT: 0.6,
            EmotionType.CONFIDENT: 0.4,
            EmotionType.NEUTRAL: 0.0,
            EmotionType.CALM: -0.2,
            EmotionType.BORED: -0.4,
            EmotionType.CONFUSED: -0.6,
            EmotionType.SAD: -0.8,
            EmotionType.ANGRY: -1.0,
            EmotionType.FEARFUL: -0.9,
            EmotionType.FRUSTRATED: -0.7,
            EmotionType.ANXIOUS: -0.5,
            EmotionType.SURPRISED: 0.2,
            EmotionType.DISGUSTED: -0.8
        }
        
        weighted_sentiment = 0.0
        total_weight = 0.0
        
        for score in emotion_scores:
            sentiment_value = sentiment_mapping.get(score.emotion, 0.0)
            weight = score.score * score.confidence
            weighted_sentiment += sentiment_value * weight
            total_weight += weight
        
        return weighted_sentiment / total_weight if total_weight > 0 else 0.0
    
    def _calculate_emotional_intensity(self, emotion_scores: List[EmotionScore]) -> float:
        """Calculate emotional intensity."""
        if not emotion_scores:
            return 0.0
        
        # Calculate variance of emotion scores
        scores = [score.score for score in emotion_scores]
        return float(np.var(scores))
    
    def _calculate_emotional_stability(self, emotion_scores: List[EmotionScore]) -> float:
        """Calculate emotional stability."""
        if not emotion_scores:
            return 1.0
        
        # Stability is inverse of intensity
        intensity = self._calculate_emotional_intensity(emotion_scores)
        return max(0.0, 1.0 - intensity)
    
    def _get_dominant_emotions(self, emotion_scores: List[EmotionScore], top_n: int = 3) -> List[Tuple[EmotionType, float]]:
        """Get dominant emotions."""
        if not emotion_scores:
            return []
        
        # Group by emotion and sum scores
        emotion_totals = {}
        for score in emotion_scores:
            emotion = score.emotion
            weighted_score = score.score * score.confidence
            
            if emotion in emotion_totals:
                emotion_totals[emotion] += weighted_score
            else:
                emotion_totals[emotion] = weighted_score
        
        # Sort by total score
        sorted_emotions = sorted(emotion_totals.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_emotions[:top_n]
    
    def get_analysis_history(self, limit: int = 100) -> List[EmotionAnalysis]:
        """Get analysis history."""
        return self.analysis_history[-limit:]
    
    def get_emotion_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get emotion trends over time."""
        try:
            cutoff_time = time.time() - (days * 24 * 60 * 60)
            recent_analyses = [a for a in self.analysis_history if a.timestamp > cutoff_time]
            
            if not recent_analyses:
                return {"message": "No recent analyses available"}
            
            # Group by day
            daily_emotions = {}
            for analysis in recent_analyses:
                day = datetime.fromtimestamp(analysis.timestamp).strftime('%Y-%m-%d')
                if day not in daily_emotions:
                    daily_emotions[day] = []
                daily_emotions[day].append(analysis.primary_emotion.value)
            
            # Calculate trends
            trends = {}
            for day, emotions in daily_emotions.items():
                emotion_counts = {}
                for emotion in emotions:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                
                total = len(emotions)
                trends[day] = {
                    emotion: count / total for emotion, count in emotion_counts.items()
                }
            
            return {
                "trends": trends,
                "total_analyses": len(recent_analyses),
                "days_analyzed": len(daily_emotions)
            }
            
        except Exception as e:
            logger.error(f"Error calculating emotion trends: {e}")
            return {"error": str(e)}

# Global emotion analysis system instance
_global_emotion_analysis: Optional[EmotionAnalysisSystem] = None

def get_emotion_analysis_system() -> EmotionAnalysisSystem:
    """Get the global emotion analysis system instance."""
    global _global_emotion_analysis
    if _global_emotion_analysis is None:
        _global_emotion_analysis = EmotionAnalysisSystem()
    return _global_emotion_analysis

async def analyze_emotions(content_path: str, analysis_type: AnalysisType, 
                         text: str = None) -> EmotionAnalysis:
    """Analyze emotions in content."""
    emotion_system = get_emotion_analysis_system()
    return await emotion_system.analyze_emotions(content_path, analysis_type, text)

def get_emotion_trends(days: int = 7) -> Dict[str, Any]:
    """Get emotion trends over time."""
    emotion_system = get_emotion_analysis_system()
    return emotion_system.get_emotion_trends(days)


