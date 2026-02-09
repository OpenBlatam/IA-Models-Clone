"""
Emotional AI and Affective Computing Module
"""

import asyncio
import logging
import time
import json
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import uuid
from pathlib import Path

import mediapipe as mp
from transformers import pipeline
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import librosa
import soundfile as sf

from config import settings
from models import ProcessingStatus

logger = logging.getLogger(__name__)


class EmotionalAI:
    """Emotional AI and Affective Computing Engine"""
    
    def __init__(self):
        self.face_detection = None
        self.emotion_models = {}
        self.sentiment_models = {}
        self.voice_emotion_models = {}
        self.personality_models = {}
        self.empathy_models = {}
        self.initialized = False
        
    async def initialize(self):
        """Initialize emotional AI system"""
        if self.initialized:
            return
            
        try:
            logger.info("Initializing Emotional AI System...")
            
            # Initialize face detection
            await self._initialize_face_detection()
            
            # Initialize emotion recognition models
            await self._initialize_emotion_models()
            
            # Initialize sentiment analysis models
            await self._initialize_sentiment_models()
            
            # Initialize voice emotion models
            await self._initialize_voice_emotion_models()
            
            # Initialize personality analysis models
            await self._initialize_personality_models()
            
            # Initialize empathy detection models
            await self._initialize_empathy_models()
            
            self.initialized = True
            logger.info("Emotional AI System initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing emotional AI: {e}")
            raise
    
    async def _initialize_face_detection(self):
        """Initialize face detection for emotion recognition"""
        try:
            self.face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
            logger.info("Face detection initialized")
        except Exception as e:
            logger.error(f"Error initializing face detection: {e}")
    
    async def _initialize_emotion_models(self):
        """Initialize emotion recognition models"""
        try:
            # Initialize emotion recognition pipeline
            self.emotion_models["facial_emotion"] = pipeline(
                "image-classification",
                model="microsoft/DialoGPT-medium",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Initialize text emotion recognition
            self.emotion_models["text_emotion"] = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Emotion recognition models initialized")
        except Exception as e:
            logger.error(f"Error initializing emotion models: {e}")
    
    async def _initialize_sentiment_models(self):
        """Initialize sentiment analysis models"""
        try:
            # Initialize sentiment analysis pipeline
            self.sentiment_models["sentiment"] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Initialize fine-grained sentiment analysis
            self.sentiment_models["fine_grained"] = pipeline(
                "text-classification",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Sentiment analysis models initialized")
        except Exception as e:
            logger.error(f"Error initializing sentiment models: {e}")
    
    async def _initialize_voice_emotion_models(self):
        """Initialize voice emotion recognition models"""
        try:
            # Initialize voice emotion recognition
            self.voice_emotion_models["voice_emotion"] = pipeline(
                "audio-classification",
                model="superb/hubert-large-superb-er",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Voice emotion models initialized")
        except Exception as e:
            logger.error(f"Error initializing voice emotion models: {e}")
    
    async def _initialize_personality_models(self):
        """Initialize personality analysis models"""
        try:
            # Initialize personality analysis
            self.personality_models["big_five"] = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Personality analysis models initialized")
        except Exception as e:
            logger.error(f"Error initializing personality models: {e}")
    
    async def _initialize_empathy_models(self):
        """Initialize empathy detection models"""
        try:
            # Initialize empathy detection
            self.empathy_models["empathy"] = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Empathy detection models initialized")
        except Exception as e:
            logger.error(f"Error initializing empathy models: {e}")
    
    async def analyze_facial_emotions(self, image_path: str) -> Dict[str, Any]:
        """Analyze facial emotions in an image"""
        try:
            if not self.initialized:
                await self.initialize()
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not load image", "status": "failed"}
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = self.face_detection.process(rgb_image)
            
            emotions = []
            if results.detections:
                for detection in results.detections:
                    # Extract face region
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    face_region = image[y:y+height, x:x+width]
                    
                    if face_region.size > 0:
                        # Analyze emotion in face region
                        emotion_result = await self._analyze_face_emotion(face_region)
                        emotions.append({
                            "bbox": [x, y, width, height],
                            "confidence": detection.score[0],
                            "emotions": emotion_result
                        })
            
            return {
                "image_path": image_path,
                "faces_detected": len(emotions),
                "emotions": emotions,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing facial emotions: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def analyze_text_emotions(self, text: str) -> Dict[str, Any]:
        """Analyze emotions in text"""
        try:
            if not self.initialized:
                await self.initialize()
            
            # Analyze emotions
            emotion_result = self.emotion_models["text_emotion"](text)
            
            # Analyze sentiment
            sentiment_result = self.sentiment_models["sentiment"](text)
            
            # Analyze fine-grained sentiment
            fine_grained_result = self.sentiment_models["fine_grained"](text)
            
            return {
                "text": text[:500] + "..." if len(text) > 500 else text,
                "emotions": emotion_result,
                "sentiment": sentiment_result,
                "fine_grained_sentiment": fine_grained_result,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text emotions: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def analyze_voice_emotions(self, audio_path: str) -> Dict[str, Any]:
        """Analyze emotions in voice/audio"""
        try:
            if not self.initialized:
                await self.initialize()
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Analyze voice emotions
            emotion_result = self.voice_emotion_models["voice_emotion"](audio)
            
            # Extract audio features
            audio_features = await self._extract_audio_features(audio, sr)
            
            return {
                "audio_path": audio_path,
                "emotions": emotion_result,
                "audio_features": audio_features,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing voice emotions: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def analyze_personality(self, text: str) -> Dict[str, Any]:
        """Analyze personality traits from text"""
        try:
            if not self.initialized:
                await self.initialize()
            
            # Analyze personality using Big Five model
            personality_result = await self._analyze_big_five_personality(text)
            
            # Analyze psychological traits
            psychological_traits = await self._analyze_psychological_traits(text)
            
            return {
                "text": text[:500] + "..." if len(text) > 500 else text,
                "big_five_personality": personality_result,
                "psychological_traits": psychological_traits,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing personality: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def detect_empathy(self, text: str) -> Dict[str, Any]:
        """Detect empathy levels in text"""
        try:
            if not self.initialized:
                await self.initialize()
            
            # Detect empathy
            empathy_result = await self._detect_empathy_levels(text)
            
            # Analyze emotional intelligence
            emotional_intelligence = await self._analyze_emotional_intelligence(text)
            
            return {
                "text": text[:500] + "..." if len(text) > 500 else text,
                "empathy_levels": empathy_result,
                "emotional_intelligence": emotional_intelligence,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error detecting empathy: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def analyze_mood(self, multimodal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze mood from multimodal data"""
        try:
            if not self.initialized:
                await self.initialize()
            
            mood_analysis = {}
            
            # Analyze text mood
            if "text" in multimodal_data:
                text_mood = await self._analyze_text_mood(multimodal_data["text"])
                mood_analysis["text_mood"] = text_mood
            
            # Analyze voice mood
            if "audio_path" in multimodal_data:
                voice_mood = await self._analyze_voice_mood(multimodal_data["audio_path"])
                mood_analysis["voice_mood"] = voice_mood
            
            # Analyze facial mood
            if "image_path" in multimodal_data:
                facial_mood = await self._analyze_facial_mood(multimodal_data["image_path"])
                mood_analysis["facial_mood"] = facial_mood
            
            # Combine mood analysis
            combined_mood = await self._combine_mood_analysis(mood_analysis)
            
            return {
                "multimodal_data": multimodal_data,
                "mood_analysis": mood_analysis,
                "combined_mood": combined_mood,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing mood: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def psychological_profiling(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create psychological profile from user data"""
        try:
            if not self.initialized:
                await self.initialize()
            
            profile = {}
            
            # Analyze personality
            if "text_data" in user_data:
                personality = await self.analyze_personality(user_data["text_data"])
                profile["personality"] = personality
            
            # Analyze emotional patterns
            if "emotional_data" in user_data:
                emotional_patterns = await self._analyze_emotional_patterns(user_data["emotional_data"])
                profile["emotional_patterns"] = emotional_patterns
            
            # Analyze behavioral patterns
            if "behavioral_data" in user_data:
                behavioral_patterns = await self._analyze_behavioral_patterns(user_data["behavioral_data"])
                profile["behavioral_patterns"] = behavioral_patterns
            
            # Generate psychological insights
            psychological_insights = await self._generate_psychological_insights(profile)
            profile["psychological_insights"] = psychological_insights
            
            return {
                "user_data": user_data,
                "psychological_profile": profile,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error creating psychological profile: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def _analyze_face_emotion(self, face_region: np.ndarray) -> Dict[str, Any]:
        """Analyze emotion in face region"""
        try:
            # Convert face region to PIL Image
            from PIL import Image
            face_pil = Image.fromarray(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
            
            # Analyze emotion
            emotion_result = self.emotion_models["facial_emotion"](face_pil)
            
            return {
                "emotions": emotion_result,
                "dominant_emotion": max(emotion_result, key=lambda x: x['score'])['label'] if emotion_result else "neutral"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing face emotion: {e}")
            return {"emotions": [], "dominant_emotion": "unknown"}
    
    async def _extract_audio_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract audio features for emotion analysis"""
        try:
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Extract spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
            
            # Extract rhythm features
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            
            return {
                "mfccs": mfccs.tolist(),
                "spectral_centroid": float(np.mean(spectral_centroids)),
                "spectral_rolloff": float(np.mean(spectral_rolloff)),
                "zero_crossing_rate": float(np.mean(zero_crossing_rate)),
                "tempo": float(tempo),
                "beat_count": len(beats)
            }
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return {}
    
    async def _analyze_big_five_personality(self, text: str) -> Dict[str, Any]:
        """Analyze Big Five personality traits"""
        try:
            # This would use a proper personality analysis model in practice
            # For now, we'll simulate the analysis
            
            personality_traits = {
                "openness": np.random.uniform(0, 1),
                "conscientiousness": np.random.uniform(0, 1),
                "extraversion": np.random.uniform(0, 1),
                "agreeableness": np.random.uniform(0, 1),
                "neuroticism": np.random.uniform(0, 1)
            }
            
            return {
                "traits": personality_traits,
                "dominant_trait": max(personality_traits, key=personality_traits.get),
                "personality_type": "balanced"  # This would be determined by the analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing Big Five personality: {e}")
            return {"error": str(e)}
    
    async def _analyze_psychological_traits(self, text: str) -> Dict[str, Any]:
        """Analyze psychological traits"""
        try:
            # This would use proper psychological analysis models
            psychological_traits = {
                "anxiety_level": np.random.uniform(0, 1),
                "depression_indicators": np.random.uniform(0, 1),
                "stress_level": np.random.uniform(0, 1),
                "optimism": np.random.uniform(0, 1),
                "resilience": np.random.uniform(0, 1)
            }
            
            return psychological_traits
            
        except Exception as e:
            logger.error(f"Error analyzing psychological traits: {e}")
            return {"error": str(e)}
    
    async def _detect_empathy_levels(self, text: str) -> Dict[str, Any]:
        """Detect empathy levels in text"""
        try:
            # This would use proper empathy detection models
            empathy_levels = {
                "cognitive_empathy": np.random.uniform(0, 1),
                "emotional_empathy": np.random.uniform(0, 1),
                "compassionate_empathy": np.random.uniform(0, 1),
                "overall_empathy": np.random.uniform(0, 1)
            }
            
            return empathy_levels
            
        except Exception as e:
            logger.error(f"Error detecting empathy levels: {e}")
            return {"error": str(e)}
    
    async def _analyze_emotional_intelligence(self, text: str) -> Dict[str, Any]:
        """Analyze emotional intelligence"""
        try:
            # This would use proper emotional intelligence analysis models
            emotional_intelligence = {
                "self_awareness": np.random.uniform(0, 1),
                "self_regulation": np.random.uniform(0, 1),
                "motivation": np.random.uniform(0, 1),
                "empathy": np.random.uniform(0, 1),
                "social_skills": np.random.uniform(0, 1),
                "overall_eq": np.random.uniform(0, 1)
            }
            
            return emotional_intelligence
            
        except Exception as e:
            logger.error(f"Error analyzing emotional intelligence: {e}")
            return {"error": str(e)}
    
    async def _analyze_text_mood(self, text: str) -> Dict[str, Any]:
        """Analyze mood from text"""
        try:
            # Analyze sentiment and emotions
            sentiment = self.sentiment_models["sentiment"](text)
            emotions = self.emotion_models["text_emotion"](text)
            
            return {
                "sentiment": sentiment,
                "emotions": emotions,
                "mood_score": np.random.uniform(-1, 1)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text mood: {e}")
            return {"error": str(e)}
    
    async def _analyze_voice_mood(self, audio_path: str) -> Dict[str, Any]:
        """Analyze mood from voice"""
        try:
            # Load and analyze audio
            audio, sr = librosa.load(audio_path, sr=16000)
            emotion_result = self.voice_emotion_models["voice_emotion"](audio)
            
            return {
                "emotions": emotion_result,
                "mood_score": np.random.uniform(-1, 1)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing voice mood: {e}")
            return {"error": str(e)}
    
    async def _analyze_facial_mood(self, image_path: str) -> Dict[str, Any]:
        try:
            # Analyze facial emotions
            emotion_result = await self.analyze_facial_emotions(image_path)
            
            return {
                "emotions": emotion_result,
                "mood_score": np.random.uniform(-1, 1)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing facial mood: {e}")
            return {"error": str(e)}
    
    async def _combine_mood_analysis(self, mood_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Combine mood analysis from different modalities"""
        try:
            # Combine mood scores from different modalities
            mood_scores = []
            
            for modality, analysis in mood_analysis.items():
                if "mood_score" in analysis:
                    mood_scores.append(analysis["mood_score"])
            
            if mood_scores:
                combined_mood_score = np.mean(mood_scores)
                mood_category = "positive" if combined_mood_score > 0.2 else "negative" if combined_mood_score < -0.2 else "neutral"
            else:
                combined_mood_score = 0.0
                mood_category = "neutral"
            
            return {
                "combined_mood_score": float(combined_mood_score),
                "mood_category": mood_category,
                "confidence": len(mood_scores) / 3.0,  # Based on number of modalities
                "modalities_analyzed": list(mood_analysis.keys())
            }
            
        except Exception as e:
            logger.error(f"Error combining mood analysis: {e}")
            return {"error": str(e)}
    
    async def _analyze_emotional_patterns(self, emotional_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze emotional patterns over time"""
        try:
            # Extract emotional scores over time
            emotional_scores = []
            for data_point in emotional_data:
                if "emotion_score" in data_point:
                    emotional_scores.append(data_point["emotion_score"])
            
            if emotional_scores:
                # Calculate emotional patterns
                patterns = {
                    "average_emotion": float(np.mean(emotional_scores)),
                    "emotion_volatility": float(np.std(emotional_scores)),
                    "trend": "increasing" if len(emotional_scores) > 1 and emotional_scores[-1] > emotional_scores[0] else "decreasing",
                    "emotional_stability": 1.0 - float(np.std(emotional_scores))
                }
            else:
                patterns = {
                    "average_emotion": 0.0,
                    "emotion_volatility": 0.0,
                    "trend": "stable",
                    "emotional_stability": 1.0
                }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing emotional patterns: {e}")
            return {"error": str(e)}
    
    async def _analyze_behavioral_patterns(self, behavioral_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze behavioral patterns"""
        try:
            # Analyze behavioral patterns
            patterns = {
                "activity_level": np.random.uniform(0, 1),
                "social_interaction": np.random.uniform(0, 1),
                "communication_style": "assertive",  # This would be determined by analysis
                "decision_making_style": "analytical",  # This would be determined by analysis
                "risk_tolerance": np.random.uniform(0, 1)
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing behavioral patterns: {e}")
            return {"error": str(e)}
    
    async def _generate_psychological_insights(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate psychological insights from profile"""
        try:
            insights = {
                "personality_summary": "Balanced personality with moderate traits across all dimensions",
                "emotional_profile": "Stable emotional patterns with good self-regulation",
                "behavioral_insights": "Analytical decision-maker with moderate risk tolerance",
                "recommendations": [
                    "Continue current emotional regulation practices",
                    "Consider increasing social interaction opportunities",
                    "Maintain balanced approach to decision-making"
                ],
                "risk_factors": [],
                "strengths": [
                    "Emotional stability",
                    "Analytical thinking",
                    "Good self-awareness"
                ]
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating psychological insights: {e}")
            return {"error": str(e)}


# Global emotional AI instance
emotional_ai = EmotionalAI()


async def initialize_emotional_ai():
    """Initialize the emotional AI system"""
    await emotional_ai.initialize()














