"""
Multimodal Analysis System
=========================

This module provides advanced multimodal analysis capabilities including:
- Image analysis and computer vision
- Video processing and analysis
- Audio processing and speech recognition
- Cross-modal understanding
- Content generation analysis
- Quality assessment across modalities
- Real-time multimodal processing
- Advanced feature extraction
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import os
import cv2
import librosa
import torch
import torchvision.transforms as transforms
from PIL import Image
import base64
import io
from transformers import (
    pipeline, AutoTokenizer, AutoModel, 
    BlipProcessor, BlipForConditionalGeneration,
    CLIPProcessor, CLIPModel
)
import whisper
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# Import our AI history analyzer components
from .ai_history_analyzer import (
    AIHistoryAnalyzer, ModelType, PerformanceMetric,
    get_ai_history_analyzer
)
from .config import get_ai_history_config

logger = logging.getLogger(__name__)


@dataclass
class ImageAnalysis:
    """Image analysis result"""
    image_path: str
    objects_detected: List[Dict[str, Any]]
    scene_description: str
    quality_metrics: Dict[str, float]
    aesthetic_score: float
    technical_quality: float
    content_safety: str
    dominant_colors: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class VideoAnalysis:
    """Video analysis result"""
    video_path: str
    duration: float
    frame_count: int
    fps: float
    scene_changes: List[float]
    objects_tracked: List[Dict[str, Any]]
    audio_analysis: Dict[str, Any]
    quality_metrics: Dict[str, float]
    content_summary: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AudioAnalysis:
    """Audio analysis result"""
    audio_path: str
    duration: float
    sample_rate: int
    transcription: str
    language: str
    sentiment: str
    emotion: str
    quality_metrics: Dict[str, float]
    speaker_diarization: List[Dict[str, Any]]
    music_analysis: Dict[str, Any]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MultimodalAnalysis:
    """Multimodal analysis result"""
    content_id: str
    modalities: List[str]
    image_analysis: Optional[ImageAnalysis] = None
    video_analysis: Optional[VideoAnalysis] = None
    audio_analysis: Optional[AudioAnalysis] = None
    cross_modal_similarity: float
    content_coherence: float
    overall_quality: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MultimodalAnalysisSystem:
    """Advanced multimodal analysis system"""
    
    def __init__(self, model_storage_path: str = "multimodal_models"):
        self.model_storage_path = model_storage_path
        self.analyzer = get_ai_history_analyzer()
        self.config = get_ai_history_config()
        
        # Analysis models
        self.image_models: Dict[str, Any] = {}
        self.video_models: Dict[str, Any] = {}
        self.audio_models: Dict[str, Any] = {}
        self.multimodal_models: Dict[str, Any] = {}
        
        # Processing tools
        self.image_processor = None
        self.video_processor = None
        self.audio_processor = None
        self.whisper_model = None
        self.clip_model = None
        self.blip_model = None
        
        # Analysis history
        self.analysis_history: List[Dict[str, Any]] = []
        
        # Ensure model storage directory exists
        os.makedirs(model_storage_path, exist_ok=True)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize multimodal analysis models"""
        try:
            # Initialize image analysis models
            try:
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                logger.info("CLIP model loaded for image analysis")
            except Exception as e:
                logger.warning(f"Could not load CLIP model: {str(e)}")
            
            try:
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                logger.info("BLIP model loaded for image captioning")
            except Exception as e:
                logger.warning(f"Could not load BLIP model: {str(e)}")
            
            # Initialize audio analysis models
            try:
                self.whisper_model = whisper.load_model("base")
                logger.info("Whisper model loaded for speech recognition")
            except Exception as e:
                logger.warning(f"Could not load Whisper model: {str(e)}")
            
            # Initialize object detection
            try:
                self.object_detector = pipeline("object-detection", model="facebook/detr-resnet-50")
                logger.info("Object detection model loaded")
            except Exception as e:
                logger.warning(f"Could not load object detection model: {str(e)}")
            
            logger.info("Multimodal analysis models initialized")
        
        except Exception as e:
            logger.error(f"Error initializing multimodal models: {str(e)}")
    
    async def analyze_image(self, image_path: str) -> ImageAnalysis:
        """Analyze image content and quality"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Object detection
            objects_detected = await self._detect_objects(image)
            
            # Scene description
            scene_description = await self._describe_scene(image)
            
            # Quality metrics
            quality_metrics = self._calculate_image_quality(image)
            
            # Aesthetic score
            aesthetic_score = self._calculate_aesthetic_score(image)
            
            # Technical quality
            technical_quality = self._calculate_technical_quality(image)
            
            # Content safety
            content_safety = self._assess_content_safety(image)
            
            # Dominant colors
            dominant_colors = self._extract_dominant_colors(image)
            
            return ImageAnalysis(
                image_path=image_path,
                objects_detected=objects_detected,
                scene_description=scene_description,
                quality_metrics=quality_metrics,
                aesthetic_score=aesthetic_score,
                technical_quality=technical_quality,
                content_safety=content_safety,
                dominant_colors=dominant_colors,
                metadata={
                    "image_size": image.size,
                    "image_mode": image.mode,
                    "analysis_timestamp": datetime.now().isoformat()
                }
            )
        
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return ImageAnalysis(
                image_path=image_path,
                objects_detected=[],
                scene_description="",
                quality_metrics={},
                aesthetic_score=0.0,
                technical_quality=0.0,
                content_safety="unknown",
                dominant_colors=[],
                metadata={"error": str(e)}
            )
    
    async def _detect_objects(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect objects in image"""
        try:
            if not hasattr(self, 'object_detector'):
                return []
            
            results = self.object_detector(image)
            objects = []
            
            for result in results:
                objects.append({
                    "label": result["label"],
                    "confidence": result["score"],
                    "bbox": result["box"]
                })
            
            return objects
        
        except Exception as e:
            logger.error(f"Error detecting objects: {str(e)}")
            return []
    
    async def _describe_scene(self, image: Image.Image) -> str:
        """Generate scene description"""
        try:
            if not self.blip_model or not self.blip_processor:
                return "Scene description not available"
            
            inputs = self.blip_processor(image, return_tensors="pt")
            out = self.blip_model.generate(**inputs, max_length=50)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            return caption
        
        except Exception as e:
            logger.error(f"Error describing scene: {str(e)}")
            return "Scene description failed"
    
    def _calculate_image_quality(self, image: Image.Image) -> Dict[str, float]:
        """Calculate image quality metrics"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Calculate various quality metrics
            quality_metrics = {}
            
            # Sharpness (Laplacian variance)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            quality_metrics["sharpness"] = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Brightness
            quality_metrics["brightness"] = np.mean(img_array)
            
            # Contrast
            quality_metrics["contrast"] = np.std(img_array)
            
            # Colorfulness
            quality_metrics["colorfulness"] = self._calculate_colorfulness(img_array)
            
            # Noise level (simplified)
            quality_metrics["noise_level"] = self._estimate_noise_level(img_array)
            
            return quality_metrics
        
        except Exception as e:
            logger.error(f"Error calculating image quality: {str(e)}")
            return {}
    
    def _calculate_colorfulness(self, image: np.ndarray) -> float:
        """Calculate colorfulness metric"""
        try:
            # Convert to Lab color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Calculate colorfulness
            a = lab[:, :, 1]
            b = lab[:, :, 2]
            
            colorfulness = np.sqrt(np.var(a) + np.var(b))
            return float(colorfulness)
        
        except Exception:
            return 0.0
    
    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise level in image"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Calculate difference
            diff = cv2.absdiff(gray, blurred)
            
            # Estimate noise
            noise_level = np.mean(diff)
            return float(noise_level)
        
        except Exception:
            return 0.0
    
    def _calculate_aesthetic_score(self, image: Image.Image) -> float:
        """Calculate aesthetic score"""
        try:
            # Simplified aesthetic scoring based on composition
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            
            # Rule of thirds
            rule_of_thirds_score = self._calculate_rule_of_thirds_score(img_array)
            
            # Color harmony
            color_harmony_score = self._calculate_color_harmony_score(img_array)
            
            # Symmetry
            symmetry_score = self._calculate_symmetry_score(img_array)
            
            # Combine scores
            aesthetic_score = (rule_of_thirds_score + color_harmony_score + symmetry_score) / 3
            return float(aesthetic_score)
        
        except Exception:
            return 0.5
    
    def _calculate_rule_of_thirds_score(self, image: np.ndarray) -> float:
        """Calculate rule of thirds score"""
        try:
            height, width = image.shape[:2]
            
            # Define rule of thirds lines
            third_h = height // 3
            third_w = width // 3
            
            # Calculate edge density at rule of thirds lines
            edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 50, 150)
            
            # Check edge density at rule of thirds lines
            score = 0.0
            for i in [third_h, 2 * third_h]:
                score += np.mean(edges[i, :])
            for j in [third_w, 2 * third_w]:
                score += np.mean(edges[:, j])
            
            return min(score / 1000, 1.0)  # Normalize
        
        except Exception:
            return 0.5
    
    def _calculate_color_harmony_score(self, image: np.ndarray) -> float:
        """Calculate color harmony score"""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Calculate hue distribution
            hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            
            # Calculate color harmony based on hue distribution
            # Simplified: prefer images with dominant hues
            max_hue_count = np.max(hue_hist)
            total_pixels = np.sum(hue_hist)
            
            if total_pixels > 0:
                harmony_score = max_hue_count / total_pixels
                return float(harmony_score)
            
            return 0.5
        
        except Exception:
            return 0.5
    
    def _calculate_symmetry_score(self, image: np.ndarray) -> float:
        """Calculate symmetry score"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Check horizontal symmetry
            height, width = gray.shape
            top_half = gray[:height//2, :]
            bottom_half = cv2.flip(gray[height//2:, :], 0)
            
            # Resize if necessary
            if top_half.shape != bottom_half.shape:
                bottom_half = cv2.resize(bottom_half, (top_half.shape[1], top_half.shape[0]))
            
            # Calculate similarity
            similarity = cv2.matchTemplate(top_half, bottom_half, cv2.TM_CCOEFF_NORMED)[0][0]
            return float(max(0, similarity))
        
        except Exception:
            return 0.5
    
    def _calculate_technical_quality(self, image: Image.Image) -> float:
        """Calculate technical quality score"""
        try:
            quality_metrics = self._calculate_image_quality(image)
            
            # Combine quality metrics
            sharpness_score = min(quality_metrics.get("sharpness", 0) / 1000, 1.0)
            contrast_score = min(quality_metrics.get("contrast", 0) / 100, 1.0)
            noise_score = max(0, 1.0 - quality_metrics.get("noise_level", 0) / 50)
            
            technical_quality = (sharpness_score + contrast_score + noise_score) / 3
            return float(technical_quality)
        
        except Exception:
            return 0.5
    
    def _assess_content_safety(self, image: Image.Image) -> str:
        """Assess content safety"""
        try:
            # Simplified content safety assessment
            # In production, this would use specialized models
            
            # Check for inappropriate content based on color analysis
            img_array = np.array(image)
            
            # Check for skin tone dominance (simplified)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            skin_mask = cv2.inRange(hsv, (0, 20, 70), (20, 255, 255))
            skin_ratio = np.sum(skin_mask > 0) / (img_array.shape[0] * img_array.shape[1])
            
            if skin_ratio > 0.3:
                return "review_required"
            else:
                return "safe"
        
        except Exception:
            return "unknown"
    
    def _extract_dominant_colors(self, image: Image.Image) -> List[str]:
        """Extract dominant colors"""
        try:
            # Resize image for faster processing
            image_small = image.resize((150, 150))
            img_array = np.array(image_small)
            
            # Reshape image to be a list of pixels
            pixels = img_array.reshape(-1, 3)
            
            # Use K-means to find dominant colors
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans.fit(pixels)
            
            # Get dominant colors
            colors = kmeans.cluster_centers_.astype(int)
            dominant_colors = [f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}" for color in colors]
            
            return dominant_colors
        
        except Exception:
            return []
    
    async def analyze_video(self, video_path: str) -> VideoAnalysis:
        """Analyze video content and quality"""
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Analyze frames
            scene_changes = await self._detect_scene_changes(cap)
            objects_tracked = await self._track_objects(cap)
            
            # Analyze audio
            audio_analysis = await self._analyze_video_audio(video_path)
            
            # Calculate quality metrics
            quality_metrics = await self._calculate_video_quality(cap)
            
            # Generate content summary
            content_summary = await self._summarize_video_content(objects_tracked, scene_changes)
            
            cap.release()
            
            return VideoAnalysis(
                video_path=video_path,
                duration=duration,
                frame_count=frame_count,
                fps=fps,
                scene_changes=scene_changes,
                objects_tracked=objects_tracked,
                audio_analysis=audio_analysis,
                quality_metrics=quality_metrics,
                content_summary=content_summary,
                metadata={
                    "analysis_timestamp": datetime.now().isoformat(),
                    "video_codec": "unknown"  # Would extract from video metadata
                }
            )
        
        except Exception as e:
            logger.error(f"Error analyzing video: {str(e)}")
            return VideoAnalysis(
                video_path=video_path,
                duration=0.0,
                frame_count=0,
                fps=0.0,
                scene_changes=[],
                objects_tracked=[],
                audio_analysis={},
                quality_metrics={},
                content_summary="",
                metadata={"error": str(e)}
            )
    
    async def _detect_scene_changes(self, cap: cv2.VideoCapture) -> List[float]:
        """Detect scene changes in video"""
        try:
            scene_changes = []
            prev_frame = None
            frame_number = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if prev_frame is not None:
                    # Calculate frame difference
                    diff = cv2.absdiff(prev_frame, frame)
                    diff_score = np.mean(diff)
                    
                    # Threshold for scene change
                    if diff_score > 30:  # Adjust threshold as needed
                        scene_changes.append(frame_number / cap.get(cv2.CAP_PROP_FPS))
                
                prev_frame = frame
                frame_number += 1
                
                # Limit analysis to avoid memory issues
                if frame_number > 1000:
                    break
            
            return scene_changes
        
        except Exception as e:
            logger.error(f"Error detecting scene changes: {str(e)}")
            return []
    
    async def _track_objects(self, cap: cv2.VideoCapture) -> List[Dict[str, Any]]:
        """Track objects in video"""
        try:
            objects_tracked = []
            frame_number = 0
            
            # Sample every 30th frame for object detection
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_number % 30 == 0:  # Sample every 30th frame
                    # Convert to PIL Image for object detection
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Detect objects
                    if hasattr(self, 'object_detector'):
                        results = self.object_detector(pil_image)
                        
                        for result in results:
                            objects_tracked.append({
                                "frame": frame_number,
                                "timestamp": frame_number / cap.get(cv2.CAP_PROP_FPS),
                                "label": result["label"],
                                "confidence": result["score"],
                                "bbox": result["box"]
                            })
                
                frame_number += 1
                
                # Limit analysis to avoid memory issues
                if frame_number > 300:  # Analyze first 300 frames
                    break
            
            return objects_tracked
        
        except Exception as e:
            logger.error(f"Error tracking objects: {str(e)}")
            return []
    
    async def _analyze_video_audio(self, video_path: str) -> Dict[str, Any]:
        """Analyze audio from video"""
        try:
            # Extract audio from video (simplified)
            # In production, would use ffmpeg or similar
            
            audio_analysis = {
                "has_audio": True,  # Would check if audio track exists
                "duration": 0.0,
                "sample_rate": 0,
                "channels": 0,
                "transcription": "",
                "language": "unknown"
            }
            
            return audio_analysis
        
        except Exception as e:
            logger.error(f"Error analyzing video audio: {str(e)}")
            return {}
    
    async def _calculate_video_quality(self, cap: cv2.VideoCapture) -> Dict[str, float]:
        """Calculate video quality metrics"""
        try:
            quality_metrics = {}
            
            # Get video properties
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            quality_metrics["resolution"] = width * height
            quality_metrics["fps"] = fps
            quality_metrics["aspect_ratio"] = width / height if height > 0 else 0
            
            # Analyze first few frames for quality
            frame_count = 0
            total_sharpness = 0
            
            while frame_count < 10:  # Analyze first 10 frames
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate sharpness
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                total_sharpness += sharpness
                
                frame_count += 1
            
            if frame_count > 0:
                quality_metrics["average_sharpness"] = total_sharpness / frame_count
            
            return quality_metrics
        
        except Exception as e:
            logger.error(f"Error calculating video quality: {str(e)}")
            return {}
    
    async def _summarize_video_content(self, objects_tracked: List[Dict], scene_changes: List[float]) -> str:
        """Summarize video content"""
        try:
            # Count object occurrences
            object_counts = {}
            for obj in objects_tracked:
                label = obj["label"]
                object_counts[label] = object_counts.get(label, 0) + 1
            
            # Create summary
            summary_parts = []
            
            if object_counts:
                top_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                summary_parts.append(f"Main objects: {', '.join([obj[0] for obj in top_objects])}")
            
            if scene_changes:
                summary_parts.append(f"Scene changes: {len(scene_changes)}")
            
            return ". ".join(summary_parts) if summary_parts else "Video content analysis completed"
        
        except Exception as e:
            logger.error(f"Error summarizing video content: {str(e)}")
            return "Content summary failed"
    
    async def analyze_audio(self, audio_path: str) -> AudioAnalysis:
        """Analyze audio content and quality"""
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Load audio
            y, sr = librosa.load(audio_path)
            duration = len(y) / sr
            
            # Speech recognition
            transcription = await self._transcribe_audio(audio_path)
            
            # Language detection
            language = await self._detect_language(transcription)
            
            # Sentiment and emotion analysis
            sentiment, emotion = await self._analyze_audio_sentiment(y, sr)
            
            # Quality metrics
            quality_metrics = self._calculate_audio_quality(y, sr)
            
            # Speaker diarization (simplified)
            speaker_diarization = await self._perform_speaker_diarization(y, sr)
            
            # Music analysis
            music_analysis = await self._analyze_music(y, sr)
            
            return AudioAnalysis(
                audio_path=audio_path,
                duration=duration,
                sample_rate=sr,
                transcription=transcription,
                language=language,
                sentiment=sentiment,
                emotion=emotion,
                quality_metrics=quality_metrics,
                speaker_diarization=speaker_diarization,
                music_analysis=music_analysis,
                metadata={
                    "analysis_timestamp": datetime.now().isoformat(),
                    "audio_format": "unknown"  # Would extract from audio metadata
                }
            )
        
        except Exception as e:
            logger.error(f"Error analyzing audio: {str(e)}")
            return AudioAnalysis(
                audio_path=audio_path,
                duration=0.0,
                sample_rate=0,
                transcription="",
                language="unknown",
                sentiment="neutral",
                emotion="neutral",
                quality_metrics={},
                speaker_diarization=[],
                music_analysis={},
                metadata={"error": str(e)}
            )
    
    async def _transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio to text"""
        try:
            if not self.whisper_model:
                return "Transcription not available"
            
            result = self.whisper_model.transcribe(audio_path)
            return result["text"]
        
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return ""
    
    async def _detect_language(self, text: str) -> str:
        """Detect language from text"""
        try:
            if not text:
                return "unknown"
            
            # Simplified language detection
            # In production, would use specialized language detection models
            
            # Basic heuristics
            if any(char in text for char in "ñáéíóúü"):
                return "spanish"
            elif any(char in text for char in "àâäéèêëïîôöùûüÿç"):
                return "french"
            elif any(char in text for char in "äöüß"):
                return "german"
            else:
                return "english"
        
        except Exception:
            return "unknown"
    
    async def _analyze_audio_sentiment(self, y: np.ndarray, sr: int) -> Tuple[str, str]:
        """Analyze sentiment and emotion from audio"""
        try:
            # Extract audio features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            
            # Simplified sentiment analysis based on audio features
            avg_mfcc = np.mean(mfccs)
            avg_spectral_centroid = np.mean(spectral_centroids)
            
            # Determine sentiment based on features
            if avg_spectral_centroid > 2000:
                sentiment = "positive"
                emotion = "happy"
            elif avg_spectral_centroid < 1000:
                sentiment = "negative"
                emotion = "sad"
            else:
                sentiment = "neutral"
                emotion = "neutral"
            
            return sentiment, emotion
        
        except Exception:
            return "neutral", "neutral"
    
    def _calculate_audio_quality(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Calculate audio quality metrics"""
        try:
            quality_metrics = {}
            
            # Signal-to-noise ratio (simplified)
            signal_power = np.mean(y ** 2)
            noise_floor = np.percentile(y, 10) ** 2
            snr = 10 * np.log10(signal_power / (noise_floor + 1e-10))
            quality_metrics["snr"] = float(snr)
            
            # Dynamic range
            dynamic_range = np.max(y) - np.min(y)
            quality_metrics["dynamic_range"] = float(dynamic_range)
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            quality_metrics["spectral_rolloff"] = float(np.mean(spectral_rolloff))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            quality_metrics["zero_crossing_rate"] = float(np.mean(zcr))
            
            return quality_metrics
        
        except Exception:
            return {}
    
    async def _perform_speaker_diarization(self, y: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Perform speaker diarization"""
        try:
            # Simplified speaker diarization
            # In production, would use specialized models like pyannote.audio
            
            # Segment audio into chunks
            chunk_length = 3.0  # 3 seconds
            chunk_samples = int(chunk_length * sr)
            
            speakers = []
            for i in range(0, len(y), chunk_samples):
                chunk = y[i:i + chunk_samples]
                if len(chunk) > 0:
                    speakers.append({
                        "start_time": i / sr,
                        "end_time": (i + len(chunk)) / sr,
                        "speaker_id": f"speaker_{i // chunk_samples}",
                        "confidence": 0.8  # Placeholder
                    })
            
            return speakers
        
        except Exception:
            return []
    
    async def _analyze_music(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze music characteristics"""
        try:
            music_analysis = {}
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            music_analysis["tempo"] = float(tempo)
            
            # Key
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            key_profile = np.mean(chroma, axis=1)
            key = np.argmax(key_profile)
            music_analysis["key"] = int(key)
            
            # Genre classification (simplified)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            
            # Simple genre classification based on MFCC features
            if mfcc_mean[0] > 0:
                music_analysis["genre"] = "electronic"
            elif mfcc_mean[1] > 0:
                music_analysis["genre"] = "classical"
            else:
                music_analysis["genre"] = "unknown"
            
            return music_analysis
        
        except Exception:
            return {}
    
    async def analyze_multimodal(self, content_path: str, content_type: str) -> MultimodalAnalysis:
        """Perform comprehensive multimodal analysis"""
        try:
            content_id = f"{content_type}_{os.path.basename(content_path)}"
            modalities = []
            
            image_analysis = None
            video_analysis = None
            audio_analysis = None
            
            # Analyze based on content type
            if content_type in ["image", "video"]:
                if content_type == "image":
                    image_analysis = await self.analyze_image(content_path)
                    modalities.append("image")
                else:
                    video_analysis = await self.analyze_video(content_path)
                    modalities.append("video")
            
            if content_type in ["audio", "video"]:
                if content_type == "audio":
                    audio_analysis = await self.analyze_audio(content_path)
                    modalities.append("audio")
                elif video_analysis and video_analysis.audio_analysis:
                    # Extract audio from video for analysis
                    audio_analysis = await self.analyze_audio(content_path)
                    modalities.append("audio")
            
            # Calculate cross-modal similarity
            cross_modal_similarity = await self._calculate_cross_modal_similarity(
                image_analysis, video_analysis, audio_analysis
            )
            
            # Calculate content coherence
            content_coherence = await self._calculate_content_coherence(
                image_analysis, video_analysis, audio_analysis
            )
            
            # Calculate overall quality
            overall_quality = await self._calculate_overall_quality(
                image_analysis, video_analysis, audio_analysis
            )
            
            result = MultimodalAnalysis(
                content_id=content_id,
                modalities=modalities,
                image_analysis=image_analysis,
                video_analysis=video_analysis,
                audio_analysis=audio_analysis,
                cross_modal_similarity=cross_modal_similarity,
                content_coherence=content_coherence,
                overall_quality=overall_quality,
                metadata={
                    "content_type": content_type,
                    "analysis_timestamp": datetime.now().isoformat()
                }
            )
            
            # Store in history
            self.analysis_history.append(asdict(result))
            
            return result
        
        except Exception as e:
            logger.error(f"Error in multimodal analysis: {str(e)}")
            return MultimodalAnalysis(
                content_id=content_path,
                modalities=[],
                cross_modal_similarity=0.0,
                content_coherence=0.0,
                overall_quality=0.0,
                metadata={"error": str(e)}
            )
    
    async def _calculate_cross_modal_similarity(self, image_analysis, video_analysis, audio_analysis) -> float:
        """Calculate cross-modal similarity"""
        try:
            similarities = []
            
            # Compare image and video content
            if image_analysis and video_analysis:
                # Compare scene descriptions
                if image_analysis.scene_description and video_analysis.content_summary:
                    # Simplified similarity calculation
                    similarity = 0.7  # Placeholder
                    similarities.append(similarity)
            
            # Compare audio and visual content
            if audio_analysis and (image_analysis or video_analysis):
                # Compare sentiment/emotion
                if audio_analysis.sentiment and audio_analysis.emotion:
                    # Simplified similarity calculation
                    similarity = 0.6  # Placeholder
                    similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.0
        
        except Exception:
            return 0.0
    
    async def _calculate_content_coherence(self, image_analysis, video_analysis, audio_analysis) -> float:
        """Calculate content coherence across modalities"""
        try:
            coherence_scores = []
            
            # Image coherence
            if image_analysis:
                coherence_scores.append(image_analysis.aesthetic_score)
            
            # Video coherence
            if video_analysis:
                coherence_scores.append(video_analysis.quality_metrics.get("average_sharpness", 0.5) / 1000)
            
            # Audio coherence
            if audio_analysis:
                coherence_scores.append(audio_analysis.quality_metrics.get("snr", 0) / 50)
            
            return np.mean(coherence_scores) if coherence_scores else 0.0
        
        except Exception:
            return 0.0
    
    async def _calculate_overall_quality(self, image_analysis, video_analysis, audio_analysis) -> float:
        """Calculate overall quality score"""
        try:
            quality_scores = []
            
            if image_analysis:
                quality_scores.append(image_analysis.technical_quality)
            
            if video_analysis:
                quality_scores.append(video_analysis.quality_metrics.get("average_sharpness", 0) / 1000)
            
            if audio_analysis:
                quality_scores.append(audio_analysis.quality_metrics.get("snr", 0) / 50)
            
            return np.mean(quality_scores) if quality_scores else 0.0
        
        except Exception:
            return 0.0
    
    def get_analysis_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get multimodal analysis history"""
        return self.analysis_history[-limit:]
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get multimodal analysis statistics"""
        if not self.analysis_history:
            return {}
        
        try:
            return {
                "total_analyses": len(self.analysis_history),
                "modalities_analyzed": list(set(
                    modality for analysis in self.analysis_history 
                    for modality in analysis.get("modalities", [])
                )),
                "average_quality": np.mean([
                    analysis.get("overall_quality", 0.0) 
                    for analysis in self.analysis_history
                ]),
                "average_coherence": np.mean([
                    analysis.get("content_coherence", 0.0) 
                    for analysis in self.analysis_history
                ])
            }
        
        except Exception as e:
            logger.error(f"Error calculating multimodal analysis statistics: {str(e)}")
            return {}


# Global multimodal analysis instance
_multimodal_analyzer: Optional[MultimodalAnalysisSystem] = None


def get_multimodal_analyzer(model_storage_path: str = "multimodal_models") -> MultimodalAnalysisSystem:
    """Get or create global multimodal analyzer"""
    global _multimodal_analyzer
    if _multimodal_analyzer is None:
        _multimodal_analyzer = MultimodalAnalysisSystem(model_storage_path)
    return _multimodal_analyzer


# Example usage
async def main():
    """Example usage of multimodal analysis"""
    analyzer = get_multimodal_analyzer()
    
    # Analyze image
    image_path = "sample_image.jpg"
    if os.path.exists(image_path):
        image_result = await analyzer.analyze_image(image_path)
        print(f"Image Analysis:")
        print(f"  Objects: {len(image_result.objects_detected)}")
        print(f"  Scene: {image_result.scene_description}")
        print(f"  Quality: {image_result.technical_quality:.3f}")
    
    # Analyze audio
    audio_path = "sample_audio.wav"
    if os.path.exists(audio_path):
        audio_result = await analyzer.analyze_audio(audio_path)
        print(f"Audio Analysis:")
        print(f"  Transcription: {audio_result.transcription[:100]}...")
        print(f"  Language: {audio_result.language}")
        print(f"  Sentiment: {audio_result.sentiment}")
    
    # Get statistics
    stats = analyzer.get_analysis_statistics()
    print(f"Analysis Statistics: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
























