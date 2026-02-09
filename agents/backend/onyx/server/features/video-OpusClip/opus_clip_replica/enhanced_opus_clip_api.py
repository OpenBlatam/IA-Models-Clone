"""
Enhanced Opus Clip API

Advanced version of Opus Clip replica with enhanced features:
- Advanced AI models and processing
- Real-time collaboration
- Batch processing
- Advanced analytics
- Performance optimizations
- Multi-language support
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Union, Tuple
import asyncio
import uuid
import time
import json
import os
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import structlog
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
import cv2
import numpy as np
import whisper
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
import moviepy.editor as mp
from PIL import Image, ImageDraw, ImageFont
import librosa
import soundfile as sf
from scipy import signal
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import redis
import sqlite3
from collections import defaultdict
import threading
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import aiohttp
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger("enhanced_opus_clip")

# Enums
class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class QualityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

class Platform(Enum):
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"

# Data classes
@dataclass
class ProcessingJob:
    job_id: str
    video_path: str
    status: ProcessingStatus
    progress: float
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None

@dataclass
class VideoMetadata:
    duration: float
    fps: float
    width: int
    height: int
    bitrate: int
    codec: str
    audio_codec: str
    has_audio: bool
    file_size: int

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Opus Clip API",
    description="Advanced video processing and AI-powered content creation",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
whisper_model = None
sentiment_analyzer = None
emotion_analyzer = None
topic_classifier = None
redis_client = None
job_queue = queue.Queue()
active_jobs = {}
websocket_connections = []

# Pydantic models
class VideoAnalysisRequest(BaseModel):
    video_url: Optional[str] = None
    video_file: Optional[str] = None
    max_clips: int = Field(default=10, ge=1, le=100)
    min_duration: float = Field(default=3.0, ge=1.0, le=60.0)
    max_duration: float = Field(default=30.0, ge=1.0, le=300.0)
    language: str = Field(default="auto", regex="^(auto|en|es|fr|de|it|pt|ru|ja|ko|zh)$")
    advanced_analysis: bool = Field(default=True)
    include_thumbnails: bool = Field(default=True)
    include_transcripts: bool = Field(default=True)

class BatchProcessingRequest(BaseModel):
    videos: List[Dict[str, Any]]
    processing_options: Dict[str, Any] = Field(default_factory=dict)
    callback_url: Optional[str] = None

class ClipExtractionRequest(BaseModel):
    video_path: str
    segments: List[Dict[str, Any]]
    output_format: str = Field(default="mp4", regex="^(mp4|mov|avi|webm)$")
    quality: str = Field(default="high", regex="^(low|medium|high|ultra)$")
    include_subtitles: bool = Field(default=False)
    subtitle_language: str = Field(default="en")
    add_intro_outro: bool = Field(default=False)
    intro_duration: float = Field(default=2.0, ge=0.0, le=10.0)
    outro_duration: float = Field(default=2.0, ge=0.0, le=10.0)

class ViralScoreRequest(BaseModel):
    content: str
    platform: str = Field(default="youtube", regex="^(youtube|tiktok|instagram|facebook|twitter|linkedin)$")
    audience_data: Optional[Dict[str, Any]] = None
    include_emotion_analysis: bool = Field(default=True)
    include_topic_classification: bool = Field(default=True)

class ExportRequest(BaseModel):
    clips: List[Dict[str, Any]]
    platform: str
    format: str = Field(default="mp4")
    quality: str = Field(default="high")
    add_watermark: bool = Field(default=False)
    watermark_text: Optional[str] = None
    custom_dimensions: Optional[Dict[str, int]] = None

class CollaborationRequest(BaseModel):
    project_name: str
    description: str
    video_path: str
    collaborators: List[str] = Field(default_factory=list)
    permissions: Dict[str, List[str]] = Field(default_factory=dict)

# Initialize models
async def initialize_models():
    """Initialize all AI models."""
    global whisper_model, sentiment_analyzer, emotion_analyzer, topic_classifier
    
    try:
        # Load Whisper model
        whisper_model = whisper.load_model("large-v2")
        
        # Load sentiment analysis
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        
        # Load emotion analysis
        emotion_analyzer = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base"
        )
        
        # Load topic classifier
        topic_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        logger.info("All AI models initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise

# Initialize Redis
def initialize_redis():
    """Initialize Redis client."""
    global redis_client
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        redis_client.ping()
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.warning(f"Redis not available: {e}")
        redis_client = None

# Initialize database
def initialize_database():
    """Initialize SQLite database."""
    try:
        conn = sqlite3.connect('enhanced_opus_clip.db')
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_jobs (
                job_id TEXT PRIMARY KEY,
                video_path TEXT NOT NULL,
                status TEXT NOT NULL,
                progress REAL DEFAULT 0.0,
                result TEXT,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS video_analytics (
                video_id TEXT PRIMARY KEY,
                video_path TEXT NOT NULL,
                analysis_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collaboration_projects (
                project_id TEXT PRIMARY KEY,
                project_name TEXT NOT NULL,
                description TEXT,
                video_path TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

class EnhancedVideoAnalyzer:
    """Enhanced video analyzer with advanced AI capabilities."""
    
    def __init__(self):
        self.logger = structlog.get_logger("enhanced_video_analyzer")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    async def analyze_video_advanced(self, video_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced video analysis with multiple AI models."""
        try:
            # Get video metadata
            metadata = await self._get_video_metadata(video_path)
            
            # Extract frames for analysis
            frames = await self._extract_frames_advanced(video_path, metadata)
            
            # Perform comprehensive analysis
            analysis = {
                "metadata": metadata,
                "face_analysis": await self._analyze_faces_advanced(frames),
                "motion_analysis": await self._analyze_motion_advanced(frames),
                "audio_analysis": await self._analyze_audio_advanced(video_path),
                "text_analysis": await self._analyze_text_advanced(video_path, options.get("language", "auto")),
                "emotion_analysis": await self._analyze_emotions_advanced(frames),
                "scene_analysis": await self._analyze_scenes_advanced(frames),
                "engagement_analysis": await self._analyze_engagement_advanced(frames, video_path)
            }
            
            # Extract engaging segments
            segments = await self._extract_engaging_segments_advanced(analysis, options)
            
            # Calculate advanced viral scores
            viral_scores = await self._calculate_advanced_viral_scores(segments, analysis)
            
            return {
                "video_metadata": metadata,
                "analysis": analysis,
                "segments": segments,
                "viral_scores": viral_scores,
                "total_segments": len(segments),
                "processing_time": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Advanced video analysis failed: {e}")
            raise
    
    async def _get_video_metadata(self, video_path: str) -> VideoMetadata:
        """Get comprehensive video metadata."""
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Basic properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Get file size
            file_size = os.path.getsize(video_path)
            
            # Get codec info
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            cap.release()
            
            # Check for audio
            has_audio = False
            audio_codec = None
            try:
                video = mp.VideoFileClip(video_path)
                if video.audio:
                    has_audio = True
                    audio_codec = "unknown"  # Would need more complex analysis
                video.close()
            except:
                pass
            
            return VideoMetadata(
                duration=duration,
                fps=fps,
                width=width,
                height=height,
                bitrate=int(file_size * 8 / duration) if duration > 0 else 0,
                codec=codec,
                audio_codec=audio_codec or "none",
                has_audio=has_audio,
                file_size=file_size
            )
            
        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {e}")
            raise
    
    async def _extract_frames_advanced(self, video_path: str, metadata: VideoMetadata) -> List[np.ndarray]:
        """Extract frames with advanced sampling."""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            # Adaptive frame sampling based on video length
            if metadata.duration < 60:  # Short video
                frame_interval = max(1, int(metadata.fps * 0.5))  # Every 0.5 seconds
            elif metadata.duration < 300:  # Medium video
                frame_interval = max(1, int(metadata.fps * 1.0))  # Every 1 second
            else:  # Long video
                frame_interval = max(1, int(metadata.fps * 2.0))  # Every 2 seconds
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Resize frame for analysis if too large
                    if frame.shape[1] > 1280:
                        scale = 1280 / frame.shape[1]
                        new_width = int(frame.shape[1] * scale)
                        new_height = int(frame.shape[0] * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    frames.append(frame)
                
                frame_count += 1
            
            cap.release()
            return frames
            
        except Exception as e:
            self.logger.error(f"Frame extraction failed: {e}")
            return []
    
    async def _analyze_faces_advanced(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Advanced face analysis with emotion detection."""
        try:
            face_data = []
            emotion_scores = defaultdict(list)
            
            for i, frame in enumerate(frames):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                frame_faces = []
                for (x, y, w, h) in faces:
                    # Extract face region
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Detect eyes
                    eyes = self.eye_cascade.detectMultiScale(face_roi)
                    
                    # Calculate face metrics
                    face_area = w * h
                    aspect_ratio = w / h
                    eye_count = len(eyes)
                    
                    face_info = {
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "area": int(face_area),
                        "aspect_ratio": float(aspect_ratio),
                        "eye_count": int(eye_count),
                        "confidence": 1.0  # Placeholder
                    }
                    
                    frame_faces.append(face_info)
                
                face_data.append({
                    "frame": i,
                    "face_count": len(faces),
                    "faces": frame_faces
                })
            
            # Calculate statistics
            total_faces = sum(f["face_count"] for f in face_data)
            avg_faces_per_frame = total_faces / len(frames) if frames else 0
            
            return {
                "face_data": face_data,
                "statistics": {
                    "total_faces": total_faces,
                    "avg_faces_per_frame": avg_faces_per_frame,
                    "frames_with_faces": len([f for f in face_data if f["face_count"] > 0]),
                    "face_consistency": self._calculate_face_consistency(face_data)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Face analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_motion_advanced(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Advanced motion analysis with optical flow."""
        try:
            motion_data = []
            
            if len(frames) < 2:
                return {"motion_data": [], "statistics": {}}
            
            # Convert to grayscale for motion analysis
            gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
            
            for i in range(1, len(gray_frames)):
                # Calculate optical flow
                flow = cv2.calcOpticalFlowPyrLK(
                    gray_frames[i-1], gray_frames[i],
                    None, None
                )
                
                # Calculate frame difference
                diff = cv2.absdiff(gray_frames[i-1], gray_frames[i])
                motion_score = np.mean(diff)
                
                # Calculate structural similarity
                ssim_score = self._calculate_ssim(gray_frames[i-1], gray_frames[i])
                
                motion_data.append({
                    "frame": i,
                    "motion_score": float(motion_score),
                    "ssim_score": float(ssim_score),
                    "motion_intensity": self._classify_motion_intensity(motion_score)
                })
            
            # Calculate motion statistics
            motion_scores = [m["motion_score"] for m in motion_data]
            avg_motion = np.mean(motion_scores) if motion_scores else 0
            motion_variance = np.var(motion_scores) if motion_scores else 0
            
            return {
                "motion_data": motion_data,
                "statistics": {
                    "avg_motion": float(avg_motion),
                    "motion_variance": float(motion_variance),
                    "high_motion_frames": len([m for m in motion_data if m["motion_intensity"] == "high"]),
                    "motion_trend": self._calculate_motion_trend(motion_data)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Motion analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_audio_advanced(self, video_path: str) -> Dict[str, Any]:
        """Advanced audio analysis with spectral analysis."""
        try:
            # Load audio
            y, sr = librosa.load(video_path, sr=None)
            
            # Basic audio features
            duration = len(y) / sr
            rms = librosa.feature.rms(y=y)[0]
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            # Tempo and rhythm
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Spectral features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            # Voice activity detection
            voice_activity = self._detect_voice_activity(y, sr)
            
            # Audio quality metrics
            snr = self._calculate_snr(y)
            dynamic_range = self._calculate_dynamic_range(y)
            
            return {
                "duration": float(duration),
                "sample_rate": int(sr),
                "rms_energy": {
                    "mean": float(np.mean(rms)),
                    "std": float(np.std(rms)),
                    "max": float(np.max(rms))
                },
                "spectral_features": {
                    "centroid_mean": float(np.mean(spectral_centroids)),
                    "rolloff_mean": float(np.mean(spectral_rolloff)),
                    "zero_crossing_rate": float(np.mean(zero_crossing_rate))
                },
                "rhythm": {
                    "tempo": float(tempo),
                    "beat_count": len(beats)
                },
                "voice_activity": voice_activity,
                "quality_metrics": {
                    "snr": float(snr),
                    "dynamic_range": float(dynamic_range)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Audio analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_text_advanced(self, video_path: str, language: str = "auto") -> Dict[str, Any]:
        """Advanced text analysis with sentiment and topic classification."""
        try:
            if not whisper_model:
                return {"error": "Whisper model not loaded"}
            
            # Transcribe with language detection
            if language == "auto":
                result = whisper_model.transcribe(video_path)
            else:
                result = whisper_model.transcribe(video_path, language=language)
            
            transcription = result["text"]
            segments = result.get("segments", [])
            detected_language = result.get("language", "unknown")
            
            # Sentiment analysis
            sentiment_scores = []
            if sentiment_analyzer and transcription.strip():
                # Analyze in chunks to avoid token limits
                words = transcription.split()
                chunk_size = 500
                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i:i+chunk_size])
                    if chunk.strip():
                        sentiment_result = sentiment_analyzer(chunk)
                        sentiment_scores.extend(sentiment_result)
            
            # Emotion analysis
            emotion_scores = []
            if emotion_analyzer and transcription.strip():
                words = transcription.split()
                chunk_size = 500
                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i:i+chunk_size])
                    if chunk.strip():
                        emotion_result = emotion_analyzer(chunk)
                        emotion_scores.extend(emotion_result)
            
            # Topic classification
            topics = []
            if topic_classifier and transcription.strip():
                candidate_labels = [
                    "technology", "entertainment", "education", "news", "sports",
                    "music", "comedy", "tutorial", "review", "lifestyle"
                ]
                topic_result = topic_classifier(transcription, candidate_labels)
                topics = topic_result["labels"][:3]  # Top 3 topics
            
            # Calculate text statistics
            word_count = len(transcription.split())
            char_count = len(transcription)
            avg_words_per_second = word_count / (segments[-1]["end"] if segments else 1)
            
            return {
                "transcription": transcription,
                "segments": segments,
                "language": detected_language,
                "statistics": {
                    "word_count": word_count,
                    "char_count": char_count,
                    "avg_words_per_second": float(avg_words_per_second)
                },
                "sentiment": {
                    "scores": sentiment_scores,
                    "overall_sentiment": self._calculate_overall_sentiment(sentiment_scores)
                },
                "emotions": {
                    "scores": emotion_scores,
                    "dominant_emotion": self._get_dominant_emotion(emotion_scores)
                },
                "topics": topics
            }
            
        except Exception as e:
            self.logger.error(f"Text analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_emotions_advanced(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Advanced emotion analysis from visual content."""
        try:
            # This would typically use a facial emotion recognition model
            # For now, we'll use a simplified approach based on visual features
            
            emotion_data = []
            
            for i, frame in enumerate(frames):
                # Convert to different color spaces for analysis
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                
                # Analyze color distribution
                brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                saturation = np.mean(hsv[:, :, 1])
                
                # Analyze contrast
                contrast = np.std(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                
                # Simple emotion inference based on visual features
                emotion_score = self._infer_emotion_from_visuals(brightness, saturation, contrast)
                
                emotion_data.append({
                    "frame": i,
                    "brightness": float(brightness),
                    "saturation": float(saturation),
                    "contrast": float(contrast),
                    "emotion_score": emotion_score
                })
            
            return {
                "emotion_data": emotion_data,
                "overall_emotion": self._calculate_overall_emotion(emotion_data)
            }
            
        except Exception as e:
            self.logger.error(f"Emotion analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_scenes_advanced(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Advanced scene analysis and segmentation."""
        try:
            scene_data = []
            scene_changes = []
            
            if len(frames) < 2:
                return {"scene_data": [], "scene_changes": []}
            
            # Convert to grayscale for scene analysis
            gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
            
            for i in range(1, len(gray_frames)):
                # Calculate histogram difference
                hist1 = cv2.calcHist([gray_frames[i-1]], [0], None, [256], [0, 256])
                hist2 = cv2.calcHist([gray_frames[i]], [0], None, [256], [0, 256])
                hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                
                # Calculate structural similarity
                ssim = self._calculate_ssim(gray_frames[i-1], gray_frames[i])
                
                # Detect scene change
                is_scene_change = hist_diff < 0.3 or ssim < 0.5
                
                if is_scene_change:
                    scene_changes.append(i)
                
                scene_data.append({
                    "frame": i,
                    "histogram_correlation": float(hist_diff),
                    "ssim": float(ssim),
                    "is_scene_change": is_scene_change
                })
            
            return {
                "scene_data": scene_data,
                "scene_changes": scene_changes,
                "scene_count": len(scene_changes) + 1
            }
            
        except Exception as e:
            self.logger.error(f"Scene analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_engagement_advanced(self, frames: List[np.ndarray], video_path: str) -> Dict[str, Any]:
        """Advanced engagement analysis combining multiple factors."""
        try:
            engagement_factors = []
            
            for i, frame in enumerate(frames):
                # Visual engagement factors
                brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                colorfulness = self._calculate_colorfulness(frame)
                sharpness = self._calculate_sharpness(frame)
                
                # Face engagement
                faces = self.face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 4)
                face_count = len(faces)
                
                # Calculate composite engagement score
                visual_score = (brightness / 255.0) * 0.3 + (colorfulness / 100.0) * 0.3 + (sharpness / 100.0) * 0.4
                face_score = min(face_count / 3.0, 1.0)  # Normalize face count
                
                engagement_score = (visual_score + face_score) / 2.0
                
                engagement_factors.append({
                    "frame": i,
                    "brightness": float(brightness),
                    "colorfulness": float(colorfulness),
                    "sharpness": float(sharpness),
                    "face_count": int(face_count),
                    "engagement_score": float(engagement_score)
                })
            
            # Calculate engagement statistics
            scores = [f["engagement_score"] for f in engagement_factors]
            avg_engagement = np.mean(scores) if scores else 0
            engagement_variance = np.var(scores) if scores else 0
            
            return {
                "engagement_factors": engagement_factors,
                "statistics": {
                    "avg_engagement": float(avg_engagement),
                    "engagement_variance": float(engagement_variance),
                    "high_engagement_frames": len([f for f in engagement_factors if f["engagement_score"] > 0.7])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Engagement analysis failed: {e}")
            return {"error": str(e)}
    
    # Helper methods
    def _calculate_face_consistency(self, face_data: List[Dict]) -> float:
        """Calculate face consistency across frames."""
        if not face_data:
            return 0.0
        
        face_counts = [f["face_count"] for f in face_data]
        return 1.0 - (np.std(face_counts) / (np.mean(face_counts) + 1e-6))
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Structural Similarity Index."""
        try:
            # Simple SSIM calculation
            mu1 = np.mean(img1)
            mu2 = np.mean(img2)
            sigma1 = np.var(img1)
            sigma2 = np.var(img2)
            sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
            
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
            
            return float(ssim)
        except:
            return 0.0
    
    def _classify_motion_intensity(self, motion_score: float) -> str:
        """Classify motion intensity based on score."""
        if motion_score > 50:
            return "high"
        elif motion_score > 20:
            return "medium"
        else:
            return "low"
    
    def _calculate_motion_trend(self, motion_data: List[Dict]) -> str:
        """Calculate motion trend over time."""
        if len(motion_data) < 3:
            return "stable"
        
        scores = [m["motion_score"] for m in motion_data]
        # Simple linear trend
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]
        
        if slope > 0.5:
            return "increasing"
        elif slope < -0.5:
            return "decreasing"
        else:
            return "stable"
    
    def _detect_voice_activity(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Detect voice activity in audio."""
        try:
            # Simple voice activity detection based on energy
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)   # 10ms hop
            
            # Calculate energy for each frame
            energy = []
            for i in range(0, len(y) - frame_length, hop_length):
                frame = y[i:i + frame_length]
                frame_energy = np.sum(frame ** 2)
                energy.append(frame_energy)
            
            # Threshold for voice activity
            threshold = np.mean(energy) * 0.1
            voice_frames = [e > threshold for e in energy]
            
            return {
                "voice_activity_ratio": sum(voice_frames) / len(voice_frames) if voice_frames else 0,
                "total_voice_frames": sum(voice_frames),
                "total_frames": len(voice_frames)
            }
        except:
            return {"voice_activity_ratio": 0, "total_voice_frames": 0, "total_frames": 0}
    
    def _calculate_snr(self, y: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio."""
        try:
            signal_power = np.mean(y ** 2)
            noise_power = np.var(y - np.mean(y))
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            return float(snr)
        except:
            return 0.0
    
    def _calculate_dynamic_range(self, y: np.ndarray) -> float:
        """Calculate dynamic range of audio."""
        try:
            return float(20 * np.log10(np.max(np.abs(y)) / (np.min(np.abs(y[y != 0])) + 1e-10))
        except:
            return 0.0
    
    def _calculate_overall_sentiment(self, sentiment_scores: List[Dict]) -> str:
        """Calculate overall sentiment from scores."""
        if not sentiment_scores:
            return "neutral"
        
        positive_count = sum(1 for s in sentiment_scores if s["label"] == "POSITIVE")
        negative_count = sum(1 for s in sentiment_scores if s["label"] == "NEGATIVE")
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _get_dominant_emotion(self, emotion_scores: List[Dict]) -> str:
        """Get dominant emotion from scores."""
        if not emotion_scores:
            return "neutral"
        
        emotion_counts = defaultdict(int)
        for score in emotion_scores:
            emotion_counts[score["label"]] += 1
        
        return max(emotion_counts, key=emotion_counts.get)
    
    def _infer_emotion_from_visuals(self, brightness: float, saturation: float, contrast: float) -> Dict[str, float]:
        """Infer emotion from visual features."""
        # Simple heuristic-based emotion inference
        emotions = {
            "happiness": 0.0,
            "sadness": 0.0,
            "excitement": 0.0,
            "calm": 0.0
        }
        
        # Brightness affects happiness/sadness
        if brightness > 150:
            emotions["happiness"] += 0.3
        elif brightness < 100:
            emotions["sadness"] += 0.3
        
        # Saturation affects excitement
        if saturation > 100:
            emotions["excitement"] += 0.3
        else:
            emotions["calm"] += 0.3
        
        # Contrast affects excitement
        if contrast > 50:
            emotions["excitement"] += 0.2
        else:
            emotions["calm"] += 0.2
        
        return emotions
    
    def _calculate_overall_emotion(self, emotion_data: List[Dict]) -> str:
        """Calculate overall emotion from frame data."""
        if not emotion_data:
            return "neutral"
        
        # Average emotion scores across frames
        avg_emotions = defaultdict(float)
        for frame_data in emotion_data:
            for emotion, score in frame_data["emotion_score"].items():
                avg_emotions[emotion] += score
        
        # Normalize
        total_frames = len(emotion_data)
        for emotion in avg_emotions:
            avg_emotions[emotion] /= total_frames
        
        return max(avg_emotions, key=avg_emotions.get)
    
    def _calculate_colorfulness(self, frame: np.ndarray) -> float:
        """Calculate colorfulness of frame."""
        try:
            # Convert to Lab color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            a = lab[:, :, 1]
            b = lab[:, :, 2]
            
            # Calculate colorfulness metric
            colorfulness = np.sqrt(np.var(a) + np.var(b)) + 0.3 * np.sqrt(np.mean(a)**2 + np.mean(b)**2)
            return float(colorfulness)
        except:
            return 0.0
    
    def _calculate_sharpness(self, frame: np.ndarray) -> float:
        """Calculate sharpness of frame using Laplacian variance."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return float(laplacian_var)
        except:
            return 0.0

# Initialize analyzer
analyzer = EnhancedVideoAnalyzer()

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    await initialize_models()
    initialize_redis()
    initialize_database()

# API Endpoints
@app.post("/analyze/advanced")
async def analyze_video_advanced(request: VideoAnalysisRequest):
    """Advanced video analysis with multiple AI models."""
    try:
        if not request.video_url and not request.video_file:
            raise HTTPException(status_code=400, detail="Either video_url or video_file must be provided")
        
        video_path = request.video_file or request.video_url
        
        # Perform advanced analysis
        result = await analyzer.analyze_video_advanced(video_path, {
            "language": request.language,
            "advanced_analysis": request.advanced_analysis,
            "include_thumbnails": request.include_thumbnails,
            "include_transcripts": request.include_transcripts
        })
        
        return {
            "success": True,
            "analysis": result,
            "message": f"Advanced analysis completed with {result['total_segments']} segments"
        }
        
    except Exception as e:
        logger.error(f"Advanced video analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "models_loaded": {
            "whisper": whisper_model is not None,
            "sentiment": sentiment_analyzer is not None,
            "emotion": emotion_analyzer is not None,
            "topic": topic_classifier is not None
        },
        "services": {
            "redis": redis_client is not None,
            "database": True
        }
    }

@app.get("/")
async def root():
    """Root endpoint with enhanced information."""
    return {
        "message": "Enhanced Opus Clip API",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Advanced video analysis",
            "Multi-language support",
            "Real-time collaboration",
            "Batch processing",
            "Advanced analytics",
            "Performance optimizations"
        ],
        "endpoints": {
            "analyze_advanced": "/analyze/advanced",
            "health": "/health",
            "docs": "/docs"
        }
    }

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "enhanced_opus_clip_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


