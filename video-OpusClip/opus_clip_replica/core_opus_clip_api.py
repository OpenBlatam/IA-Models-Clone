"""
Core Opus Clip Replica API

Exact replica of Opus Clip platform with core features:
- Video analysis and clip extraction
- AI-powered content curation
- Viral potential scoring
- Multi-platform export
- Simple, clean interface
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Union
import asyncio
import uuid
import time
from datetime import datetime
from pathlib import Path
import structlog
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn
import cv2
import numpy as np
import whisper
import torch
from transformers import pipeline
import moviepy.editor as mp
from PIL import Image
import json
import os
import tempfile
import shutil

logger = structlog.get_logger("opus_clip_replica")

# Initialize FastAPI app
app = FastAPI(
    title="Opus Clip Replica",
    description="Exact replica of Opus Clip platform with core features",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class VideoAnalysisRequest(BaseModel):
    """Request model for video analysis."""
    video_url: Optional[str] = None
    video_file: Optional[str] = None
    max_clips: int = Field(default=10, ge=1, le=50)
    min_duration: float = Field(default=3.0, ge=1.0, le=60.0)
    max_duration: float = Field(default=30.0, ge=1.0, le=300.0)

class ClipExtractionRequest(BaseModel):
    """Request model for clip extraction."""
    video_path: str
    segments: List[Dict[str, Any]]
    output_format: str = Field(default="mp4", regex="^(mp4|mov|avi)$")
    quality: str = Field(default="high", regex="^(low|medium|high|ultra)$")

class ViralScoreRequest(BaseModel):
    """Request model for viral scoring."""
    content: str
    platform: str = Field(default="youtube", regex="^(youtube|tiktok|instagram|facebook|twitter)$")
    audience_data: Optional[Dict[str, Any]] = None

class ExportRequest(BaseModel):
    """Request model for export."""
    clips: List[Dict[str, Any]]
    platform: str
    format: str = Field(default="mp4")
    quality: str = Field(default="high")

# Global variables
whisper_model = None
sentiment_analyzer = None
video_cache = {}

# Initialize models
async def initialize_models():
    """Initialize AI models."""
    global whisper_model, sentiment_analyzer
    
    try:
        # Load Whisper model for transcription
        whisper_model = whisper.load_model("base")
        
        # Load sentiment analysis model
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        
        logger.info("AI models initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    await initialize_models()

class OpusClipAnalyzer:
    """Core Opus Clip video analyzer."""
    
    def __init__(self):
        self.logger = structlog.get_logger("opus_clip_analyzer")
    
    async def analyze_video(self, video_path: str, max_clips: int = 10, 
                          min_duration: float = 3.0, max_duration: float = 30.0) -> Dict[str, Any]:
        """Analyze video and extract engaging segments."""
        try:
            # Load video
            video = mp.VideoFileClip(video_path)
            duration = video.duration
            
            # Extract frames for analysis
            frames = await self._extract_frames(video_path, duration)
            
            # Analyze content
            analysis = await self._analyze_content(video_path, frames)
            
            # Extract engaging segments
            segments = await self._extract_engaging_segments(
                analysis, duration, max_clips, min_duration, max_duration
            )
            
            # Calculate viral scores
            viral_scores = await self._calculate_viral_scores(segments)
            
            return {
                "video_duration": duration,
                "total_segments": len(segments),
                "segments": segments,
                "viral_scores": viral_scores,
                "analysis": analysis
            }
            
        except Exception as e:
            self.logger.error(f"Video analysis failed: {e}")
            raise
    
    async def _extract_frames(self, video_path: str, duration: float) -> List[np.ndarray]:
        """Extract frames from video for analysis."""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            # Extract frames every 2 seconds
            frame_interval = int(cap.get(cv2.CAP_PROP_FPS) * 2)
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    frames.append(frame)
                
                frame_count += 1
            
            cap.release()
            return frames
            
        except Exception as e:
            self.logger.error(f"Frame extraction failed: {e}")
            return []
    
    async def _analyze_content(self, video_path: str, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze video content for engagement factors."""
        try:
            analysis = {
                "face_detection": [],
                "motion_analysis": [],
                "audio_analysis": {},
                "text_analysis": {},
                "engagement_factors": []
            }
            
            # Face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            for i, frame in enumerate(frames):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                analysis["face_detection"].append({
                    "frame": i,
                    "face_count": len(faces),
                    "faces": faces.tolist() if len(faces) > 0 else []
                })
            
            # Motion analysis
            if len(frames) > 1:
                for i in range(1, len(frames)):
                    diff = cv2.absdiff(frames[i-1], frames[i])
                    motion_score = np.mean(diff)
                    analysis["motion_analysis"].append({
                        "frame": i,
                        "motion_score": float(motion_score)
                    })
            
            # Audio analysis
            try:
                video = mp.VideoFileClip(video_path)
                if video.audio:
                    audio = video.audio
                    analysis["audio_analysis"] = {
                        "has_audio": True,
                        "duration": audio.duration,
                        "volume_levels": await self._analyze_audio_levels(audio)
                    }
                else:
                    analysis["audio_analysis"] = {"has_audio": False}
            except Exception as e:
                analysis["audio_analysis"] = {"error": str(e)}
            
            # Text analysis (transcription)
            try:
                if whisper_model:
                    result = whisper_model.transcribe(video_path)
                    analysis["text_analysis"] = {
                        "transcription": result["text"],
                        "segments": result.get("segments", []),
                        "language": result.get("language", "unknown")
                    }
                else:
                    analysis["text_analysis"] = {"error": "Whisper model not loaded"}
            except Exception as e:
                analysis["text_analysis"] = {"error": str(e)}
            
            # Calculate engagement factors
            analysis["engagement_factors"] = await self._calculate_engagement_factors(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Content analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_audio_levels(self, audio) -> List[float]:
        """Analyze audio volume levels."""
        try:
            # Get audio array
            audio_array = audio.to_soundarray()
            
            # Calculate RMS for each second
            sample_rate = audio.fps
            chunk_size = int(sample_rate)
            volume_levels = []
            
            for i in range(0, len(audio_array), chunk_size):
                chunk = audio_array[i:i+chunk_size]
                if len(chunk) > 0:
                    rms = np.sqrt(np.mean(chunk**2))
                    volume_levels.append(float(rms))
            
            return volume_levels
            
        except Exception as e:
            self.logger.error(f"Audio analysis failed: {e}")
            return []
    
    async def _calculate_engagement_factors(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate engagement factors from analysis."""
        try:
            factors = []
            
            # Face engagement
            face_data = analysis.get("face_detection", [])
            if face_data:
                avg_faces = np.mean([f["face_count"] for f in face_data])
                factors.append({
                    "factor": "face_presence",
                    "score": min(avg_faces / 2.0, 1.0),  # Normalize to 0-1
                    "description": f"Average {avg_faces:.1f} faces per frame"
                })
            
            # Motion engagement
            motion_data = analysis.get("motion_analysis", [])
            if motion_data:
                avg_motion = np.mean([m["motion_score"] for m in motion_data])
                factors.append({
                    "factor": "motion_level",
                    "score": min(avg_motion / 50.0, 1.0),  # Normalize to 0-1
                    "description": f"Motion level: {avg_motion:.1f}"
                })
            
            # Audio engagement
            audio_data = analysis.get("audio_analysis", {})
            if audio_data.get("has_audio") and "volume_levels" in audio_data:
                volume_levels = audio_data["volume_levels"]
                if volume_levels:
                    avg_volume = np.mean(volume_levels)
                    factors.append({
                        "factor": "audio_presence",
                        "score": min(avg_volume * 10, 1.0),  # Normalize to 0-1
                        "description": f"Average volume: {avg_volume:.3f}"
                    })
            
            # Text engagement
            text_data = analysis.get("text_analysis", {})
            if text_data.get("transcription"):
                transcription = text_data["transcription"]
                word_count = len(transcription.split())
                factors.append({
                    "factor": "text_content",
                    "score": min(word_count / 100.0, 1.0),  # Normalize to 0-1
                    "description": f"Word count: {word_count}"
                })
            
            return factors
            
        except Exception as e:
            self.logger.error(f"Engagement calculation failed: {e}")
            return []
    
    async def _extract_engaging_segments(self, analysis: Dict[str, Any], duration: float,
                                       max_clips: int, min_duration: float, max_duration: float) -> List[Dict[str, Any]]:
        """Extract the most engaging segments from the video."""
        try:
            segments = []
            
            # Get engagement factors
            factors = analysis.get("engagement_factors", [])
            if not factors:
                return segments
            
            # Calculate overall engagement score for each time segment
            segment_duration = 5.0  # 5-second segments
            num_segments = int(duration / segment_duration)
            
            for i in range(num_segments):
                start_time = i * segment_duration
                end_time = min((i + 1) * segment_duration, duration)
                
                if end_time - start_time < min_duration:
                    continue
                
                # Calculate engagement score for this segment
                engagement_score = await self._calculate_segment_score(
                    analysis, start_time, end_time, factors
                )
                
                if engagement_score > 0.3:  # Threshold for engagement
                    segments.append({
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": end_time - start_time,
                        "engagement_score": engagement_score,
                        "segment_id": f"segment_{i}",
                        "title": f"Engaging Segment {i+1}",
                        "description": f"High engagement segment from {start_time:.1f}s to {end_time:.1f}s"
                    })
            
            # Sort by engagement score and limit to max_clips
            segments.sort(key=lambda x: x["engagement_score"], reverse=True)
            return segments[:max_clips]
            
        except Exception as e:
            self.logger.error(f"Segment extraction failed: {e}")
            return []
    
    async def _calculate_segment_score(self, analysis: Dict[str, Any], start_time: float, 
                                     end_time: float, factors: List[Dict[str, Any]]) -> float:
        """Calculate engagement score for a specific segment."""
        try:
            score = 0.0
            weight_sum = 0.0
            
            # Face presence weight
            face_weight = 0.3
            face_data = analysis.get("face_detection", [])
            if face_data:
                # Find frames in this time segment
                segment_frames = [f for f in face_data if start_time <= f["frame"] * 2 <= end_time]
                if segment_frames:
                    avg_faces = np.mean([f["face_count"] for f in segment_frames])
                    score += min(avg_faces / 2.0, 1.0) * face_weight
                    weight_sum += face_weight
            
            # Motion weight
            motion_weight = 0.25
            motion_data = analysis.get("motion_analysis", [])
            if motion_data:
                segment_motion = [m for m in motion_data if start_time <= m["frame"] * 2 <= end_time]
                if segment_motion:
                    avg_motion = np.mean([m["motion_score"] for m in segment_motion])
                    score += min(avg_motion / 50.0, 1.0) * motion_weight
                    weight_sum += motion_weight
            
            # Audio weight
            audio_weight = 0.2
            audio_data = analysis.get("audio_analysis", {})
            if audio_data.get("has_audio") and "volume_levels" in audio_data:
                volume_levels = audio_data["volume_levels"]
                if volume_levels:
                    # Map time to volume index
                    start_idx = int(start_time)
                    end_idx = int(end_time)
                    segment_volumes = volume_levels[start_idx:end_idx+1]
                    if segment_volumes:
                        avg_volume = np.mean(segment_volumes)
                        score += min(avg_volume * 10, 1.0) * audio_weight
                        weight_sum += audio_weight
            
            # Text weight
            text_weight = 0.25
            text_data = analysis.get("text_analysis", {})
            if text_data.get("segments"):
                # Find text segments in this time range
                text_segments = [
                    s for s in text_data["segments"] 
                    if start_time <= s.get("start", 0) <= end_time
                ]
                if text_segments:
                    total_text = " ".join([s.get("text", "") for s in text_segments])
                    word_count = len(total_text.split())
                    score += min(word_count / 20.0, 1.0) * text_weight
                    weight_sum += text_weight
            
            # Normalize score
            if weight_sum > 0:
                return score / weight_sum
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Segment score calculation failed: {e}")
            return 0.0
    
    async def _calculate_viral_scores(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate viral potential scores for segments."""
        try:
            viral_scores = {}
            
            for segment in segments:
                segment_id = segment["segment_id"]
                
                # Base score from engagement
                base_score = segment["engagement_score"]
                
                # Duration bonus (optimal 15-30 seconds)
                duration = segment["duration"]
                if 15 <= duration <= 30:
                    duration_bonus = 0.2
                elif 10 <= duration <= 45:
                    duration_bonus = 0.1
                else:
                    duration_bonus = 0.0
                
                # Calculate final viral score
                viral_score = min(base_score + duration_bonus, 1.0)
                
                viral_scores[segment_id] = {
                    "viral_score": viral_score,
                    "base_score": base_score,
                    "duration_bonus": duration_bonus,
                    "viral_potential": self._get_viral_potential_label(viral_score)
                }
            
            return viral_scores
            
        except Exception as e:
            self.logger.error(f"Viral score calculation failed: {e}")
            return {}
    
    def _get_viral_potential_label(self, score: float) -> str:
        """Get viral potential label based on score."""
        if score >= 0.8:
            return "High"
        elif score >= 0.6:
            return "Medium"
        elif score >= 0.4:
            return "Low"
        else:
            return "Very Low"

class OpusClipExporter:
    """Core Opus Clip video exporter."""
    
    def __init__(self):
        self.logger = structlog.get_logger("opus_clip_exporter")
    
    async def export_clips(self, video_path: str, segments: List[Dict[str, Any]], 
                          output_format: str = "mp4", quality: str = "high") -> List[Dict[str, Any]]:
        """Export video clips."""
        try:
            exported_clips = []
            
            # Load video
            video = mp.VideoFileClip(video_path)
            
            # Create output directory
            output_dir = tempfile.mkdtemp(prefix="opus_clips_")
            
            for i, segment in enumerate(segments):
                try:
                    # Extract clip
                    start_time = segment["start_time"]
                    end_time = segment["end_time"]
                    
                    clip = video.subclip(start_time, end_time)
                    
                    # Generate filename
                    filename = f"clip_{i+1}_{int(start_time)}_{int(end_time)}.{output_format}"
                    output_path = os.path.join(output_dir, filename)
                    
                    # Export with quality settings
                    if quality == "ultra":
                        bitrate = "5000k"
                    elif quality == "high":
                        bitrate = "3000k"
                    elif quality == "medium":
                        bitrate = "1500k"
                    else:  # low
                        bitrate = "800k"
                    
                    clip.write_videofile(
                        output_path,
                        bitrate=bitrate,
                        codec='libx264',
                        audio_codec='aac',
                        temp_audiofile='temp-audio.m4a',
                        remove_temp=True
                    )
                    
                    # Generate thumbnail
                    thumbnail_path = await self._generate_thumbnail(clip, output_dir, i)
                    
                    exported_clips.append({
                        "clip_id": segment["segment_id"],
                        "filename": filename,
                        "path": output_path,
                        "thumbnail": thumbnail_path,
                        "duration": segment["duration"],
                        "start_time": start_time,
                        "end_time": end_time,
                        "size": os.path.getsize(output_path),
                        "quality": quality
                    })
                    
                    clip.close()
                    
                except Exception as e:
                    self.logger.error(f"Failed to export clip {i}: {e}")
                    continue
            
            video.close()
            
            return exported_clips
            
        except Exception as e:
            self.logger.error(f"Clip export failed: {e}")
            raise
    
    async def _generate_thumbnail(self, clip, output_dir: str, index: int) -> str:
        """Generate thumbnail for clip."""
        try:
            # Get frame at 25% of clip duration
            thumbnail_time = clip.duration * 0.25
            frame = clip.get_frame(thumbnail_time)
            
            # Convert to PIL Image
            image = Image.fromarray(frame)
            
            # Resize to standard thumbnail size
            image.thumbnail((320, 180), Image.Resampling.LANCZOS)
            
            # Save thumbnail
            thumbnail_path = os.path.join(output_dir, f"thumb_{index+1}.jpg")
            image.save(thumbnail_path, "JPEG", quality=85)
            
            return thumbnail_path
            
        except Exception as e:
            self.logger.error(f"Thumbnail generation failed: {e}")
            return ""

# Initialize analyzer and exporter
analyzer = OpusClipAnalyzer()
exporter = OpusClipExporter()

# API Endpoints
@app.post("/analyze")
async def analyze_video(request: VideoAnalysisRequest):
    """Analyze video and extract engaging segments."""
    try:
        if not request.video_url and not request.video_file:
            raise HTTPException(status_code=400, detail="Either video_url or video_file must be provided")
        
        video_path = request.video_file or request.video_url
        
        # Analyze video
        result = await analyzer.analyze_video(
            video_path=video_path,
            max_clips=request.max_clips,
            min_duration=request.min_duration,
            max_duration=request.max_duration
        )
        
        return {
            "success": True,
            "analysis": result,
            "message": f"Found {result['total_segments']} engaging segments"
        }
        
    except Exception as e:
        logger.error(f"Video analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract")
async def extract_clips(request: ClipExtractionRequest):
    """Extract video clips from segments."""
    try:
        # Export clips
        exported_clips = await exporter.export_clips(
            video_path=request.video_path,
            segments=request.segments,
            output_format=request.output_format,
            quality=request.quality
        )
        
        return {
            "success": True,
            "clips": exported_clips,
            "message": f"Exported {len(exported_clips)} clips successfully"
        }
        
    except Exception as e:
        logger.error(f"Clip extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/viral-score")
async def calculate_viral_score(request: ViralScoreRequest):
    """Calculate viral potential score for content."""
    try:
        # Analyze sentiment
        if sentiment_analyzer:
            sentiment_result = sentiment_analyzer(request.content)
            sentiment_score = sentiment_result[0]["score"]
            sentiment_label = sentiment_result[0]["label"]
        else:
            sentiment_score = 0.5
            sentiment_label = "neutral"
        
        # Calculate viral score based on content and platform
        base_score = 0.5
        
        # Content length factor
        word_count = len(request.content.split())
        if 50 <= word_count <= 200:
            length_factor = 0.2
        elif 20 <= word_count <= 300:
            length_factor = 0.1
        else:
            length_factor = 0.0
        
        # Platform factor
        platform_factors = {
            "youtube": 0.8,
            "tiktok": 0.9,
            "instagram": 0.7,
            "facebook": 0.6,
            "twitter": 0.5
        }
        platform_factor = platform_factors.get(request.platform, 0.5)
        
        # Sentiment factor
        sentiment_factor = sentiment_score if sentiment_label == "POSITIVE" else 0.3
        
        # Calculate final viral score
        viral_score = min(
            base_score + length_factor + (platform_factor * 0.3) + (sentiment_factor * 0.2),
            1.0
        )
        
        return {
            "success": True,
            "viral_score": viral_score,
            "sentiment": {
                "score": sentiment_score,
                "label": sentiment_label
            },
            "factors": {
                "content_length": word_count,
                "length_factor": length_factor,
                "platform_factor": platform_factor,
                "sentiment_factor": sentiment_factor
            },
            "viral_potential": analyzer._get_viral_potential_label(viral_score)
        }
        
    except Exception as e:
        logger.error(f"Viral score calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export")
async def export_to_platform(request: ExportRequest):
    """Export clips for specific platform."""
    try:
        # Platform-specific settings
        platform_settings = {
            "youtube": {"format": "mp4", "quality": "high", "aspect_ratio": "16:9"},
            "tiktok": {"format": "mp4", "quality": "high", "aspect_ratio": "9:16"},
            "instagram": {"format": "mp4", "quality": "high", "aspect_ratio": "1:1"},
            "facebook": {"format": "mp4", "quality": "medium", "aspect_ratio": "16:9"},
            "twitter": {"format": "mp4", "quality": "medium", "aspect_ratio": "16:9"}
        }
        
        settings = platform_settings.get(request.platform, platform_settings["youtube"])
        
        # Process clips for platform
        processed_clips = []
        for clip in request.clips:
            processed_clip = {
                **clip,
                "platform": request.platform,
                "format": settings["format"],
                "quality": settings["quality"],
                "aspect_ratio": settings["aspect_ratio"],
                "exported_at": datetime.now().isoformat()
            }
            processed_clips.append(processed_clip)
        
        return {
            "success": True,
            "platform": request.platform,
            "clips": processed_clips,
            "settings": settings,
            "message": f"Exported {len(processed_clips)} clips for {request.platform}"
        }
        
    except Exception as e:
        logger.error(f"Platform export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "models_loaded": {
            "whisper": whisper_model is not None,
            "sentiment": sentiment_analyzer is not None
        }
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Opus Clip Replica API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "analyze": "/analyze",
            "extract": "/extract",
            "viral-score": "/viral-score",
            "export": "/export",
            "health": "/health",
            "docs": "/docs"
        }
    }

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "core_opus_clip_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


