"""
Refactored Opus Clip Analyzer

Enhanced video analyzer with:
- BaseProcessor architecture
- Async processing
- Error handling and retries
- Performance monitoring
- Caching
- Modular design
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import asyncio
import time
import cv2
import numpy as np
import whisper
import torch
from transformers import pipeline
import moviepy.editor as mp
from PIL import Image
import librosa
import soundfile as sf
import structlog
from pathlib import Path

from ..core.base_processor import BaseProcessor, ProcessorResult, ProcessorConfig
from ..core.config_manager import ConfigManager

logger = structlog.get_logger("refactored_analyzer")

class RefactoredOpusClipAnalyzer(BaseProcessor):
    """
    Refactored Opus Clip video analyzer.
    
    Features:
    - Async processing with BaseProcessor
    - Error handling and retries
    - Performance monitoring
    - Caching
    - Modular design
    """
    
    def __init__(self, config: Optional[ProcessorConfig] = None, 
                 app_config: Optional[ConfigManager] = None):
        """Initialize the refactored analyzer."""
        super().__init__(config)
        self.app_config = app_config
        self.logger = structlog.get_logger("refactored_analyzer")
        
        # AI models (lazy loading)
        self.whisper_model = None
        self.sentiment_analyzer = None
        self.face_cascade = None
        
        # Video processing components
        self.video_cache = {}
        
        self.logger.info("Initialized RefactoredOpusClipAnalyzer")
    
    async def _load_models(self):
        """Load AI models lazily."""
        try:
            if self.whisper_model is None:
                model_name = self.app_config.ai.whisper_model if self.app_config else "base"
                self.whisper_model = whisper.load_model(model_name)
                self.logger.info(f"Loaded Whisper model: {model_name}")
            
            if self.sentiment_analyzer is None:
                model_name = self.app_config.ai.sentiment_model if self.app_config else "cardiffnlp/twitter-roberta-base-sentiment-latest"
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model=model_name
                )
                self.logger.info(f"Loaded sentiment model: {model_name}")
            
            if self.face_cascade is None:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                self.logger.info("Loaded face cascade classifier")
                
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise
    
    async def _process_impl(self, input_data: Dict[str, Any]) -> ProcessorResult:
        """Process video analysis with enhanced error handling."""
        start_time = time.time()
        
        try:
            # Extract input parameters
            video_path = input_data.get("video_path")
            max_clips = input_data.get("max_clips", 10)
            min_duration = input_data.get("min_duration", 3.0)
            max_duration = input_data.get("max_duration", 30.0)
            
            if not video_path:
                return ProcessorResult(
                    success=False,
                    error="video_path is required"
                )
            
            # Validate video file
            if not Path(video_path).exists():
                return ProcessorResult(
                    success=False,
                    error=f"Video file not found: {video_path}"
                )
            
            # Load models if needed
            await self._load_models()
            
            # Load video
            video = mp.VideoFileClip(video_path)
            duration = video.duration
            
            # Validate duration
            if duration > (self.app_config.video.max_duration if self.app_config else 300.0):
                video.close()
                return ProcessorResult(
                    success=False,
                    error=f"Video too long: {duration}s (max: {self.app_config.video.max_duration if self.app_config else 300.0}s)"
                )
            
            # Extract frames for analysis
            frames = await self._extract_frames_async(video_path, duration)
            
            # Analyze content
            analysis = await self._analyze_content_async(video_path, frames)
            
            # Extract engaging segments
            segments = await self._extract_engaging_segments_async(
                analysis, duration, max_clips, min_duration, max_duration
            )
            
            # Calculate viral scores
            viral_scores = await self._calculate_viral_scores_async(segments)
            
            # Clean up
            video.close()
            
            processing_time = time.time() - start_time
            
            # Prepare result
            result_data = {
                "video_duration": duration,
                "total_segments": len(segments),
                "segments": segments,
                "viral_scores": viral_scores,
                "analysis": analysis,
                "processing_time": processing_time
            }
            
            return ProcessorResult(
                success=True,
                data=result_data,
                processing_time=processing_time,
                metadata={
                    "video_path": video_path,
                    "max_clips": max_clips,
                    "min_duration": min_duration,
                    "max_duration": max_duration
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Video analysis failed: {e}")
            return ProcessorResult(
                success=False,
                error=str(e),
                processing_time=processing_time
            )
    
    async def _extract_frames_async(self, video_path: str, duration: float) -> List[np.ndarray]:
        """Extract frames asynchronously."""
        try:
            # Run frame extraction in thread pool
            loop = asyncio.get_event_loop()
            frames = await loop.run_in_executor(
                None, 
                self._extract_frames_sync, 
                video_path, 
                duration
            )
            return frames
            
        except Exception as e:
            self.logger.error(f"Frame extraction failed: {e}")
            return []
    
    def _extract_frames_sync(self, video_path: str, duration: float) -> List[np.ndarray]:
        """Extract frames synchronously."""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            # Frame sampling interval (every 2 seconds)
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
    
    async def _analyze_content_async(self, video_path: str, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze content asynchronously."""
        try:
            analysis = {
                "face_detection": [],
                "motion_analysis": [],
                "audio_analysis": {},
                "text_analysis": {},
                "engagement_factors": []
            }
            
            # Run analysis tasks in parallel
            tasks = [
                self._analyze_faces_async(frames),
                self._analyze_motion_async(frames),
                self._analyze_audio_async(video_path),
                self._analyze_text_async(video_path)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            face_results, motion_results, audio_results, text_results = results
            
            if not isinstance(face_results, Exception):
                analysis["face_detection"] = face_results
            
            if not isinstance(motion_results, Exception):
                analysis["motion_analysis"] = motion_results
            
            if not isinstance(audio_results, Exception):
                analysis["audio_analysis"] = audio_results
            
            if not isinstance(text_results, Exception):
                analysis["text_analysis"] = text_results
            
            # Calculate engagement factors
            analysis["engagement_factors"] = await self._calculate_engagement_factors_async(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Content analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_faces_async(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Analyze faces asynchronously."""
        try:
            # Run face detection in thread pool
            loop = asyncio.get_event_loop()
            face_results = await loop.run_in_executor(
                None,
                self._analyze_faces_sync,
                frames
            )
            return face_results
            
        except Exception as e:
            self.logger.error(f"Face analysis failed: {e}")
            return []
    
    def _analyze_faces_sync(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Analyze faces synchronously."""
        try:
            face_results = []
            
            for i, frame in enumerate(frames):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                face_results.append({
                    "frame": i,
                    "face_count": len(faces),
                    "faces": faces.tolist() if len(faces) > 0 else []
                })
            
            return face_results
            
        except Exception as e:
            self.logger.error(f"Face analysis failed: {e}")
            return []
    
    async def _analyze_motion_async(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Analyze motion asynchronously."""
        try:
            # Run motion analysis in thread pool
            loop = asyncio.get_event_loop()
            motion_results = await loop.run_in_executor(
                None,
                self._analyze_motion_sync,
                frames
            )
            return motion_results
            
        except Exception as e:
            self.logger.error(f"Motion analysis failed: {e}")
            return []
    
    def _analyze_motion_sync(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Analyze motion synchronously."""
        try:
            motion_results = []
            
            if len(frames) > 1:
                for i in range(1, len(frames)):
                    diff = cv2.absdiff(frames[i-1], frames[i])
                    motion_score = np.mean(diff)
                    motion_results.append({
                        "frame": i,
                        "motion_score": float(motion_score)
                    })
            
            return motion_results
            
        except Exception as e:
            self.logger.error(f"Motion analysis failed: {e}")
            return []
    
    async def _analyze_audio_async(self, video_path: str) -> Dict[str, Any]:
        """Analyze audio asynchronously."""
        try:
            # Run audio analysis in thread pool
            loop = asyncio.get_event_loop()
            audio_results = await loop.run_in_executor(
                None,
                self._analyze_audio_sync,
                video_path
            )
            return audio_results
            
        except Exception as e:
            self.logger.error(f"Audio analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_audio_sync(self, video_path: str) -> Dict[str, Any]:
        """Analyze audio synchronously."""
        try:
            video = mp.VideoFileClip(video_path)
            
            if video.audio:
                audio = video.audio
                volume_levels = self._analyze_audio_levels(audio)
                
                audio_results = {
                    "has_audio": True,
                    "duration": audio.duration,
                    "volume_levels": volume_levels
                }
                
                audio.close()
            else:
                audio_results = {"has_audio": False}
            
            video.close()
            return audio_results
            
        except Exception as e:
            self.logger.error(f"Audio analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_audio_levels(self, audio) -> List[float]:
        """Analyze audio levels."""
        try:
            audio_array = audio.to_soundarray()
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
            self.logger.error(f"Audio level analysis failed: {e}")
            return []
    
    async def _analyze_text_async(self, video_path: str) -> Dict[str, Any]:
        """Analyze text asynchronously."""
        try:
            # Run text analysis in thread pool
            loop = asyncio.get_event_loop()
            text_results = await loop.run_in_executor(
                None,
                self._analyze_text_sync,
                video_path
            )
            return text_results
            
        except Exception as e:
            self.logger.error(f"Text analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_text_sync(self, video_path: str) -> Dict[str, Any]:
        """Analyze text synchronously."""
        try:
            if self.whisper_model:
                result = self.whisper_model.transcribe(video_path)
                return {
                    "transcription": result["text"],
                    "segments": result.get("segments", []),
                    "language": result.get("language", "unknown")
                }
            else:
                return {"error": "Whisper model not loaded"}
                
        except Exception as e:
            self.logger.error(f"Text analysis failed: {e}")
            return {"error": str(e)}
    
    async def _calculate_engagement_factors_async(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate engagement factors asynchronously."""
        try:
            # Run engagement calculation in thread pool
            loop = asyncio.get_event_loop()
            factors = await loop.run_in_executor(
                None,
                self._calculate_engagement_factors_sync,
                analysis
            )
            return factors
            
        except Exception as e:
            self.logger.error(f"Engagement calculation failed: {e}")
            return []
    
    def _calculate_engagement_factors_sync(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate engagement factors synchronously."""
        try:
            factors = []
            
            # Face engagement
            face_data = analysis.get("face_detection", [])
            if face_data:
                avg_faces = np.mean([f["face_count"] for f in face_data])
                factors.append({
                    "factor": "face_presence",
                    "score": min(avg_faces / 2.0, 1.0),
                    "description": f"Average {avg_faces:.1f} faces per frame"
                })
            
            # Motion engagement
            motion_data = analysis.get("motion_analysis", [])
            if motion_data:
                avg_motion = np.mean([m["motion_score"] for m in motion_data])
                factors.append({
                    "factor": "motion_level",
                    "score": min(avg_motion / 50.0, 1.0),
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
                        "score": min(avg_volume * 10, 1.0),
                        "description": f"Average volume: {avg_volume:.3f}"
                    })
            
            # Text engagement
            text_data = analysis.get("text_analysis", {})
            if text_data.get("transcription"):
                transcription = text_data["transcription"]
                word_count = len(transcription.split())
                factors.append({
                    "factor": "text_content",
                    "score": min(word_count / 100.0, 1.0),
                    "description": f"Word count: {word_count}"
                })
            
            return factors
            
        except Exception as e:
            self.logger.error(f"Engagement calculation failed: {e}")
            return []
    
    async def _extract_engaging_segments_async(self, analysis: Dict[str, Any], 
                                             duration: float, max_clips: int,
                                             min_duration: float, max_duration: float) -> List[Dict[str, Any]]:
        """Extract engaging segments asynchronously."""
        try:
            # Run segment extraction in thread pool
            loop = asyncio.get_event_loop()
            segments = await loop.run_in_executor(
                None,
                self._extract_engaging_segments_sync,
                analysis, duration, max_clips, min_duration, max_duration
            )
            return segments
            
        except Exception as e:
            self.logger.error(f"Segment extraction failed: {e}")
            return []
    
    def _extract_engaging_segments_sync(self, analysis: Dict[str, Any], duration: float,
                                      max_clips: int, min_duration: float, max_duration: float) -> List[Dict[str, Any]]:
        """Extract engaging segments synchronously."""
        try:
            segments = []
            factors = analysis.get("engagement_factors", [])
            
            if not factors:
                return segments
            
            # Calculate segment duration
            segment_duration = self.app_config.video.segment_duration if self.app_config else 5.0
            num_segments = int(duration / segment_duration)
            
            for i in range(num_segments):
                start_time = i * segment_duration
                end_time = min((i + 1) * segment_duration, duration)
                
                if end_time - start_time < min_duration:
                    continue
                
                # Calculate engagement score
                engagement_score = self._calculate_segment_score(
                    analysis, start_time, end_time, factors
                )
                
                threshold = self.app_config.video.engagement_threshold if self.app_config else 0.3
                if engagement_score > threshold:
                    segments.append({
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": end_time - start_time,
                        "engagement_score": engagement_score,
                        "segment_id": f"segment_{i}",
                        "title": f"Engaging Segment {i+1}",
                        "description": f"High engagement segment from {start_time:.1f}s to {end_time:.1f}s"
                    })
            
            # Sort by engagement score and limit
            segments.sort(key=lambda x: x["engagement_score"], reverse=True)
            return segments[:max_clips]
            
        except Exception as e:
            self.logger.error(f"Segment extraction failed: {e}")
            return []
    
    def _calculate_segment_score(self, analysis: Dict[str, Any], start_time: float, 
                               end_time: float, factors: List[Dict[str, Any]]) -> float:
        """Calculate engagement score for a segment."""
        try:
            score = 0.0
            weight_sum = 0.0
            
            # Face presence weight
            face_weight = 0.3
            face_data = analysis.get("face_detection", [])
            if face_data:
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
    
    async def _calculate_viral_scores_async(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate viral scores asynchronously."""
        try:
            # Run viral score calculation in thread pool
            loop = asyncio.get_event_loop()
            viral_scores = await loop.run_in_executor(
                None,
                self._calculate_viral_scores_sync,
                segments
            )
            return viral_scores
            
        except Exception as e:
            self.logger.error(f"Viral score calculation failed: {e}")
            return {}
    
    def _calculate_viral_scores_sync(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate viral scores synchronously."""
        try:
            viral_scores = {}
            
            for segment in segments:
                segment_id = segment["segment_id"]
                
                # Base score from engagement
                base_score = segment["engagement_score"]
                
                # Duration bonus
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
        """Get viral potential label."""
        if score >= 0.8:
            return "High"
        elif score >= 0.6:
            return "Medium"
        elif score >= 0.4:
            return "Low"
        else:
            return "Very Low"


