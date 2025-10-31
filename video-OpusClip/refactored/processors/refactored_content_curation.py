"""
Refactored Content Curation Engine

Improved implementation with better architecture, performance optimization,
and enhanced error handling.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any, Tuple, Union
import asyncio
import numpy as np
import cv2
import librosa
import torch
import torch.nn as nn
from dataclasses import dataclass
from enum import Enum
import structlog
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor
import pickle
from collections import defaultdict

from ..core.base_processor import BaseProcessor, ProcessorConfig, ProcessingResult, ProcessorStatus
from ..core.config_manager import config_manager
from ..error_handling import ErrorHandler, ProcessingError, ValidationError

logger = structlog.get_logger("refactored_content_curation")
error_handler = ErrorHandler()

class EngagementType(Enum):
    """Types of engagement signals detected in video content."""
    VISUAL_ATTENTION = "visual_attention"
    AUDIO_ENERGY = "audio_energy"
    SPEECH_CLARITY = "speech_clarity"
    EMOTIONAL_INTENSITY = "emotional_intensity"
    ACTION_MOMENTS = "action_moments"
    TRANSITION_POINTS = "transition_points"
    FACIAL_EXPRESSIONS = "facial_expressions"
    TEXT_OVERLAY = "text_overlay"

@dataclass
class EngagementScore:
    """Engagement score for a specific time segment."""
    timestamp: float
    duration: float
    score: float
    engagement_type: EngagementType
    confidence: float
    metadata: Dict[str, Any]
    frame_index: int = 0

@dataclass
class VideoSegment:
    """A segment of video identified as potentially engaging."""
    start_time: float
    end_time: float
    duration: float
    engagement_scores: List[EngagementScore]
    overall_score: float
    content_type: str
    metadata: Dict[str, Any]
    segment_id: str = ""
    quality_metrics: Dict[str, float] = None

@dataclass
class ContentCurationConfig:
    """Configuration for content curation."""
    min_segment_duration: float = 3.0
    max_segment_duration: float = 30.0
    engagement_threshold: float = 0.6
    viral_threshold: float = 0.7
    smoothing_window: int = 5
    max_clips: int = 10
    min_clips: int = 1
    target_duration: float = 12.0
    duration_tolerance: float = 3.0
    enable_ai_analysis: bool = True
    enable_audio_analysis: bool = True
    enable_visual_analysis: bool = True
    cache_enabled: bool = True

class EnhancedEngagementAnalyzer:
    """Enhanced engagement analyzer with improved algorithms."""
    
    def __init__(self, config: ContentCurationConfig):
        self.config = config
        self.visual_model = None
        self.audio_model = None
        self.face_detector = None
        self.text_detector = None
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models for engagement analysis."""
        try:
            # Load visual attention model
            self.visual_model = self._load_visual_model()
            
            # Load audio analysis model
            self.audio_model = self._load_audio_model()
            
            # Load face detection model
            self.face_detector = self._load_face_detector()
            
            # Load text detection model
            self.text_detector = self._load_text_detector()
            
            logger.info("Enhanced engagement analysis models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load engagement models: {e}")
            raise ProcessingError(f"Model loading failed: {e}")
    
    def _load_visual_model(self):
        """Load visual attention detection model."""
        # Placeholder - would load actual computer vision model
        return EnhancedVisualAnalyzer()
    
    def _load_audio_model(self):
        """Load audio analysis model."""
        # Placeholder - would load actual audio analysis model
        return EnhancedAudioAnalyzer()
    
    def _load_face_detector(self):
        """Load face detection model."""
        # Placeholder - would load actual face detection model
        return EnhancedFaceDetector()
    
    def _load_text_detector(self):
        """Load text detection model."""
        # Placeholder - would load actual text detection model
        return EnhancedTextDetector()
    
    async def analyze_frames(self, frames: List[np.ndarray], fps: float = 30.0) -> List[EngagementScore]:
        """Analyze video frames for engagement signals."""
        try:
            scores = []
            
            # Process frames in parallel for better performance
            with ThreadPoolExecutor(max_workers=4) as executor:
                tasks = []
                for i, frame in enumerate(frames):
                    task = executor.submit(self._analyze_single_frame, frame, i, fps)
                    tasks.append(task)
                
                # Collect results
                for task in asyncio.as_completed(tasks):
                    frame_scores = await task
                    scores.extend(frame_scores)
            
            # Sort by timestamp
            scores.sort(key=lambda x: x.timestamp)
            
            return scores
            
        except Exception as e:
            logger.error(f"Frame analysis failed: {e}")
            raise ProcessingError(f"Frame analysis failed: {e}")
    
    async def _analyze_single_frame(self, frame: np.ndarray, frame_index: int, fps: float) -> List[EngagementScore]:
        """Analyze a single frame for engagement signals."""
        try:
            timestamp = frame_index / fps
            scores = []
            
            # Visual attention analysis
            attention_score = await self._analyze_visual_attention(frame)
            scores.append(EngagementScore(
                timestamp=timestamp,
                duration=1.0 / fps,
                score=attention_score,
                engagement_type=EngagementType.VISUAL_ATTENTION,
                confidence=0.8,
                metadata={"frame_index": frame_index},
                frame_index=frame_index
            ))
            
            # Action detection
            action_score = await self._detect_action_moments(frame)
            scores.append(EngagementScore(
                timestamp=timestamp,
                duration=1.0 / fps,
                score=action_score,
                engagement_type=EngagementType.ACTION_MOMENTS,
                confidence=0.7,
                metadata={"frame_index": frame_index},
                frame_index=frame_index
            ))
            
            # Emotional intensity
            emotion_score = await self._analyze_emotional_intensity(frame)
            scores.append(EngagementScore(
                timestamp=timestamp,
                duration=1.0 / fps,
                score=emotion_score,
                engagement_type=EngagementType.EMOTIONAL_INTENSITY,
                confidence=0.6,
                metadata={"frame_index": frame_index},
                frame_index=frame_index
            ))
            
            # Facial expressions
            facial_score = await self._analyze_facial_expressions(frame)
            scores.append(EngagementScore(
                timestamp=timestamp,
                duration=1.0 / fps,
                score=facial_score,
                engagement_type=EngagementType.FACIAL_EXPRESSIONS,
                confidence=0.7,
                metadata={"frame_index": frame_index},
                frame_index=frame_index
            ))
            
            # Text overlay detection
            text_score = await self._detect_text_overlay(frame)
            scores.append(EngagementScore(
                timestamp=timestamp,
                duration=1.0 / fps,
                score=text_score,
                engagement_type=EngagementType.TEXT_OVERLAY,
                confidence=0.8,
                metadata={"frame_index": frame_index},
                frame_index=frame_index
            ))
            
            return scores
            
        except Exception as e:
            logger.error(f"Single frame analysis failed: {e}")
            return []
    
    async def _analyze_visual_attention(self, frame: np.ndarray) -> float:
        """Analyze visual attention in a frame."""
        try:
            # Use enhanced visual analyzer
            return await self.visual_model.analyze_attention(frame)
        except Exception as e:
            logger.error(f"Visual attention analysis failed: {e}")
            return 0.5
    
    async def _detect_action_moments(self, frame: np.ndarray) -> float:
        """Detect action moments in a frame."""
        try:
            # Use enhanced visual analyzer
            return await self.visual_model.detect_action(frame)
        except Exception as e:
            logger.error(f"Action detection failed: {e}")
            return 0.0
    
    async def _analyze_emotional_intensity(self, frame: np.ndarray) -> float:
        """Analyze emotional intensity in a frame."""
        try:
            # Use enhanced visual analyzer
            return await self.visual_model.analyze_emotion(frame)
        except Exception as e:
            logger.error(f"Emotional intensity analysis failed: {e}")
            return 0.5
    
    async def _analyze_facial_expressions(self, frame: np.ndarray) -> float:
        """Analyze facial expressions in a frame."""
        try:
            # Use face detector
            return await self.face_detector.analyze_expressions(frame)
        except Exception as e:
            logger.error(f"Facial expression analysis failed: {e}")
            return 0.0
    
    async def _detect_text_overlay(self, frame: np.ndarray) -> float:
        """Detect text overlay in a frame."""
        try:
            # Use text detector
            return await self.text_detector.detect_text(frame)
        except Exception as e:
            logger.error(f"Text overlay detection failed: {e}")
            return 0.0
    
    async def analyze_audio(self, audio_path: str) -> List[EngagementScore]:
        """Analyze audio for engagement signals."""
        try:
            # Use enhanced audio analyzer
            return await self.audio_model.analyze_audio(audio_path)
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return []

class EnhancedSegmentDetector:
    """Enhanced segment detector with improved algorithms."""
    
    def __init__(self, config: ContentCurationConfig):
        self.config = config
        self.segment_cache = {}
    
    async def detect_segments(self, 
                            visual_scores: List[EngagementScore],
                            audio_scores: List[EngagementScore]) -> List[VideoSegment]:
        """Detect high-engagement segments from visual and audio scores."""
        try:
            # Combine and align scores
            combined_scores = await self._combine_scores(visual_scores, audio_scores)
            
            # Smooth scores to reduce noise
            smoothed_scores = await self._smooth_scores(combined_scores)
            
            # Find peaks and valleys
            peaks = await self._find_engagement_peaks(smoothed_scores)
            
            # Create segments from peaks
            segments = await self._create_segments(peaks, smoothed_scores)
            
            # Filter and optimize segments
            optimized_segments = await self._optimize_segments(segments)
            
            # Calculate quality metrics
            for segment in optimized_segments:
                segment.quality_metrics = await self._calculate_quality_metrics(segment)
            
            return optimized_segments
            
        except Exception as e:
            logger.error(f"Segment detection failed: {e}")
            raise ProcessingError(f"Segment detection failed: {e}")
    
    async def _combine_scores(self, 
                            visual_scores: List[EngagementScore],
                            audio_scores: List[EngagementScore]) -> List[EngagementScore]:
        """Combine visual and audio scores with intelligent weighting."""
        try:
            combined = []
            
            # Create time-based mapping
            visual_map = defaultdict(list)
            audio_map = defaultdict(list)
            
            for score in visual_scores:
                visual_map[score.timestamp].append(score)
            
            for score in audio_scores:
                audio_map[score.timestamp].append(score)
            
            # Get all unique timestamps
            all_timestamps = set(visual_map.keys()) | set(audio_map.keys())
            
            for timestamp in sorted(all_timestamps):
                visual_scores_at_time = visual_map[timestamp]
                audio_scores_at_time = audio_map[timestamp]
                
                # Calculate combined score with intelligent weighting
                combined_score = await self._calculate_combined_score(
                    visual_scores_at_time, audio_scores_at_time
                )
                
                # Create combined engagement score
                combined.append(EngagementScore(
                    timestamp=timestamp,
                    duration=visual_scores_at_time[0].duration if visual_scores_at_time else 1.0/30.0,
                    score=combined_score,
                    engagement_type=EngagementType.VISUAL_ATTENTION,  # Default type
                    confidence=0.8,
                    metadata={
                        "visual_scores": [s.score for s in visual_scores_at_time],
                        "audio_scores": [s.score for s in audio_scores_at_time],
                        "combined": True
                    }
                ))
            
            return combined
            
        except Exception as e:
            logger.error(f"Score combination failed: {e}")
            raise ProcessingError(f"Score combination failed: {e}")
    
    async def _calculate_combined_score(self, 
                                      visual_scores: List[EngagementScore],
                                      audio_scores: List[EngagementScore]) -> float:
        """Calculate combined score with intelligent weighting."""
        try:
            # Weight different engagement types
            type_weights = {
                EngagementType.VISUAL_ATTENTION: 0.25,
                EngagementType.ACTION_MOMENTS: 0.20,
                EngagementType.EMOTIONAL_INTENSITY: 0.20,
                EngagementType.FACIAL_EXPRESSIONS: 0.15,
                EngagementType.TEXT_OVERLAY: 0.10,
                EngagementType.AUDIO_ENERGY: 0.10
            }
            
            combined_score = 0.0
            total_weight = 0.0
            
            # Process visual scores
            for score in visual_scores:
                weight = type_weights.get(score.engagement_type, 0.1)
                combined_score += score.score * weight * score.confidence
                total_weight += weight * score.confidence
            
            # Process audio scores
            for score in audio_scores:
                weight = type_weights.get(score.engagement_type, 0.1)
                combined_score += score.score * weight * score.confidence
                total_weight += weight * score.confidence
            
            return combined_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Combined score calculation failed: {e}")
            return 0.5
    
    async def _smooth_scores(self, scores: List[EngagementScore]) -> List[EngagementScore]:
        """Smooth engagement scores to reduce noise."""
        try:
            if len(scores) < self.config.smoothing_window:
                return scores
            
            smoothed = []
            
            for i in range(len(scores)):
                start_idx = max(0, i - self.config.smoothing_window // 2)
                end_idx = min(len(scores), i + self.config.smoothing_window // 2 + 1)
                
                window_scores = scores[start_idx:end_idx]
                smoothed_score = np.mean([s.score for s in window_scores])
                
                # Calculate confidence based on score consistency
                score_variance = np.var([s.score for s in window_scores])
                confidence = max(0.1, 1.0 - min(score_variance, 1.0))
                
                smoothed.append(EngagementScore(
                    timestamp=scores[i].timestamp,
                    duration=scores[i].duration,
                    score=smoothed_score,
                    engagement_type=scores[i].engagement_type,
                    confidence=confidence,
                    metadata={**scores[i].metadata, "smoothed": True},
                    frame_index=scores[i].frame_index
                ))
            
            return smoothed
            
        except Exception as e:
            logger.error(f"Score smoothing failed: {e}")
            return scores
    
    async def _find_engagement_peaks(self, scores: List[EngagementScore]) -> List[int]:
        """Find peaks in engagement scores using advanced algorithms."""
        try:
            if not scores:
                return []
            
            score_values = [s.score for s in scores]
            peaks = []
            
            # Use scipy's peak detection for better accuracy
            try:
                from scipy.signal import find_peaks
                peak_indices, properties = find_peaks(
                    score_values,
                    height=self.config.engagement_threshold,
                    distance=5,  # Minimum distance between peaks
                    prominence=0.1  # Minimum prominence
                )
                peaks = peak_indices.tolist()
            except ImportError:
                # Fallback to simple peak detection
                for i in range(1, len(score_values) - 1):
                    if (score_values[i] > score_values[i-1] and 
                        score_values[i] > score_values[i+1] and
                        score_values[i] > self.config.engagement_threshold):
                        peaks.append(i)
            
            return peaks
            
        except Exception as e:
            logger.error(f"Peak finding failed: {e}")
            return []
    
    async def _create_segments(self, 
                             peaks: List[int], 
                             scores: List[EngagementScore]) -> List[VideoSegment]:
        """Create video segments from engagement peaks."""
        try:
            segments = []
            
            for peak_idx in peaks:
                if peak_idx >= len(scores):
                    continue
                
                peak_score = scores[peak_idx]
                
                # Find segment boundaries using advanced algorithms
                start_idx = await self._find_segment_start(peak_idx, scores)
                end_idx = await self._find_segment_end(peak_idx, scores)
                
                if start_idx is None or end_idx is None:
                    continue
                
                # Calculate segment properties
                segment_scores = scores[start_idx:end_idx+1]
                overall_score = np.mean([s.score for s in segment_scores])
                
                start_time = scores[start_idx].timestamp
                end_time = scores[end_idx].timestamp + scores[end_idx].duration
                duration = end_time - start_time
                
                # Check duration constraints
                if duration < self.config.min_segment_duration or duration > self.config.max_segment_duration:
                    continue
                
                # Generate segment ID
                segment_id = f"segment_{int(start_time)}_{int(end_time)}"
                
                segment = VideoSegment(
                    segment_id=segment_id,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    engagement_scores=segment_scores,
                    overall_score=overall_score,
                    content_type="engagement_peak",
                    metadata={
                        "peak_timestamp": peak_score.timestamp,
                        "peak_score": peak_score.score,
                        "segment_length": len(segment_scores),
                        "score_variance": np.var([s.score for s in segment_scores])
                    }
                )
                
                segments.append(segment)
            
            return segments
            
        except Exception as e:
            logger.error(f"Segment creation failed: {e}")
            return []
    
    async def _find_segment_start(self, peak_idx: int, scores: List[EngagementScore]) -> Optional[int]:
        """Find the start of a segment around a peak using advanced algorithms."""
        try:
            peak_score = scores[peak_idx].score
            threshold = peak_score * 0.7  # 70% of peak score
            
            # Search backwards from peak
            for i in range(peak_idx, -1, -1):
                if scores[i].score < threshold:
                    return i + 1
            
            return 0
            
        except Exception as e:
            logger.error(f"Segment start finding failed: {e}")
            return None
    
    async def _find_segment_end(self, peak_idx: int, scores: List[EngagementScore]) -> Optional[int]:
        """Find the end of a segment around a peak using advanced algorithms."""
        try:
            peak_score = scores[peak_idx].score
            threshold = peak_score * 0.7  # 70% of peak score
            
            # Search forwards from peak
            for i in range(peak_idx, len(scores)):
                if scores[i].score < threshold:
                    return i - 1
            
            return len(scores) - 1
            
        except Exception as e:
            logger.error(f"Segment end finding failed: {e}")
            return None
    
    async def _optimize_segments(self, segments: List[VideoSegment]) -> List[VideoSegment]:
        """Optimize segments by removing overlaps and low-quality segments."""
        try:
            if not segments:
                return []
            
            # Sort by score (highest first)
            sorted_segments = sorted(segments, key=lambda s: s.overall_score, reverse=True)
            
            optimized = []
            used_times = []
            
            for segment in sorted_segments:
                # Check for overlap with existing segments
                overlap = False
                for used_start, used_end in used_times:
                    if (segment.start_time < used_end and segment.end_time > used_start):
                        overlap = True
                        break
                
                if not overlap and segment.overall_score > self.config.engagement_threshold:
                    optimized.append(segment)
                    used_times.append((segment.start_time, segment.end_time))
            
            # Limit to max clips
            return optimized[:self.config.max_clips]
            
        except Exception as e:
            logger.error(f"Segment optimization failed: {e}")
            return segments
    
    async def _calculate_quality_metrics(self, segment: VideoSegment) -> Dict[str, float]:
        """Calculate quality metrics for a segment."""
        try:
            scores = [s.score for s in segment.engagement_scores]
            
            return {
                "average_score": np.mean(scores),
                "max_score": np.max(scores),
                "min_score": np.min(scores),
                "score_variance": np.var(scores),
                "score_consistency": 1.0 - min(np.var(scores), 1.0),
                "duration_score": 1.0 - abs(segment.duration - self.config.target_duration) / self.config.target_duration,
                "overall_quality": np.mean([
                    np.mean(scores),
                    1.0 - min(np.var(scores), 1.0),
                    1.0 - abs(segment.duration - self.config.target_duration) / self.config.target_duration
                ])
            }
            
        except Exception as e:
            logger.error(f"Quality metrics calculation failed: {e}")
            return {"overall_quality": 0.5}

class RefactoredContentCurationEngine(BaseProcessor):
    """Refactored content curation engine with improved architecture."""
    
    def __init__(self):
        config = ProcessorConfig(
            name="content_curation_engine",
            version="2.0.0",
            enabled=config_manager.is_feature_enabled("content_curation"),
            max_workers=config_manager.get("processing.max_workers", 4),
            timeout=config_manager.get("processing.timeout", 300.0),
            retry_attempts=config_manager.get("processing.retry_attempts", 3),
            cache_enabled=config_manager.get("cache.enabled", True),
            cache_ttl=config_manager.get("cache.ttl", 3600.0)
        )
        
        super().__init__(config)
        
        # Initialize components
        self.curation_config = self._load_curation_config()
        self.engagement_analyzer = EnhancedEngagementAnalyzer(self.curation_config)
        self.segment_detector = EnhancedSegmentDetector(self.curation_config)
        
        self.logger = structlog.get_logger("refactored_content_curation")
    
    def _load_curation_config(self) -> ContentCurationConfig:
        """Load content curation configuration."""
        feature_config = config_manager.get_processor_config("content_curation")
        
        return ContentCurationConfig(
            min_segment_duration=feature_config.get("min_duration", 3.0),
            max_segment_duration=feature_config.get("max_duration", 60.0),
            engagement_threshold=feature_config.get("engagement_threshold", 0.6),
            viral_threshold=feature_config.get("viral_threshold", 0.7),
            max_clips=feature_config.get("max_clips", 10),
            target_duration=feature_config.get("target_duration", 12.0),
            enable_ai_analysis=feature_config.get("enable_ai_analysis", True),
            enable_audio_analysis=feature_config.get("enable_audio_analysis", True),
            enable_visual_analysis=feature_config.get("enable_visual_analysis", True)
        )
    
    async def _initialize_impl(self) -> bool:
        """Initialize the content curation engine."""
        try:
            # Initialize engagement analyzer
            await self.engagement_analyzer._load_models()
            
            self.logger.info("Refactored content curation engine initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Content curation engine initialization failed: {e}")
            return False
    
    async def _process_impl(self, input_data: Dict[str, Any]) -> ProcessingResult:
        """Process video for content curation."""
        try:
            video_path = input_data.get("video_path")
            if not video_path:
                raise ValidationError("video_path is required")
            
            self.logger.info(f"Starting content curation for video: {video_path}")
            
            # Extract frames and audio
            frames, fps = await self._extract_frames(video_path)
            audio_path = await self._extract_audio(video_path)
            
            # Analyze visual content
            visual_scores = []
            if self.curation_config.enable_visual_analysis:
                visual_scores = await self.engagement_analyzer.analyze_frames(frames, fps)
            
            # Analyze audio content
            audio_scores = []
            if self.curation_config.enable_audio_analysis:
                audio_scores = await self.engagement_analyzer.analyze_audio(audio_path)
            
            # Detect segments
            segments = await self.segment_detector.detect_segments(visual_scores, audio_scores)
            
            # Convert segments to result format
            clips = []
            for segment in segments:
                clips.append({
                    "segment_id": segment.segment_id,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "duration": segment.duration,
                    "score": segment.overall_score,
                    "content_type": segment.content_type,
                    "quality_metrics": segment.quality_metrics,
                    "metadata": segment.metadata
                })
            
            # Calculate processing statistics
            processing_stats = {
                "total_frames": len(frames),
                "fps": fps,
                "visual_scores": len(visual_scores),
                "audio_scores": len(audio_scores),
                "segments_found": len(segments),
                "clips_generated": len(clips),
                "average_clip_duration": np.mean([c["duration"] for c in clips]) if clips else 0,
                "best_clip_score": max([c["score"] for c in clips]) if clips else 0
            }
            
            result_data = {
                "clips": clips,
                "processing_stats": processing_stats,
                "config_used": {
                    "min_segment_duration": self.curation_config.min_segment_duration,
                    "max_segment_duration": self.curation_config.max_segment_duration,
                    "engagement_threshold": self.curation_config.engagement_threshold,
                    "max_clips": self.curation_config.max_clips
                }
            }
            
            return ProcessingResult(
                success=True,
                processor_name=self.config.name,
                processing_time=time.time() - (self.start_time or time.time()),
                result_data=result_data,
                metadata={
                    "video_path": video_path,
                    "fps": fps,
                    "total_frames": len(frames)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Content curation processing failed: {e}")
            return ProcessingResult(
                success=False,
                processor_name=self.config.name,
                processing_time=time.time() - (self.start_time or time.time()),
                result_data={},
                error_message=str(e)
            )
    
    async def _extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], float]:
        """Extract frames from video with improved performance."""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ProcessingError(f"Could not open video file: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frames = []
            frame_count = 0
            
            # Extract frames with sampling for long videos
            sample_rate = max(1, total_frames // 1000)  # Sample every N frames for long videos
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_rate == 0:
                    frames.append(frame)
                
                frame_count += 1
                
                # Limit frames for very long videos
                if len(frames) >= 2000:  # Max 2000 frames
                    break
            
            cap.release()
            
            self.logger.info(f"Extracted {len(frames)} frames from video (fps: {fps})")
            return frames, fps
            
        except Exception as e:
            self.logger.error(f"Frame extraction failed: {e}")
            raise ProcessingError(f"Frame extraction failed: {e}")
    
    async def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video with improved error handling."""
        try:
            audio_path = f"/tmp/audio_{int(time.time())}.wav"
            
            import subprocess
            result = subprocess.run([
                "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", 
                "-ar", "44100", "-ac", "2", audio_path, "-y"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                raise ProcessingError(f"Audio extraction failed: {result.stderr}")
            
            return audio_path
            
        except Exception as e:
            self.logger.error(f"Audio extraction failed: {e}")
            raise ProcessingError(f"Audio extraction failed: {e}")

# Placeholder classes for enhanced analyzers
class EnhancedVisualAnalyzer:
    """Enhanced visual analyzer placeholder."""
    async def analyze_attention(self, frame: np.ndarray) -> float:
        return 0.5
    
    async def detect_action(self, frame: np.ndarray) -> float:
        return 0.0
    
    async def analyze_emotion(self, frame: np.ndarray) -> float:
        return 0.5

class EnhancedAudioAnalyzer:
    """Enhanced audio analyzer placeholder."""
    async def analyze_audio(self, audio_path: str) -> List[EngagementScore]:
        return []

class EnhancedFaceDetector:
    """Enhanced face detector placeholder."""
    async def analyze_expressions(self, frame: np.ndarray) -> float:
        return 0.0

class EnhancedTextDetector:
    """Enhanced text detector placeholder."""
    async def detect_text(self, frame: np.ndarray) -> float:
        return 0.0

# Export the main class
__all__ = [
    "RefactoredContentCurationEngine",
    "ContentCurationConfig",
    "EngagementScore",
    "VideoSegment",
    "EngagementType"
]


