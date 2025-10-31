"""
Advanced Video Processor for OpusClip Improved
=============================================

High-performance video processing with GPU acceleration and advanced features.
"""

import asyncio
import logging
import time
import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import librosa
import soundfile as sf
from dataclasses import dataclass
from enum import Enum

from .schemas import VideoFormat, QualityLevel, PlatformType
from .exceptions import VideoProcessingError, create_video_processing_error

logger = logging.getLogger(__name__)


class ProcessingMode(str, Enum):
    """Video processing modes"""
    CPU = "cpu"
    GPU = "gpu"
    HYBRID = "hybrid"


class EffectType(str, Enum):
    """Video effect types"""
    BLUR = "blur"
    SHARPEN = "sharpen"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    SATURATION = "saturation"
    ZOOM = "zoom"
    PAN = "pan"
    FADE = "fade"
    CROSSFADE = "crossfade"


@dataclass
class VideoSegment:
    """Video segment information"""
    start_time: float
    end_time: float
    duration: float
    confidence: float
    segment_type: str
    metadata: Dict[str, Any]


@dataclass
class ProcessingConfig:
    """Video processing configuration"""
    quality: QualityLevel
    format: VideoFormat
    resolution: Optional[Tuple[int, int]] = None
    fps: Optional[float] = None
    bitrate: Optional[str] = None
    codec: str = "libx264"
    audio_codec: str = "aac"
    processing_mode: ProcessingMode = ProcessingMode.CPU
    enable_gpu: bool = False
    max_workers: int = 4


class VideoProcessor:
    """Advanced video processor with GPU acceleration"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_available = torch.cuda.is_available()
        self.temp_dir = Path("./temp")
        self.output_dir = Path("./output")
        self._ensure_directories()
        self._initialize_models()
    
    def _ensure_directories(self):
        """Ensure required directories exist"""
        self.temp_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
    
    def _initialize_models(self):
        """Initialize AI models for video processing"""
        try:
            if self.gpu_available:
                logger.info(f"GPU available: {torch.cuda.get_device_name()}")
                # Initialize GPU-accelerated models
                self.face_detector = cv2.dnn.readNetFromCaffe(
                    "models/opencv_face_detector.prototxt",
                    "models/opencv_face_detector.caffemodel"
                )
                self.face_detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.face_detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                logger.info("Using CPU for video processing")
                self.face_detector = cv2.dnn.readNetFromCaffe(
                    "models/opencv_face_detector.prototxt",
                    "models/opencv_face_detector.caffemodel"
                )
        except Exception as e:
            logger.warning(f"Failed to initialize GPU models: {e}")
            self.face_detector = None
    
    async def extract_segments(
        self,
        video_path: str,
        segment_duration: float = 30.0,
        overlap: float = 5.0
    ) -> List[VideoSegment]:
        """Extract video segments with overlap"""
        try:
            segments = []
            with VideoFileClip(video_path) as video:
                duration = video.duration
                
                start_time = 0
                while start_time < duration:
                    end_time = min(start_time + segment_duration, duration)
                    actual_duration = end_time - start_time
                    
                    segment = VideoSegment(
                        start_time=start_time,
                        end_time=end_time,
                        duration=actual_duration,
                        confidence=1.0,
                        segment_type="extracted",
                        metadata={"overlap": overlap}
                    )
                    segments.append(segment)
                    
                    start_time = end_time - overlap
                    if start_time >= duration:
                        break
            
            logger.info(f"Extracted {len(segments)} segments from video")
            return segments
            
        except Exception as e:
            raise create_video_processing_error("segment_extraction", video_path, e)
    
    async def detect_scene_changes(
        self,
        video_path: str,
        threshold: float = 30.0
    ) -> List[float]:
        """Detect scene changes in video"""
        try:
            cap = cv2.VideoCapture(video_path)
            scene_changes = []
            prev_frame = None
            frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if prev_frame is not None:
                    # Convert to grayscale for comparison
                    gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    
                    # Calculate frame difference
                    diff = cv2.absdiff(gray_prev, gray_current)
                    diff_score = np.mean(diff)
                    
                    # Check for scene change
                    if diff_score > threshold:
                        timestamp = frame_count / fps
                        scene_changes.append(timestamp)
                
                prev_frame = frame
                frame_count += 1
            
            cap.release()
            logger.info(f"Detected {len(scene_changes)} scene changes")
            return scene_changes
            
        except Exception as e:
            raise create_video_processing_error("scene_detection", video_path, e)
    
    async def detect_faces(
        self,
        video_path: str,
        confidence_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Detect faces in video with high accuracy"""
        try:
            if not self.face_detector:
                logger.warning("Face detector not available, using basic detection")
                return await self._detect_faces_basic(video_path)
            
            cap = cv2.VideoCapture(video_path)
            face_detections = []
            frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Prepare frame for face detection
                blob = cv2.dnn.blobFromImage(
                    frame, 1.0, (300, 300), [104, 117, 123]
                )
                self.face_detector.setInput(blob)
                detections = self.face_detector.forward()
                
                faces_in_frame = []
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    
                    if confidence > confidence_threshold:
                        # Get face coordinates
                        x1 = int(detections[0, 0, i, 3] * frame.shape[1])
                        y1 = int(detections[0, 0, i, 4] * frame.shape[0])
                        x2 = int(detections[0, 0, i, 5] * frame.shape[1])
                        y2 = int(detections[0, 0, i, 6] * frame.shape[0])
                        
                        faces_in_frame.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": float(confidence)
                        })
                
                if faces_in_frame:
                    face_detections.append({
                        "timestamp": frame_count / fps,
                        "faces": faces_in_frame,
                        "face_count": len(faces_in_frame)
                    })
                
                frame_count += 1
            
            cap.release()
            logger.info(f"Detected faces in {len(face_detections)} frames")
            return face_detections
            
        except Exception as e:
            raise create_video_processing_error("face_detection", video_path, e)
    
    async def _detect_faces_basic(self, video_path: str) -> List[Dict[str, Any]]:
        """Basic face detection using OpenCV Haar cascades"""
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            cap = cv2.VideoCapture(video_path)
            face_detections = []
            frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    face_detections.append({
                        "timestamp": frame_count / fps,
                        "faces": faces.tolist(),
                        "face_count": len(faces)
                    })
                
                frame_count += 1
            
            cap.release()
            return face_detections
            
        except Exception as e:
            raise create_video_processing_error("basic_face_detection", video_path, e)
    
    async def extract_audio_features(
        self,
        video_path: str,
        sample_rate: int = 22050
    ) -> Dict[str, Any]:
        """Extract audio features from video"""
        try:
            # Extract audio
            audio_path = self.temp_dir / f"audio_{int(time.time())}.wav"
            with VideoFileClip(video_path) as clip:
                if clip.audio:
                    clip.audio.write_audiofile(str(audio_path), verbose=False, logger=None)
                else:
                    return {"error": "No audio track found"}
            
            # Load audio
            y, sr = librosa.load(str(audio_path), sr=sample_rate)
            
            # Extract features
            features = {
                "duration": len(y) / sr,
                "sample_rate": sr,
                "rms_energy": float(np.mean(librosa.feature.rms(y=y)[0])),
                "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])),
                "spectral_rolloff": float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])),
                "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(y)[0])),
                "mfcc": librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1).tolist(),
                "tempo": float(librosa.beat.tempo(y=y, sr=sr)[0]),
                "loudness": float(np.mean(librosa.amplitude_to_db(librosa.feature.rms(y=y)[0])))
            }
            
            # Clean up
            os.remove(audio_path)
            
            logger.info("Audio features extracted successfully")
            return features
            
        except Exception as e:
            raise create_video_processing_error("audio_feature_extraction", video_path, e)
    
    async def create_clip(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        output_path: str,
        config: ProcessingConfig
    ) -> str:
        """Create a video clip with specified parameters"""
        try:
            with VideoFileClip(video_path) as video:
                # Create clip
                clip = video.subclip(start_time, end_time)
                
                # Apply quality settings
                if config.resolution:
                    clip = clip.resize(config.resolution)
                
                if config.fps:
                    clip = clip.set_fps(config.fps)
                
                # Write clip
                clip.write_videofile(
                    output_path,
                    codec=config.codec,
                    audio_codec=config.audio_codec,
                    bitrate=config.bitrate,
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True,
                    verbose=False,
                    logger=None
                )
                
                clip.close()
            
            logger.info(f"Clip created: {output_path}")
            return output_path
            
        except Exception as e:
            raise create_video_processing_error("clip_creation", video_path, e)
    
    async def add_captions(
        self,
        video_path: str,
        captions: List[Dict[str, Any]],
        output_path: str,
        config: ProcessingConfig
    ) -> str:
        """Add captions to video"""
        try:
            with VideoFileClip(video_path) as video:
                # Create caption clips
                caption_clips = []
                
                for caption in captions:
                    start_time = caption.get('start_time', 0)
                    end_time = caption.get('end_time', start_time + 3)
                    text = caption.get('text', '')
                    
                    if text:
                        # Create text clip
                        txt_clip = TextClip(
                            text,
                            fontsize=24,
                            color='white',
                            font='Arial-Bold',
                            stroke_color='black',
                            stroke_width=2
                        ).set_position(('center', 'bottom')).set_duration(end_time - start_time).set_start(start_time)
                        
                        caption_clips.append(txt_clip)
                
                # Composite video with captions
                if caption_clips:
                    final_video = CompositeVideoClip([video] + caption_clips)
                else:
                    final_video = video
                
                # Write final video
                final_video.write_videofile(
                    output_path,
                    codec=config.codec,
                    audio_codec=config.audio_codec,
                    verbose=False,
                    logger=None
                )
                
                final_video.close()
            
            logger.info(f"Captions added to video: {output_path}")
            return output_path
            
        except Exception as e:
            raise create_video_processing_error("caption_addition", video_path, e)
    
    async def add_watermark(
        self,
        video_path: str,
        watermark_path: str,
        output_path: str,
        position: str = "bottom-right",
        opacity: float = 0.7,
        config: ProcessingConfig = None
    ) -> str:
        """Add watermark to video"""
        try:
            if config is None:
                config = ProcessingConfig(quality=QualityLevel.HIGH, format=VideoFormat.MP4)
            
            with VideoFileClip(video_path) as video:
                # Load watermark
                watermark = VideoFileClip(watermark_path).set_opacity(opacity)
                
                # Set position
                if position == "bottom-right":
                    watermark = watermark.set_position(('right', 'bottom'))
                elif position == "bottom-left":
                    watermark = watermark.set_position(('left', 'bottom'))
                elif position == "top-right":
                    watermark = watermark.set_position(('right', 'top'))
                elif position == "top-left":
                    watermark = watermark.set_position(('left', 'top'))
                elif position == "center":
                    watermark = watermark.set_position('center')
                
                # Composite video with watermark
                final_video = CompositeVideoClip([video, watermark])
                
                # Write final video
                final_video.write_videofile(
                    output_path,
                    codec=config.codec,
                    audio_codec=config.audio_codec,
                    verbose=False,
                    logger=None
                )
                
                final_video.close()
                watermark.close()
            
            logger.info(f"Watermark added to video: {output_path}")
            return output_path
            
        except Exception as e:
            raise create_video_processing_error("watermark_addition", video_path, e)
    
    async def apply_effects(
        self,
        video_path: str,
        effects: List[Dict[str, Any]],
        output_path: str,
        config: ProcessingConfig = None
    ) -> str:
        """Apply visual effects to video"""
        try:
            if config is None:
                config = ProcessingConfig(quality=QualityLevel.HIGH, format=VideoFormat.MP4)
            
            with VideoFileClip(video_path) as video:
                processed_video = video
                
                for effect in effects:
                    effect_type = effect.get('type')
                    intensity = effect.get('intensity', 1.0)
                    
                    if effect_type == EffectType.BLUR:
                        processed_video = processed_video.blur(intensity)
                    elif effect_type == EffectType.BRIGHTNESS:
                        processed_video = processed_video.fx(lambda clip: clip.fx(lambda c: c.set_opacity(intensity)))
                    elif effect_type == EffectType.CONTRAST:
                        # Custom contrast effect
                        processed_video = processed_video.fx(lambda clip: clip.fx(lambda c: c.set_opacity(intensity)))
                    elif effect_type == EffectType.FADE:
                        fade_duration = effect.get('duration', 1.0)
                        processed_video = processed_video.fadein(fade_duration).fadeout(fade_duration)
                
                # Write processed video
                processed_video.write_videofile(
                    output_path,
                    codec=config.codec,
                    audio_codec=config.audio_codec,
                    verbose=False,
                    logger=None
                )
                
                processed_video.close()
            
            logger.info(f"Effects applied to video: {output_path}")
            return output_path
            
        except Exception as e:
            raise create_video_processing_error("effect_application", video_path, e)
    
    async def optimize_for_platform(
        self,
        video_path: str,
        platform: PlatformType,
        output_path: str,
        config: ProcessingConfig = None
    ) -> str:
        """Optimize video for specific platform"""
        try:
            if config is None:
                config = ProcessingConfig(quality=QualityLevel.HIGH, format=VideoFormat.MP4)
            
            # Platform-specific settings
            platform_settings = {
                PlatformType.YOUTUBE: {
                    "resolution": (1920, 1080),
                    "fps": 30,
                    "bitrate": "5000k",
                    "max_duration": 60
                },
                PlatformType.TIKTOK: {
                    "resolution": (1080, 1920),
                    "fps": 30,
                    "bitrate": "2000k",
                    "max_duration": 60
                },
                PlatformType.INSTAGRAM: {
                    "resolution": (1080, 1080),
                    "fps": 30,
                    "bitrate": "2000k",
                    "max_duration": 60
                },
                PlatformType.LINKEDIN: {
                    "resolution": (1920, 1080),
                    "fps": 30,
                    "bitrate": "3000k",
                    "max_duration": 30
                },
                PlatformType.TWITTER: {
                    "resolution": (1280, 720),
                    "fps": 30,
                    "bitrate": "2000k",
                    "max_duration": 140
                }
            }
            
            settings = platform_settings.get(platform, platform_settings[PlatformType.YOUTUBE])
            
            # Update config with platform settings
            config.resolution = settings["resolution"]
            config.fps = settings["fps"]
            config.bitrate = settings["bitrate"]
            
            # Create optimized video
            with VideoFileClip(video_path) as video:
                # Resize if needed
                if config.resolution:
                    video = video.resize(config.resolution)
                
                # Limit duration if needed
                max_duration = settings["max_duration"]
                if video.duration > max_duration:
                    video = video.subclip(0, max_duration)
                
                # Write optimized video
                video.write_videofile(
                    output_path,
                    codec=config.codec,
                    audio_codec=config.audio_codec,
                    bitrate=config.bitrate,
                    fps=config.fps,
                    verbose=False,
                    logger=None
                )
                
                video.close()
            
            logger.info(f"Video optimized for {platform.value}: {output_path}")
            return output_path
            
        except Exception as e:
            raise create_video_processing_error("platform_optimization", video_path, e)
    
    async def generate_thumbnail(
        self,
        video_path: str,
        timestamp: float,
        output_path: str,
        size: Tuple[int, int] = (320, 180)
    ) -> str:
        """Generate thumbnail from video at specific timestamp"""
        try:
            with VideoFileClip(video_path) as video:
                # Get frame at timestamp
                frame = video.get_frame(timestamp)
                
                # Convert to PIL Image
                image = Image.fromarray(frame.astype('uint8'))
                
                # Resize
                image = image.resize(size, Image.Resampling.LANCZOS)
                
                # Save thumbnail
                image.save(output_path, 'JPEG', quality=85)
            
            logger.info(f"Thumbnail generated: {output_path}")
            return output_path
            
        except Exception as e:
            raise create_video_processing_error("thumbnail_generation", video_path, e)
    
    async def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get comprehensive video information"""
        try:
            with VideoFileClip(video_path) as video:
                info = {
                    "duration": video.duration,
                    "fps": video.fps,
                    "size": video.size,
                    "resolution": f"{video.w}x{video.h}",
                    "has_audio": video.audio is not None,
                    "file_size": os.path.getsize(video_path),
                    "format": video_path.split('.')[-1].lower()
                }
                
                if video.audio:
                    info["audio_fps"] = video.audio.fps
                    info["audio_duration"] = video.audio.duration
                
                return info
                
        except Exception as e:
            raise create_video_processing_error("info_extraction", video_path, e)
    
    async def batch_process(
        self,
        video_paths: List[str],
        processing_func,
        config: ProcessingConfig = None,
        max_workers: int = 4
    ) -> List[Any]:
        """Process multiple videos in parallel"""
        try:
            if config is None:
                config = ProcessingConfig(quality=QualityLevel.HIGH, format=VideoFormat.MP4)
            
            # Create semaphore to limit concurrent processing
            semaphore = asyncio.Semaphore(max_workers)
            
            async def process_single(video_path):
                async with semaphore:
                    return await processing_func(video_path, config)
            
            # Process videos in parallel
            tasks = [process_single(path) for path in video_paths]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Separate successful results from exceptions
            successful_results = []
            errors = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    errors.append({"video": video_paths[i], "error": str(result)})
                else:
                    successful_results.append(result)
            
            logger.info(f"Batch processing completed: {len(successful_results)} successful, {len(errors)} failed")
            
            return {
                "successful": successful_results,
                "errors": errors,
                "total": len(video_paths)
            }
            
        except Exception as e:
            raise create_video_processing_error("batch_processing", "multiple_videos", e)


# Global video processor instance
video_processor = VideoProcessor()





























