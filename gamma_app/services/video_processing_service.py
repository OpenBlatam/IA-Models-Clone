"""
Gamma App - Advanced Video Processing Service
Advanced video processing with AI-powered features, effects, and optimization
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import numpy as np
from pathlib import Path
import sqlite3
import redis
from collections import defaultdict
import cv2
import ffmpeg
import moviepy.editor as mp
from moviepy.video.fx import resize, crop, rotate, fadein, fadeout
from moviepy.audio.fx import volumex
import librosa
import soundfile as sf
from PIL import Image, ImageFilter, ImageEnhance
import torch
import torchvision.transforms as transforms
from transformers import pipeline
import openai
import anthropic
from scipy import signal
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)

class VideoFormat(Enum):
    """Video formats"""
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    MKV = "mkv"
    WEBM = "webm"
    FLV = "flv"
    WMV = "wmv"

class AudioFormat(Enum):
    """Audio formats"""
    MP3 = "mp3"
    WAV = "wav"
    AAC = "aac"
    FLAC = "flac"
    OGG = "ogg"
    M4A = "m4a"

class VideoEffect(Enum):
    """Video effects"""
    BLUR = "blur"
    SHARPEN = "sharpen"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    SATURATION = "saturation"
    HUE = "hue"
    GRAYSCALE = "grayscale"
    SEPIA = "sepia"
    VINTAGE = "vintage"
    NEGATIVE = "negative"

class ProcessingQuality(Enum):
    """Processing quality levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

@dataclass
class VideoMetadata:
    """Video metadata"""
    duration: float
    fps: float
    width: int
    height: int
    bitrate: int
    codec: str
    format: str
    file_size: int
    created_at: datetime
    modified_at: datetime

@dataclass
class AudioMetadata:
    """Audio metadata"""
    duration: float
    sample_rate: int
    channels: int
    bitrate: int
    codec: str
    format: str
    file_size: int

@dataclass
class ProcessingJob:
    """Video processing job"""
    job_id: str
    input_path: str
    output_path: str
    operation: str
    parameters: Dict[str, Any]
    status: str = "pending"
    progress: float = 0.0
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

class AdvancedVideoProcessingService:
    """Advanced video processing service with AI capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config.get("database_path", "video_processing.db")
        self.redis_client = None
        self.processing_jobs = {}
        self.ai_models = {}
        self.effect_templates = {}
        
        # Initialize components
        self._init_database()
        self._init_redis()
        self._init_ai_models()
        self._init_effect_templates()
    
    def _init_database(self):
        """Initialize video processing database"""
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create processing jobs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_jobs (
                    job_id TEXT PRIMARY KEY,
                    input_path TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    progress REAL DEFAULT 0.0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    started_at DATETIME,
                    completed_at DATETIME,
                    error_message TEXT
                )
            """)
            
            # Create video metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS video_metadata (
                    video_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    thumbnail_path TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create processing history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_history (
                    history_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    input_metadata TEXT,
                    output_metadata TEXT,
                    processing_time REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES processing_jobs (job_id)
                )
            """)
            
            conn.commit()
        
        logger.info("Video processing database initialized")
    
    def _init_redis(self):
        """Initialize Redis for caching"""
        try:
            redis_url = self.config.get("redis_url", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            logger.info("Redis client initialized for video processing")
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
    
    def _init_ai_models(self):
        """Initialize AI models for video processing"""
        try:
            # Initialize object detection model
            self.ai_models["object_detection"] = pipeline(
                "object-detection",
                model="facebook/detr-resnet-50",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Initialize scene classification model
            self.ai_models["scene_classification"] = pipeline(
                "image-classification",
                model="microsoft/resnet-50",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Initialize face detection model
            self.ai_models["face_detection"] = pipeline(
                "face-detection",
                model="microsoft/DialoGPT-medium",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("AI models initialized for video processing")
        except Exception as e:
            logger.warning(f"AI models initialization failed: {e}")
    
    def _init_effect_templates(self):
        """Initialize effect templates"""
        self.effect_templates = {
            "vintage": {
                "brightness": 0.8,
                "contrast": 1.2,
                "saturation": 0.7,
                "hue": 0.1,
                "blur": 0.5
            },
            "cinematic": {
                "brightness": 0.9,
                "contrast": 1.3,
                "saturation": 0.8,
                "hue": -0.05,
                "blur": 0.2
            },
            "dramatic": {
                "brightness": 0.7,
                "contrast": 1.5,
                "saturation": 1.1,
                "hue": 0.0,
                "blur": 0.0
            },
            "soft": {
                "brightness": 1.1,
                "contrast": 0.9,
                "saturation": 0.8,
                "hue": 0.0,
                "blur": 0.8
            }
        }
    
    async def process_video(
        self,
        input_path: str,
        output_path: str,
        operation: str,
        parameters: Dict[str, Any],
        quality: ProcessingQuality = ProcessingQuality.HIGH
    ) -> str:
        """Process video with specified operation"""
        
        job_id = str(uuid.uuid4())
        
        # Create processing job
        job = ProcessingJob(
            job_id=job_id,
            input_path=input_path,
            output_path=output_path,
            operation=operation,
            parameters=parameters,
            created_at=datetime.now()
        )
        
        # Store job
        self.processing_jobs[job_id] = job
        await self._store_processing_job(job)
        
        # Start processing in background
        asyncio.create_task(self._process_video_async(job, quality))
        
        logger.info(f"Video processing job created: {job_id}")
        return job_id
    
    async def _process_video_async(self, job: ProcessingJob, quality: ProcessingQuality):
        """Process video asynchronously"""
        
        try:
            job.status = "processing"
            job.started_at = datetime.now()
            await self._update_processing_job(job)
            
            # Get input video metadata
            input_metadata = await self._get_video_metadata(job.input_path)
            
            # Process based on operation
            if job.operation == "resize":
                await self._resize_video(job, input_metadata, quality)
            elif job.operation == "crop":
                await self._crop_video(job, input_metadata, quality)
            elif job.operation == "rotate":
                await self._rotate_video(job, input_metadata, quality)
            elif job.operation == "apply_effect":
                await self._apply_video_effect(job, input_metadata, quality)
            elif job.operation == "extract_audio":
                await self._extract_audio(job, input_metadata, quality)
            elif job.operation == "add_audio":
                await self._add_audio(job, input_metadata, quality)
            elif job.operation == "create_thumbnail":
                await self._create_thumbnail(job, input_metadata, quality)
            elif job.operation == "stabilize":
                await self._stabilize_video(job, input_metadata, quality)
            elif job.operation == "enhance":
                await self._enhance_video(job, input_metadata, quality)
            elif job.operation == "detect_objects":
                await self._detect_objects_in_video(job, input_metadata, quality)
            elif job.operation == "generate_subtitles":
                await self._generate_subtitles(job, input_metadata, quality)
            elif job.operation == "compress":
                await self._compress_video(job, input_metadata, quality)
            else:
                raise ValueError(f"Unsupported operation: {job.operation}")
            
            job.status = "completed"
            job.progress = 100.0
            job.completed_at = datetime.now()
            await self._update_processing_job(job)
            
            logger.info(f"Video processing completed: {job.job_id}")
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.now()
            await self._update_processing_job(job)
            
            logger.error(f"Video processing failed: {job.job_id} - {e}")
    
    async def _resize_video(
        self,
        job: ProcessingJob,
        input_metadata: VideoMetadata,
        quality: ProcessingQuality
    ):
        """Resize video"""
        
        width = job.parameters.get("width", input_metadata.width)
        height = job.parameters.get("height", input_metadata.height)
        
        # Load video
        video = mp.VideoFileClip(job.input_path)
        
        # Resize video
        resized_video = video.resize((width, height))
        
        # Set quality parameters
        quality_params = self._get_quality_parameters(quality)
        
        # Write output
        resized_video.write_videofile(
            job.output_path,
            codec='libx264',
            audio_codec='aac',
            **quality_params
        )
        
        # Clean up
        video.close()
        resized_video.close()
        
        job.progress = 100.0
    
    async def _crop_video(
        self,
        job: ProcessingJob,
        input_metadata: VideoMetadata,
        quality: ProcessingQuality
    ):
        """Crop video"""
        
        x1 = job.parameters.get("x1", 0)
        y1 = job.parameters.get("y1", 0)
        x2 = job.parameters.get("x2", input_metadata.width)
        y2 = job.parameters.get("y2", input_metadata.height)
        
        # Load video
        video = mp.VideoFileClip(job.input_path)
        
        # Crop video
        cropped_video = video.crop(x1=x1, y1=y1, x2=x2, y2=y2)
        
        # Set quality parameters
        quality_params = self._get_quality_parameters(quality)
        
        # Write output
        cropped_video.write_videofile(
            job.output_path,
            codec='libx264',
            audio_codec='aac',
            **quality_params
        )
        
        # Clean up
        video.close()
        cropped_video.close()
        
        job.progress = 100.0
    
    async def _rotate_video(
        self,
        job: ProcessingJob,
        input_metadata: VideoMetadata,
        quality: ProcessingQuality
    ):
        """Rotate video"""
        
        angle = job.parameters.get("angle", 90)
        
        # Load video
        video = mp.VideoFileClip(job.input_path)
        
        # Rotate video
        rotated_video = video.rotate(angle)
        
        # Set quality parameters
        quality_params = self._get_quality_parameters(quality)
        
        # Write output
        rotated_video.write_videofile(
            job.output_path,
            codec='libx264',
            audio_codec='aac',
            **quality_params
        )
        
        # Clean up
        video.close()
        rotated_video.close()
        
        job.progress = 100.0
    
    async def _apply_video_effect(
        self,
        job: ProcessingJob,
        input_metadata: VideoMetadata,
        quality: ProcessingQuality
    ):
        """Apply video effect"""
        
        effect_name = job.parameters.get("effect", "vintage")
        effect_params = job.parameters.get("parameters", {})
        
        # Get effect template
        template = self.effect_templates.get(effect_name, {})
        template.update(effect_params)
        
        # Load video
        video = mp.VideoFileClip(job.input_path)
        
        # Apply effects
        processed_video = video
        
        if template.get("brightness"):
            processed_video = processed_video.fx(lambda clip: self._adjust_brightness(clip, template["brightness"]))
        
        if template.get("contrast"):
            processed_video = processed_video.fx(lambda clip: self._adjust_contrast(clip, template["contrast"]))
        
        if template.get("saturation"):
            processed_video = processed_video.fx(lambda clip: self._adjust_saturation(clip, template["saturation"]))
        
        if template.get("blur"):
            processed_video = processed_video.fx(lambda clip: self._apply_blur(clip, template["blur"]))
        
        # Set quality parameters
        quality_params = self._get_quality_parameters(quality)
        
        # Write output
        processed_video.write_videofile(
            job.output_path,
            codec='libx264',
            audio_codec='aac',
            **quality_params
        )
        
        # Clean up
        video.close()
        processed_video.close()
        
        job.progress = 100.0
    
    async def _extract_audio(
        self,
        job: ProcessingJob,
        input_metadata: VideoMetadata,
        quality: ProcessingQuality
    ):
        """Extract audio from video"""
        
        audio_format = job.parameters.get("format", "mp3")
        bitrate = job.parameters.get("bitrate", 192)
        
        # Load video
        video = mp.VideoFileClip(job.input_path)
        
        # Extract audio
        audio = video.audio
        
        if audio:
            # Write audio file
            audio.write_audiofile(
                job.output_path,
                bitrate=f"{bitrate}k",
                verbose=False,
                logger=None
            )
            
            audio.close()
        
        video.close()
        
        job.progress = 100.0
    
    async def _add_audio(
        self,
        job: ProcessingJob,
        input_metadata: VideoMetadata,
        quality: ProcessingQuality
    ):
        """Add audio to video"""
        
        audio_path = job.parameters.get("audio_path")
        volume = job.parameters.get("volume", 1.0)
        loop = job.parameters.get("loop", False)
        
        if not audio_path or not Path(audio_path).exists():
            raise ValueError("Audio file not found")
        
        # Load video and audio
        video = mp.VideoFileClip(job.input_path)
        audio = mp.AudioFileClip(audio_path)
        
        # Adjust audio volume
        if volume != 1.0:
            audio = audio.fx(volumex, volume)
        
        # Set audio duration to match video
        if loop and audio.duration < video.duration:
            # Loop audio to match video duration
            loops_needed = int(video.duration / audio.duration) + 1
            audio = mp.concatenate_audioclips([audio] * loops_needed)
        
        # Trim audio to video duration
        audio = audio.subclip(0, video.duration)
        
        # Set audio to video
        final_video = video.set_audio(audio)
        
        # Set quality parameters
        quality_params = self._get_quality_parameters(quality)
        
        # Write output
        final_video.write_videofile(
            job.output_path,
            codec='libx264',
            audio_codec='aac',
            **quality_params
        )
        
        # Clean up
        video.close()
        audio.close()
        final_video.close()
        
        job.progress = 100.0
    
    async def _create_thumbnail(
        self,
        job: ProcessingJob,
        input_metadata: VideoMetadata,
        quality: ProcessingQuality
    ):
        """Create video thumbnail"""
        
        timestamp = job.parameters.get("timestamp", 0)
        width = job.parameters.get("width", 320)
        height = job.parameters.get("height", 240)
        
        # Load video
        video = mp.VideoFileClip(job.input_path)
        
        # Get frame at timestamp
        frame = video.get_frame(timestamp)
        
        # Convert to PIL Image
        image = Image.fromarray(frame)
        
        # Resize if needed
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        
        # Save thumbnail
        image.save(job.output_path, "JPEG", quality=95)
        
        video.close()
        
        job.progress = 100.0
    
    async def _stabilize_video(
        self,
        job: ProcessingJob,
        input_metadata: VideoMetadata,
        quality: ProcessingQuality
    ):
        """Stabilize video using OpenCV"""
        
        # Open input video
        cap = cv2.VideoCapture(job.input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(job.output_path, fourcc, fps, (width, height))
        
        # Initialize stabilizer
        stabilizer = cv2.createStabilizer()
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Stabilize frame
            stabilized_frame = stabilizer.stabilize(frame)
            
            # Write stabilized frame
            out.write(stabilized_frame)
            
            # Update progress
            frame_count += 1
            job.progress = (frame_count / total_frames) * 100
            await self._update_processing_job(job)
        
        # Clean up
        cap.release()
        out.release()
        
        job.progress = 100.0
    
    async def _enhance_video(
        self,
        job: ProcessingJob,
        input_metadata: VideoMetadata,
        quality: ProcessingQuality
    ):
        """Enhance video quality using AI"""
        
        enhancement_type = job.parameters.get("type", "upscale")
        
        if enhancement_type == "upscale":
            await self._upscale_video(job, input_metadata, quality)
        elif enhancement_type == "denoise":
            await self._denoise_video(job, input_metadata, quality)
        elif enhancement_type == "sharpen":
            await self._sharpen_video(job, input_metadata, quality)
        else:
            raise ValueError(f"Unsupported enhancement type: {enhancement_type}")
    
    async def _upscale_video(
        self,
        job: ProcessingJob,
        input_metadata: VideoMetadata,
        quality: ProcessingQuality
    ):
        """Upscale video using AI"""
        
        scale_factor = job.parameters.get("scale_factor", 2)
        
        # Load video
        video = mp.VideoFileClip(job.input_path)
        
        # Get new dimensions
        new_width = int(input_metadata.width * scale_factor)
        new_height = int(input_metadata.height * scale_factor)
        
        # Upscale video
        upscaled_video = video.resize((new_width, new_height))
        
        # Set quality parameters
        quality_params = self._get_quality_parameters(quality)
        
        # Write output
        upscaled_video.write_videofile(
            job.output_path,
            codec='libx264',
            audio_codec='aac',
            **quality_params
        )
        
        # Clean up
        video.close()
        upscaled_video.close()
        
        job.progress = 100.0
    
    async def _denoise_video(
        self,
        job: ProcessingJob,
        input_metadata: VideoMetadata,
        quality: ProcessingQuality
    ):
        """Denoise video using OpenCV"""
        
        # Open input video
        cap = cv2.VideoCapture(job.input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(job.output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply denoising
            denoised_frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
            
            # Write denoised frame
            out.write(denoised_frame)
            
            # Update progress
            frame_count += 1
            job.progress = (frame_count / total_frames) * 100
            await self._update_processing_job(job)
        
        # Clean up
        cap.release()
        out.release()
        
        job.progress = 100.0
    
    async def _sharpen_video(
        self,
        job: ProcessingJob,
        input_metadata: VideoMetadata,
        quality: ProcessingQuality
    ):
        """Sharpen video"""
        
        strength = job.parameters.get("strength", 1.0)
        
        # Load video
        video = mp.VideoFileClip(job.input_path)
        
        # Apply sharpening effect
        sharpened_video = video.fx(lambda clip: self._apply_sharpening(clip, strength))
        
        # Set quality parameters
        quality_params = self._get_quality_parameters(quality)
        
        # Write output
        sharpened_video.write_videofile(
            job.output_path,
            codec='libx264',
            audio_codec='aac',
            **quality_params
        )
        
        # Clean up
        video.close()
        sharpened_video.close()
        
        job.progress = 100.0
    
    async def _detect_objects_in_video(
        self,
        job: ProcessingJob,
        input_metadata: VideoMetadata,
        quality: ProcessingQuality
    ):
        """Detect objects in video using AI"""
        
        # Load video
        video = mp.VideoFileClip(job.input_path)
        
        # Sample frames for object detection
        sample_rate = job.parameters.get("sample_rate", 30)  # Every 30 frames
        detections = []
        
        frame_count = 0
        total_frames = int(video.duration * video.fps)
        
        for frame in video.iter_frames():
            if frame_count % sample_rate == 0:
                # Convert frame to PIL Image
                image = Image.fromarray(frame)
                
                # Detect objects
                objects = self.ai_models["object_detection"](image)
                
                detections.append({
                    "frame": frame_count,
                    "timestamp": frame_count / video.fps,
                    "objects": objects
                })
            
            frame_count += 1
            
            # Update progress
            job.progress = (frame_count / total_frames) * 100
            await self._update_processing_job(job)
        
        # Save detections to JSON file
        with open(job.output_path, 'w') as f:
            json.dump(detections, f, indent=2)
        
        video.close()
        
        job.progress = 100.0
    
    async def _generate_subtitles(
        self,
        job: ProcessingJob,
        input_metadata: VideoMetadata,
        quality: ProcessingQuality
    ):
        """Generate subtitles using AI"""
        
        # Extract audio
        video = mp.VideoFileClip(job.input_path)
        audio = video.audio
        
        if not audio:
            raise ValueError("No audio track found in video")
        
        # Save temporary audio file
        temp_audio_path = f"temp_audio_{uuid.uuid4().hex}.wav"
        audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
        
        try:
            # Transcribe audio using OpenAI Whisper
            if self.config.get("openai_api_key"):
                with open(temp_audio_path, 'rb') as f:
                    transcript = openai.Audio.transcribe("whisper-1", f)
                
                # Generate subtitles with timestamps
                subtitles = await self._create_subtitle_timestamps(
                    transcript.text, input_metadata.duration
                )
            else:
                # Fallback: create dummy subtitles
                subtitles = [
                    {"start": 0, "end": input_metadata.duration, "text": "Subtitles not available"}
                ]
            
            # Save subtitles
            with open(job.output_path, 'w') as f:
                json.dump(subtitles, f, indent=2)
            
        finally:
            # Clean up
            Path(temp_audio_path).unlink()
            video.close()
            audio.close()
        
        job.progress = 100.0
    
    async def _compress_video(
        self,
        job: ProcessingJob,
        input_metadata: VideoMetadata,
        quality: ProcessingQuality
    ):
        """Compress video"""
        
        target_size = job.parameters.get("target_size_mb", 10)
        compression_ratio = job.parameters.get("compression_ratio", 0.8)
        
        # Calculate target bitrate
        target_bitrate = int((target_size * 8 * 1024 * 1024) / input_metadata.duration)
        
        # Load video
        video = mp.VideoFileClip(job.input_path)
        
        # Compress video
        compressed_video = video
        
        # Set quality parameters for compression
        quality_params = {
            "bitrate": f"{target_bitrate}k",
            "audio_bitrate": "128k",
            "verbose": False,
            "logger": None
        }
        
        # Write compressed video
        compressed_video.write_videofile(
            job.output_path,
            codec='libx264',
            audio_codec='aac',
            **quality_params
        )
        
        # Clean up
        video.close()
        compressed_video.close()
        
        job.progress = 100.0
    
    def _adjust_brightness(self, clip, factor):
        """Adjust video brightness"""
        def adjust_frame(frame):
            return np.clip(frame * factor, 0, 255).astype(np.uint8)
        return clip.fl_image(adjust_frame)
    
    def _adjust_contrast(self, clip, factor):
        """Adjust video contrast"""
        def adjust_frame(frame):
            return np.clip((frame - 128) * factor + 128, 0, 255).astype(np.uint8)
        return clip.fl_image(adjust_frame)
    
    def _adjust_saturation(self, clip, factor):
        """Adjust video saturation"""
        def adjust_frame(frame):
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return clip.fl_image(adjust_frame)
    
    def _apply_blur(self, clip, strength):
        """Apply blur effect"""
        def blur_frame(frame):
            return cv2.GaussianBlur(frame, (0, 0), strength)
        return clip.fl_image(blur_frame)
    
    def _apply_sharpening(self, clip, strength):
        """Apply sharpening effect"""
        def sharpen_frame(frame):
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            return cv2.filter2D(frame, -1, kernel * strength)
        return clip.fl_image(sharpen_frame)
    
    def _get_quality_parameters(self, quality: ProcessingQuality) -> Dict[str, Any]:
        """Get quality parameters for video processing"""
        
        quality_params = {
            ProcessingQuality.LOW: {
                "bitrate": "500k",
                "audio_bitrate": "64k",
                "preset": "ultrafast"
            },
            ProcessingQuality.MEDIUM: {
                "bitrate": "1000k",
                "audio_bitrate": "128k",
                "preset": "fast"
            },
            ProcessingQuality.HIGH: {
                "bitrate": "2000k",
                "audio_bitrate": "192k",
                "preset": "medium"
            },
            ProcessingQuality.ULTRA: {
                "bitrate": "4000k",
                "audio_bitrate": "320k",
                "preset": "slow"
            }
        }
        
        return quality_params.get(quality, quality_params[ProcessingQuality.HIGH])
    
    async def _get_video_metadata(self, video_path: str) -> VideoMetadata:
        """Get video metadata"""
        
        try:
            # Use ffprobe to get metadata
            probe = ffmpeg.probe(video_path)
            video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            
            metadata = VideoMetadata(
                duration=float(probe['format']['duration']),
                fps=eval(video_stream['r_frame_rate']),
                width=int(video_stream['width']),
                height=int(video_stream['height']),
                bitrate=int(probe['format'].get('bit_rate', 0)),
                codec=video_stream['codec_name'],
                format=probe['format']['format_name'],
                file_size=int(probe['format']['size']),
                created_at=datetime.now(),
                modified_at=datetime.now()
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get video metadata: {e}")
            # Return default metadata
            return VideoMetadata(
                duration=0.0,
                fps=30.0,
                width=1920,
                height=1080,
                bitrate=0,
                codec="unknown",
                format="unknown",
                file_size=0,
                created_at=datetime.now(),
                modified_at=datetime.now()
            )
    
    async def _create_subtitle_timestamps(
        self,
        text: str,
        duration: float
    ) -> List[Dict[str, Any]]:
        """Create subtitle timestamps"""
        
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        # Calculate time per sentence
        time_per_sentence = duration / len(sentences)
        
        subtitles = []
        current_time = 0.0
        
        for sentence in sentences:
            end_time = min(current_time + time_per_sentence, duration)
            
            subtitles.append({
                "start": current_time,
                "end": end_time,
                "text": sentence.strip()
            })
            
            current_time = end_time
        
        return subtitles
    
    async def _store_processing_job(self, job: ProcessingJob):
        """Store processing job in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO processing_jobs
                (job_id, input_path, output_path, operation, parameters, status, progress, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.job_id,
                job.input_path,
                job.output_path,
                job.operation,
                json.dumps(job.parameters),
                job.status,
                job.progress,
                job.created_at.isoformat()
            ))
            conn.commit()
    
    async def _update_processing_job(self, job: ProcessingJob):
        """Update processing job in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE processing_jobs
                SET status = ?, progress = ?, started_at = ?, completed_at = ?, error_message = ?
                WHERE job_id = ?
            """, (
                job.status,
                job.progress,
                job.started_at.isoformat() if job.started_at else None,
                job.completed_at.isoformat() if job.completed_at else None,
                job.error_message,
                job.job_id
            ))
            conn.commit()
    
    async def get_processing_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Get processing job by ID"""
        
        return self.processing_jobs.get(job_id)
    
    async def list_processing_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[ProcessingJob]:
        """List processing jobs"""
        
        jobs = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM processing_jobs"
            params = []
            
            if status:
                query += " WHERE status = ?"
                params.append(status)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            for row in rows:
                job = ProcessingJob(
                    job_id=row[0],
                    input_path=row[1],
                    output_path=row[2],
                    operation=row[3],
                    parameters=json.loads(row[4]),
                    status=row[5],
                    progress=row[6],
                    created_at=datetime.fromisoformat(row[7]),
                    started_at=datetime.fromisoformat(row[8]) if row[8] else None,
                    completed_at=datetime.fromisoformat(row[9]) if row[9] else None,
                    error_message=row[10]
                )
                jobs.append(job)
        
        return jobs
    
    async def cancel_processing_job(self, job_id: str) -> bool:
        """Cancel processing job"""
        
        job = self.processing_jobs.get(job_id)
        if job and job.status == "processing":
            job.status = "cancelled"
            job.completed_at = datetime.now()
            await self._update_processing_job(job)
            return True
        
        return False
    
    async def get_video_analytics(self, video_path: str) -> Dict[str, Any]:
        """Get video analytics"""
        
        try:
            # Get video metadata
            metadata = await self._get_video_metadata(video_path)
            
            # Analyze video content
            analytics = {
                "metadata": asdict(metadata),
                "content_analysis": await self._analyze_video_content(video_path),
                "quality_metrics": await self._calculate_quality_metrics(video_path),
                "generated_at": datetime.now().isoformat()
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Video analytics failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_video_content(self, video_path: str) -> Dict[str, Any]:
        """Analyze video content"""
        
        # Load video
        video = mp.VideoFileClip(video_path)
        
        # Sample frames for analysis
        sample_frames = []
        for i, frame in enumerate(video.iter_frames()):
            if i % 30 == 0:  # Sample every 30 frames
                sample_frames.append(frame)
                if len(sample_frames) >= 10:  # Limit to 10 frames
                    break
        
        # Analyze frames
        scene_types = []
        object_counts = []
        
        for frame in sample_frames:
            # Convert to PIL Image
            image = Image.fromarray(frame)
            
            # Scene classification
            if "scene_classification" in self.ai_models:
                scene = self.ai_models["scene_classification"](image)
                scene_types.append(scene[0]["label"])
            
            # Object detection
            if "object_detection" in self.ai_models:
                objects = self.ai_models["object_detection"](image)
                object_counts.append(len(objects))
        
        video.close()
        
        return {
            "scene_types": list(set(scene_types)),
            "average_objects_per_frame": np.mean(object_counts) if object_counts else 0,
            "frames_analyzed": len(sample_frames)
        }
    
    async def _calculate_quality_metrics(self, video_path: str) -> Dict[str, Any]:
        """Calculate video quality metrics"""
        
        # Load video
        video = mp.VideoFileClip(video_path)
        
        # Calculate metrics
        metrics = {
            "resolution": f"{video.w}x{video.h}",
            "aspect_ratio": video.w / video.h,
            "duration": video.duration,
            "fps": video.fps,
            "has_audio": video.audio is not None,
            "file_size_mb": Path(video_path).stat().st_size / (1024 * 1024)
        }
        
        video.close()
        
        return metrics
    
    async def cleanup(self):
        """Cleanup resources"""
        
        if self.redis_client:
            self.redis_client.close()
        
        # Clean up AI models
        for model in self.ai_models.values():
            if hasattr(model, 'cleanup'):
                model.cleanup()
        
        logger.info("Video processing service cleanup completed")

# Global instance
video_processing_service = None

async def get_video_processing_service() -> AdvancedVideoProcessingService:
    """Get global video processing service instance"""
    global video_processing_service
    if not video_processing_service:
        config = {
            "database_path": "data/video_processing.db",
            "redis_url": "redis://localhost:6379",
            "openai_api_key": "your-openai-key"
        }
        video_processing_service = AdvancedVideoProcessingService(config)
    return video_processing_service



