"""
Gamma App - Video Utilities
Advanced video processing and manipulation utilities
"""

import io
import base64
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import moviepy.editor as mp
from moviepy.video.fx import resize, speedx, fadein, fadeout
import ffmpeg
import subprocess
import tempfile
import os

logger = logging.getLogger(__name__)

class VideoFormat(Enum):
    """Video formats"""
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    WMV = "wmv"
    FLV = "flv"
    WEBM = "webm"
    MKV = "mkv"

class VideoQuality(Enum):
    """Video quality levels"""
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
    format: str
    codec: str
    bitrate: int
    file_size: int
    has_audio: bool
    audio_codec: Optional[str] = None
    audio_bitrate: Optional[int] = None

@dataclass
class VideoFrame:
    """Video frame data"""
    frame_number: int
    timestamp: float
    image: np.ndarray
    width: int
    height: int

class VideoProcessor:
    """Advanced video processing class"""
    
    def __init__(self):
        self.supported_formats = ['mp4', 'avi', 'mov', 'wmv', 'flv', 'webm', 'mkv']
        self.max_file_size = 500 * 1024 * 1024  # 500MB
    
    def get_video_info(self, video_path: str) -> VideoMetadata:
        """Get comprehensive video information"""
        try:
            # Use ffprobe to get detailed video info
            probe = ffmpeg.probe(video_path)
            
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
            
            if not video_stream:
                raise ValueError("No video stream found")
            
            duration = float(probe['format']['duration'])
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            fps = eval(video_stream['r_frame_rate'])
            codec = video_stream['codec_name']
            bitrate = int(probe['format'].get('bit_rate', 0))
            file_size = int(probe['format']['size'])
            
            has_audio = audio_stream is not None
            audio_codec = audio_stream['codec_name'] if audio_stream else None
            audio_bitrate = int(audio_stream.get('bit_rate', 0)) if audio_stream else None
            
            return VideoMetadata(
                duration=duration,
                fps=fps,
                width=width,
                height=height,
                format=Path(video_path).suffix.lower(),
                codec=codec,
                bitrate=bitrate,
                file_size=file_size,
                has_audio=has_audio,
                audio_codec=audio_codec,
                audio_bitrate=audio_bitrate
            )
            
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            raise
    
    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        frame_interval: int = 1,
        format: str = "jpg"
    ) -> List[str]:
        """Extract frames from video"""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            extracted_frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    output_file = output_dir / f"frame_{frame_count:06d}.{format}"
                    cv2.imwrite(str(output_file), frame)
                    extracted_frames.append(str(output_file))
                
                frame_count += 1
            
            cap.release()
            return extracted_frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            raise
    
    def create_video_from_frames(
        self,
        frames_dir: str,
        output_path: str,
        fps: float = 30.0,
        codec: str = "mp4v"
    ) -> str:
        """Create video from frames"""
        try:
            frames_dir = Path(frames_dir)
            frame_files = sorted(frames_dir.glob("*.jpg"))
            
            if not frame_files:
                raise ValueError("No frame files found")
            
            # Read first frame to get dimensions
            first_frame = cv2.imread(str(frame_files[0]))
            height, width, _ = first_frame.shape
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Write frames
            for frame_file in frame_files:
                frame = cv2.imread(str(frame_file))
                out.write(frame)
            
            out.release()
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating video from frames: {e}")
            raise
    
    def resize_video(
        self,
        video_path: str,
        output_path: str,
        width: int,
        height: int,
        maintain_aspect: bool = True
    ) -> str:
        """Resize video"""
        try:
            if maintain_aspect:
                # Use moviepy for aspect ratio preservation
                clip = mp.VideoFileClip(video_path)
                resized_clip = clip.resize((width, height))
                resized_clip.write_videofile(output_path)
                clip.close()
                resized_clip.close()
            else:
                # Use ffmpeg for direct resize
                (
                    ffmpeg
                    .input(video_path)
                    .filter('scale', width, height)
                    .output(output_path)
                    .overwrite_output()
                    .run()
                )
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error resizing video: {e}")
            raise
    
    def compress_video(
        self,
        video_path: str,
        output_path: str,
        quality: VideoQuality = VideoQuality.MEDIUM
    ) -> str:
        """Compress video"""
        try:
            quality_settings = {
                VideoQuality.LOW: {'crf': 28, 'preset': 'fast'},
                VideoQuality.MEDIUM: {'crf': 23, 'preset': 'medium'},
                VideoQuality.HIGH: {'crf': 18, 'preset': 'slow'},
                VideoQuality.ULTRA: {'crf': 15, 'preset': 'veryslow'}
            }
            
            settings = quality_settings[quality]
            
            (
                ffmpeg
                .input(video_path)
                .output(
                    output_path,
                    vcodec='libx264',
                    crf=settings['crf'],
                    preset=settings['preset'],
                    acodec='aac'
                )
                .overwrite_output()
                .run()
            )
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error compressing video: {e}")
            raise
    
    def add_watermark(
        self,
        video_path: str,
        output_path: str,
        watermark_text: str,
        position: str = "bottom-right",
        opacity: float = 0.7
    ) -> str:
        """Add text watermark to video"""
        try:
            clip = mp.VideoFileClip(video_path)
            
            # Create text clip
            txt_clip = mp.TextClip(
                watermark_text,
                fontsize=50,
                color='white',
                font='Arial-Bold'
            ).set_duration(clip.duration)
            
            # Position watermark
            if position == "top-left":
                txt_clip = txt_clip.set_position(('left', 'top'))
            elif position == "top-right":
                txt_clip = txt_clip.set_position(('right', 'top'))
            elif position == "bottom-left":
                txt_clip = txt_clip.set_position(('left', 'bottom'))
            else:  # bottom-right
                txt_clip = txt_clip.set_position(('right', 'bottom'))
            
            # Set opacity
            txt_clip = txt_clip.set_opacity(opacity)
            
            # Composite video
            final_clip = mp.CompositeVideoClip([clip, txt_clip])
            final_clip.write_videofile(output_path)
            
            # Clean up
            clip.close()
            txt_clip.close()
            final_clip.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error adding watermark: {e}")
            raise
    
    def add_image_watermark(
        self,
        video_path: str,
        output_path: str,
        watermark_image_path: str,
        position: str = "bottom-right",
        opacity: float = 0.7,
        scale: float = 0.1
    ) -> str:
        """Add image watermark to video"""
        try:
            clip = mp.VideoFileClip(video_path)
            
            # Load watermark image
            watermark = mp.ImageClip(watermark_image_path)
            
            # Scale watermark
            watermark = watermark.resize(scale)
            
            # Position watermark
            if position == "top-left":
                watermark = watermark.set_position(('left', 'top'))
            elif position == "top-right":
                watermark = watermark.set_position(('right', 'top'))
            elif position == "bottom-left":
                watermark = watermark.set_position(('left', 'bottom'))
            else:  # bottom-right
                watermark = watermark.set_position(('right', 'bottom'))
            
            # Set duration and opacity
            watermark = watermark.set_duration(clip.duration).set_opacity(opacity)
            
            # Composite video
            final_clip = mp.CompositeVideoClip([clip, watermark])
            final_clip.write_videofile(output_path)
            
            # Clean up
            clip.close()
            watermark.close()
            final_clip.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error adding image watermark: {e}")
            raise
    
    def trim_video(
        self,
        video_path: str,
        output_path: str,
        start_time: float,
        end_time: float
    ) -> str:
        """Trim video to specific time range"""
        try:
            clip = mp.VideoFileClip(video_path)
            trimmed_clip = clip.subclip(start_time, end_time)
            trimmed_clip.write_videofile(output_path)
            
            # Clean up
            clip.close()
            trimmed_clip.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error trimming video: {e}")
            raise
    
    def merge_videos(
        self,
        video_paths: List[str],
        output_path: str,
        method: str = "concatenate"
    ) -> str:
        """Merge multiple videos"""
        try:
            if method == "concatenate":
                clips = [mp.VideoFileClip(path) for path in video_paths]
                final_clip = mp.concatenate_videoclips(clips)
                final_clip.write_videofile(output_path)
                
                # Clean up
                for clip in clips:
                    clip.close()
                final_clip.close()
                
            elif method == "side_by_side":
                clips = [mp.VideoFileClip(path) for path in video_paths]
                # Resize clips to same height
                target_height = min(clip.h for clip in clips)
                resized_clips = [clip.resize(height=target_height) for clip in clips]
                
                final_clip = mp.concatenate_videoclips(resized_clips, method="compose")
                final_clip.write_videofile(output_path)
                
                # Clean up
                for clip in clips + resized_clips:
                    clip.close()
                final_clip.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error merging videos: {e}")
            raise
    
    def extract_audio(
        self,
        video_path: str,
        output_path: str,
        format: str = "mp3"
    ) -> str:
        """Extract audio from video"""
        try:
            clip = mp.VideoFileClip(video_path)
            audio = clip.audio
            audio.write_audiofile(output_path)
            
            # Clean up
            clip.close()
            audio.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            raise
    
    def add_audio(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        replace: bool = False
    ) -> str:
        """Add audio to video"""
        try:
            video_clip = mp.VideoFileClip(video_path)
            audio_clip = mp.AudioFileClip(audio_path)
            
            if replace:
                # Replace existing audio
                final_clip = video_clip.set_audio(audio_clip)
            else:
                # Mix with existing audio
                if video_clip.audio:
                    mixed_audio = mp.CompositeAudioClip([video_clip.audio, audio_clip])
                    final_clip = video_clip.set_audio(mixed_audio)
                else:
                    final_clip = video_clip.set_audio(audio_clip)
            
            final_clip.write_videofile(output_path)
            
            # Clean up
            video_clip.close()
            audio_clip.close()
            final_clip.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error adding audio: {e}")
            raise
    
    def change_speed(
        self,
        video_path: str,
        output_path: str,
        speed_factor: float
    ) -> str:
        """Change video playback speed"""
        try:
            clip = mp.VideoFileClip(video_path)
            speeded_clip = clip.fx(speedx, speed_factor)
            speeded_clip.write_videofile(output_path)
            
            # Clean up
            clip.close()
            speeded_clip.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error changing speed: {e}")
            raise
    
    def add_fade_effects(
        self,
        video_path: str,
        output_path: str,
        fade_in_duration: float = 0,
        fade_out_duration: float = 0
    ) -> str:
        """Add fade in/out effects"""
        try:
            clip = mp.VideoFileClip(video_path)
            
            if fade_in_duration > 0:
                clip = clip.fx(fadein, fade_in_duration)
            
            if fade_out_duration > 0:
                clip = clip.fx(fadeout, fade_out_duration)
            
            clip.write_videofile(output_path)
            
            # Clean up
            clip.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error adding fade effects: {e}")
            raise
    
    def convert_format(
        self,
        video_path: str,
        output_path: str,
        target_format: VideoFormat
    ) -> str:
        """Convert video to different format"""
        try:
            clip = mp.VideoFileClip(video_path)
            clip.write_videofile(output_path, codec='libx264')
            
            # Clean up
            clip.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error converting format: {e}")
            raise
    
    def create_thumbnail(
        self,
        video_path: str,
        output_path: str,
        timestamp: float = 0,
        width: int = 320,
        height: int = 240
    ) -> str:
        """Create video thumbnail"""
        try:
            clip = mp.VideoFileClip(video_path)
            frame = clip.get_frame(timestamp)
            
            # Convert to PIL Image
            image = Image.fromarray(frame)
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            image.save(output_path)
            
            # Clean up
            clip.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating thumbnail: {e}")
            raise
    
    def detect_scenes(
        self,
        video_path: str,
        threshold: float = 30.0
    ) -> List[Dict[str, Any]]:
        """Detect scene changes in video"""
        try:
            cap = cv2.VideoCapture(video_path)
            scenes = []
            prev_frame = None
            frame_count = 0
            scene_start = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Calculate difference
                    diff = cv2.absdiff(prev_frame, gray)
                    mean_diff = np.mean(diff)
                    
                    if mean_diff > threshold:
                        # Scene change detected
                        scenes.append({
                            'start_frame': scene_start,
                            'end_frame': frame_count - 1,
                            'start_time': scene_start / cap.get(cv2.CAP_PROP_FPS),
                            'end_time': (frame_count - 1) / cap.get(cv2.CAP_PROP_FPS),
                            'duration': (frame_count - scene_start) / cap.get(cv2.CAP_PROP_FPS)
                        })
                        scene_start = frame_count
                
                prev_frame = gray
                frame_count += 1
            
            # Add final scene
            if frame_count > scene_start:
                scenes.append({
                    'start_frame': scene_start,
                    'end_frame': frame_count - 1,
                    'start_time': scene_start / cap.get(cv2.CAP_PROP_FPS),
                    'end_time': (frame_count - 1) / cap.get(cv2.CAP_PROP_FPS),
                    'duration': (frame_count - scene_start) / cap.get(cv2.CAP_PROP_FPS)
                })
            
            cap.release()
            return scenes
            
        except Exception as e:
            logger.error(f"Error detecting scenes: {e}")
            raise
    
    def stabilize_video(
        self,
        video_path: str,
        output_path: str
    ) -> str:
        """Stabilize shaky video"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Initialize stabilizer
            prev_gray = None
            transform = np.eye(2, 3, dtype=np.float32)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_gray is not None:
                    # Detect features
                    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
                    
                    if prev_pts is not None:
                        # Track features
                        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
                        
                        # Select good points
                        idx = np.where(status == 1)[0]
                        prev_pts = prev_pts[idx]
                        curr_pts = curr_pts[idx]
                        
                        if len(prev_pts) > 4:
                            # Estimate transformation
                            transform_matrix = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]
                            
                            if transform_matrix is not None:
                                # Apply transformation
                                frame = cv2.warpAffine(frame, transform_matrix, (width, height))
                
                out.write(frame)
                prev_gray = gray
            
            cap.release()
            out.release()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error stabilizing video: {e}")
            raise
    
    def validate_video(self, video_path: str) -> Dict[str, Any]:
        """Validate video file"""
        try:
            validation_result = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'info': {}
            }
            
            # Check if file exists
            if not Path(video_path).exists():
                validation_result['valid'] = False
                validation_result['errors'].append("File does not exist")
                return validation_result
            
            # Check file size
            file_size = Path(video_path).stat().st_size
            if file_size > self.max_file_size:
                validation_result['warnings'].append(f"File size ({file_size} bytes) exceeds maximum ({self.max_file_size} bytes)")
            
            # Check file extension
            file_ext = Path(video_path).suffix.lower()
            if file_ext not in ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv']:
                validation_result['warnings'].append(f"File extension {file_ext} may not be supported")
            
            # Try to open video
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    validation_result['valid'] = False
                    validation_result['errors'].append("Cannot open video file")
                else:
                    validation_result['info']['fps'] = cap.get(cv2.CAP_PROP_FPS)
                    validation_result['info']['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    validation_result['info']['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    validation_result['info']['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    validation_result['info']['duration'] = validation_result['info']['frame_count'] / validation_result['info']['fps']
                cap.release()
                
            except Exception as e:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Cannot open video: {str(e)}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating video: {e}")
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': [],
                'info': {}
            }

# Global video processor instance
video_processor = VideoProcessor()

def get_video_info(video_path: str) -> VideoMetadata:
    """Get video info using global processor"""
    return video_processor.get_video_info(video_path)

def extract_video_frames(video_path: str, output_dir: str, frame_interval: int = 1, format: str = "jpg") -> List[str]:
    """Extract frames using global processor"""
    return video_processor.extract_frames(video_path, output_dir, frame_interval, format)

def create_video_from_frames(frames_dir: str, output_path: str, fps: float = 30.0, codec: str = "mp4v") -> str:
    """Create video from frames using global processor"""
    return video_processor.create_video_from_frames(frames_dir, output_path, fps, codec)

def resize_video(video_path: str, output_path: str, width: int, height: int, maintain_aspect: bool = True) -> str:
    """Resize video using global processor"""
    return video_processor.resize_video(video_path, output_path, width, height, maintain_aspect)

def compress_video(video_path: str, output_path: str, quality: VideoQuality = VideoQuality.MEDIUM) -> str:
    """Compress video using global processor"""
    return video_processor.compress_video(video_path, output_path, quality)

def add_video_watermark(video_path: str, output_path: str, watermark_text: str, position: str = "bottom-right", opacity: float = 0.7) -> str:
    """Add watermark using global processor"""
    return video_processor.add_watermark(video_path, output_path, watermark_text, position, opacity)

def trim_video(video_path: str, output_path: str, start_time: float, end_time: float) -> str:
    """Trim video using global processor"""
    return video_processor.trim_video(video_path, output_path, start_time, end_time)

def merge_videos(video_paths: List[str], output_path: str, method: str = "concatenate") -> str:
    """Merge videos using global processor"""
    return video_processor.merge_videos(video_paths, output_path, method)

def extract_audio_from_video(video_path: str, output_path: str, format: str = "mp3") -> str:
    """Extract audio using global processor"""
    return video_processor.extract_audio(video_path, output_path, format)

def create_video_thumbnail(video_path: str, output_path: str, timestamp: float = 0, width: int = 320, height: int = 240) -> str:
    """Create thumbnail using global processor"""
    return video_processor.create_thumbnail(video_path, output_path, timestamp, width, height)

























