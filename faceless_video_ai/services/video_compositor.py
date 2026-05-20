"""
Video Compositor Service
Composites final video from images, audio, and subtitles
"""

import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import subprocess
import json

from .utils.error_handler import (
    VideoCompositionError,
    handle_ffmpeg_error,
    validate_file_path,
    retry_on_failure,
)

logger = logging.getLogger(__name__)


class VideoCompositor:
    """Composites final video from all components"""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("/tmp/faceless_video/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def composite_video(
        self,
        image_sequence: List[Dict[str, Any]],
        audio_path: str,
        subtitles: List[Dict[str, Any]],
        video_config: Dict[str, Any],
        subtitle_config: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Composite final video from all components
        
        Args:
            image_sequence: List of image frames with timing
            audio_path: Path to audio file
            subtitles: List of subtitle entries
            video_config: Video configuration
            subtitle_config: Subtitle configuration
            output_path: Output video path
            
        Returns:
            Path to final video file
        """
        if output_path is None:
            output_path = self.output_dir / "final_video.mp4"
        
        logger.info(f"Compositing video: {output_path}")
        
        # Create video from images
        video_with_images = await self._create_video_from_images(
            image_sequence=image_sequence,
            video_config=video_config
        )
        
        # Add audio
        video_with_audio = await self._add_audio_to_video(
            video_path=video_with_images,
            audio_path=audio_path
        )
        
        # Add subtitles
        if subtitles and subtitle_config.get("enabled", True):
            final_video = await self._add_subtitles_to_video(
                video_path=video_with_audio,
                subtitles=subtitles,
                subtitle_config=subtitle_config
            )
        else:
            final_video = video_with_audio
        
        # Move to final output path
        if final_video != output_path:
            final_video.rename(output_path)
        
        logger.info(f"Video composited successfully: {output_path}")
        return output_path

    @retry_on_failure(max_retries=2, delay=2.0, exceptions=(RuntimeError,))
    async def _create_video_from_images(
        self,
        image_sequence: List[Dict[str, Any]],
        video_config: Dict[str, Any]
    ) -> Path:
        """Create video from image sequence using ffmpeg with validation"""
        if not image_sequence:
            raise VideoCompositionError("Image sequence is empty")
        
        # Validate all images exist
        for frame_data in image_sequence:
            image_path = frame_data.get("image_path")
            if image_path:
                validate_file_path(image_path, "image")
        
        resolution = video_config.get("resolution", "1920x1080")
        fps = video_config.get("fps", 30)
        
        output_path = self.output_dir / "video_images.mp4"
        
        # Create concat file for ffmpeg
        concat_file = self.output_dir / "concat_list.txt"
        try:
            with open(concat_file, 'w', encoding='utf-8') as f:
                for frame_data in image_sequence:
                    image_path = frame_data["image_path"]
                    duration = frame_data["duration"]
                    # Use absolute path and escape single quotes
                    abs_path = str(Path(image_path).absolute()).replace("'", "'\\''")
                    f.write(f"file '{abs_path}'\n")
                    f.write(f"duration {duration}\n")
                # Repeat last frame
                if image_sequence:
                    last_image = image_sequence[-1]["image_path"]
                    abs_path = str(Path(last_image).absolute()).replace("'", "'\\''")
                    f.write(f"file '{abs_path}'\n")
        except Exception as e:
            raise VideoCompositionError(f"Failed to create concat file: {str(e)}")
        
        # Use ffmpeg to create video
        width, height = map(int, resolution.split('x'))
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                   f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
            "-r", str(fps),
            "-pix_fmt", "yuv420p",
            "-y",
            str(output_path)
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown FFmpeg error"
                logger.error(f"FFmpeg error: {error_msg}")
                raise handle_ffmpeg_error(RuntimeError(error_msg), "video creation")
            
            # Validate output file
            validate_file_path(str(output_path), "video")
            
            logger.info(f"Created video from images: {output_path}")
            return output_path
            
        except FileNotFoundError:
            raise handle_ffmpeg_error(FileNotFoundError("ffmpeg"), "video creation")
        except Exception as e:
            if isinstance(e, VideoCompositionError):
                raise
            raise handle_ffmpeg_error(e, "video creation")

    async def _add_audio_to_video(
        self,
        video_path: Path,
        audio_path: str
    ) -> Path:
        """Add audio track to video"""
        output_path = self.output_dir / "video_with_audio.mp4"
        
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            "-y",
            str(output_path)
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"FFmpeg error: {stderr.decode()}")
                raise RuntimeError(f"Failed to add audio: {stderr.decode()}")
            
            logger.info(f"Added audio to video: {output_path}")
            return output_path
        except FileNotFoundError:
            logger.error("FFmpeg not found. Please install ffmpeg.")
            raise RuntimeError("FFmpeg is required for video composition")

    async def _add_subtitles_to_video(
        self,
        video_path: Path,
        subtitles: List[Dict[str, Any]],
        subtitle_config: Dict[str, Any]
    ) -> Path:
        """Add subtitles overlay to video"""
        output_path = self.output_dir / "final_video.mp4"
        
        # Create SRT file
        srt_path = self.output_dir / "subtitles.srt"
        from .subtitle_generator import SubtitleGenerator
        subtitle_gen = SubtitleGenerator()
        subtitle_gen.export_srt(subtitles, srt_path)
        
        # Build subtitle filter
        font_size = subtitle_config.get("font_size", 48)
        font_color = subtitle_config.get("font_color", "#FFFFFF")
        position = subtitle_config.get("position", "bottom")
        
        # Calculate Y position
        y_positions = {
            "top": "h-th-20",
            "center": "(h-text_h)/2",
            "bottom": "h-th-20",
        }
        y_pos = y_positions.get(position, "h-th-20")
        
        # Subtitle filter
        subtitle_filter = f"subtitles={srt_path}:force_style='FontSize={font_size},PrimaryColour={font_color},Alignment=2'"
        
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", subtitle_filter,
            "-c:a", "copy",
            "-y",
            str(output_path)
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"FFmpeg error: {stderr.decode()}")
                raise RuntimeError(f"Failed to add subtitles: {stderr.decode()}")
            
            logger.info(f"Added subtitles to video: {output_path}")
            return output_path
        except FileNotFoundError:
            logger.error("FFmpeg not found. Please install ffmpeg.")
            raise RuntimeError("FFmpeg is required for video composition")

