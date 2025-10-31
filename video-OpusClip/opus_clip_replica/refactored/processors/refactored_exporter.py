"""
Refactored Opus Clip Exporter

Enhanced video exporter with:
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
import os
import tempfile
from pathlib import Path
import structlog
import moviepy.editor as mp
from PIL import Image
import cv2
import numpy as np

from ..core.base_processor import BaseProcessor, ProcessorResult, ProcessorConfig
from ..core.config_manager import ConfigManager

logger = structlog.get_logger("refactored_exporter")

class RefactoredOpusClipExporter(BaseProcessor):
    """
    Refactored Opus Clip video exporter.
    
    Features:
    - Async processing with BaseProcessor
    - Error handling and retries
    - Performance monitoring
    - Caching
    - Modular design
    """
    
    def __init__(self, config: Optional[ProcessorConfig] = None,
                 app_config: Optional[ConfigManager] = None):
        """Initialize the refactored exporter."""
        super().__init__(config)
        self.app_config = app_config
        self.logger = structlog.get_logger("refactored_exporter")
        
        # Export settings
        self.output_dir = None
        self.quality_settings = {}
        
        # Initialize quality settings
        self._initialize_quality_settings()
        
        self.logger.info("Initialized RefactoredOpusClipExporter")
    
    def _initialize_quality_settings(self):
        """Initialize quality settings from config."""
        if self.app_config and hasattr(self.app_config, 'video'):
            self.quality_settings = self.app_config.video.quality_settings
        else:
            self.quality_settings = {
                "low": {"bitrate": "800k", "resolution": "480p"},
                "medium": {"bitrate": "1500k", "resolution": "720p"},
                "high": {"bitrate": "3000k", "resolution": "1080p"},
                "ultra": {"bitrate": "5000k", "resolution": "4k"}
            }
    
    async def _process_impl(self, input_data: Dict[str, Any]) -> ProcessorResult:
        """Process video export with enhanced error handling."""
        start_time = time.time()
        
        try:
            # Extract input parameters
            video_path = input_data.get("video_path")
            segments = input_data.get("segments", [])
            output_format = input_data.get("output_format", "mp4")
            quality = input_data.get("quality", "high")
            output_dir = input_data.get("output_dir")
            
            if not video_path:
                return ProcessorResult(
                    success=False,
                    error="video_path is required"
                )
            
            if not segments:
                return ProcessorResult(
                    success=False,
                    error="segments are required"
                )
            
            # Validate video file
            if not Path(video_path).exists():
                return ProcessorResult(
                    success=False,
                    error=f"Video file not found: {video_path}"
                )
            
            # Validate output format
            if output_format not in ["mp4", "mov", "avi"]:
                return ProcessorResult(
                    success=False,
                    error=f"Unsupported output format: {output_format}"
                )
            
            # Validate quality
            if quality not in self.quality_settings:
                return ProcessorResult(
                    success=False,
                    error=f"Unsupported quality: {quality}"
                )
            
            # Create output directory
            if output_dir:
                self.output_dir = Path(output_dir)
                self.output_dir.mkdir(parents=True, exist_ok=True)
            else:
                self.output_dir = Path(tempfile.mkdtemp(prefix="opus_clips_"))
            
            # Export clips
            exported_clips = await self._export_clips_async(
                video_path, segments, output_format, quality
            )
            
            processing_time = time.time() - start_time
            
            # Prepare result
            result_data = {
                "exported_clips": exported_clips,
                "total_clips": len(exported_clips),
                "output_directory": str(self.output_dir),
                "processing_time": processing_time
            }
            
            return ProcessorResult(
                success=True,
                data=result_data,
                processing_time=processing_time,
                metadata={
                    "video_path": video_path,
                    "output_format": output_format,
                    "quality": quality,
                    "segments_count": len(segments)
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Video export failed: {e}")
            return ProcessorResult(
                success=False,
                error=str(e),
                processing_time=processing_time
            )
    
    async def _export_clips_async(self, video_path: str, segments: List[Dict[str, Any]],
                                 output_format: str, quality: str) -> List[Dict[str, Any]]:
        """Export video clips asynchronously."""
        try:
            # Run export in thread pool
            loop = asyncio.get_event_loop()
            exported_clips = await loop.run_in_executor(
                None,
                self._export_clips_sync,
                video_path, segments, output_format, quality
            )
            return exported_clips
            
        except Exception as e:
            self.logger.error(f"Clip export failed: {e}")
            return []
    
    def _export_clips_sync(self, video_path: str, segments: List[Dict[str, Any]],
                          output_format: str, quality: str) -> List[Dict[str, Any]]:
        """Export video clips synchronously."""
        try:
            exported_clips = []
            
            # Load video
            video = mp.VideoFileClip(video_path)
            
            # Get quality settings
            quality_config = self.quality_settings.get(quality, self.quality_settings["high"])
            bitrate = quality_config["bitrate"]
            
            for i, segment in enumerate(segments):
                try:
                    # Extract clip
                    start_time = segment["start_time"]
                    end_time = segment["end_time"]
                    
                    clip = video.subclip(start_time, end_time)
                    
                    # Generate filename
                    filename = f"clip_{i+1}_{int(start_time)}_{int(end_time)}.{output_format}"
                    output_path = self.output_dir / filename
                    
                    # Export with quality settings
                    clip.write_videofile(
                        str(output_path),
                        bitrate=bitrate,
                        codec='libx264',
                        audio_codec='aac',
                        temp_audiofile='temp-audio.m4a',
                        remove_temp=True,
                        verbose=False,
                        logger=None
                    )
                    
                    # Generate thumbnail
                    thumbnail_path = self._generate_thumbnail_sync(clip, i)
                    
                    # Get file size
                    file_size = output_path.stat().st_size
                    
                    exported_clips.append({
                        "clip_id": segment.get("segment_id", f"clip_{i+1}"),
                        "filename": filename,
                        "path": str(output_path),
                        "thumbnail": thumbnail_path,
                        "duration": segment["duration"],
                        "start_time": start_time,
                        "end_time": end_time,
                        "size": file_size,
                        "quality": quality,
                        "format": output_format
                    })
                    
                    clip.close()
                    
                except Exception as e:
                    self.logger.error(f"Failed to export clip {i}: {e}")
                    continue
            
            video.close()
            return exported_clips
            
        except Exception as e:
            self.logger.error(f"Clip export failed: {e}")
            return []
    
    def _generate_thumbnail_sync(self, clip, index: int) -> str:
        """Generate thumbnail synchronously."""
        try:
            # Get frame at 25% of clip duration
            thumbnail_time = clip.duration * 0.25
            frame = clip.get_frame(thumbnail_time)
            
            # Convert to PIL Image
            image = Image.fromarray(frame)
            
            # Resize to standard thumbnail size
            image.thumbnail((320, 180), Image.Resampling.LANCZOS)
            
            # Save thumbnail
            thumbnail_path = self.output_dir / f"thumb_{index+1}.jpg"
            image.save(thumbnail_path, "JPEG", quality=85)
            
            return str(thumbnail_path)
            
        except Exception as e:
            self.logger.error(f"Thumbnail generation failed: {e}")
            return ""
    
    async def export_single_clip(self, video_path: str, start_time: float, end_time: float,
                                output_format: str = "mp4", quality: str = "high",
                                output_dir: Optional[str] = None) -> ProcessorResult:
        """Export a single clip."""
        segment = {
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "segment_id": "single_clip"
        }
        
        input_data = {
            "video_path": video_path,
            "segments": [segment],
            "output_format": output_format,
            "quality": quality,
            "output_dir": output_dir
        }
        
        return await self.process(input_data)
    
    async def export_batch_clips(self, video_path: str, segments: List[Dict[str, Any]],
                                output_format: str = "mp4", quality: str = "high",
                                output_dir: Optional[str] = None,
                                batch_size: int = 5) -> List[ProcessorResult]:
        """Export clips in batches for better performance."""
        try:
            results = []
            
            # Process segments in batches
            for i in range(0, len(segments), batch_size):
                batch_segments = segments[i:i+batch_size]
                
                input_data = {
                    "video_path": video_path,
                    "segments": batch_segments,
                    "output_format": output_format,
                    "quality": quality,
                    "output_dir": output_dir
                }
                
                result = await self.process(input_data)
                results.append(result)
                
                # Small delay between batches to prevent resource exhaustion
                if i + batch_size < len(segments):
                    await asyncio.sleep(0.1)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch export failed: {e}")
            return []
    
    async def get_export_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get export progress for a job."""
        # This would be implemented with job tracking
        # For now, return basic status
        return {
            "job_id": job_id,
            "status": "processing",
            "progress": 0.0
        }
    
    async def cancel_export(self, job_id: str) -> bool:
        """Cancel an export job."""
        return await self.cancel_job(job_id)
    
    async def cleanup_old_exports(self, max_age_hours: int = 24):
        """Clean up old export files."""
        try:
            if not self.output_dir or not self.output_dir.exists():
                return
            
            import time
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            cleaned_count = 0
            for file_path in self.output_dir.iterdir():
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        file_path.unlink()
                        cleaned_count += 1
            
            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} old export files")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    async def get_export_statistics(self) -> Dict[str, Any]:
        """Get export statistics."""
        try:
            if not self.output_dir or not self.output_dir.exists():
                return {"total_files": 0, "total_size": 0}
            
            total_files = 0
            total_size = 0
            
            for file_path in self.output_dir.iterdir():
                if file_path.is_file():
                    total_files += 1
                    total_size += file_path.stat().st_size
            
            return {
                "total_files": total_files,
                "total_size": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "output_directory": str(self.output_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get export statistics: {e}")
            return {"error": str(e)}


