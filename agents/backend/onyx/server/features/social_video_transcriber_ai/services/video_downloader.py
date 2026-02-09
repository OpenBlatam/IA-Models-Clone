"""
Video Downloader Service
Downloads videos from TikTok, Instagram, and YouTube using yt-dlp
"""

import os
import re
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from uuid import UUID
import yt_dlp

from ..config.settings import get_settings
from ..core.models import SupportedPlatform

logger = logging.getLogger(__name__)


class VideoDownloader:
    """Service for downloading videos from social media platforms"""
    
    PLATFORM_PATTERNS = {
        SupportedPlatform.YOUTUBE: [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+',
            r'(?:https?://)?(?:www\.)?youtu\.be/[\w-]+',
            r'(?:https?://)?(?:www\.)?youtube\.com/shorts/[\w-]+',
        ],
        SupportedPlatform.TIKTOK: [
            r'(?:https?://)?(?:www\.)?tiktok\.com/@[\w.-]+/video/\d+',
            r'(?:https?://)?(?:vm\.)?tiktok\.com/[\w]+',
            r'(?:https?://)?(?:www\.)?tiktok\.com/t/[\w]+',
        ],
        SupportedPlatform.INSTAGRAM: [
            r'(?:https?://)?(?:www\.)?instagram\.com/reel/[\w-]+',
            r'(?:https?://)?(?:www\.)?instagram\.com/p/[\w-]+',
            r'(?:https?://)?(?:www\.)?instagram\.com/tv/[\w-]+',
        ],
    }
    
    def __init__(self):
        self.settings = get_settings()
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories"""
        for dir_path in [
            self.settings.downloads_dir,
            self.settings.audio_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def detect_platform(self, url: str) -> SupportedPlatform:
        """Detect which platform a URL is from"""
        for platform, patterns in self.PLATFORM_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, url, re.IGNORECASE):
                    return platform
        raise ValueError(f"Unsupported URL format: {url}")
    
    def _get_yt_dlp_options(self, job_id: UUID, extract_audio: bool = True) -> Dict[str, Any]:
        """Get yt-dlp options for downloading"""
        output_template = os.path.join(
            self.settings.downloads_dir,
            f"{job_id}.%(ext)s"
        )
        
        options = {
            'format': 'bestaudio/best' if extract_audio else 'best',
            'outtmpl': output_template,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'socket_timeout': self.settings.download_timeout,
            'retries': 3,
            'nocheckcertificate': True,
        }
        
        if extract_audio:
            options['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }]
        
        return options
    
    async def get_video_info(self, url: str) -> Dict[str, Any]:
        """
        Get video information without downloading
        
        Args:
            url: Video URL
            
        Returns:
            Dict with video metadata
        """
        options = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'skip_download': True,
        }
        
        def _extract_info():
            with yt_dlp.YoutubeDL(options) as ydl:
                return ydl.extract_info(url, download=False)
        
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, _extract_info)
        
        return {
            'title': info.get('title'),
            'duration': info.get('duration'),
            'author': info.get('uploader') or info.get('channel'),
            'description': info.get('description'),
            'thumbnail': info.get('thumbnail'),
            'view_count': info.get('view_count'),
            'like_count': info.get('like_count'),
            'upload_date': info.get('upload_date'),
            'platform': self.detect_platform(url).value,
        }
    
    async def download_video(
        self,
        url: str,
        job_id: UUID,
        extract_audio: bool = True,
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Download video and optionally extract audio
        
        Args:
            url: Video URL
            job_id: Unique job identifier
            extract_audio: Whether to extract audio for transcription
            
        Returns:
            Tuple of (file_path, video_info)
        """
        logger.info(f"Downloading video: {url} (job_id: {job_id})")
        
        platform = self.detect_platform(url)
        options = self._get_yt_dlp_options(job_id, extract_audio)
        
        def _download():
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=True)
                return info
        
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, _download)
        
        # Check for duration limit
        duration = info.get('duration', 0)
        if duration > self.settings.max_video_duration:
            raise ValueError(
                f"Video duration ({duration}s) exceeds maximum allowed "
                f"({self.settings.max_video_duration}s)"
            )
        
        # Find the downloaded file
        if extract_audio:
            audio_path = Path(self.settings.downloads_dir) / f"{job_id}.wav"
            if not audio_path.exists():
                # Try other audio formats
                for ext in ['m4a', 'mp3', 'opus', 'webm']:
                    alt_path = Path(self.settings.downloads_dir) / f"{job_id}.{ext}"
                    if alt_path.exists():
                        audio_path = alt_path
                        break
            file_path = audio_path
        else:
            # Find video file
            for ext in ['mp4', 'webm', 'mkv']:
                video_path = Path(self.settings.downloads_dir) / f"{job_id}.{ext}"
                if video_path.exists():
                    file_path = video_path
                    break
            else:
                raise FileNotFoundError(f"Downloaded file not found for job {job_id}")
        
        video_info = {
            'title': info.get('title'),
            'duration': info.get('duration'),
            'author': info.get('uploader') or info.get('channel'),
            'description': info.get('description'),
            'thumbnail': info.get('thumbnail'),
            'platform': platform.value,
        }
        
        logger.info(f"Download complete: {file_path}")
        return file_path, video_info
    
    async def cleanup(self, job_id: UUID):
        """Remove downloaded files for a job"""
        patterns = [
            f"{job_id}.*",
        ]
        
        for pattern in patterns:
            for file_path in Path(self.settings.downloads_dir).glob(pattern):
                try:
                    file_path.unlink()
                    logger.debug(f"Deleted: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete {file_path}: {e}")


_video_downloader: Optional[VideoDownloader] = None


def get_video_downloader() -> VideoDownloader:
    """Get video downloader singleton"""
    global _video_downloader
    if _video_downloader is None:
        _video_downloader = VideoDownloader()
    return _video_downloader












