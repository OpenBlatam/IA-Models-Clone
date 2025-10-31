"""
Mobile API for Ultimate Opus Clip

Optimized API specifically designed for mobile applications
with lightweight responses, efficient data transfer, and
mobile-specific features.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union
import asyncio
import time
import json
import base64
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
from PIL import Image
import io
import aiofiles

logger = structlog.get_logger("mobile_api")

class MobilePlatform(Enum):
    """Supported mobile platforms."""
    IOS = "ios"
    ANDROID = "android"
    FLUTTER = "flutter"
    REACT_NATIVE = "react_native"

class CompressionLevel(Enum):
    """Compression levels for mobile optimization."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class QualityLevel(Enum):
    """Quality levels for mobile optimization."""
    LOW = "low"      # 480p
    MEDIUM = "medium"  # 720p
    HIGH = "high"    # 1080p
    ULTRA = "ultra"  # 4K

@dataclass
class MobileConfig:
    """Configuration for mobile API."""
    max_file_size_mb: int = 100
    max_duration_seconds: int = 300
    supported_formats: List[str] = None
    compression_level: CompressionLevel = CompressionLevel.MEDIUM
    quality_level: QualityLevel = QualityLevel.MEDIUM
    enable_offline_mode: bool = True
    enable_progressive_upload: bool = True
    enable_thumbnail_generation: bool = True
    thumbnail_size: tuple = (320, 240)
    cache_duration: int = 3600  # 1 hour

@dataclass
class MobileVideoRequest:
    """Mobile-optimized video request."""
    video_data: str  # Base64 encoded video
    platform: MobilePlatform
    quality: QualityLevel
    compression: CompressionLevel
    duration: float
    file_size: int
    metadata: Dict[str, Any] = None

@dataclass
class MobileVideoResponse:
    """Mobile-optimized video response."""
    job_id: str
    status: str
    progress: float
    thumbnail_url: Optional[str] = None
    preview_url: Optional[str] = None
    download_url: Optional[str] = None
    estimated_time: Optional[float] = None
    file_size: Optional[int] = None
    quality: Optional[str] = None
    compression_ratio: Optional[float] = None

@dataclass
class MobileJobStatus:
    """Mobile job status."""
    job_id: str
    status: str
    progress: float
    current_step: str
    estimated_completion: Optional[float] = None
    error_message: Optional[str] = None
    result_urls: Dict[str, str] = None

class MobileAPI:
    """Mobile-optimized API for Ultimate Opus Clip."""
    
    def __init__(self, config: MobileConfig = None):
        self.config = config or MobileConfig()
        self.app = FastAPI(
            title="Ultimate Opus Clip Mobile API",
            description="Mobile-optimized API for video processing",
            version="2.0.0"
        )
        self.jobs: Dict[str, MobileJobStatus] = {}
        self.security = HTTPBearer(auto_error=False)
        
        # Initialize supported formats
        if self.config.supported_formats is None:
            self.config.supported_formats = ["mp4", "mov", "avi", "mkv"]
        
        self._setup_middleware()
        self._setup_routes()
        
        logger.info("Mobile API initialized")
    
    def _setup_middleware(self):
        """Setup middleware for mobile optimization."""
        # CORS for mobile apps
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Gzip compression for mobile data savings
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.post("/mobile/upload", response_model=MobileVideoResponse)
        async def upload_video_mobile(
            video_data: str = Form(...),
            platform: str = Form(...),
            quality: str = Form("medium"),
            compression: str = Form("medium"),
            duration: float = Form(...),
            file_size: int = Form(...),
            metadata: str = Form("{}")
        ):
            """Upload video for mobile processing."""
            try:
                # Validate platform
                try:
                    platform_enum = MobilePlatform(platform)
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid platform")
                
                # Validate quality
                try:
                    quality_enum = QualityLevel(quality)
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid quality level")
                
                # Validate compression
                try:
                    compression_enum = CompressionLevel(compression)
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid compression level")
                
                # Validate file size
                if file_size > self.config.max_file_size_mb * 1024 * 1024:
                    raise HTTPException(
                        status_code=413, 
                        detail=f"File too large. Max size: {self.config.max_file_size_mb}MB"
                    )
                
                # Validate duration
                if duration > self.config.max_duration_seconds:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Video too long. Max duration: {self.config.max_duration_seconds}s"
                    )
                
                # Decode video data
                try:
                    video_bytes = base64.b64decode(video_data)
                except Exception:
                    raise HTTPException(status_code=400, detail="Invalid video data")
                
                # Generate job ID
                job_id = f"mobile_{int(time.time())}_{platform_enum.value}"
                
                # Create job status
                job_status = MobileJobStatus(
                    job_id=job_id,
                    status="uploaded",
                    progress=0.0,
                    current_step="Processing video data"
                )
                self.jobs[job_id] = job_status
                
                # Start processing asynchronously
                asyncio.create_task(self._process_mobile_video(
                    job_id, video_bytes, platform_enum, quality_enum, compression_enum
                ))
                
                # Generate thumbnail if enabled
                thumbnail_url = None
                if self.config.enable_thumbnail_generation:
                    thumbnail_url = await self._generate_thumbnail(video_bytes, job_id)
                
                return MobileVideoResponse(
                    job_id=job_id,
                    status="processing",
                    progress=0.0,
                    thumbnail_url=thumbnail_url,
                    estimated_time=self._estimate_processing_time(duration, quality_enum)
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error in mobile upload: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/mobile/status/{job_id}", response_model=MobileJobStatus)
        async def get_job_status(job_id: str):
            """Get job status for mobile."""
            if job_id not in self.jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            
            return self.jobs[job_id]
        
        @self.app.get("/mobile/download/{job_id}")
        async def download_result(job_id: str, format: str = "mp4"):
            """Download processed video for mobile."""
            if job_id not in self.jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            
            job = self.jobs[job_id]
            if job.status != "completed":
                raise HTTPException(status_code=400, detail="Job not completed")
            
            # Return file path (in production, this would be a cloud URL)
            file_path = f"outputs/{job_id}.{format}"
            if not Path(file_path).exists():
                raise HTTPException(status_code=404, detail="Result file not found")
            
            return FileResponse(
                file_path,
                media_type="video/mp4",
                filename=f"processed_{job_id}.{format}"
            )
        
        @self.app.get("/mobile/health")
        async def health_check():
            """Health check for mobile."""
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "version": "2.0.0",
                "mobile_optimized": True
            }
        
        @self.app.get("/mobile/config")
        async def get_mobile_config():
            """Get mobile configuration."""
            return {
                "max_file_size_mb": self.config.max_file_size_mb,
                "max_duration_seconds": self.config.max_duration_seconds,
                "supported_formats": self.config.supported_formats,
                "quality_levels": [q.value for q in QualityLevel],
                "compression_levels": [c.value for c in CompressionLevel],
                "thumbnail_size": self.config.thumbnail_size,
                "offline_mode": self.config.enable_offline_mode
            }
        
        @self.app.post("/mobile/batch")
        async def upload_batch_mobile(
            videos: List[str] = Form(...),
            platform: str = Form(...),
            quality: str = Form("medium"),
            compression: str = Form("medium")
        ):
            """Upload multiple videos for batch processing."""
            try:
                platform_enum = MobilePlatform(platform)
                quality_enum = QualityLevel(quality)
                compression_enum = CompressionLevel(compression)
                
                job_ids = []
                
                for i, video_data in enumerate(videos):
                    try:
                        video_bytes = base64.b64decode(video_data)
                        job_id = f"mobile_batch_{int(time.time())}_{i}"
                        
                        job_status = MobileJobStatus(
                            job_id=job_id,
                            status="uploaded",
                            progress=0.0,
                            current_step="Processing video data"
                        )
                        self.jobs[job_id] = job_status
                        
                        asyncio.create_task(self._process_mobile_video(
                            job_id, video_bytes, platform_enum, quality_enum, compression_enum
                        ))
                        
                        job_ids.append(job_id)
                        
                    except Exception as e:
                        logger.error(f"Error processing video {i}: {e}")
                        continue
                
                return {
                    "batch_id": f"batch_{int(time.time())}",
                    "job_ids": job_ids,
                    "total_videos": len(videos),
                    "successful_uploads": len(job_ids)
                }
                
            except Exception as e:
                logger.error(f"Error in batch upload: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/mobile/thumbnails/{job_id}")
        async def get_thumbnail(job_id: str):
            """Get video thumbnail for mobile."""
            if job_id not in self.jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            
            thumbnail_path = f"thumbnails/{job_id}.jpg"
            if not Path(thumbnail_path).exists():
                raise HTTPException(status_code=404, detail="Thumbnail not found")
            
            return FileResponse(
                thumbnail_path,
                media_type="image/jpeg",
                filename=f"thumbnail_{job_id}.jpg"
            )
    
    async def _process_mobile_video(
        self, 
        job_id: str, 
        video_bytes: bytes, 
        platform: MobilePlatform, 
        quality: QualityLevel, 
        compression: CompressionLevel
    ):
        """Process video for mobile."""
        try:
            job = self.jobs[job_id]
            job.status = "processing"
            job.progress = 10.0
            job.current_step = "Analyzing video"
            
            # Save video temporarily
            temp_path = f"temp/{job_id}_input.mp4"
            Path("temp").mkdir(exist_ok=True)
            
            async with aiofiles.open(temp_path, 'wb') as f:
                await f.write(video_bytes)
            
            job.progress = 30.0
            job.current_step = "Processing video"
            
            # Simulate video processing
            await asyncio.sleep(2)  # Simulate processing time
            
            job.progress = 70.0
            job.current_step = "Optimizing for mobile"
            
            # Apply mobile optimizations
            optimized_path = await self._optimize_for_mobile(
                temp_path, platform, quality, compression
            )
            
            job.progress = 90.0
            job.current_step = "Finalizing"
            
            # Move to output directory
            output_path = f"outputs/{job_id}.mp4"
            Path("outputs").mkdir(exist_ok=True)
            
            async with aiofiles.open(optimized_path, 'rb') as src:
                async with aiofiles.open(output_path, 'wb') as dst:
                    await dst.write(await src.read())
            
            # Cleanup temp file
            Path(temp_path).unlink(missing_ok=True)
            Path(optimized_path).unlink(missing_ok=True)
            
            job.status = "completed"
            job.progress = 100.0
            job.current_step = "Completed"
            job.result_urls = {
                "download": f"/mobile/download/{job_id}",
                "thumbnail": f"/mobile/thumbnails/{job_id}"
            }
            
            logger.info(f"Mobile video processing completed for job {job_id}")
            
        except Exception as e:
            logger.error(f"Error processing mobile video {job_id}: {e}")
            job = self.jobs[job_id]
            job.status = "failed"
            job.error_message = str(e)
            job.current_step = "Failed"
    
    async def _optimize_for_mobile(
        self, 
        video_path: str, 
        platform: MobilePlatform, 
        quality: QualityLevel, 
        compression: CompressionLevel
    ) -> str:
        """Optimize video for mobile platform."""
        # This is a placeholder - in production, use actual video processing
        # For now, just copy the file
        optimized_path = f"temp/{Path(video_path).stem}_optimized.mp4"
        
        async with aiofiles.open(video_path, 'rb') as src:
            async with aiofiles.open(optimized_path, 'wb') as dst:
                await dst.write(await src.read())
        
        return optimized_path
    
    async def _generate_thumbnail(self, video_bytes: bytes, job_id: str) -> str:
        """Generate thumbnail for mobile."""
        try:
            # This is a placeholder - in production, use actual video thumbnail generation
            # For now, create a simple placeholder image
            thumbnail_path = f"thumbnails/{job_id}.jpg"
            Path("thumbnails").mkdir(exist_ok=True)
            
            # Create a simple placeholder thumbnail
            img = Image.new('RGB', self.config.thumbnail_size, color='gray')
            img.save(thumbnail_path, 'JPEG')
            
            return f"/mobile/thumbnails/{job_id}"
            
        except Exception as e:
            logger.error(f"Error generating thumbnail: {e}")
            return None
    
    def _estimate_processing_time(self, duration: float, quality: QualityLevel) -> float:
        """Estimate processing time based on video duration and quality."""
        base_time = duration * 0.1  # Base processing time
        
        # Adjust based on quality
        quality_multiplier = {
            QualityLevel.LOW: 0.5,
            QualityLevel.MEDIUM: 1.0,
            QualityLevel.HIGH: 1.5,
            QualityLevel.ULTRA: 2.0
        }
        
        return base_time * quality_multiplier.get(quality, 1.0)

class MobileAPIResponse:
    """Mobile-optimized API response wrapper."""
    
    @staticmethod
    def success(data: Any, message: str = "Success") -> JSONResponse:
        """Create success response."""
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": message,
                "data": data,
                "timestamp": time.time()
            }
        )
    
    @staticmethod
    def error(message: str, code: int = 400, details: Any = None) -> JSONResponse:
        """Create error response."""
        return JSONResponse(
            status_code=code,
            content={
                "success": False,
                "message": message,
                "details": details,
                "timestamp": time.time()
            }
        )
    
    @staticmethod
    def mobile_optimized(data: Any, compression_ratio: float = None) -> JSONResponse:
        """Create mobile-optimized response."""
        response_data = {
            "success": True,
            "data": data,
            "mobile_optimized": True,
            "timestamp": time.time()
        }
        
        if compression_ratio:
            response_data["compression_ratio"] = compression_ratio
        
        return JSONResponse(content=response_data)

# Global mobile API instance
_global_mobile_api: Optional[MobileAPI] = None

def get_mobile_api() -> MobileAPI:
    """Get the global mobile API instance."""
    global _global_mobile_api
    if _global_mobile_api is None:
        _global_mobile_api = MobileAPI()
    return _global_mobile_api

def run_mobile_api(host: str = "0.0.0.0", port: int = 8002):
    """Run the mobile API server."""
    mobile_api = get_mobile_api()
    
    logger.info(f"Starting Mobile API on {host}:{port}")
    
    uvicorn.run(
        mobile_api.app,
        host=host,
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    run_mobile_api()


