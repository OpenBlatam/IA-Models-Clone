"""
Advanced Media Service for comprehensive media management
"""

import asyncio
import json
import os
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union, BinaryIO
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, text
from dataclasses import dataclass
from enum import Enum
import hashlib
import mimetypes
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip
import librosa
import soundfile as sf
import requests
from bs4 import BeautifulSoup
import aiofiles
import aiohttp
from pathlib import Path
import magic
import exifread
import ffmpeg
import whisper
from transformers import pipeline
import torch
from sentence_transformers import SentenceTransformer
import face_recognition
import pytesseract
from gtts import gTTS
import speech_recognition as sr

from ..models.database import MediaFile, MediaMetadata, MediaProcessing, MediaTag, MediaCollection
from ..core.exceptions import DatabaseError, ValidationError


class MediaType(Enum):
    """Media type enumeration."""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    ARCHIVE = "archive"
    OTHER = "other"


class MediaStatus(Enum):
    """Media status enumeration."""
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"
    DELETED = "deleted"


class ProcessingType(Enum):
    """Processing type enumeration."""
    THUMBNAIL = "thumbnail"
    RESIZE = "resize"
    COMPRESS = "compress"
    WATERMARK = "watermark"
    FILTER = "filter"
    OCR = "ocr"
    TRANSCRIPTION = "transcription"
    FACE_DETECTION = "face_detection"
    OBJECT_DETECTION = "object_detection"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    TRANSLATION = "translation"
    ENHANCEMENT = "enhancement"


@dataclass
class MediaMetadata:
    """Media metadata structure."""
    file_id: str
    filename: str
    original_filename: str
    file_size: int
    mime_type: str
    media_type: MediaType
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[float] = None
    bitrate: Optional[int] = None
    fps: Optional[float] = None
    channels: Optional[int] = None
    sample_rate: Optional[int] = None
    exif_data: Optional[Dict[str, Any]] = None
    color_profile: Optional[str] = None
    compression: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class ProcessingResult:
    """Processing result structure."""
    success: bool
    output_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None


class AdvancedMediaService:
    """Service for advanced media management operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.media_cache = {}
        self.processing_queue = []
        self.ai_models = {}
        self._initialize_ai_models()
        self._initialize_directories()
    
    def _initialize_ai_models(self):
        """Initialize AI models for media processing."""
        try:
            # Initialize Whisper for speech-to-text
            try:
                self.ai_models['whisper'] = whisper.load_model("base")
            except Exception as e:
                print(f"Warning: Could not load Whisper model: {e}")
            
            # Initialize object detection model
            try:
                self.ai_models['object_detection'] = pipeline(
                    "object-detection",
                    model="facebook/detr-resnet-50",
                    device=0 if torch.cuda.is_available() else -1
                )
            except Exception as e:
                print(f"Warning: Could not load object detection model: {e}")
            
            # Initialize image captioning model
            try:
                self.ai_models['image_captioning'] = pipeline(
                    "image-to-text",
                    model="nlpconnect/vit-gpt2-image-captioning",
                    device=0 if torch.cuda.is_available() else -1
                )
            except Exception as e:
                print(f"Warning: Could not load image captioning model: {e}")
            
            # Initialize sentence transformer for similarity
            try:
                self.ai_models['sentence_transformer'] = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Warning: Could not load sentence transformer: {e}")
                
        except Exception as e:
            print(f"Warning: Could not initialize AI models: {e}")
    
    def _initialize_directories(self):
        """Initialize media storage directories."""
        try:
            base_dir = Path("media")
            directories = [
                "uploads",
                "processed",
                "thumbnails",
                "watermarks",
                "transcriptions",
                "ocr",
                "temp"
            ]
            
            for directory in directories:
                dir_path = base_dir / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                
        except Exception as e:
            print(f"Warning: Could not initialize directories: {e}")
    
    async def upload_media(
        self,
        file_data: bytes,
        filename: str,
        user_id: str = None,
        collection_id: str = None,
        tags: List[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Upload and process media file."""
        try:
            # Generate unique file ID
            file_id = str(uuid.uuid4())
            
            # Detect file type
            mime_type = magic.from_buffer(file_data, mime=True)
            media_type = self._get_media_type(mime_type)
            
            # Validate file
            validation_result = await self._validate_media(file_data, mime_type, media_type)
            if not validation_result["valid"]:
                raise ValidationError(f"Media validation failed: {validation_result['errors']}")
            
            # Save original file
            file_path = await self._save_file(file_data, file_id, "uploads")
            
            # Extract metadata
            media_metadata = await self._extract_metadata(file_data, filename, mime_type, media_type)
            
            # Create database record
            media_file = MediaFile(
                file_id=file_id,
                filename=filename,
                original_filename=filename,
                file_path=str(file_path),
                file_size=len(file_data),
                mime_type=mime_type,
                media_type=media_type.value,
                user_id=user_id,
                collection_id=collection_id,
                status=MediaStatus.UPLOADING.value,
                metadata=media_metadata.__dict__,
                created_at=datetime.utcnow()
            )
            
            self.session.add(media_file)
            await self.session.commit()
            
            # Start background processing
            asyncio.create_task(self._process_media_background(file_id))
            
            return {
                "success": True,
                "file_id": file_id,
                "metadata": media_metadata.__dict__,
                "message": "Media uploaded successfully"
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to upload media: {str(e)}")
    
    def _get_media_type(self, mime_type: str) -> MediaType:
        """Determine media type from MIME type."""
        if mime_type.startswith('image/'):
            return MediaType.IMAGE
        elif mime_type.startswith('video/'):
            return MediaType.VIDEO
        elif mime_type.startswith('audio/'):
            return MediaType.AUDIO
        elif mime_type in ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            return MediaType.DOCUMENT
        elif mime_type in ['application/zip', 'application/x-rar-compressed', 'application/x-7z-compressed']:
            return MediaType.ARCHIVE
        else:
            return MediaType.OTHER
    
    async def _validate_media(
        self,
        file_data: bytes,
        mime_type: str,
        media_type: MediaType
    ) -> Dict[str, Any]:
        """Validate media file."""
        try:
            errors = []
            
            # Check file size (max 100MB)
            max_size = 100 * 1024 * 1024  # 100MB
            if len(file_data) > max_size:
                errors.append(f"File size exceeds maximum limit of {max_size} bytes")
            
            # Check MIME type
            allowed_types = {
                MediaType.IMAGE: ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/svg+xml'],
                MediaType.VIDEO: ['video/mp4', 'video/avi', 'video/mov', 'video/wmv', 'video/webm'],
                MediaType.AUDIO: ['audio/mp3', 'audio/wav', 'audio/ogg', 'audio/m4a', 'audio/flac'],
                MediaType.DOCUMENT: ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
                MediaType.ARCHIVE: ['application/zip', 'application/x-rar-compressed', 'application/x-7z-compressed']
            }
            
            if media_type in allowed_types and mime_type not in allowed_types[media_type]:
                errors.append(f"Unsupported MIME type: {mime_type}")
            
            # Additional validation based on media type
            if media_type == MediaType.IMAGE:
                try:
                    with Image.open(io.BytesIO(file_data)) as img:
                        img.verify()
                except Exception:
                    errors.append("Invalid image file")
            
            elif media_type == MediaType.VIDEO:
                try:
                    # Basic video validation
                    pass
                except Exception:
                    errors.append("Invalid video file")
            
            elif media_type == MediaType.AUDIO:
                try:
                    # Basic audio validation
                    pass
                except Exception:
                    errors.append("Invalid audio file")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors
            }
            
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"]
            }
    
    async def _save_file(self, file_data: bytes, file_id: str, directory: str) -> Path:
        """Save file to storage."""
        try:
            file_path = Path("media") / directory / f"{file_id}"
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_data)
            return file_path
        except Exception as e:
            raise DatabaseError(f"Failed to save file: {str(e)}")
    
    async def _extract_metadata(
        self,
        file_data: bytes,
        filename: str,
        mime_type: str,
        media_type: MediaType
    ) -> MediaMetadata:
        """Extract metadata from media file."""
        try:
            metadata = MediaMetadata(
                file_id="",
                filename=filename,
                original_filename=filename,
                file_size=len(file_data),
                mime_type=mime_type,
                media_type=media_type,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            if media_type == MediaType.IMAGE:
                metadata = await self._extract_image_metadata(file_data, metadata)
            elif media_type == MediaType.VIDEO:
                metadata = await self._extract_video_metadata(file_data, metadata)
            elif media_type == MediaType.AUDIO:
                metadata = await self._extract_audio_metadata(file_data, metadata)
            
            return metadata
            
        except Exception as e:
            return metadata  # Return basic metadata if extraction fails
    
    async def _extract_image_metadata(self, file_data: bytes, metadata: MediaMetadata) -> MediaMetadata:
        """Extract image metadata."""
        try:
            with Image.open(io.BytesIO(file_data)) as img:
                metadata.width = img.width
                metadata.height = img.height
                metadata.color_profile = img.mode
                
                # Extract EXIF data
                if hasattr(img, '_getexif') and img._getexif():
                    exif_data = {}
                    for tag, value in img._getexif().items():
                        exif_data[tag] = str(value)
                    metadata.exif_data = exif_data
                
                # Extract compression info
                if hasattr(img, 'format'):
                    metadata.compression = img.format
            
            return metadata
            
        except Exception as e:
            return metadata
    
    async def _extract_video_metadata(self, file_data: bytes, metadata: MediaMetadata) -> MediaMetadata:
        """Extract video metadata."""
        try:
            # Save temporary file for processing
            temp_path = Path("media/temp") / f"temp_{uuid.uuid4()}"
            async with aiofiles.open(temp_path, 'wb') as f:
                await f.write(file_data)
            
            try:
                # Use ffmpeg to extract metadata
                probe = ffmpeg.probe(str(temp_path))
                video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
                
                if video_stream:
                    metadata.width = int(video_stream.get('width', 0))
                    metadata.height = int(video_stream.get('height', 0))
                    metadata.fps = eval(video_stream.get('r_frame_rate', '0/1'))
                    metadata.bitrate = int(video_stream.get('bit_rate', 0))
                
                # Get duration
                metadata.duration = float(probe.get('format', {}).get('duration', 0))
                
            finally:
                # Clean up temporary file
                if temp_path.exists():
                    temp_path.unlink()
            
            return metadata
            
        except Exception as e:
            return metadata
    
    async def _extract_audio_metadata(self, file_data: bytes, metadata: MediaMetadata) -> MediaMetadata:
        """Extract audio metadata."""
        try:
            # Save temporary file for processing
            temp_path = Path("media/temp") / f"temp_{uuid.uuid4()}"
            async with aiofiles.open(temp_path, 'wb') as f:
                await f.write(file_data)
            
            try:
                # Use librosa to extract audio metadata
                y, sr = librosa.load(str(temp_path), sr=None)
                metadata.sample_rate = sr
                metadata.channels = 1 if y.ndim == 1 else y.shape[0]
                metadata.duration = len(y) / sr
                
                # Get bitrate from file info
                info = sf.info(str(temp_path))
                metadata.bitrate = info.samplerate * info.channels * 16  # Approximate
                
            finally:
                # Clean up temporary file
                if temp_path.exists():
                    temp_path.unlink()
            
            return metadata
            
        except Exception as e:
            return metadata
    
    async def _process_media_background(self, file_id: str):
        """Process media file in background."""
        try:
            # Get media file
            media_query = select(MediaFile).where(MediaFile.file_id == file_id)
            media_result = await self.session.execute(media_query)
            media_file = media_result.scalar_one_or_none()
            
            if not media_file:
                return
            
            # Update status to processing
            media_file.status = MediaStatus.PROCESSING.value
            await self.session.commit()
            
            # Process based on media type
            if media_file.media_type == MediaType.IMAGE.value:
                await self._process_image(media_file)
            elif media_file.media_type == MediaType.VIDEO.value:
                await self._process_video(media_file)
            elif media_file.media_type == MediaType.AUDIO.value:
                await self._process_audio(media_file)
            
            # Update status to ready
            media_file.status = MediaStatus.READY.value
            media_file.updated_at = datetime.utcnow()
            await self.session.commit()
            
        except Exception as e:
            # Update status to error
            media_file.status = MediaStatus.ERROR.value
            media_file.updated_at = datetime.utcnow()
            await self.session.commit()
            print(f"Error processing media {file_id}: {e}")
    
    async def _process_image(self, media_file: MediaFile):
        """Process image file."""
        try:
            # Generate thumbnail
            await self._generate_thumbnail(media_file)
            
            # Generate multiple sizes
            await self._generate_sizes(media_file)
            
            # Extract text using OCR
            await self._extract_text_ocr(media_file)
            
            # Detect faces
            await self._detect_faces(media_file)
            
            # Detect objects
            await self._detect_objects(media_file)
            
            # Generate image caption
            await self._generate_image_caption(media_file)
            
        except Exception as e:
            print(f"Error processing image {media_file.file_id}: {e}")
    
    async def _process_video(self, media_file: MediaFile):
        """Process video file."""
        try:
            # Generate video thumbnail
            await self._generate_video_thumbnail(media_file)
            
            # Extract audio for transcription
            await self._extract_audio_from_video(media_file)
            
            # Transcribe audio
            await self._transcribe_audio(media_file)
            
            # Generate video summary
            await self._generate_video_summary(media_file)
            
        except Exception as e:
            print(f"Error processing video {media_file.file_id}: {e}")
    
    async def _process_audio(self, media_file: MediaFile):
        """Process audio file."""
        try:
            # Transcribe audio
            await self._transcribe_audio(media_file)
            
            # Extract audio features
            await self._extract_audio_features(media_file)
            
            # Generate audio summary
            await self._generate_audio_summary(media_file)
            
        except Exception as e:
            print(f"Error processing audio {media_file.file_id}: {e}")
    
    async def _generate_thumbnail(self, media_file: MediaFile) -> ProcessingResult:
        """Generate thumbnail for image."""
        try:
            start_time = datetime.now()
            
            # Load image
            with Image.open(media_file.file_path) as img:
                # Create thumbnail
                img.thumbnail((300, 300), Image.Resampling.LANCZOS)
                
                # Save thumbnail
                thumbnail_path = Path("media/thumbnails") / f"{media_file.file_id}_thumb.jpg"
                img.save(thumbnail_path, "JPEG", quality=85)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                success=True,
                output_path=str(thumbnail_path),
                processing_time=processing_time
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )
    
    async def _generate_sizes(self, media_file: MediaFile):
        """Generate multiple sizes for image."""
        try:
            sizes = [
                (150, 150, "small"),
                (300, 300, "medium"),
                (600, 600, "large"),
                (1200, 1200, "xlarge")
            ]
            
            with Image.open(media_file.file_path) as img:
                for width, height, size_name in sizes:
                    # Resize image
                    resized = img.resize((width, height), Image.Resampling.LANCZOS)
                    
                    # Save resized image
                    size_path = Path("media/processed") / f"{media_file.file_id}_{size_name}.jpg"
                    resized.save(size_path, "JPEG", quality=85)
            
        except Exception as e:
            print(f"Error generating sizes for {media_file.file_id}: {e}")
    
    async def _extract_text_ocr(self, media_file: MediaFile) -> ProcessingResult:
        """Extract text from image using OCR."""
        try:
            start_time = datetime.now()
            
            # Load image
            with Image.open(media_file.file_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Extract text using Tesseract
                text = pytesseract.image_to_string(img)
                
                # Save extracted text
                ocr_path = Path("media/ocr") / f"{media_file.file_id}_ocr.txt"
                async with aiofiles.open(ocr_path, 'w', encoding='utf-8') as f:
                    await f.write(text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                success=True,
                output_path=str(ocr_path),
                metadata={"extracted_text": text},
                processing_time=processing_time
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )
    
    async def _detect_faces(self, media_file: MediaFile) -> ProcessingResult:
        """Detect faces in image."""
        try:
            start_time = datetime.now()
            
            # Load image
            image = face_recognition.load_image_file(media_file.file_path)
            
            # Find face locations
            face_locations = face_recognition.face_locations(image)
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                success=True,
                metadata={
                    "face_count": len(face_locations),
                    "face_locations": face_locations,
                    "face_encodings": [encoding.tolist() for encoding in face_encodings]
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )
    
    async def _detect_objects(self, media_file: MediaFile) -> ProcessingResult:
        """Detect objects in image."""
        try:
            start_time = datetime.now()
            
            if 'object_detection' not in self.ai_models:
                return ProcessingResult(
                    success=False,
                    error="Object detection model not available"
                )
            
            # Load image
            with Image.open(media_file.file_path) as img:
                # Detect objects
                results = self.ai_models['object_detection'](img)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                success=True,
                metadata={
                    "objects": results,
                    "object_count": len(results)
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )
    
    async def _generate_image_caption(self, media_file: MediaFile) -> ProcessingResult:
        """Generate caption for image."""
        try:
            start_time = datetime.now()
            
            if 'image_captioning' not in self.ai_models:
                return ProcessingResult(
                    success=False,
                    error="Image captioning model not available"
                )
            
            # Load image
            with Image.open(media_file.file_path) as img:
                # Generate caption
                result = self.ai_models['image_captioning'](img)
                caption = result[0]['generated_text']
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                success=True,
                metadata={"caption": caption},
                processing_time=processing_time
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )
    
    async def _generate_video_thumbnail(self, media_file: MediaFile) -> ProcessingResult:
        """Generate thumbnail for video."""
        try:
            start_time = datetime.now()
            
            # Load video
            video = VideoFileClip(media_file.file_path)
            
            # Get frame at 10% of video duration
            frame_time = video.duration * 0.1
            frame = video.get_frame(frame_time)
            
            # Convert to PIL Image
            img = Image.fromarray(frame)
            
            # Create thumbnail
            img.thumbnail((300, 300), Image.Resampling.LANCZOS)
            
            # Save thumbnail
            thumbnail_path = Path("media/thumbnails") / f"{media_file.file_id}_video_thumb.jpg"
            img.save(thumbnail_path, "JPEG", quality=85)
            
            # Close video
            video.close()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                success=True,
                output_path=str(thumbnail_path),
                processing_time=processing_time
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )
    
    async def _extract_audio_from_video(self, media_file: MediaFile) -> ProcessingResult:
        """Extract audio from video."""
        try:
            start_time = datetime.now()
            
            # Load video
            video = VideoFileClip(media_file.file_path)
            
            # Extract audio
            audio = video.audio
            
            # Save audio
            audio_path = Path("media/temp") / f"{media_file.file_id}_audio.wav"
            audio.write_audiofile(str(audio_path), verbose=False, logger=None)
            
            # Close video and audio
            audio.close()
            video.close()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                success=True,
                output_path=str(audio_path),
                processing_time=processing_time
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )
    
    async def _transcribe_audio(self, media_file: MediaFile) -> ProcessingResult:
        """Transcribe audio to text."""
        try:
            start_time = datetime.now()
            
            if 'whisper' not in self.ai_models:
                return ProcessingResult(
                    success=False,
                    error="Whisper model not available"
                )
            
            # Transcribe audio
            result = self.ai_models['whisper'].transcribe(media_file.file_path)
            transcription = result["text"]
            
            # Save transcription
            transcription_path = Path("media/transcriptions") / f"{media_file.file_id}_transcription.txt"
            async with aiofiles.open(transcription_path, 'w', encoding='utf-8') as f:
                await f.write(transcription)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                success=True,
                output_path=str(transcription_path),
                metadata={"transcription": transcription},
                processing_time=processing_time
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )
    
    async def _generate_video_summary(self, media_file: MediaFile) -> ProcessingResult:
        """Generate summary for video."""
        try:
            start_time = datetime.now()
            
            # This would implement video summarization
            # For now, return placeholder
            summary = f"Video summary for {media_file.filename}"
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                success=True,
                metadata={"summary": summary},
                processing_time=processing_time
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )
    
    async def _extract_audio_features(self, media_file: MediaFile) -> ProcessingResult:
        """Extract audio features."""
        try:
            start_time = datetime.now()
            
            # Load audio
            y, sr = librosa.load(media_file.file_path)
            
            # Extract features
            features = {
                "tempo": librosa.beat.tempo(y=y, sr=sr)[0],
                "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
                "spectral_rolloff": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
                "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(y)),
                "mfcc": np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1).tolist()
            }
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                success=True,
                metadata={"audio_features": features},
                processing_time=processing_time
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )
    
    async def _generate_audio_summary(self, media_file: MediaFile) -> ProcessingResult:
        """Generate summary for audio."""
        try:
            start_time = datetime.now()
            
            # This would implement audio summarization
            # For now, return placeholder
            summary = f"Audio summary for {media_file.filename}"
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                success=True,
                metadata={"summary": summary},
                processing_time=processing_time
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=str(e)
            )
    
    async def get_media(
        self,
        file_id: str,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Get media file by ID."""
        try:
            # Check cache first
            if file_id in self.media_cache:
                cached_media = self.media_cache[file_id]
                if datetime.now() - cached_media["timestamp"] < timedelta(hours=1):
                    return {
                        "success": True,
                        "data": cached_media["data"]
                    }
            
            # Get from database
            media_query = select(MediaFile).where(MediaFile.file_id == file_id)
            media_result = await self.session.execute(media_query)
            media_file = media_result.scalar_one_or_none()
            
            if not media_file:
                raise ValidationError(f"Media file with ID {file_id} not found")
            
            result_data = {
                "file_id": media_file.file_id,
                "filename": media_file.filename,
                "original_filename": media_file.original_filename,
                "file_path": media_file.file_path,
                "file_size": media_file.file_size,
                "mime_type": media_file.mime_type,
                "media_type": media_file.media_type,
                "user_id": media_file.user_id,
                "collection_id": media_file.collection_id,
                "status": media_file.status,
                "created_at": media_file.created_at.isoformat(),
                "updated_at": media_file.updated_at.isoformat()
            }
            
            if include_metadata:
                result_data["metadata"] = media_file.metadata
            
            # Cache result
            self.media_cache[file_id] = {
                "data": result_data,
                "timestamp": datetime.now()
            }
            
            return {
                "success": True,
                "data": result_data
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get media: {str(e)}")
    
    async def list_media(
        self,
        media_type: Optional[MediaType] = None,
        status: Optional[MediaStatus] = None,
        user_id: Optional[str] = None,
        collection_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "created_at",
        sort_order: str = "desc"
    ) -> Dict[str, Any]:
        """List media files with filtering and pagination."""
        try:
            # Build query
            query = select(MediaFile)
            
            if media_type:
                query = query.where(MediaFile.media_type == media_type.value)
            
            if status:
                query = query.where(MediaFile.status == status.value)
            
            if user_id:
                query = query.where(MediaFile.user_id == user_id)
            
            if collection_id:
                query = query.where(MediaFile.collection_id == collection_id)
            
            # Add sorting
            if sort_order.lower() == "desc":
                query = query.order_by(desc(getattr(MediaFile, sort_by)))
            else:
                query = query.order_by(getattr(MediaFile, sort_by))
            
            # Add pagination
            offset = (page - 1) * page_size
            query = query.offset(offset).limit(page_size)
            
            # Execute query
            result = await self.session.execute(query)
            media_list = result.scalars().all()
            
            # Get total count
            count_query = select(func.count(MediaFile.id))
            if media_type:
                count_query = count_query.where(MediaFile.media_type == media_type.value)
            if status:
                count_query = count_query.where(MediaFile.status == status.value)
            if user_id:
                count_query = count_query.where(MediaFile.user_id == user_id)
            if collection_id:
                count_query = count_query.where(MediaFile.collection_id == collection_id)
            
            count_result = await self.session.execute(count_query)
            total_count = count_result.scalar()
            
            # Format results
            formatted_media = []
            for media in media_list:
                formatted_media.append({
                    "file_id": media.file_id,
                    "filename": media.filename,
                    "original_filename": media.original_filename,
                    "file_size": media.file_size,
                    "mime_type": media.mime_type,
                    "media_type": media.media_type,
                    "user_id": media.user_id,
                    "collection_id": media.collection_id,
                    "status": media.status,
                    "created_at": media.created_at.isoformat(),
                    "updated_at": media.updated_at.isoformat()
                })
            
            return {
                "success": True,
                "data": {
                    "media": formatted_media,
                    "total": total_count,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": (total_count + page_size - 1) // page_size
                }
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to list media: {str(e)}")
    
    async def delete_media(self, file_id: str) -> Dict[str, Any]:
        """Delete media file."""
        try:
            # Get media file
            media_query = select(MediaFile).where(MediaFile.file_id == file_id)
            media_result = await self.session.execute(media_query)
            media_file = media_result.scalar_one_or_none()
            
            if not media_file:
                raise ValidationError(f"Media file with ID {file_id} not found")
            
            # Soft delete by changing status
            media_file.status = MediaStatus.DELETED.value
            media_file.updated_at = datetime.utcnow()
            
            await self.session.commit()
            
            # Remove from cache
            if file_id in self.media_cache:
                del self.media_cache[file_id]
            
            return {
                "success": True,
                "file_id": file_id,
                "message": "Media deleted successfully"
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to delete media: {str(e)}")
    
    async def get_media_stats(self) -> Dict[str, Any]:
        """Get media statistics."""
        try:
            # Get total media count
            total_query = select(func.count(MediaFile.id))
            total_result = await self.session.execute(total_query)
            total_media = total_result.scalar()
            
            # Get media by type
            type_query = select(
                MediaFile.media_type,
                func.count(MediaFile.id).label('count')
            ).group_by(MediaFile.media_type)
            
            type_result = await self.session.execute(type_query)
            media_by_type = {row[0]: row[1] for row in type_result}
            
            # Get media by status
            status_query = select(
                MediaFile.status,
                func.count(MediaFile.id).label('count')
            ).group_by(MediaFile.status)
            
            status_result = await self.session.execute(status_query)
            media_by_status = {row[0]: row[1] for row in status_result}
            
            # Get average file size
            avg_query = select(func.avg(MediaFile.file_size))
            avg_result = await self.session.execute(avg_query)
            avg_file_size = float(avg_result.scalar() or 0)
            
            return {
                "success": True,
                "data": {
                    "total_media": total_media,
                    "media_by_type": media_by_type,
                    "media_by_status": media_by_status,
                    "average_file_size": avg_file_size,
                    "cache_size": len(self.media_cache),
                    "ai_models_loaded": len(self.ai_models)
                }
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get media stats: {str(e)}")
























