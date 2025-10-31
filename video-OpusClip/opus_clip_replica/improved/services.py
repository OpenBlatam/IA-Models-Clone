"""
Core Services for OpusClip Improved
==================================

Advanced video processing and AI-powered content creation services.
"""

import asyncio
import logging
import time
import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from uuid import UUID, uuid4
import json
import hashlib
from pathlib import Path

import cv2
import numpy as np
import librosa
import whisper
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip
import torch
from transformers import pipeline
import openai
import anthropic
from google.cloud import videointelligence
import redis
import aiofiles
import aiohttp
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete

from .schemas import (
    VideoAnalysisRequest, VideoAnalysisResponse, ClipGenerationRequest,
    ClipGenerationResponse, ClipExportRequest, ClipExportResponse,
    BatchProcessingRequest, BatchProcessingResponse, ProjectRequest,
    ProjectResponse, AnalyticsRequest, AnalyticsResponse,
    ProcessingStatus, ClipType, PlatformType, QualityLevel, AIProvider
)
from .exceptions import (
    VideoProcessingError, VideoAnalysisError, ClipGenerationError,
    AIProviderError, ValidationError, FileError, DatabaseError,
    create_video_processing_error, create_ai_provider_error
)
from .database import get_database_session
from .ai_engine import AIEngine
from .video_processor import VideoProcessor
from .analytics import AnalyticsEngine

logger = logging.getLogger(__name__)


class OpusClipService:
    """Main service for OpusClip operations"""
    
    def __init__(self):
        self.ai_engine = AIEngine()
        self.video_processor = VideoProcessor()
        self.analytics_engine = AnalyticsEngine()
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.temp_dir = Path("./temp")
        self.output_dir = Path("./output")
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure required directories exist"""
        self.temp_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
    
    async def analyze_video(self, request: VideoAnalysisRequest) -> VideoAnalysisResponse:
        """Analyze video and extract insights"""
        start_time = time.time()
        analysis_id = uuid4()
        
        try:
            logger.info(f"Starting video analysis {analysis_id}")
            
            # Download/prepare video
            video_path = await self._prepare_video(request)
            
            # Extract video metadata
            metadata = await self._extract_video_metadata(video_path)
            
            # Perform analysis based on request options
            analysis_results = {}
            
            if request.extract_transcript:
                analysis_results['transcript'] = await self._extract_transcript(video_path, request.language)
            
            if request.analyze_sentiment:
                analysis_results['sentiment_scores'] = await self._analyze_sentiment(
                    analysis_results.get('transcript', '')
                )
            
            if request.detect_scenes:
                analysis_results['scene_changes'] = await self._detect_scene_changes(video_path)
            
            if request.detect_faces:
                analysis_results['face_detections'] = await self._detect_faces(video_path)
            
            if request.detect_objects:
                analysis_results['object_detections'] = await self._detect_objects(video_path)
            
            # Extract key moments
            analysis_results['key_moments'] = await self._extract_key_moments(
                video_path, analysis_results.get('transcript', '')
            )
            
            # AI-powered insights
            ai_insights = await self._generate_ai_insights(
                video_path, analysis_results, request.ai_provider
            )
            
            # Calculate viral potential
            viral_potential = await self._calculate_viral_potential(
                analysis_results, ai_insights
            )
            
            processing_time = time.time() - start_time
            
            response = VideoAnalysisResponse(
                analysis_id=analysis_id,
                status=ProcessingStatus.COMPLETED,
                duration=metadata['duration'],
                fps=metadata['fps'],
                resolution=metadata['resolution'],
                format=metadata['format'],
                file_size=metadata['file_size'],
                transcript=analysis_results.get('transcript'),
                sentiment_scores=analysis_results.get('sentiment_scores'),
                key_moments=analysis_results.get('key_moments'),
                scene_changes=analysis_results.get('scene_changes'),
                face_detections=analysis_results.get('face_detections'),
                object_detections=analysis_results.get('object_detections'),
                content_summary=ai_insights.get('summary'),
                topics=ai_insights.get('topics'),
                emotions=ai_insights.get('emotions'),
                viral_potential=viral_potential,
                processing_time=processing_time
            )
            
            # Cache results
            await self._cache_analysis_results(analysis_id, response)
            
            logger.info(f"Video analysis completed {analysis_id} in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Video analysis failed {analysis_id}: {e}")
            raise VideoAnalysisError(
                message=f"Video analysis failed: {str(e)}",
                analysis_id=str(analysis_id),
                details={"error": str(e)}
            )
    
    async def generate_clips(self, request: ClipGenerationRequest) -> ClipGenerationResponse:
        """Generate clips from analyzed video"""
        start_time = time.time()
        generation_id = uuid4()
        
        try:
            logger.info(f"Starting clip generation {generation_id}")
            
            # Get analysis results
            analysis_results = await self._get_analysis_results(request.analysis_id)
            if not analysis_results:
                raise ClipGenerationError(
                    message="Analysis results not found",
                    generation_id=str(generation_id),
                    details={"analysis_id": str(request.analysis_id)}
                )
            
            # Get video path
            video_path = await self._get_video_path(request.analysis_id)
            
            # Generate clips based on type and preferences
            clips = await self._generate_clips_from_analysis(
                video_path, analysis_results, request
            )
            
            processing_time = time.time() - start_time
            
            response = ClipGenerationResponse(
                generation_id=generation_id,
                analysis_id=request.analysis_id,
                status=ProcessingStatus.COMPLETED,
                clips=clips,
                processing_time=processing_time
            )
            
            # Cache results
            await self._cache_generation_results(generation_id, response)
            
            logger.info(f"Clip generation completed {generation_id} in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Clip generation failed {generation_id}: {e}")
            raise ClipGenerationError(
                message=f"Clip generation failed: {str(e)}",
                generation_id=str(generation_id),
                details={"error": str(e)}
            )
    
    async def export_clips(self, request: ClipExportRequest) -> ClipExportResponse:
        """Export clips in specified format and quality"""
        start_time = time.time()
        export_id = uuid4()
        
        try:
            logger.info(f"Starting clip export {export_id}")
            
            # Get generation results
            generation_results = await self._get_generation_results(request.generation_id)
            if not generation_results:
                raise ClipGenerationError(
                    message="Generation results not found",
                    generation_id=str(request.generation_id),
                    details={"generation_id": str(request.generation_id)}
                )
            
            # Filter clips to export
            clips_to_export = [
                clip for clip in generation_results['clips']
                if clip['id'] in request.clip_ids
            ]
            
            # Export clips
            exported_files = []
            download_urls = []
            
            for clip in clips_to_export:
                exported_file = await self._export_single_clip(
                    clip, request, export_id
                )
                exported_files.append(exported_file)
                download_urls.append(exported_file['download_url'])
            
            processing_time = time.time() - start_time
            
            response = ClipExportResponse(
                export_id=export_id,
                generation_id=request.generation_id,
                status=ProcessingStatus.COMPLETED,
                exported_files=exported_files,
                download_urls=download_urls,
                processing_time=processing_time
            )
            
            logger.info(f"Clip export completed {export_id} in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Clip export failed {export_id}: {e}")
            raise ClipGenerationError(
                message=f"Clip export failed: {str(e)}",
                generation_id=str(export_id),
                details={"error": str(e)}
            )
    
    async def process_batch(self, request: BatchProcessingRequest) -> BatchProcessingResponse:
        """Process multiple videos in batch"""
        start_time = time.time()
        batch_id = uuid4()
        
        try:
            logger.info(f"Starting batch processing {batch_id}")
            
            # Process videos
            if request.parallel_processing:
                # Process in parallel with concurrency limit
                semaphore = asyncio.Semaphore(request.max_concurrent)
                tasks = [
                    self._process_single_video(video, request.clip_types, semaphore)
                    for video in request.videos
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Process sequentially
                results = []
                for video in request.videos:
                    result = await self._process_single_video(video, request.clip_types)
                    results.append(result)
            
            # Separate successful and failed results
            analysis_results = []
            generation_results = []
            failed_count = 0
            
            for result in results:
                if isinstance(result, Exception):
                    failed_count += 1
                    logger.error(f"Batch processing failed: {result}")
                else:
                    analysis_results.append(result['analysis'])
                    generation_results.append(result['generation'])
            
            processing_time = time.time() - start_time
            
            response = BatchProcessingResponse(
                batch_id=batch_id,
                status=ProcessingStatus.COMPLETED,
                total_videos=len(request.videos),
                completed_videos=len(request.videos) - failed_count,
                failed_videos=failed_count,
                analysis_results=analysis_results,
                generation_results=generation_results,
                total_processing_time=processing_time
            )
            
            # Send notification if requested
            if request.notify_on_completion and request.webhook_url:
                await self._send_batch_notification(request.webhook_url, response)
            
            logger.info(f"Batch processing completed {batch_id} in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Batch processing failed {batch_id}: {e}")
            raise ClipGenerationError(
                message=f"Batch processing failed: {str(e)}",
                generation_id=str(batch_id),
                details={"error": str(e)}
            )
    
    async def create_project(self, request: ProjectRequest) -> ProjectResponse:
        """Create a new project"""
        try:
            project_id = uuid4()
            
            # Create project in database
            async with get_database_session() as session:
                # Implementation would create project record
                pass
            
            response = ProjectResponse(
                project_id=project_id,
                name=request.name,
                description=request.description,
                default_clip_duration=request.default_clip_duration,
                default_quality=request.default_quality,
                default_platforms=request.default_platforms,
                collaborators=request.collaborators or [],
                is_public=request.is_public
            )
            
            logger.info(f"Project created {project_id}")
            return response
            
        except Exception as e:
            logger.error(f"Project creation failed: {e}")
            raise DatabaseError(
                message=f"Project creation failed: {str(e)}",
                operation="create_project",
                details={"error": str(e)}
            )
    
    async def get_analytics(self, request: AnalyticsRequest) -> AnalyticsResponse:
        """Get analytics for project or system"""
        try:
            analytics_id = uuid4()
            
            # Get analytics data
            metrics = await self.analytics_engine.get_metrics(
                project_id=request.project_id,
                date_range=request.date_range,
                metrics=request.metrics
            )
            
            # Get trends
            trends = await self.analytics_engine.get_trends(
                project_id=request.project_id,
                date_range=request.date_range,
                group_by=request.group_by
            )
            
            # Generate insights
            insights = await self.analytics_engine.generate_insights(metrics, trends)
            
            # Generate recommendations
            recommendations = await self.analytics_engine.generate_recommendations(
                metrics, trends, insights
            )
            
            response = AnalyticsResponse(
                analytics_id=analytics_id,
                metrics=metrics,
                trends=trends,
                insights=insights,
                recommendations=recommendations,
                date_range=request.date_range or {
                    "start": datetime.utcnow() - timedelta(days=30),
                    "end": datetime.utcnow()
                }
            )
            
            logger.info(f"Analytics generated {analytics_id}")
            return response
            
        except Exception as e:
            logger.error(f"Analytics generation failed: {e}")
            raise DatabaseError(
                message=f"Analytics generation failed: {str(e)}",
                operation="get_analytics",
                details={"error": str(e)}
            )
    
    # Helper methods
    async def _prepare_video(self, request: VideoAnalysisRequest) -> str:
        """Prepare video for processing"""
        if request.video_path:
            return request.video_path
        elif request.video_url:
            return await self._download_video(request.video_url)
        elif request.video_file:
            return await self._save_base64_video(request.video_file)
        else:
            raise ValidationError("No video source provided")
    
    async def _download_video(self, url: str) -> str:
        """Download video from URL"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        video_path = self.temp_dir / f"video_{uuid4()}.mp4"
                        async with aiofiles.open(video_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                await f.write(chunk)
                        return str(video_path)
                    else:
                        raise FileError(f"Failed to download video: HTTP {response.status}")
        except Exception as e:
            raise FileError(f"Video download failed: {str(e)}")
    
    async def _save_base64_video(self, base64_data: str) -> str:
        """Save base64 encoded video to file"""
        try:
            video_data = base64.b64decode(base64_data)
            video_path = self.temp_dir / f"video_{uuid4()}.mp4"
            async with aiofiles.open(video_path, 'wb') as f:
                await f.write(video_data)
            return str(video_path)
        except Exception as e:
            raise FileError(f"Base64 video save failed: {str(e)}")
    
    async def _extract_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata"""
        try:
            with VideoFileClip(video_path) as clip:
                return {
                    'duration': clip.duration,
                    'fps': clip.fps,
                    'resolution': f"{clip.w}x{clip.h}",
                    'format': 'mp4',  # Would detect actual format
                    'file_size': os.path.getsize(video_path)
                }
        except Exception as e:
            raise VideoProcessingError(f"Metadata extraction failed: {str(e)}")
    
    async def _extract_transcript(self, video_path: str, language: str) -> str:
        """Extract transcript using Whisper"""
        try:
            # Extract audio
            audio_path = self.temp_dir / f"audio_{uuid4()}.wav"
            with VideoFileClip(video_path) as clip:
                clip.audio.write_audiofile(str(audio_path))
            
            # Transcribe
            model = whisper.load_model("base")
            result = model.transcribe(str(audio_path), language=language)
            
            # Clean up
            os.remove(audio_path)
            
            return result["text"]
        except Exception as e:
            raise VideoProcessingError(f"Transcript extraction failed: {str(e)}")
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        try:
            sentiment_pipeline = pipeline("sentiment-analysis")
            result = sentiment_pipeline(text)
            return {
                'positive': result[0]['score'] if result[0]['label'] == 'POSITIVE' else 1 - result[0]['score'],
                'negative': result[0]['score'] if result[0]['label'] == 'NEGATIVE' else 1 - result[0]['score'],
                'neutral': 1 - abs(result[0]['score'] - 0.5) * 2
            }
        except Exception as e:
            raise VideoProcessingError(f"Sentiment analysis failed: {str(e)}")
    
    async def _detect_scene_changes(self, video_path: str) -> List[float]:
        """Detect scene changes in video"""
        try:
            cap = cv2.VideoCapture(video_path)
            scene_changes = []
            prev_frame = None
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if prev_frame is not None:
                    # Calculate frame difference
                    diff = cv2.absdiff(prev_frame, frame)
                    diff_score = np.mean(diff)
                    
                    # Threshold for scene change
                    if diff_score > 30:  # Adjust threshold as needed
                        scene_changes.append(frame_count / cap.get(cv2.CAP_PROP_FPS))
                
                prev_frame = frame
                frame_count += 1
            
            cap.release()
            return scene_changes
        except Exception as e:
            raise VideoProcessingError(f"Scene detection failed: {str(e)}")
    
    async def _detect_faces(self, video_path: str) -> List[Dict[str, Any]]:
        """Detect faces in video"""
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            cap = cv2.VideoCapture(video_path)
            face_detections = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    face_detections.append({
                        'timestamp': frame_count / cap.get(cv2.CAP_PROP_FPS),
                        'face_count': len(faces),
                        'faces': faces.tolist()
                    })
                
                frame_count += 1
            
            cap.release()
            return face_detections
        except Exception as e:
            raise VideoProcessingError(f"Face detection failed: {str(e)}")
    
    async def _detect_objects(self, video_path: str) -> List[Dict[str, Any]]:
        """Detect objects in video"""
        try:
            # Use YOLO or similar for object detection
            # This is a simplified implementation
            cap = cv2.VideoCapture(video_path)
            object_detections = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Simplified object detection
                # In production, would use trained model
                objects = ['person', 'car', 'bicycle']  # Placeholder
                
                if objects:
                    object_detections.append({
                        'timestamp': frame_count / cap.get(cv2.CAP_PROP_FPS),
                        'objects': objects
                    })
                
                frame_count += 1
            
            cap.release()
            return object_detections
        except Exception as e:
            raise VideoProcessingError(f"Object detection failed: {str(e)}")
    
    async def _extract_key_moments(self, video_path: str, transcript: str) -> List[Dict[str, Any]]:
        """Extract key moments from video"""
        try:
            # Use AI to identify key moments
            key_moments = await self.ai_engine.identify_key_moments(
                video_path=video_path,
                transcript=transcript
            )
            return key_moments
        except Exception as e:
            raise VideoProcessingError(f"Key moment extraction failed: {str(e)}")
    
    async def _generate_ai_insights(self, video_path: str, analysis_results: Dict, provider: AIProvider) -> Dict[str, Any]:
        """Generate AI-powered insights"""
        try:
            insights = await self.ai_engine.generate_insights(
                video_path=video_path,
                analysis_results=analysis_results,
                provider=provider
            )
            return insights
        except Exception as e:
            raise AIProviderError(f"AI insights generation failed: {str(e)}")
    
    async def _calculate_viral_potential(self, analysis_results: Dict, ai_insights: Dict) -> float:
        """Calculate viral potential score"""
        try:
            # Calculate based on various factors
            score = 0.0
            
            # Sentiment factor
            if analysis_results.get('sentiment_scores'):
                positive_score = analysis_results['sentiment_scores'].get('positive', 0)
                score += positive_score * 0.3
            
            # Key moments factor
            if analysis_results.get('key_moments'):
                score += min(len(analysis_results['key_moments']) * 0.1, 0.3)
            
            # AI insights factor
            if ai_insights.get('topics'):
                trending_topics = ['viral', 'trending', 'popular', 'funny', 'amazing']
                topic_score = sum(1 for topic in ai_insights['topics'] if any(t in topic.lower() for t in trending_topics))
                score += min(topic_score * 0.1, 0.2)
            
            # Face detection factor
            if analysis_results.get('face_detections'):
                score += min(len(analysis_results['face_detections']) * 0.05, 0.2)
            
            return min(score, 1.0)
        except Exception as e:
            logger.error(f"Viral potential calculation failed: {e}")
            return 0.5  # Default score
    
    async def _cache_analysis_results(self, analysis_id: UUID, results: VideoAnalysisResponse):
        """Cache analysis results"""
        try:
            cache_key = f"analysis:{analysis_id}"
            cache_data = results.model_dump_json()
            self.redis_client.setex(cache_key, 3600, cache_data)  # 1 hour TTL
        except Exception as e:
            logger.error(f"Failed to cache analysis results: {e}")
    
    async def _get_analysis_results(self, analysis_id: UUID) -> Optional[Dict]:
        """Get cached analysis results"""
        try:
            cache_key = f"analysis:{analysis_id}"
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception as e:
            logger.error(f"Failed to get analysis results: {e}")
            return None
    
    async def _get_video_path(self, analysis_id: UUID) -> str:
        """Get video path for analysis ID"""
        # Implementation would retrieve video path from database or cache
        # This is a placeholder
        return f"./temp/video_{analysis_id}.mp4"
    
    async def _generate_clips_from_analysis(self, video_path: str, analysis_results: Dict, request: ClipGenerationRequest) -> List[Dict[str, Any]]:
        """Generate clips from analysis results"""
        try:
            clips = []
            
            # Get key moments
            key_moments = analysis_results.get('key_moments', [])
            
            # Generate clips based on type
            if request.clip_type == ClipType.HIGHLIGHT:
                clips = await self._generate_highlight_clips(video_path, key_moments, request)
            elif request.clip_type == ClipType.VIRAL:
                clips = await self._generate_viral_clips(video_path, analysis_results, request)
            elif request.clip_type == ClipType.EDUCATIONAL:
                clips = await self._generate_educational_clips(video_path, analysis_results, request)
            else:
                clips = await self._generate_default_clips(video_path, key_moments, request)
            
            return clips[:request.max_clips]
        except Exception as e:
            raise ClipGenerationError(f"Clip generation failed: {str(e)}")
    
    async def _generate_highlight_clips(self, video_path: str, key_moments: List[Dict], request: ClipGenerationRequest) -> List[Dict[str, Any]]:
        """Generate highlight clips"""
        clips = []
        
        for i, moment in enumerate(key_moments[:request.max_clips]):
            clip_id = uuid4()
            start_time = moment.get('start_time', 0)
            end_time = min(start_time + request.target_duration, moment.get('end_time', start_time + request.target_duration))
            
            # Create clip
            clip_path = await self._create_video_clip(
                video_path, start_time, end_time, clip_id, request
            )
            
            clips.append({
                'id': str(clip_id),
                'type': 'highlight',
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'file_path': clip_path,
                'download_url': f"/download/{clip_id}",
                'thumbnail_url': f"/thumbnail/{clip_id}",
                'metadata': {
                    'moment_type': moment.get('type', 'unknown'),
                    'confidence': moment.get('confidence', 0.5),
                    'description': moment.get('description', '')
                }
            })
        
        return clips
    
    async def _generate_viral_clips(self, video_path: str, analysis_results: Dict, request: ClipGenerationRequest) -> List[Dict[str, Any]]:
        """Generate viral clips"""
        # Implementation for viral clip generation
        # Would use AI to identify most engaging segments
        return await self._generate_highlight_clips(video_path, analysis_results.get('key_moments', []), request)
    
    async def _generate_educational_clips(self, video_path: str, analysis_results: Dict, request: ClipGenerationRequest) -> List[Dict[str, Any]]:
        """Generate educational clips"""
        # Implementation for educational clip generation
        # Would focus on informative segments
        return await self._generate_highlight_clips(video_path, analysis_results.get('key_moments', []), request)
    
    async def _generate_default_clips(self, video_path: str, key_moments: List[Dict], request: ClipGenerationRequest) -> List[Dict[str, Any]]:
        """Generate default clips"""
        return await self._generate_highlight_clips(video_path, key_moments, request)
    
    async def _create_video_clip(self, video_path: str, start_time: float, end_time: float, clip_id: UUID, request: ClipGenerationRequest) -> str:
        """Create a video clip"""
        try:
            output_path = self.output_dir / f"clip_{clip_id}.mp4"
            
            with VideoFileClip(video_path) as video:
                clip = video.subclip(start_time, end_time)
                
                # Add captions if requested
                if request.add_captions:
                    clip = await self._add_captions_to_clip(clip, start_time, end_time)
                
                # Add watermark if requested
                if request.add_watermark:
                    clip = await self._add_watermark_to_clip(clip)
                
                # Write clip
                clip.write_videofile(
                    str(output_path),
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True
                )
                
                clip.close()
            
            return str(output_path)
        except Exception as e:
            raise VideoProcessingError(f"Clip creation failed: {str(e)}")
    
    async def _add_captions_to_clip(self, clip, start_time: float, end_time: float):
        """Add captions to clip"""
        # Implementation for adding captions
        # Would extract transcript for the time range and create subtitle overlay
        return clip
    
    async def _add_watermark_to_clip(self, clip):
        """Add watermark to clip"""
        # Implementation for adding watermark
        # Would create a watermark overlay
        return clip
    
    async def _cache_generation_results(self, generation_id: UUID, results: ClipGenerationResponse):
        """Cache generation results"""
        try:
            cache_key = f"generation:{generation_id}"
            cache_data = results.model_dump_json()
            self.redis_client.setex(cache_key, 3600, cache_data)  # 1 hour TTL
        except Exception as e:
            logger.error(f"Failed to cache generation results: {e}")
    
    async def _get_generation_results(self, generation_id: UUID) -> Optional[Dict]:
        """Get cached generation results"""
        try:
            cache_key = f"generation:{generation_id}"
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception as e:
            logger.error(f"Failed to get generation results: {e}")
            return None
    
    async def _export_single_clip(self, clip: Dict, request: ClipExportRequest, export_id: UUID) -> Dict[str, Any]:
        """Export a single clip"""
        try:
            # Implementation for exporting clip in specified format
            # Would handle format conversion, quality adjustment, platform optimization
            
            export_path = self.output_dir / f"export_{export_id}_{clip['id']}.{request.format.value}"
            
            # Placeholder implementation
            # In production, would use FFmpeg for format conversion
            
            return {
                'clip_id': clip['id'],
                'export_path': str(export_path),
                'download_url': f"/download/export/{export_id}/{clip['id']}",
                'format': request.format.value,
                'quality': request.quality.value,
                'file_size': 0,  # Would calculate actual size
                'duration': clip.get('duration', 0)
            }
        except Exception as e:
            raise VideoProcessingError(f"Clip export failed: {str(e)}")
    
    async def _process_single_video(self, video_request: VideoAnalysisRequest, clip_types: List[ClipType], semaphore: Optional[asyncio.Semaphore] = None) -> Dict[str, Any]:
        """Process a single video in batch"""
        async def _process():
            # Analyze video
            analysis = await self.analyze_video(video_request)
            
            # Generate clips for each type
            generations = []
            for clip_type in clip_types:
                generation_request = ClipGenerationRequest(
                    analysis_id=analysis.analysis_id,
                    clip_type=clip_type,
                    target_duration=30,
                    max_clips=3
                )
                generation = await self.generate_clips(generation_request)
                generations.append(generation)
            
            return {
                'analysis': analysis,
                'generation': generations
            }
        
        if semaphore:
            async with semaphore:
                return await _process()
        else:
            return await _process()
    
    async def _send_batch_notification(self, webhook_url: str, response: BatchProcessingResponse):
        """Send batch completion notification"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    'batch_id': str(response.batch_id),
                    'status': response.status.value,
                    'total_videos': response.total_videos,
                    'completed_videos': response.completed_videos,
                    'failed_videos': response.failed_videos,
                    'processing_time': response.total_processing_time
                }
                
                async with session.post(webhook_url, json=payload) as response:
                    if response.status != 200:
                        logger.error(f"Webhook notification failed: {response.status}")
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")


# Global service instance
opus_clip_service = OpusClipService()






























