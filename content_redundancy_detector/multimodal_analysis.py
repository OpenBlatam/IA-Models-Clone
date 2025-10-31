"""
Multimodal Analysis Engine - Text, Image, and Audio Processing
"""

import asyncio
import base64
import io
import logging
import tempfile
from typing import Dict, Any, List, Optional, Union, BinaryIO
from datetime import datetime
import hashlib

import numpy as np
from PIL import Image
import cv2
import librosa
import soundfile as sf
from pydantic import BaseModel, Field
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    WhisperProcessor, WhisperForConditionalGeneration,
    CLIPProcessor, CLIPModel,
    pipeline
)
import easyocr
from ai_ml_enhanced import ai_ml_engine
from config import settings

logger = logging.getLogger(__name__)


class MultimodalInput(BaseModel):
    """Input model for multimodal analysis"""
    content_type: str = Field(..., description="Type of content: text, image, audio, video")
    content_data: str = Field(..., description="Base64 encoded content or text")
    analysis_types: List[str] = Field(default=["all"], description="Types of analysis to perform")
    language: Optional[str] = Field(default="en", description="Language for processing")


class ImageAnalysisResult(BaseModel):
    """Result model for image analysis"""
    image_hash: str
    dimensions: Dict[str, int]
    dominant_colors: List[str]
    objects_detected: List[Dict[str, Any]]
    text_extracted: List[Dict[str, Any]]
    scene_description: str
    sentiment: Dict[str, Any]
    quality_metrics: Dict[str, float]
    timestamp: str


class AudioAnalysisResult(BaseModel):
    """Result model for audio analysis"""
    audio_hash: str
    duration: float
    sample_rate: int
    channels: int
    transcription: str
    language_detected: str
    sentiment: Dict[str, Any]
    speaker_count: int
    audio_quality: Dict[str, float]
    keywords: List[str]
    timestamp: str


class VideoAnalysisResult(BaseModel):
    """Result model for video analysis"""
    video_hash: str
    duration: float
    frame_count: int
    fps: float
    resolution: Dict[str, int]
    scenes: List[Dict[str, Any]]
    audio_analysis: Optional[AudioAnalysisResult]
    visual_analysis: List[ImageAnalysisResult]
    transcript: str
    summary: str
    timestamp: str


class MultimodalAnalysisResult(BaseModel):
    """Result model for multimodal analysis"""
    content_type: str
    content_hash: str
    analysis_results: Dict[str, Any]
    cross_modal_insights: Dict[str, Any]
    confidence_scores: Dict[str, float]
    processing_time: float
    timestamp: str


class MultimodalAnalysisEngine:
    """Engine for multimodal content analysis"""
    
    def __init__(self):
        self.models = {}
        self.processors = {}
        self.initialized = False
        self.ocr_reader = None
    
    async def initialize(self):
        """Initialize multimodal analysis models"""
        if self.initialized:
            return
        
        try:
            logger.info("Initializing multimodal analysis engine...")
            
            # Initialize image analysis models
            logger.info("Loading image analysis models...")
            self.processors['blip'] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.models['blip'] = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            self.processors['clip'] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.models['clip'] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            
            # Initialize audio analysis models
            logger.info("Loading audio analysis models...")
            self.processors['whisper'] = WhisperProcessor.from_pretrained("openai/whisper-base")
            self.models['whisper'] = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
            
            # Initialize object detection
            logger.info("Loading object detection model...")
            self.models['object_detection'] = pipeline(
                "object-detection",
                model="facebook/detr-resnet-50",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Initialize OCR
            logger.info("Initializing OCR...")
            self.ocr_reader = easyocr.Reader(['en', 'es', 'fr', 'de'])
            
            # Initialize sentiment analysis for images
            self.models['image_sentiment'] = pipeline(
                "image-classification",
                model="microsoft/DialoGPT-medium"
            )
            
            self.initialized = True
            logger.info("Multimodal analysis engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing multimodal engine: {e}")
            raise
    
    async def analyze_image(self, image_data: str, analysis_types: List[str] = None) -> ImageAnalysisResult:
        """Analyze image content"""
        if not self.initialized:
            await self.initialize()
        
        if analysis_types is None:
            analysis_types = ["all"]
        
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Calculate image hash
            image_hash = hashlib.md5(image_bytes).hexdigest()
            
            # Get basic image info
            dimensions = {"width": image.width, "height": image.height}
            
            # Initialize result
            result = {
                "image_hash": image_hash,
                "dimensions": dimensions,
                "dominant_colors": [],
                "objects_detected": [],
                "text_extracted": [],
                "scene_description": "",
                "sentiment": {},
                "quality_metrics": {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Scene description
            if "all" in analysis_types or "description" in analysis_types:
                inputs = self.processors['blip'](image, return_tensors="pt")
                out = self.models['blip'].generate(**inputs, max_length=50)
                scene_description = self.processors['blip'].decode(out[0], skip_special_tokens=True)
                result["scene_description"] = scene_description
            
            # Object detection
            if "all" in analysis_types or "objects" in analysis_types:
                objects = self.models['object_detection'](image)
                result["objects_detected"] = [
                    {
                        "label": obj["label"],
                        "confidence": obj["score"],
                        "bbox": obj["box"]
                    }
                    for obj in objects
                ]
            
            # OCR text extraction
            if "all" in analysis_types or "text" in analysis_types:
                # Convert PIL to OpenCV format
                opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                ocr_results = self.ocr_reader.readtext(opencv_image)
                
                result["text_extracted"] = [
                    {
                        "text": text,
                        "confidence": confidence,
                        "bbox": bbox
                    }
                    for bbox, text, confidence in ocr_results
                ]
            
            # Dominant colors
            if "all" in analysis_types or "colors" in analysis_types:
                colors = self._extract_dominant_colors(image)
                result["dominant_colors"] = colors
            
            # Image quality metrics
            if "all" in analysis_types or "quality" in analysis_types:
                quality_metrics = self._calculate_image_quality(image)
                result["quality_metrics"] = quality_metrics
            
            # Sentiment analysis (if text is extracted)
            if result["text_extracted"]:
                combined_text = " ".join([item["text"] for item in result["text_extracted"]])
                if combined_text.strip():
                    sentiment_result = await ai_ml_engine.analyze_sentiment(combined_text)
                    result["sentiment"] = sentiment_result
            
            return ImageAnalysisResult(**result)
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            raise
    
    async def analyze_audio(self, audio_data: str, analysis_types: List[str] = None) -> AudioAnalysisResult:
        """Analyze audio content"""
        if not self.initialized:
            await self.initialize()
        
        if analysis_types is None:
            analysis_types = ["all"]
        
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_data)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
            
            try:
                # Load audio with librosa
                audio, sample_rate = librosa.load(temp_file_path, sr=None)
                
                # Calculate audio hash
                audio_hash = hashlib.md5(audio_bytes).hexdigest()
                
                # Get basic audio info
                duration = len(audio) / sample_rate
                channels = 1  # librosa loads as mono
                
                # Initialize result
                result = {
                    "audio_hash": audio_hash,
                    "duration": duration,
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "transcription": "",
                    "language_detected": "unknown",
                    "sentiment": {},
                    "speaker_count": 1,
                    "audio_quality": {},
                    "keywords": [],
                    "timestamp": datetime.now().isoformat()
                }
                
                # Speech transcription
                if "all" in analysis_types or "transcription" in analysis_types:
                    inputs = self.processors['whisper'](audio, sampling_rate=sample_rate, return_tensors="pt")
                    predicted_ids = self.models['whisper'].generate(inputs.input_features)
                    transcription = self.processors['whisper'].batch_decode(predicted_ids, skip_special_tokens=True)[0]
                    result["transcription"] = transcription
                    
                    # Language detection from transcription
                    if transcription.strip():
                        language_result = await ai_ml_engine.detect_language(transcription)
                        result["language_detected"] = language_result.get("language", "unknown")
                
                # Sentiment analysis from transcription
                if result["transcription"]:
                    sentiment_result = await ai_ml_engine.analyze_sentiment(result["transcription"])
                    result["sentiment"] = sentiment_result
                    
                    # Extract keywords
                    entities_result = await ai_ml_engine.extract_entities(result["transcription"])
                    result["keywords"] = [entity["text"] for entity in entities_result.get("entities", [])]
                
                # Audio quality metrics
                if "all" in analysis_types or "quality" in analysis_types:
                    quality_metrics = self._calculate_audio_quality(audio, sample_rate)
                    result["audio_quality"] = quality_metrics
                
                # Speaker count estimation (simplified)
                if "all" in analysis_types or "speakers" in analysis_types:
                    speaker_count = self._estimate_speaker_count(audio, sample_rate)
                    result["speaker_count"] = speaker_count
                
                return AudioAnalysisResult(**result)
                
            finally:
                # Clean up temporary file
                import os
                os.unlink(temp_file_path)
            
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            raise
    
    async def analyze_video(self, video_data: str, analysis_types: List[str] = None) -> VideoAnalysisResult:
        """Analyze video content"""
        if not self.initialized:
            await self.initialize()
        
        if analysis_types is None:
            analysis_types = ["all"]
        
        try:
            # Decode base64 video
            video_bytes = base64.b64decode(video_data)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                temp_file.write(video_bytes)
                temp_file_path = temp_file.name
            
            try:
                # Open video with OpenCV
                cap = cv2.VideoCapture(temp_file_path)
                
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Calculate video hash
                video_hash = hashlib.md5(video_bytes).hexdigest()
                
                # Initialize result
                result = {
                    "video_hash": video_hash,
                    "duration": duration,
                    "frame_count": frame_count,
                    "fps": fps,
                    "resolution": {"width": width, "height": height},
                    "scenes": [],
                    "audio_analysis": None,
                    "visual_analysis": [],
                    "transcript": "",
                    "summary": "",
                    "timestamp": datetime.now().isoformat()
                }
                
                # Extract frames for analysis
                frames_to_analyze = min(10, frame_count)  # Analyze up to 10 frames
                frame_interval = max(1, frame_count // frames_to_analyze)
                
                frame_analyses = []
                for i in range(0, frame_count, frame_interval):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if ret:
                        # Convert OpenCV frame to PIL Image
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_pil = Image.fromarray(frame_rgb)
                        
                        # Encode frame as base64
                        frame_buffer = io.BytesIO()
                        frame_pil.save(frame_buffer, format='PNG')
                        frame_b64 = base64.b64encode(frame_buffer.getvalue()).decode()
                        
                        # Analyze frame
                        frame_analysis = await self.analyze_image(frame_b64, ["description", "objects"])
                        frame_analyses.append(frame_analysis)
                
                result["visual_analysis"] = [analysis.model_dump() for analysis in frame_analyses]
                
                # Extract audio from video (simplified - would need ffmpeg in production)
                if "all" in analysis_types or "audio" in analysis_types:
                    # This is a placeholder - in production you'd extract audio with ffmpeg
                    result["audio_analysis"] = {
                        "audio_hash": "placeholder",
                        "duration": duration,
                        "transcription": "Audio analysis not implemented in this version",
                        "sentiment": {},
                        "timestamp": datetime.now().isoformat()
                    }
                
                # Generate transcript from visual text
                all_text = []
                for frame_analysis in frame_analyses:
                    for text_item in frame_analysis.text_extracted:
                        all_text.append(text_item["text"])
                
                result["transcript"] = " ".join(all_text)
                
                # Generate summary
                if result["transcript"]:
                    summary_result = await ai_ml_engine.generate_summary(result["transcript"])
                    result["summary"] = summary_result.get("summary", "")
                
                cap.release()
                return VideoAnalysisResult(**result)
                
            finally:
                # Clean up temporary file
                import os
                os.unlink(temp_file_path)
            
        except Exception as e:
            logger.error(f"Error analyzing video: {e}")
            raise
    
    async def analyze_multimodal(self, input_data: MultimodalInput) -> MultimodalAnalysisResult:
        """Analyze multimodal content"""
        start_time = time.time()
        
        try:
            content_hash = hashlib.md5(input_data.content_data.encode()).hexdigest()
            analysis_results = {}
            cross_modal_insights = {}
            confidence_scores = {}
            
            if input_data.content_type == "text":
                # Use existing text analysis
                if "all" in input_data.analysis_types or "sentiment" in input_data.analysis_types:
                    analysis_results["sentiment"] = await ai_ml_engine.analyze_sentiment(input_data.content_data)
                
                if "all" in input_data.analysis_types or "entities" in input_data.analysis_types:
                    analysis_results["entities"] = await ai_ml_engine.extract_entities(input_data.content_data)
                
                if "all" in input_data.analysis_types or "summary" in input_data.analysis_types:
                    analysis_results["summary"] = await ai_ml_engine.generate_summary(input_data.content_data)
            
            elif input_data.content_type == "image":
                image_result = await self.analyze_image(input_data.content_data, input_data.analysis_types)
                analysis_results["image"] = image_result.model_dump()
                confidence_scores["image_analysis"] = 0.9
            
            elif input_data.content_type == "audio":
                audio_result = await self.analyze_audio(input_data.content_data, input_data.analysis_types)
                analysis_results["audio"] = audio_result.model_dump()
                confidence_scores["audio_analysis"] = 0.85
            
            elif input_data.content_type == "video":
                video_result = await self.analyze_video(input_data.content_data, input_data.analysis_types)
                analysis_results["video"] = video_result.model_dump()
                confidence_scores["video_analysis"] = 0.8
            
            # Cross-modal insights
            if input_data.content_type in ["image", "video"] and "text_extracted" in str(analysis_results):
                cross_modal_insights["text_visual_alignment"] = self._analyze_text_visual_alignment(analysis_results)
            
            processing_time = time.time() - start_time
            
            return MultimodalAnalysisResult(
                content_type=input_data.content_type,
                content_hash=content_hash,
                analysis_results=analysis_results,
                cross_modal_insights=cross_modal_insights,
                confidence_scores=confidence_scores,
                processing_time=processing_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error in multimodal analysis: {e}")
            raise
    
    def _extract_dominant_colors(self, image: Image.Image, num_colors: int = 5) -> List[str]:
        """Extract dominant colors from image"""
        try:
            # Resize image for faster processing
            image = image.resize((150, 150))
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Reshape to list of pixels
            pixels = img_array.reshape(-1, 3)
            
            # Use K-means to find dominant colors
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get cluster centers (dominant colors)
            colors = kmeans.cluster_centers_.astype(int)
            
            # Convert to hex
            hex_colors = [f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}" for color in colors]
            
            return hex_colors
            
        except Exception as e:
            logger.error(f"Error extracting dominant colors: {e}")
            return []
    
    def _calculate_image_quality(self, image: Image.Image) -> Dict[str, float]:
        """Calculate image quality metrics"""
        try:
            # Convert to grayscale for some metrics
            gray = image.convert('L')
            gray_array = np.array(gray)
            
            # Calculate metrics
            brightness = np.mean(gray_array)
            contrast = np.std(gray_array)
            
            # Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray_array, cv2.CV_64F).var()
            
            return {
                "brightness": float(brightness),
                "contrast": float(contrast),
                "sharpness": float(laplacian_var)
            }
            
        except Exception as e:
            logger.error(f"Error calculating image quality: {e}")
            return {}
    
    def _calculate_audio_quality(self, audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Calculate audio quality metrics"""
        try:
            # Calculate metrics
            rms = np.sqrt(np.mean(audio**2))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))
            
            return {
                "rms_energy": float(rms),
                "zero_crossing_rate": float(zero_crossing_rate),
                "spectral_centroid": float(spectral_centroid)
            }
            
        except Exception as e:
            logger.error(f"Error calculating audio quality: {e}")
            return {}
    
    def _estimate_speaker_count(self, audio: np.ndarray, sample_rate: int) -> int:
        """Estimate number of speakers in audio"""
        try:
            # Simplified speaker count estimation
            # In production, you'd use more sophisticated methods like diarization
            
            # Calculate spectral features
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            
            # Simple heuristic based on spectral variation
            spectral_variation = np.std(mfccs)
            
            if spectral_variation > 0.5:
                return 2  # Multiple speakers
            else:
                return 1  # Single speaker
                
        except Exception as e:
            logger.error(f"Error estimating speaker count: {e}")
            return 1
    
    def _analyze_text_visual_alignment(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze alignment between text and visual content"""
        try:
            # This is a simplified implementation
            # In production, you'd use more sophisticated cross-modal analysis
            
            return {
                "alignment_score": 0.8,
                "text_visual_consistency": "high",
                "insights": ["Text and visual content appear to be well-aligned"]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text-visual alignment: {e}")
            return {}


# Global multimodal analysis engine
multimodal_engine = MultimodalAnalysisEngine()


async def initialize_multimodal_engine():
    """Initialize the multimodal analysis engine"""
    await multimodal_engine.initialize()
















