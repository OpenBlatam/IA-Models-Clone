"""
Advanced Audio and Video Processing Module
"""

import asyncio
import logging
import os
import tempfile
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import uuid

import librosa
import soundfile as sf
import whisper
import speech_recognition as sr
from pydub import AudioSegment
import moviepy.editor as mp
import ffmpeg
import cv2
import numpy as np
from transformers import pipeline
import torch

from config import settings
from models import ProcessingStatus

logger = logging.getLogger(__name__)


class AudioVideoProcessor:
    """Advanced Audio and Video Processing Engine"""
    
    def __init__(self):
        self.whisper_model = None
        self.speech_recognizer = sr.Recognizer()
        self.audio_classifier = None
        self.video_analyzer = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize the audio/video processor"""
        if self.initialized:
            return
            
        try:
            logger.info("Initializing Audio/Video Processor...")
            
            # Initialize Whisper model for speech recognition
            self.whisper_model = whisper.load_model("base")
            
            # Initialize audio classification pipeline
            self.audio_classifier = pipeline(
                "audio-classification",
                model="facebook/wav2vec2-base",
                return_all_scores=True
            )
            
            # Initialize video analysis components
            self.video_analyzer = VideoAnalyzer()
            await self.video_analyzer.initialize()
            
            self.initialized = True
            logger.info("Audio/Video Processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing audio/video processor: {e}")
            raise
    
    async def process_audio(self, file_path: str, analysis_types: List[str] = None) -> Dict[str, Any]:
        """Process audio file with various analysis types"""
        if not self.initialized:
            await self.initialize()
        
        if analysis_types is None:
            analysis_types = ["transcription", "classification", "features"]
        
        start_time = time.time()
        results = {
            "file_path": file_path,
            "processing_time": 0.0,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Load audio file
            audio_data, sample_rate = librosa.load(file_path, sr=None)
            
            # Perform requested analyses
            if "transcription" in analysis_types:
                results["transcription"] = await self._transcribe_audio(file_path)
            
            if "classification" in analysis_types:
                results["classification"] = await self._classify_audio(audio_data, sample_rate)
            
            if "features" in analysis_types:
                results["features"] = await self._extract_audio_features(audio_data, sample_rate)
            
            if "sentiment" in analysis_types:
                results["sentiment"] = await self._analyze_audio_sentiment(file_path)
            
            if "speaker_diarization" in analysis_types:
                results["speaker_diarization"] = await self._perform_speaker_diarization(file_path)
            
            results["processing_time"] = time.time() - start_time
            results["status"] = "completed"
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            results["processing_time"] = time.time() - start_time
        
        return results
    
    async def process_video(self, file_path: str, analysis_types: List[str] = None) -> Dict[str, Any]:
        """Process video file with various analysis types"""
        if not self.initialized:
            await self.initialize()
        
        if analysis_types is None:
            analysis_types = ["transcription", "scene_analysis", "object_detection"]
        
        start_time = time.time()
        results = {
            "file_path": file_path,
            "processing_time": 0.0,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Extract audio from video for transcription
            if "transcription" in analysis_types:
                results["transcription"] = await self._extract_and_transcribe_audio(file_path)
            
            # Analyze video content
            if "scene_analysis" in analysis_types:
                results["scene_analysis"] = await self._analyze_video_scenes(file_path)
            
            if "object_detection" in analysis_types:
                results["object_detection"] = await self._detect_objects_in_video(file_path)
            
            if "face_detection" in analysis_types:
                results["face_detection"] = await self._detect_faces_in_video(file_path)
            
            if "motion_analysis" in analysis_types:
                results["motion_analysis"] = await self._analyze_motion(file_path)
            
            if "quality_analysis" in analysis_types:
                results["quality_analysis"] = await self._analyze_video_quality(file_path)
            
            results["processing_time"] = time.time() - start_time
            results["status"] = "completed"
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            results["processing_time"] = time.time() - start_time
        
        return results
    
    async def _transcribe_audio(self, file_path: str) -> Dict[str, Any]:
        """Transcribe audio using Whisper"""
        try:
            # Use Whisper for transcription
            result = self.whisper_model.transcribe(file_path)
            
            return {
                "text": result["text"],
                "language": result.get("language", "unknown"),
                "segments": result.get("segments", []),
                "confidence": np.mean([seg.get("avg_logprob", 0) for seg in result.get("segments", [])]) if result.get("segments") else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return {"text": "", "language": "unknown", "segments": [], "confidence": 0.0}
    
    async def _classify_audio(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Classify audio content"""
        try:
            # Convert to the format expected by the model
            audio_tensor = torch.tensor(audio_data).float()
            
            # Perform classification
            results = self.audio_classifier(audio_tensor)
            
            return {
                "predictions": results,
                "top_prediction": max(results, key=lambda x: x["score"]) if results else None
            }
            
        except Exception as e:
            logger.error(f"Error classifying audio: {e}")
            return {"predictions": [], "top_prediction": None}
    
    async def _extract_audio_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract audio features using librosa"""
        try:
            features = {}
            
            # Spectral features
            features["spectral_centroid"] = float(np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)))
            features["spectral_rolloff"] = float(np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)))
            features["spectral_bandwidth"] = float(np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)))
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            features["mfcc_mean"] = [float(np.mean(mfcc)) for mfcc in mfccs]
            features["mfcc_std"] = [float(np.std(mfcc)) for mfcc in mfccs]
            
            # Rhythm features
            tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
            features["tempo"] = float(tempo)
            features["beat_count"] = len(beats)
            
            # Zero crossing rate
            features["zero_crossing_rate"] = float(np.mean(librosa.feature.zero_crossing_rate(audio_data)))
            
            # RMS energy
            features["rms_energy"] = float(np.mean(librosa.feature.rms(y=audio_data)))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return {}
    
    async def _analyze_audio_sentiment(self, file_path: str) -> Dict[str, Any]:
        """Analyze sentiment from audio transcription"""
        try:
            # First transcribe the audio
            transcription = await self._transcribe_audio(file_path)
            
            if not transcription["text"]:
                return {"sentiment": "neutral", "confidence": 0.0}
            
            # Use a sentiment analysis model on the transcribed text
            sentiment_pipeline = pipeline("sentiment-analysis")
            result = sentiment_pipeline(transcription["text"])
            
            return {
                "sentiment": result[0]["label"],
                "confidence": result[0]["score"],
                "transcribed_text": transcription["text"]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing audio sentiment: {e}")
            return {"sentiment": "neutral", "confidence": 0.0}
    
    async def _perform_speaker_diarization(self, file_path: str) -> Dict[str, Any]:
        """Perform speaker diarization (identify different speakers)"""
        try:
            # This is a simplified implementation
            # In production, you'd use specialized libraries like pyannote.audio
            
            # Load audio
            audio_data, sample_rate = librosa.load(file_path, sr=16000)
            
            # Simple speaker change detection based on spectral changes
            frame_length = int(0.025 * sample_rate)  # 25ms frames
            hop_length = int(0.010 * sample_rate)    # 10ms hop
            
            # Extract features for each frame
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, 
                                       n_mfcc=13, hop_length=hop_length)
            
            # Simple clustering to identify speaker changes
            from sklearn.cluster import KMeans
            
            # Transpose to get frames as rows
            mfccs_t = mfccs.T
            
            # Use K-means to cluster frames (assuming 2 speakers)
            kmeans = KMeans(n_clusters=2, random_state=42)
            speaker_labels = kmeans.fit_predict(mfccs_t)
            
            # Convert frame labels to time segments
            segments = []
            current_speaker = speaker_labels[0]
            start_time = 0
            
            for i, speaker in enumerate(speaker_labels):
                if speaker != current_speaker:
                    segments.append({
                        "speaker": int(current_speaker),
                        "start_time": start_time * hop_length / sample_rate,
                        "end_time": i * hop_length / sample_rate
                    })
                    current_speaker = speaker
                    start_time = i
            
            # Add final segment
            segments.append({
                "speaker": int(current_speaker),
                "start_time": start_time * hop_length / sample_rate,
                "end_time": len(speaker_labels) * hop_length / sample_rate
            })
            
            return {
                "speakers": list(set(speaker_labels)),
                "segments": segments,
                "total_speakers": len(set(speaker_labels))
            }
            
        except Exception as e:
            logger.error(f"Error performing speaker diarization: {e}")
            return {"speakers": [], "segments": [], "total_speakers": 0}
    
    async def _extract_and_transcribe_audio(self, video_path: str) -> Dict[str, Any]:
        """Extract audio from video and transcribe it"""
        try:
            # Extract audio from video
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            # Use ffmpeg to extract audio
            (
                ffmpeg
                .input(video_path)
                .output(temp_audio_path, acodec='pcm_s16le', ac=1, ar='16k')
                .overwrite_output()
                .run(quiet=True)
            )
            
            # Transcribe the extracted audio
            transcription = await self._transcribe_audio(temp_audio_path)
            
            # Clean up temp file
            os.unlink(temp_audio_path)
            
            return transcription
            
        except Exception as e:
            logger.error(f"Error extracting and transcribing audio: {e}")
            return {"text": "", "language": "unknown", "segments": [], "confidence": 0.0}
    
    async def _analyze_video_scenes(self, video_path: str) -> Dict[str, Any]:
        """Analyze video scenes and detect scene changes"""
        try:
            cap = cv2.VideoCapture(video_path)
            scenes = []
            prev_frame = None
            scene_start = 0
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale for comparison
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Calculate difference between frames
                    diff = cv2.absdiff(prev_frame, gray)
                    diff_score = np.mean(diff)
                    
                    # If difference is significant, it's a scene change
                    if diff_score > 30:  # Threshold for scene change
                        scenes.append({
                            "scene_id": len(scenes),
                            "start_frame": scene_start,
                            "end_frame": frame_count - 1,
                            "start_time": scene_start / cap.get(cv2.CAP_PROP_FPS),
                            "end_time": (frame_count - 1) / cap.get(cv2.CAP_PROP_FPS)
                        })
                        scene_start = frame_count
                
                prev_frame = gray
                frame_count += 1
            
            # Add final scene
            if frame_count > scene_start:
                scenes.append({
                    "scene_id": len(scenes),
                    "start_frame": scene_start,
                    "end_frame": frame_count - 1,
                    "start_time": scene_start / cap.get(cv2.CAP_PROP_FPS),
                    "end_time": (frame_count - 1) / cap.get(cv2.CAP_PROP_FPS)
                })
            
            cap.release()
            
            return {
                "total_scenes": len(scenes),
                "scenes": scenes,
                "total_frames": frame_count,
                "fps": cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing video scenes: {e}")
            return {"total_scenes": 0, "scenes": [], "total_frames": 0, "fps": 0}
    
    async def _detect_objects_in_video(self, video_path: str) -> Dict[str, Any]:
        """Detect objects in video frames"""
        try:
            # This would use a proper object detection model like YOLO
            # For now, we'll use OpenCV's built-in face detection as an example
            
            cap = cv2.VideoCapture(video_path)
            detections = []
            frame_count = 0
            
            # Load face cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect faces in frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    detections.append({
                        "frame": frame_count,
                        "timestamp": frame_count / cap.get(cv2.CAP_PROP_FPS),
                        "objects": [{"type": "face", "bbox": face.tolist()} for face in faces]
                    })
                
                frame_count += 1
            
            cap.release()
            
            return {
                "total_detections": len(detections),
                "detections": detections,
                "object_types": ["face"]  # Would be expanded with more object types
            }
            
        except Exception as e:
            logger.error(f"Error detecting objects in video: {e}")
            return {"total_detections": 0, "detections": [], "object_types": []}
    
    async def _detect_faces_in_video(self, video_path: str) -> Dict[str, Any]:
        """Detect and analyze faces in video"""
        try:
            cap = cv2.VideoCapture(video_path)
            face_detections = []
            frame_count = 0
            
            # Load face cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    face_detections.append({
                        "frame": frame_count,
                        "timestamp": frame_count / cap.get(cv2.CAP_PROP_FPS),
                        "faces": [{"bbox": face.tolist(), "confidence": 0.9} for face in faces]
                    })
                
                frame_count += 1
            
            cap.release()
            
            return {
                "total_faces_detected": sum(len(det["faces"]) for det in face_detections),
                "frames_with_faces": len(face_detections),
                "detections": face_detections
            }
            
        except Exception as e:
            logger.error(f"Error detecting faces in video: {e}")
            return {"total_faces_detected": 0, "frames_with_faces": 0, "detections": []}
    
    async def _analyze_motion(self, video_path: str) -> Dict[str, Any]:
        """Analyze motion in video"""
        try:
            cap = cv2.VideoCapture(video_path)
            motion_data = []
            prev_frame = None
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Calculate optical flow
                    flow = cv2.calcOpticalFlowPyrLK(prev_frame, gray, None, None)
                    
                    # Calculate motion magnitude
                    motion_magnitude = np.mean(np.sqrt(flow[0]**2 + flow[1]**2)) if flow[0] is not None else 0
                    
                    motion_data.append({
                        "frame": frame_count,
                        "timestamp": frame_count / cap.get(cv2.CAP_PROP_FPS),
                        "motion_magnitude": float(motion_magnitude)
                    })
                
                prev_frame = gray
                frame_count += 1
            
            cap.release()
            
            # Calculate motion statistics
            motion_values = [m["motion_magnitude"] for m in motion_data]
            
            return {
                "total_frames": frame_count,
                "motion_data": motion_data,
                "average_motion": float(np.mean(motion_values)) if motion_values else 0.0,
                "max_motion": float(np.max(motion_values)) if motion_values else 0.0,
                "motion_variance": float(np.var(motion_values)) if motion_values else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing motion: {e}")
            return {"total_frames": 0, "motion_data": [], "average_motion": 0.0, "max_motion": 0.0, "motion_variance": 0.0}
    
    async def _analyze_video_quality(self, video_path: str) -> Dict[str, Any]:
        """Analyze video quality metrics"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Sample frames for quality analysis
            sample_frames = []
            step = max(1, frame_count // 10)  # Sample 10 frames
            
            for i in range(0, frame_count, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    sample_frames.append(frame)
            
            cap.release()
            
            # Analyze quality metrics
            quality_metrics = {}
            
            if sample_frames:
                # Calculate sharpness (Laplacian variance)
                sharpness_values = []
                for frame in sample_frames:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                    sharpness_values.append(sharpness)
                
                quality_metrics["sharpness"] = {
                    "average": float(np.mean(sharpness_values)),
                    "std": float(np.std(sharpness_values)),
                    "min": float(np.min(sharpness_values)),
                    "max": float(np.max(sharpness_values))
                }
                
                # Calculate brightness
                brightness_values = [np.mean(frame) for frame in sample_frames]
                quality_metrics["brightness"] = {
                    "average": float(np.mean(brightness_values)),
                    "std": float(np.std(brightness_values))
                }
                
                # Calculate contrast
                contrast_values = []
                for frame in sample_frames:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    contrast = gray.std()
                    contrast_values.append(contrast)
                
                quality_metrics["contrast"] = {
                    "average": float(np.mean(contrast_values)),
                    "std": float(np.std(contrast_values))
                }
            
            return {
                "resolution": f"{width}x{height}",
                "fps": fps,
                "duration": duration,
                "frame_count": frame_count,
                "quality_metrics": quality_metrics
            }
            
        except Exception as e:
            logger.error(f"Error analyzing video quality: {e}")
            return {"resolution": "unknown", "fps": 0, "duration": 0, "frame_count": 0, "quality_metrics": {}}


class VideoAnalyzer:
    """Advanced video analysis component"""
    
    def __init__(self):
        self.initialized = False
    
    async def initialize(self):
        """Initialize video analyzer"""
        try:
            # Initialize any video analysis models here
            self.initialized = True
            logger.info("Video Analyzer initialized")
        except Exception as e:
            logger.error(f"Error initializing video analyzer: {e}")
            raise


# Global audio/video processor instance
audio_video_processor = AudioVideoProcessor()


async def initialize_audio_video_processor():
    """Initialize the audio/video processor"""
    await audio_video_processor.initialize()














