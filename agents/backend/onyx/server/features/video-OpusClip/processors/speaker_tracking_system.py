"""
Speaker Tracking System

Ensures the speaker's face is always centered and properly framed in the video.
This is essential for professional-looking short-form content.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any, Tuple, Union
import asyncio
import numpy as np
import cv2
import torch
import torch.nn as nn
from dataclasses import dataclass
from enum import Enum
import structlog
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor
import math

from ..models.video_models import VideoClipRequest, VideoClipResponse
from ..error_handling import ErrorHandler, ProcessingError, ValidationError

logger = structlog.get_logger("speaker_tracking_system")
error_handler = ErrorHandler()

class TrackingStatus(Enum):
    """Status of speaker tracking."""
    TRACKING = "tracking"
    LOST = "lost"
    FOUND = "found"
    SWITCHING = "switching"

@dataclass
class Face:
    """Represents a detected face."""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    landmarks: List[Tuple[int, int]]
    face_id: Optional[int] = None
    is_speaking: bool = False
    gaze_direction: Optional[Tuple[float, float]] = None

@dataclass
class TrackingFrame:
    """A frame with speaker tracking information."""
    frame: np.ndarray
    timestamp: float
    primary_speaker: Optional[Face]
    all_faces: List[Face]
    tracking_status: TrackingStatus
    crop_region: Optional[Tuple[int, int, int, int]] = None
    zoom_factor: float = 1.0
    pan_offset: Tuple[int, int] = (0, 0)

@dataclass
class TrackingConfig:
    """Configuration for speaker tracking."""
    min_face_size: int = 50
    max_face_size: int = 500
    confidence_threshold: float = 0.7
    tracking_threshold: float = 0.6
    max_tracking_distance: int = 100
    smoothing_factor: float = 0.7
    zoom_smoothing: float = 0.8
    pan_smoothing: float = 0.8

class FaceDetector:
    """Detects faces in video frames."""
    
    def __init__(self, config: TrackingConfig = None):
        self.config = config or TrackingConfig()
        self.face_cascade = None
        self.landmark_detector = None
        self._load_models()
    
    def _load_models(self):
        """Load face detection models."""
        try:
            # Load OpenCV Haar cascade for face detection
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Load landmark detection model (placeholder)
            self.landmark_detector = self._load_landmark_model()
            
            logger.info("Face detection models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load face detection models: {e}")
            raise ProcessingError(f"Face detection model loading failed: {e}")
    
    def _load_landmark_model(self):
        """Load facial landmark detection model."""
        # Placeholder - would load actual landmark detection model
        # For now, return a simple landmark detector
        return SimpleLandmarkDetector()
    
    async def detect_faces(self, frame: np.ndarray) -> List[Face]:
        """Detect faces in a frame."""
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.config.min_face_size, self.config.min_face_size),
                maxSize=(self.config.max_face_size, self.config.max_face_size)
            )
            
            detected_faces = []
            
            for i, (x, y, w, h) in enumerate(faces):
                # Calculate confidence based on face size and position
                confidence = self._calculate_face_confidence(x, y, w, h, frame.shape)
                
                if confidence >= self.config.confidence_threshold:
                    # Detect landmarks
                    landmarks = await self._detect_landmarks(frame, x, y, w, h)
                    
                    # Detect if face is speaking
                    is_speaking = await self._detect_speaking(frame, x, y, w, h)
                    
                    # Detect gaze direction
                    gaze_direction = await self._detect_gaze_direction(landmarks)
                    
                    face = Face(
                        bbox=(x, y, w, h),
                        confidence=confidence,
                        landmarks=landmarks,
                        face_id=i,
                        is_speaking=is_speaking,
                        gaze_direction=gaze_direction
                    )
                    
                    detected_faces.append(face)
            
            return detected_faces
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            raise ProcessingError(f"Face detection failed: {e}")
    
    def _calculate_face_confidence(self, x: int, y: int, w: int, h: int, frame_shape: Tuple) -> float:
        """Calculate confidence score for a detected face."""
        try:
            # Size confidence (prefer medium-sized faces)
            size_ratio = (w * h) / (frame_shape[0] * frame_shape[1])
            size_confidence = 1.0 - abs(size_ratio - 0.1) * 5  # Optimal around 10% of frame
            
            # Position confidence (prefer center of frame)
            center_x = x + w // 2
            center_y = y + h // 2
            frame_center_x = frame_shape[1] // 2
            frame_center_y = frame_shape[0] // 2
            
            distance_from_center = math.sqrt(
                (center_x - frame_center_x) ** 2 + (center_y - frame_center_y) ** 2
            )
            max_distance = math.sqrt(frame_center_x ** 2 + frame_center_y ** 2)
            position_confidence = 1.0 - (distance_from_center / max_distance)
            
            # Aspect ratio confidence (prefer square-ish faces)
            aspect_ratio = w / h
            aspect_confidence = 1.0 - abs(aspect_ratio - 1.0) * 0.5
            
            # Combine confidence factors
            confidence = (
                size_confidence * 0.4 +
                position_confidence * 0.3 +
                aspect_confidence * 0.3
            )
            
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Face confidence calculation failed: {e}")
            return 0.5
    
    async def _detect_landmarks(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> List[Tuple[int, int]]:
        """Detect facial landmarks."""
        try:
            # Extract face region
            face_region = frame[y:y+h, x:x+w]
            
            # Use landmark detector (placeholder implementation)
            landmarks = self.landmark_detector.detect(face_region)
            
            # Convert to full frame coordinates
            full_frame_landmarks = [(x + lx, y + ly) for lx, ly in landmarks]
            
            return full_frame_landmarks
            
        except Exception as e:
            logger.error(f"Landmark detection failed: {e}")
            return []
    
    async def _detect_speaking(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        """Detect if a face is speaking."""
        try:
            # Extract mouth region
            mouth_y = y + int(h * 0.6)
            mouth_h = int(h * 0.3)
            mouth_region = frame[mouth_y:mouth_y+mouth_h, x:x+w]
            
            # Convert to grayscale
            gray_mouth = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate mouth movement (simple approach)
            # This is a placeholder - would use more sophisticated mouth detection
            mouth_variance = np.var(gray_mouth)
            speaking_threshold = 1000  # Adjust based on testing
            
            return mouth_variance > speaking_threshold
            
        except Exception as e:
            logger.error(f"Speaking detection failed: {e}")
            return False
    
    async def _detect_gaze_direction(self, landmarks: List[Tuple[int, int]]) -> Optional[Tuple[float, float]]:
        """Detect gaze direction from facial landmarks."""
        try:
            if len(landmarks) < 2:
                return None
            
            # Simple gaze detection based on eye landmarks
            # This is a placeholder - would use more sophisticated gaze detection
            left_eye = landmarks[0] if len(landmarks) > 0 else (0, 0)
            right_eye = landmarks[1] if len(landmarks) > 1 else (0, 0)
            
            # Calculate gaze direction (simplified)
            eye_center_x = (left_eye[0] + right_eye[0]) / 2
            eye_center_y = (left_eye[1] + right_eye[1]) / 2
            
            # Normalize gaze direction
            gaze_x = (eye_center_x - 320) / 320  # Assuming 640px width
            gaze_y = (eye_center_y - 240) / 240  # Assuming 480px height
            
            return (gaze_x, gaze_y)
            
        except Exception as e:
            logger.error(f"Gaze detection failed: {e}")
            return None

class ObjectTracker:
    """Tracks faces across frames."""
    
    def __init__(self, config: TrackingConfig = None):
        self.config = config or TrackingConfig()
        self.tracked_faces: Dict[int, Face] = {}
        self.next_face_id = 0
        self.tracking_history: List[Dict] = []
    
    async def track_faces(self, 
                         current_faces: List[Face], 
                         previous_faces: List[Face] = None) -> List[Face]:
        """Track faces across frames."""
        try:
            if not current_faces:
                return []
            
            # Initialize tracking if no previous faces
            if not previous_faces:
                for face in current_faces:
                    face.face_id = self.next_face_id
                    self.tracked_faces[self.next_face_id] = face
                    self.next_face_id += 1
                return current_faces
            
            # Match current faces with previous faces
            matched_faces = await self._match_faces(current_faces, previous_faces)
            
            # Update tracking history
            self._update_tracking_history(matched_faces)
            
            return matched_faces
            
        except Exception as e:
            logger.error(f"Face tracking failed: {e}")
            raise ProcessingError(f"Face tracking failed: {e}")
    
    async def _match_faces(self, 
                          current_faces: List[Face], 
                          previous_faces: List[Face]) -> List[Face]:
        """Match current faces with previous faces."""
        try:
            matched_faces = []
            used_previous_ids = set()
            
            for current_face in current_faces:
                best_match = None
                best_distance = float('inf')
                
                for prev_face in previous_faces:
                    if prev_face.face_id in used_previous_ids:
                        continue
                    
                    # Calculate distance between faces
                    distance = self._calculate_face_distance(current_face, prev_face)
                    
                    if (distance < best_distance and 
                        distance < self.config.max_tracking_distance):
                        best_match = prev_face
                        best_distance = distance
                
                if best_match:
                    # Update face with previous ID
                    current_face.face_id = best_match.face_id
                    used_previous_ids.add(best_match.face_id)
                else:
                    # New face
                    current_face.face_id = self.next_face_id
                    self.next_face_id += 1
                
                matched_faces.append(current_face)
            
            return matched_faces
            
        except Exception as e:
            logger.error(f"Face matching failed: {e}")
            raise ProcessingError(f"Face matching failed: {e}")
    
    def _calculate_face_distance(self, face1: Face, face2: Face) -> float:
        """Calculate distance between two faces."""
        try:
            # Calculate center points
            center1 = (
                face1.bbox[0] + face1.bbox[2] // 2,
                face1.bbox[1] + face1.bbox[3] // 2
            )
            center2 = (
                face2.bbox[0] + face2.bbox[2] // 2,
                face2.bbox[1] + face2.bbox[3] // 2
            )
            
            # Calculate Euclidean distance
            distance = math.sqrt(
                (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
            )
            
            return distance
            
        except Exception as e:
            logger.error(f"Face distance calculation failed: {e}")
            return float('inf')
    
    def _update_tracking_history(self, faces: List[Face]):
        """Update tracking history for analysis."""
        try:
            history_entry = {
                "timestamp": time.time(),
                "faces": [
                    {
                        "face_id": face.face_id,
                        "bbox": face.bbox,
                        "confidence": face.confidence,
                        "is_speaking": face.is_speaking
                    }
                    for face in faces
                ]
            }
            
            self.tracking_history.append(history_entry)
            
            # Keep only recent history (last 100 frames)
            if len(self.tracking_history) > 100:
                self.tracking_history = self.tracking_history[-100:]
                
        except Exception as e:
            logger.error(f"Tracking history update failed: {e}")
    
    async def identify_primary_speaker(self, faces: List[Face]) -> Optional[Face]:
        """Identify the primary speaker from tracked faces."""
        try:
            if not faces:
                return None
            
            # Filter faces that are speaking
            speaking_faces = [face for face in faces if face.is_speaking]
            
            if not speaking_faces:
                # If no one is speaking, return the largest face
                return max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
            
            # Among speaking faces, choose the one with highest confidence
            primary_speaker = max(speaking_faces, key=lambda f: f.confidence)
            
            return primary_speaker
            
        except Exception as e:
            logger.error(f"Primary speaker identification failed: {e}")
            return None

class AutoFramer:
    """Automatically frames the video to keep the speaker centered."""
    
    def __init__(self, config: TrackingConfig = None):
        self.config = config or TrackingConfig()
        self.target_aspect_ratio = 9/16  # Vertical video aspect ratio
        self.min_zoom = 1.0
        self.max_zoom = 3.0
        self.smooth_zoom = 1.0
        self.smooth_pan = (0, 0)
    
    async def frame_speaker(self, 
                           frame: np.ndarray, 
                           speaker: Optional[Face],
                           frame_width: int = 1080,
                           frame_height: int = 1920) -> TrackingFrame:
        """Frame the video to keep the speaker centered."""
        try:
            if not speaker:
                # No speaker detected, return original frame
                return TrackingFrame(
                    frame=frame,
                    timestamp=time.time(),
                    primary_speaker=None,
                    all_faces=[],
                    tracking_status=TrackingStatus.LOST
                )
            
            # Calculate crop region
            crop_region = await self._calculate_crop_region(
                frame, speaker, frame_width, frame_height
            )
            
            # Apply smoothing
            crop_region = await self._apply_smoothing(crop_region)
            
            # Crop and resize frame
            cropped_frame = await self._crop_and_resize(frame, crop_region, frame_width, frame_height)
            
            # Calculate zoom and pan for metadata
            zoom_factor = await self._calculate_zoom_factor(crop_region, frame.shape)
            pan_offset = await self._calculate_pan_offset(crop_region, frame.shape)
            
            return TrackingFrame(
                frame=cropped_frame,
                timestamp=time.time(),
                primary_speaker=speaker,
                all_faces=[speaker],
                tracking_status=TrackingStatus.TRACKING,
                crop_region=crop_region,
                zoom_factor=zoom_factor,
                pan_offset=pan_offset
            )
            
        except Exception as e:
            logger.error(f"Speaker framing failed: {e}")
            raise ProcessingError(f"Speaker framing failed: {e}")
    
    async def _calculate_crop_region(self, 
                                   frame: np.ndarray, 
                                   speaker: Face,
                                   target_width: int,
                                   target_height: int) -> Tuple[int, int, int, int]:
        """Calculate the crop region to center the speaker."""
        try:
            frame_h, frame_w = frame.shape[:2]
            
            # Calculate speaker center
            speaker_center_x = speaker.bbox[0] + speaker.bbox[2] // 2
            speaker_center_y = speaker.bbox[1] + speaker.bbox[3] // 2
            
            # Calculate target crop size based on speaker size
            speaker_size = max(speaker.bbox[2], speaker.bbox[3])
            crop_size = int(speaker_size * 2.5)  # 2.5x speaker size for context
            
            # Ensure crop size is within frame bounds
            crop_size = min(crop_size, min(frame_w, frame_h))
            
            # Calculate crop region center
            crop_center_x = speaker_center_x
            crop_center_y = speaker_center_y
            
            # Calculate crop region bounds
            crop_x = max(0, crop_center_x - crop_size // 2)
            crop_y = max(0, crop_center_y - crop_size // 2)
            crop_w = min(crop_size, frame_w - crop_x)
            crop_h = min(crop_size, frame_h - crop_y)
            
            # Adjust for aspect ratio
            target_aspect = target_width / target_height
            current_aspect = crop_w / crop_h
            
            if current_aspect > target_aspect:
                # Too wide, reduce width
                new_w = int(crop_h * target_aspect)
                crop_x += (crop_w - new_w) // 2
                crop_w = new_w
            else:
                # Too tall, reduce height
                new_h = int(crop_w / target_aspect)
                crop_y += (crop_h - new_h) // 2
                crop_h = new_h
            
            return (crop_x, crop_y, crop_w, crop_h)
            
        except Exception as e:
            logger.error(f"Crop region calculation failed: {e}")
            return (0, 0, frame.shape[1], frame.shape[0])
    
    async def _apply_smoothing(self, crop_region: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Apply smoothing to crop region to reduce jitter."""
        try:
            x, y, w, h = crop_region
            
            # Smooth position changes
            if hasattr(self, 'last_crop_region'):
                last_x, last_y, last_w, last_h = self.last_crop_region
                
                # Apply smoothing factor
                x = int(x * self.config.pan_smoothing + last_x * (1 - self.config.pan_smoothing))
                y = int(y * self.config.pan_smoothing + last_y * (1 - self.config.pan_smoothing))
                w = int(w * self.config.zoom_smoothing + last_w * (1 - self.config.zoom_smoothing))
                h = int(h * self.config.zoom_smoothing + last_h * (1 - self.config.zoom_smoothing))
            
            # Store for next frame
            self.last_crop_region = (x, y, w, h)
            
            return (x, y, w, h)
            
        except Exception as e:
            logger.error(f"Crop smoothing failed: {e}")
            return crop_region
    
    async def _crop_and_resize(self, 
                              frame: np.ndarray, 
                              crop_region: Tuple[int, int, int, int],
                              target_width: int,
                              target_height: int) -> np.ndarray:
        """Crop and resize frame to target dimensions."""
        try:
            x, y, w, h = crop_region
            
            # Crop the frame
            cropped = frame[y:y+h, x:x+w]
            
            # Resize to target dimensions
            resized = cv2.resize(cropped, (target_width, target_height))
            
            return resized
            
        except Exception as e:
            logger.error(f"Crop and resize failed: {e}")
            return frame
    
    async def _calculate_zoom_factor(self, 
                                   crop_region: Tuple[int, int, int, int],
                                   frame_shape: Tuple) -> float:
        """Calculate zoom factor for metadata."""
        try:
            x, y, w, h = crop_region
            frame_h, frame_w = frame_shape[:2]
            
            # Calculate zoom based on crop size relative to frame size
            crop_area = w * h
            frame_area = frame_w * frame_h
            zoom_factor = math.sqrt(frame_area / crop_area)
            
            return min(max(zoom_factor, self.min_zoom), self.max_zoom)
            
        except Exception as e:
            logger.error(f"Zoom factor calculation failed: {e}")
            return 1.0
    
    async def _calculate_pan_offset(self, 
                                  crop_region: Tuple[int, int, int, int],
                                  frame_shape: Tuple) -> Tuple[int, int]:
        """Calculate pan offset for metadata."""
        try:
            x, y, w, h = crop_region
            frame_h, frame_w = frame_shape[:2]
            
            # Calculate offset from frame center
            crop_center_x = x + w // 2
            crop_center_y = y + h // 2
            frame_center_x = frame_w // 2
            frame_center_y = frame_h // 2
            
            pan_x = crop_center_x - frame_center_x
            pan_y = crop_center_y - frame_center_y
            
            return (pan_x, pan_y)
            
        except Exception as e:
            logger.error(f"Pan offset calculation failed: {e}")
            return (0, 0)

class SpeakerTrackingSystem:
    """Main speaker tracking system that orchestrates the entire process."""
    
    def __init__(self, config: TrackingConfig = None):
        self.config = config or TrackingConfig()
        self.face_detector = FaceDetector(config)
        self.object_tracker = ObjectTracker(config)
        self.auto_framer = AutoFramer(config)
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    async def track_speaker(self, 
                           video_frames: List[np.ndarray],
                           target_width: int = 1080,
                           target_height: int = 1920) -> List[TrackingFrame]:
        """Track speaker across video frames."""
        try:
            logger.info(f"Starting speaker tracking for {len(video_frames)} frames")
            start_time = time.time()
            
            tracked_frames = []
            previous_faces = []
            
            for i, frame in enumerate(video_frames):
                # Detect faces in current frame
                current_faces = await self.face_detector.detect_faces(frame)
                
                # Track faces across frames
                tracked_faces = await self.object_tracker.track_faces(current_faces, previous_faces)
                
                # Identify primary speaker
                primary_speaker = await self.object_tracker.identify_primary_speaker(tracked_faces)
                
                # Frame the speaker
                tracking_frame = await self.auto_framer.frame_speaker(
                    frame, primary_speaker, target_width, target_height
                )
                
                tracked_frames.append(tracking_frame)
                previous_faces = tracked_faces
                
                # Log progress
                if i % 30 == 0:  # Every 30 frames
                    logger.info(f"Processed {i+1}/{len(video_frames)} frames")
            
            processing_time = time.time() - start_time
            logger.info(f"Speaker tracking completed in {processing_time:.2f}s")
            
            return tracked_frames
            
        except Exception as e:
            logger.error(f"Speaker tracking failed: {e}")
            raise ProcessingError(f"Speaker tracking failed: {e}")
    
    async def process_video(self, 
                           video_path: str,
                           output_path: str,
                           target_width: int = 1080,
                           target_height: int = 1920) -> Dict[str, Any]:
        """Process a video file with speaker tracking."""
        try:
            logger.info(f"Processing video: {video_path}")
            
            # Extract frames
            frames = await self._extract_frames(video_path)
            
            # Track speaker
            tracked_frames = await self.track_speaker(frames, target_width, target_height)
            
            # Create output video
            await self._create_output_video(tracked_frames, output_path)
            
            # Generate tracking report
            report = await self._generate_tracking_report(tracked_frames)
            
            return {
                "input_video": video_path,
                "output_video": output_path,
                "frames_processed": len(tracked_frames),
                "tracking_report": report,
                "target_resolution": f"{target_width}x{target_height}",
                "processing_time": time.time() - time.time()
            }
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise ProcessingError(f"Video processing failed: {e}")
    
    async def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from video."""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            return frames
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            raise ProcessingError(f"Frame extraction failed: {e}")
    
    async def _create_output_video(self, 
                                 tracked_frames: List[TrackingFrame], 
                                 output_path: str):
        """Create output video from tracked frames."""
        try:
            if not tracked_frames:
                return
            
            # Get video properties from first frame
            first_frame = tracked_frames[0].frame
            height, width = first_frame.shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
            
            # Write frames
            for tracking_frame in tracked_frames:
                out.write(tracking_frame.frame)
            
            out.release()
            
        except Exception as e:
            logger.error(f"Output video creation failed: {e}")
            raise ProcessingError(f"Output video creation failed: {e}")
    
    async def _generate_tracking_report(self, tracked_frames: List[TrackingFrame]) -> Dict[str, Any]:
        """Generate tracking report."""
        try:
            total_frames = len(tracked_frames)
            tracking_frames = len([f for f in tracked_frames if f.tracking_status == TrackingStatus.TRACKING])
            lost_frames = len([f for f in tracked_frames if f.tracking_status == TrackingStatus.LOST])
            
            # Calculate average zoom and pan
            zoom_factors = [f.zoom_factor for f in tracked_frames if f.zoom_factor is not None]
            pan_offsets = [f.pan_offset for f in tracked_frames if f.pan_offset is not None]
            
            avg_zoom = np.mean(zoom_factors) if zoom_factors else 1.0
            avg_pan_x = np.mean([p[0] for p in pan_offsets]) if pan_offsets else 0
            avg_pan_y = np.mean([p[1] for p in pan_offsets]) if pan_offsets else 0
            
            return {
                "total_frames": total_frames,
                "tracking_success_rate": tracking_frames / total_frames if total_frames > 0 else 0,
                "lost_frames": lost_frames,
                "average_zoom": avg_zoom,
                "average_pan": (avg_pan_x, avg_pan_y),
                "tracking_quality": "high" if tracking_frames / total_frames > 0.9 else "medium" if tracking_frames / total_frames > 0.7 else "low"
            }
            
        except Exception as e:
            logger.error(f"Tracking report generation failed: {e}")
            return {"error": str(e)}

# Placeholder classes
class SimpleLandmarkDetector:
    """Simple landmark detector placeholder."""
    def detect(self, face_region: np.ndarray) -> List[Tuple[int, int]]:
        """Detect facial landmarks (placeholder implementation)."""
        # Return dummy landmarks
        h, w = face_region.shape[:2]
        return [
            (w//4, h//3),      # Left eye
            (3*w//4, h//3),    # Right eye
            (w//2, 2*h//3),    # Nose
            (w//3, 5*h//6),    # Left mouth corner
            (2*w//3, 5*h//6)   # Right mouth corner
        ]

# Export the main class
__all__ = ["SpeakerTrackingSystem", "FaceDetector", "ObjectTracker", "AutoFramer", "TrackingConfig"]


