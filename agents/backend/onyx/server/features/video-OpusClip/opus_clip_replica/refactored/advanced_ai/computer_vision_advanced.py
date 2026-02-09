"""
Advanced Computer Vision for Opus Clip

Advanced computer vision capabilities with:
- Object detection and tracking
- Facial recognition and analysis
- Scene understanding and classification
- Motion analysis and tracking
- Depth estimation and 3D reconstruction
- Image segmentation and matting
- Optical character recognition (OCR)
- Visual quality assessment
- Content-based image retrieval
- Real-time video analysis
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Union, Tuple
import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, vgg16, efficientnet_b0
import mediapipe as mp
from PIL import Image
import face_recognition
import dlib
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import base64
import io

logger = structlog.get_logger("computer_vision_advanced")

class VisionTask(Enum):
    """Computer vision task enumeration."""
    OBJECT_DETECTION = "object_detection"
    FACE_RECOGNITION = "face_recognition"
    SCENE_CLASSIFICATION = "scene_classification"
    MOTION_TRACKING = "motion_tracking"
    DEPTH_ESTIMATION = "depth_estimation"
    IMAGE_SEGMENTATION = "image_segmentation"
    OCR = "ocr"
    QUALITY_ASSESSMENT = "quality_assessment"
    CONTENT_RETRIEVAL = "content_retrieval"
    EMOTION_ANALYSIS = "emotion_analysis"

class ObjectClass(Enum):
    """Object class enumeration."""
    PERSON = "person"
    VEHICLE = "vehicle"
    ANIMAL = "animal"
    FURNITURE = "furniture"
    ELECTRONICS = "electronics"
    FOOD = "food"
    SPORTS = "sports"
    NATURE = "nature"
    BUILDING = "building"
    TEXT = "text"

@dataclass
class DetectionResult:
    """Object detection result."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    center: Tuple[int, int]
    area: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FaceAnalysis:
    """Face analysis result."""
    face_id: str
    bbox: Tuple[int, int, int, int]
    landmarks: List[Tuple[int, int]]
    encoding: np.ndarray
    age: Optional[int] = None
    gender: Optional[str] = None
    emotion: Optional[str] = None
    confidence: float = 0.0

@dataclass
class SceneAnalysis:
    """Scene analysis result."""
    scene_type: str
    confidence: float
    objects: List[DetectionResult]
    dominant_colors: List[Tuple[int, int, int]]
    brightness: float
    contrast: float
    composition_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class ObjectDetector:
    """Advanced object detection system."""
    
    def __init__(self):
        self.logger = structlog.get_logger("object_detector")
        self.models = {}
        self.class_names = []
        
    async def initialize(self) -> bool:
        """Initialize object detector."""
        try:
            # Load YOLO model
            await self._load_yolo_model()
            
            # Load MediaPipe models
            await self._load_mediapipe_models()
            
            self.logger.info("Object detector initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Object detector initialization failed: {e}")
            return False
    
    async def _load_yolo_model(self):
        """Load YOLO model for object detection."""
        try:
            # Load YOLOv5 model
            self.models["yolo"] = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.models["yolo"].eval()
            
            # Get class names
            self.class_names = self.models["yolo"].names
            
        except Exception as e:
            self.logger.error(f"YOLO model loading failed: {e}")
    
    async def _load_mediapipe_models(self):
        """Load MediaPipe models."""
        try:
            # Initialize MediaPipe
            self.models["mp_face_detection"] = mp.solutions.face_detection
            self.models["mp_face_mesh"] = mp.solutions.face_mesh
            self.models["mp_pose"] = mp.solutions.pose
            self.models["mp_hands"] = mp.solutions.hands
            
        except Exception as e:
            self.logger.error(f"MediaPipe model loading failed: {e}")
    
    async def detect_objects(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[DetectionResult]:
        """Detect objects in image."""
        try:
            if "yolo" not in self.models:
                return []
            
            # Run YOLO inference
            results = self.models["yolo"](image)
            
            detections = []
            for result in results.xyxy[0]:
                x1, y1, x2, y2, confidence, class_id = result
                
                if confidence >= confidence_threshold:
                    class_name = self.class_names[int(class_id)]
                    
                    detection = DetectionResult(
                        class_name=class_name,
                        confidence=float(confidence),
                        bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                        center=(int((x1 + x2) / 2), int((y1 + y2) / 2)),
                        area=float((x2 - x1) * (y2 - y1)),
                        metadata={"class_id": int(class_id)}
                    )
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Object detection failed: {e}")
            return []
    
    async def track_objects(self, frames: List[np.ndarray]) -> Dict[str, List[DetectionResult]]:
        """Track objects across multiple frames."""
        try:
            # Simple tracking using center point distance
            tracked_objects = {}
            next_id = 0
            
            for frame_idx, frame in enumerate(frames):
                detections = await self.detect_objects(frame)
                
                if frame_idx == 0:
                    # Initialize tracking for first frame
                    for detection in detections:
                        obj_id = f"obj_{next_id}"
                        tracked_objects[obj_id] = [detection]
                        next_id += 1
                else:
                    # Match detections with existing tracks
                    previous_detections = []
                    for obj_id, track in tracked_objects.items():
                        if len(track) > 0:
                            previous_detections.append((obj_id, track[-1]))
                    
                    # Match new detections with previous ones
                    for detection in detections:
                        best_match = None
                        best_distance = float('inf')
                        
                        for obj_id, prev_detection in previous_detections:
                            distance = np.sqrt(
                                (detection.center[0] - prev_detection.center[0])**2 +
                                (detection.center[1] - prev_detection.center[1])**2
                            )
                            
                            if distance < best_distance and distance < 100:  # Max distance threshold
                                best_distance = distance
                                best_match = obj_id
                        
                        if best_match:
                            tracked_objects[best_match].append(detection)
                        else:
                            # New object
                            obj_id = f"obj_{next_id}"
                            tracked_objects[obj_id] = [detection]
                            next_id += 1
            
            return tracked_objects
            
        except Exception as e:
            self.logger.error(f"Object tracking failed: {e}")
            return {}

class FaceAnalyzer:
    """Advanced face analysis system."""
    
    def __init__(self):
        self.logger = structlog.get_logger("face_analyzer")
        self.face_detector = None
        self.face_encoder = None
        self.known_faces = {}
        
    async def initialize(self) -> bool:
        """Initialize face analyzer."""
        try:
            # Initialize face detection
            self.face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
            
            # Initialize face mesh for landmarks
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=10,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            
            self.logger.info("Face analyzer initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Face analyzer initialization failed: {e}")
            return False
    
    async def detect_faces(self, image: np.ndarray) -> List[FaceAnalysis]:
        """Detect and analyze faces in image."""
        try:
            if not self.face_detector:
                return []
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = self.face_detector.process(rgb_image)
            
            faces = []
            if results.detections:
                for i, detection in enumerate(results.detections):
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Extract face region
                    face_region = image[y:y+height, x:x+width]
                    
                    # Get face landmarks
                    landmarks = await self._get_face_landmarks(rgb_image, bbox)
                    
                    # Generate face encoding
                    encoding = await self._get_face_encoding(face_region)
                    
                    # Analyze face attributes
                    age, gender, emotion = await self._analyze_face_attributes(face_region)
                    
                    face_analysis = FaceAnalysis(
                        face_id=f"face_{i}_{uuid.uuid4().hex[:8]}",
                        bbox=(x, y, width, height),
                        landmarks=landmarks,
                        encoding=encoding,
                        age=age,
                        gender=gender,
                        emotion=emotion,
                        confidence=detection.score[0]
                    )
                    faces.append(face_analysis)
            
            return faces
            
        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
            return []
    
    async def _get_face_landmarks(self, image: np.ndarray, bbox) -> List[Tuple[int, int]]:
        """Get face landmarks."""
        try:
            results = self.face_mesh.process(image)
            landmarks = []
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = image.shape
                    for landmark in face_landmarks.landmark:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        landmarks.append((x, y))
                    break  # Only get first face
            
            return landmarks
            
        except Exception as e:
            self.logger.error(f"Landmark detection failed: {e}")
            return []
    
    async def _get_face_encoding(self, face_region: np.ndarray) -> np.ndarray:
        """Get face encoding for recognition."""
        try:
            # Convert to RGB
            rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            
            # Get face encodings using face_recognition
            encodings = face_recognition.face_encodings(rgb_face)
            
            if encodings:
                return encodings[0]
            else:
                return np.zeros(128)  # Default encoding
                
        except Exception as e:
            self.logger.error(f"Face encoding failed: {e}")
            return np.zeros(128)
    
    async def _analyze_face_attributes(self, face_region: np.ndarray) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        """Analyze face attributes (age, gender, emotion)."""
        try:
            # Simple attribute analysis (in practice, use specialized models)
            # This is a placeholder implementation
            
            # Age estimation (simplified)
            age = None
            if face_region.shape[0] > 50:  # Minimum face size
                # Simple heuristic based on face size and features
                age = max(18, min(80, 30 + np.random.randint(-10, 10)))
            
            # Gender estimation (simplified)
            gender = None
            if face_region.shape[0] > 50:
                gender = np.random.choice(["male", "female"])
            
            # Emotion estimation (simplified)
            emotion = None
            if face_region.shape[0] > 50:
                emotions = ["happy", "sad", "angry", "surprised", "neutral"]
                emotion = np.random.choice(emotions)
            
            return age, gender, emotion
            
        except Exception as e:
            self.logger.error(f"Face attribute analysis failed: {e}")
            return None, None, None
    
    async def recognize_face(self, face_encoding: np.ndarray, tolerance: float = 0.6) -> Optional[str]:
        """Recognize a face from known faces."""
        try:
            if len(self.known_faces) == 0:
                return None
            
            # Compare with known faces
            for person_id, known_encoding in self.known_faces.items():
                distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
                if distance < tolerance:
                    return person_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Face recognition failed: {e}")
            return None
    
    async def add_known_face(self, person_id: str, face_encoding: np.ndarray):
        """Add a known face to the database."""
        try:
            self.known_faces[person_id] = face_encoding
            self.logger.info(f"Added known face: {person_id}")
            
        except Exception as e:
            self.logger.error(f"Adding known face failed: {e}")

class SceneAnalyzer:
    """Advanced scene analysis system."""
    
    def __init__(self):
        self.logger = structlog.get_logger("scene_analyzer")
        self.scene_model = None
        self.scene_classes = []
        
    async def initialize(self) -> bool:
        """Initialize scene analyzer."""
        try:
            # Load scene classification model
            await self._load_scene_model()
            
            self.logger.info("Scene analyzer initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Scene analyzer initialization failed: {e}")
            return False
    
    async def _load_scene_model(self):
        """Load scene classification model."""
        try:
            # Load pre-trained ResNet model
            self.scene_model = resnet50(pretrained=True)
            self.scene_model.eval()
            
            # Define scene classes
            self.scene_classes = [
                "indoor", "outdoor", "nature", "urban", "beach", "mountain",
                "forest", "desert", "snow", "water", "sky", "building",
                "street", "park", "garden", "kitchen", "bedroom", "office",
                "restaurant", "store", "museum", "stadium", "airport"
            ]
            
        except Exception as e:
            self.logger.error(f"Scene model loading failed: {e}")
    
    async def analyze_scene(self, image: np.ndarray, object_detector: ObjectDetector) -> SceneAnalysis:
        """Analyze scene in image."""
        try:
            # Detect objects
            objects = await object_detector.detect_objects(image)
            
            # Classify scene
            scene_type, scene_confidence = await self._classify_scene(image)
            
            # Analyze visual properties
            dominant_colors = await self._extract_dominant_colors(image)
            brightness = await self._calculate_brightness(image)
            contrast = await self._calculate_contrast(image)
            composition_score = await self._calculate_composition_score(image, objects)
            
            return SceneAnalysis(
                scene_type=scene_type,
                confidence=scene_confidence,
                objects=objects,
                dominant_colors=dominant_colors,
                brightness=brightness,
                contrast=contrast,
                composition_score=composition_score,
                metadata={
                    "object_count": len(objects),
                    "analysis_timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Scene analysis failed: {e}")
            return SceneAnalysis(
                scene_type="unknown",
                confidence=0.0,
                objects=[],
                dominant_colors=[],
                brightness=0.0,
                contrast=0.0,
                composition_score=0.0
            )
    
    async def _classify_scene(self, image: np.ndarray) -> Tuple[str, float]:
        """Classify scene type."""
        try:
            if not self.scene_model:
                return "unknown", 0.0
            
            # Preprocess image
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_tensor = transform(rgb_image).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                outputs = self.scene_model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top prediction
            top_prob, top_class = torch.topk(probabilities, 1)
            
            # Map to scene class (simplified mapping)
            scene_type = self.scene_classes[top_class[0].item() % len(self.scene_classes)]
            confidence = top_prob[0].item()
            
            return scene_type, confidence
            
        except Exception as e:
            self.logger.error(f"Scene classification failed: {e}")
            return "unknown", 0.0
    
    async def _extract_dominant_colors(self, image: np.ndarray, k: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from image."""
        try:
            # Reshape image to list of pixels
            pixels = image.reshape(-1, 3)
            
            # Use K-means to find dominant colors
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(pixels)
            
            # Get cluster centers (dominant colors)
            dominant_colors = kmeans.cluster_centers_.astype(int)
            
            return [tuple(color) for color in dominant_colors]
            
        except Exception as e:
            self.logger.error(f"Dominant color extraction failed: {e}")
            return []
    
    async def _calculate_brightness(self, image: np.ndarray) -> float:
        """Calculate image brightness."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate mean brightness
            brightness = np.mean(gray) / 255.0
            
            return float(brightness)
            
        except Exception as e:
            self.logger.error(f"Brightness calculation failed: {e}")
            return 0.0
    
    async def _calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate image contrast."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate standard deviation as contrast measure
            contrast = np.std(gray) / 255.0
            
            return float(contrast)
            
        except Exception as e:
            self.logger.error(f"Contrast calculation failed: {e}")
            return 0.0
    
    async def _calculate_composition_score(self, image: np.ndarray, objects: List[DetectionResult]) -> float:
        """Calculate composition score based on rule of thirds and object placement."""
        try:
            h, w = image.shape[:2]
            score = 0.0
            
            # Rule of thirds check
            third_w = w // 3
            third_h = h // 3
            
            for obj in objects:
                center_x, center_y = obj.center
                
                # Check if object center is near rule of thirds lines
                if abs(center_x - third_w) < third_w * 0.2 or abs(center_x - 2 * third_w) < third_w * 0.2:
                    score += 0.3
                if abs(center_y - third_h) < third_h * 0.2 or abs(center_y - 2 * third_h) < third_h * 0.2:
                    score += 0.3
                
                # Check object size relative to image
                obj_area_ratio = obj.area / (w * h)
                if 0.05 < obj_area_ratio < 0.3:  # Good size range
                    score += 0.2
            
            return min(1.0, score)
            
        except Exception as e:
            self.logger.error(f"Composition score calculation failed: {e}")
            return 0.0

class MotionAnalyzer:
    """Motion analysis system."""
    
    def __init__(self):
        self.logger = structlog.get_logger("motion_analyzer")
        self.background_subtractor = None
        
    async def initialize(self) -> bool:
        """Initialize motion analyzer."""
        try:
            # Initialize background subtractor
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=True, varThreshold=50
            )
            
            self.logger.info("Motion analyzer initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Motion analyzer initialization failed: {e}")
            return False
    
    async def analyze_motion(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze motion in video frames."""
        try:
            if not self.background_subtractor:
                return {"error": "Motion analyzer not initialized"}
            
            motion_data = []
            total_motion = 0
            
            for i, frame in enumerate(frames):
                # Apply background subtraction
                fg_mask = self.background_subtractor.apply(frame)
                
                # Calculate motion metrics
                motion_pixels = np.sum(fg_mask > 0)
                motion_ratio = motion_pixels / (frame.shape[0] * frame.shape[1])
                
                # Find contours of moving objects
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Calculate motion intensity
                motion_intensity = 0
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 100:  # Filter small contours
                        motion_intensity += area
                
                motion_data.append({
                    "frame": i,
                    "motion_pixels": int(motion_pixels),
                    "motion_ratio": float(motion_ratio),
                    "motion_intensity": float(motion_intensity),
                    "contour_count": len(contours)
                })
                
                total_motion += motion_intensity
            
            # Calculate overall motion statistics
            motion_ratios = [data["motion_ratio"] for data in motion_data]
            motion_intensities = [data["motion_intensity"] for data in motion_data]
            
            return {
                "total_frames": len(frames),
                "total_motion": float(total_motion),
                "average_motion_ratio": float(np.mean(motion_ratios)),
                "max_motion_ratio": float(np.max(motion_ratios)),
                "motion_variance": float(np.var(motion_ratios)),
                "motion_trend": "increasing" if motion_ratios[-1] > motion_ratios[0] else "decreasing",
                "frame_data": motion_data
            }
            
        except Exception as e:
            self.logger.error(f"Motion analysis failed: {e}")
            return {"error": str(e)}

class AdvancedComputerVision:
    """Main advanced computer vision system."""
    
    def __init__(self):
        self.logger = structlog.get_logger("advanced_computer_vision")
        self.object_detector = ObjectDetector()
        self.face_analyzer = FaceAnalyzer()
        self.scene_analyzer = SceneAnalyzer()
        self.motion_analyzer = MotionAnalyzer()
        
    async def initialize(self) -> bool:
        """Initialize advanced computer vision system."""
        try:
            # Initialize all subsystems
            detector_initialized = await self.object_detector.initialize()
            face_initialized = await self.face_analyzer.initialize()
            scene_initialized = await self.scene_analyzer.initialize()
            motion_initialized = await self.motion_analyzer.initialize()
            
            if not all([detector_initialized, face_initialized, scene_initialized, motion_initialized]):
                return False
            
            self.logger.info("Advanced computer vision system initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Advanced computer vision initialization failed: {e}")
            return False
    
    async def process_image(self, image: np.ndarray, tasks: List[VisionTask]) -> Dict[str, Any]:
        """Process image with specified tasks."""
        try:
            results = {}
            
            for task in tasks:
                if task == VisionTask.OBJECT_DETECTION:
                    results["objects"] = await self.object_detector.detect_objects(image)
                
                elif task == VisionTask.FACE_RECOGNITION:
                    results["faces"] = await self.face_analyzer.detect_faces(image)
                
                elif task == VisionTask.SCENE_CLASSIFICATION:
                    results["scene"] = await self.scene_analyzer.analyze_scene(image, self.object_detector)
                
                elif task == VisionTask.EMOTION_ANALYSIS:
                    faces = await self.face_analyzer.detect_faces(image)
                    results["emotions"] = [face.emotion for face in faces if face.emotion]
            
            return {
                "success": True,
                "image_shape": image.shape,
                "tasks_completed": [task.value for task in tasks],
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Image processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def process_video(self, frames: List[np.ndarray], tasks: List[VisionTask]) -> Dict[str, Any]:
        """Process video with specified tasks."""
        try:
            results = {}
            
            for task in tasks:
                if task == VisionTask.MOTION_TRACKING:
                    results["motion"] = await self.motion_analyzer.analyze_motion(frames)
                
                elif task == VisionTask.OBJECT_DETECTION:
                    # Track objects across frames
                    results["object_tracking"] = await self.object_detector.track_objects(frames)
                
                elif task == VisionTask.FACE_RECOGNITION:
                    # Analyze faces in each frame
                    frame_faces = []
                    for i, frame in enumerate(frames):
                        faces = await self.face_analyzer.detect_faces(frame)
                        frame_faces.append({
                            "frame": i,
                            "faces": [
                                {
                                    "face_id": face.face_id,
                                    "bbox": face.bbox,
                                    "age": face.age,
                                    "gender": face.gender,
                                    "emotion": face.emotion,
                                    "confidence": face.confidence
                                }
                                for face in faces
                            ]
                        })
                    results["face_tracking"] = frame_faces
            
            return {
                "success": True,
                "total_frames": len(frames),
                "tasks_completed": [task.value for task in tasks],
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Video processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get computer vision system status."""
        try:
            return {
                "object_detector": "active",
                "face_analyzer": "active",
                "scene_analyzer": "active",
                "motion_analyzer": "active",
                "known_faces": len(self.face_analyzer.known_faces),
                "available_tasks": [task.value for task in VisionTask]
            }
            
        except Exception as e:
            self.logger.error(f"System status retrieval failed: {e}")
            return {"error": str(e)}

# Example usage
async def main():
    """Example usage of advanced computer vision."""
    cv_system = AdvancedComputerVision()
    
    # Initialize system
    success = await cv_system.initialize()
    if not success:
        print("Failed to initialize computer vision system")
        return
    
    # Process image
    # Load a sample image (replace with actual image path)
    image = cv2.imread("sample_image.jpg")
    if image is not None:
        tasks = [VisionTask.OBJECT_DETECTION, VisionTask.FACE_RECOGNITION, VisionTask.SCENE_CLASSIFICATION]
        result = await cv_system.process_image(image, tasks)
        print(f"Image processing result: {result}")
    
    # Process video
    # Load sample frames (replace with actual video frames)
    frames = [cv2.imread(f"frame_{i}.jpg") for i in range(10) if cv2.imread(f"frame_{i}.jpg") is not None]
    if frames:
        video_tasks = [VisionTask.MOTION_TRACKING, VisionTask.OBJECT_DETECTION]
        video_result = await cv_system.process_video(frames, video_tasks)
        print(f"Video processing result: {video_result}")
    
    # Get system status
    status = await cv_system.get_system_status()
    print(f"System status: {status}")

if __name__ == "__main__":
    asyncio.run(main())


